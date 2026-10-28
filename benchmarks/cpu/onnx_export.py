"""Export ModernBERT-base to ONNX as the CPU engine's ONNX baseline, then build int8 variants.

    python benchmarks/cpu/onnx_export.py [--dir benchmarks/cpu/onnx_models] [--only int8,int8_w8]

Needs onnxruntime (measured with 1.31.0), onnx and onnxscript next to torch and transformers.
Each file is built only when missing.

fp32.onnx       dynamo export, opset 18, dynamic batch and sequence
fp32_opt.onnx   onnxruntime.transformers optimizer (model_type bert) on top; SkipLayerNorm fusion
                disabled because it indexes a LayerNorm bias that ModernBERT does not have
Dynamic int8 (int8 weights, per-tensor dynamic int8 activations), all from fp32_opt:
  int8          quantize_dynamic defaults; the baseline the engine's published tables use
  int8_pc       per-channel weights
  int8_lin      only MatMuls with an initializer input; byte-identical to int8_proj, because the two
                rotary MatMuls it also selects cannot be quantized (their initializer is input A)
  int8_lin_u8   int8_lin with uint8 weights (U8U8)
  int8_rr       reduce_range=True (7-bit weights)
  int8_proj     exactly the 88 projection MatMuls
  int8_proj_pc  int8_proj with per-channel weights
Weight-only int8 (MatMulNBits, 8-bit, symmetric, blocks of 128 along K):
  int8_w8       FP32 activations and compute
  int8_w8a8     accuracy_level=4: activations quantized to int8 per block for the int8 kernels
The defaults quantize only the 88 projections too: attention and rotary MatMuls stay FP32 in
every dynamic variant; the default additionally quantizes the embedding Gather.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch import nn

MODEL = "answerdotai/ModernBERT-base"
DYNAMIC = ("int8", "int8_pc", "int8_lin", "int8_lin_u8", "int8_rr", "int8_proj", "int8_proj_pc")
WEIGHT_ONLY = ("int8_w8", "int8_w8a8")


class Wrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def export(path):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()
    # Example longer than the 128-token local window so no length branch is specialised away.
    ids = torch.randint(5, model.config.vocab_size, (2, 256))
    mask = torch.ones_like(ids)
    mask[1, 200:] = 0
    dyn = {"input_ids": {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},
           "attention_mask": {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC}}
    with torch.no_grad():
        torch.onnx.export(Wrap(model).eval(), (ids, mask), path, dynamo=True, opset_version=18,
                          input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
                          dynamic_shapes=dyn, external_data=False)
    return model.config


def optimize(src, dst, config):
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.optimizer import optimize_model

    options = FusionOptions("bert")
    options.enable_skip_layer_norm = False
    m = optimize_model(src, model_type="bert", num_heads=config.num_attention_heads,
                       hidden_size=config.hidden_size, optimization_options=options)
    print("optimizer fused ops:", {k: v for k, v in m.get_fused_operator_statistics().items() if v}, flush=True)
    m.save_model_to_file(dst)


def matmul_sets(path):
    import onnx

    graph = onnx.load(path, load_external_data=False).graph
    inits = {i.name: i for i in graph.initializer}
    consts = {o for n in graph.node if n.op_type == "Constant" for o in n.output}
    matmuls = [n for n in graph.node if n.op_type == "MatMul"]
    lin = [n.name for n in matmuls if any(i in inits or i in consts for i in n.input)]
    proj = [n.name for n in matmuls if any(i in inits and len(inits[i].dims) == 2 for i in n.input)]
    assert len(proj) == 88, len(proj)
    return lin, proj


def dynamic(name, src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    lin, proj = matmul_sets(src)
    kwargs = {"int8": {}, "int8_pc": {"per_channel": True},
              "int8_lin": {"nodes_to_quantize": lin},
              "int8_lin_u8": {"nodes_to_quantize": lin, "weight_type": QuantType.QUInt8},
              "int8_rr": {"reduce_range": True},
              "int8_proj": {"nodes_to_quantize": proj},
              "int8_proj_pc": {"nodes_to_quantize": proj, "per_channel": True}}[name]
    quantize_dynamic(src, dst, **{"weight_type": QuantType.QInt8, **kwargs})


def weight_only(name, src, dst):
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    quantizer = MatMulNBitsQuantizer(onnx.load(src), bits=8, block_size=128, is_symmetric=True,
                                     accuracy_level=4 if name == "int8_w8a8" else None)
    quantizer.process()
    quantizer.model.save_model_to_file(dst, use_external_data_format=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).parent / "onnx_models"))
    ap.add_argument("--only", default=",".join(DYNAMIC + WEIGHT_ONLY), help="int8 variants to build")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    path = lambda name: os.path.join(a.dir, f"{name}.onnx")
    config = None
    if not os.path.exists(path("fp32")):
        t = time.perf_counter()
        config = export(path("fp32"))
        print(f"exported fp32.onnx in {time.perf_counter() - t:.0f}s", flush=True)
    if not os.path.exists(path("fp32_opt")):
        if config is None:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(MODEL)
        optimize(path("fp32"), path("fp32_opt"), config)
    for name in filter(None, a.only.split(",")):
        if os.path.exists(path(name)):
            continue
        build = dynamic if name in DYNAMIC else weight_only if name in WEIGHT_ONLY else None
        if build is None:
            raise SystemExit(f"unknown variant {name!r}; choose from {DYNAMIC + WEIGHT_ONLY}")
        build(name, path("fp32_opt"), path(name))
        print(f"built {name}.onnx ({os.path.getsize(path(name)) / 2**20:.1f} MiB)", flush=True)
    print("EXPORT DONE", flush=True)


if __name__ == "__main__":
    main()
