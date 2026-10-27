"""Export ModernBERT-base (FP32) to ONNX with dynamic batch/sequence, then derive variants.

Variants:
  fp32.onnx            raw torch.onnx (dynamo) export
  fp32_opt.onnx        + ORT transformers optimizer (model_type=bert; fuses LN/GELU where it can)
  int8.onnx            quantize_dynamic(fp32_opt, QInt8 weights, per-tensor)   -- ORT default recipe
  int8_pc.onnx         quantize_dynamic(fp32_opt, QInt8 weights, per-channel)
"""
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel

out = Path(sys.argv[1])
attn = sys.argv[2] if len(sys.argv) > 2 else "sdpa"
out.mkdir(parents=True, exist_ok=True)

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation=attn).float().eval()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


ids = torch.randint(5, 50000, (2, 64))
mask = torch.ones_like(ids)
mask[0, 40:] = 0
batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq", max=8192)
t = time.time()
with torch.no_grad():
    prog = torch.onnx.export(Wrap(model).eval(), (ids, mask), dynamo=True, opset_version=18,
                             input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
                             dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
                             external_data=True)
prog.optimize()
prog.save(str(out / "fp32.onnx"), external_data=True)
print("export", time.time() - t, "s")

with torch.no_grad():
    ref = Wrap(model)(ids, mask)
import onnxruntime as ort
s = ort.InferenceSession(str(out / "fp32.onnx"), providers=["CPUExecutionProvider"])
got = torch.from_numpy(s.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0])
m = mask.bool()
print("fp32 parity max abs", (got - ref)[m].abs().max().item())
