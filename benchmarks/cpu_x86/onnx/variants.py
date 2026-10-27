"""Derive optimized and int8 ONNX variants from fp32.onnx."""
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions

d = Path(sys.argv[1])
opts = FusionOptions("bert")
opts.enable_skip_layer_norm = False
m = optimizer.optimize_model(str(d / "fp32.onnx"), model_type="bert", num_heads=12, hidden_size=768,
                             optimization_options=opts, opt_level=0, use_gpu=False)
print("fused ops:", m.get_fused_operator_statistics())
m.save_model_to_file(str(d / "fp32_opt.onnx"), use_external_data_format=True, all_tensors_to_one_file=True)
for name, pc in (("int8", False), ("int8_pc", True)):
    quantize_dynamic(str(d / "fp32_opt.onnx"), str(d / f"{name}.onnx"), weight_type=QuantType.QInt8, per_channel=pc)
    ops = {}
    for n in onnx.load(str(d / f"{name}.onnx"), load_external_data=False).graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(name, sorted(ops.items(), key=lambda kv: -kv[1])[:14])
