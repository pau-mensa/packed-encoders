"""Packed engine accuracy on real text vs the HF FP32 forward: the metrics of onnx_accuracy_text.py.

python packed_accuracy_text.py TEXT_FILE [THREADS]   (e.g. README.md 8, pinned with taskset -c 0-7)
"""
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import packed_encoders as pe

threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
torch.set_num_threads(threads)
paras = [p.strip() for p in re.split(r"\n\s*\n", open(sys.argv[1]).read()) if len(p.strip()) > 80][:24]
tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
enc = tok(paras, padding=True, truncation=True, max_length=512, return_tensors="pt")
ids, mask = enc["input_ids"], enc["attention_mask"]
m = mask.bool()
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa").float().eval()
with torch.no_grad():
    ref = model(input_ids=ids, attention_mask=mask).last_hidden_state
refp = (ref * m[..., None]).sum(1) / m.sum(1, keepdim=True)
print(f"{len(paras)} real-text paragraphs, {int(m.sum())} tokens, lengths {m.sum(1).min().item()}-{m.sum(1).max().item()}")
for precision in ("fp32", "int8"):
    pe.pack(model, engine=pe.ModernBertCPU(threads=threads), precision=precision)
    report = pe.get_engine(model).state.validation_report
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=mask).last_hidden_state
    pe.unpack(model)
    outp = (out * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    cos = F.cosine_similarity(out[m], ref[m], dim=-1)
    pc = F.cosine_similarity(outp, refp, dim=-1)
    print(f"{report.precision:5s} token cos median {cos.median():.5f} p5 {cos.quantile(.05):.4f} min {cos.min():.4f} "
          f"relL2 {((out[m] - ref[m]).norm() / ref[m].norm()).item():.2e} | mean-pool cos min {pc.min():.5f} "
          f"median {pc.median():.5f} | pack gate: median cos {report.median_cosine:.5f}, relL2 {report.relative_l2:.2e}")
