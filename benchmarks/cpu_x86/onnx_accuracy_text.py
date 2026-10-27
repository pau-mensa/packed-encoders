import re, sys, torch, torch.nn.functional as F, onnxruntime as ort
from transformers import AutoModel, AutoTokenizer
torch.set_num_threads(16)
text = open(sys.argv[2]).read()
paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 80][:24]
tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
enc = tok(paras, padding=True, truncation=True, max_length=512, return_tensors="pt")
ids, mask = enc["input_ids"], enc["attention_mask"]; m = mask.bool()
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa").float().eval()
with torch.no_grad(): ref = model(input_ids=ids, attention_mask=mask).last_hidden_state
refp = (ref * m[..., None]).sum(1) / m.sum(1, keepdim=True)
print(f"{len(paras)} real-text paragraphs, {int(m.sum())} tokens, lengths {m.sum(1).min().item()}-{m.sum(1).max().item()}")
for v in ("fp32_opt", "int8", "int8_pc"):
    s = ort.InferenceSession(f"{sys.argv[1]}/{v}.onnx", providers=["CPUExecutionProvider"])
    out = torch.from_numpy(s.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0])
    outp = (out * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    cos = F.cosine_similarity(out[m], ref[m], dim=-1); pc = F.cosine_similarity(outp, refp, dim=-1)
    print(f"{v:9s} token cos median {cos.median():.5f} p5 {cos.quantile(.05):.4f} min {cos.min():.4f} relL2 {((out[m]-ref[m]).norm()/ref[m].norm()).item():.2e} | mean-pool cos min {pc.min():.5f} median {pc.median():.5f}")
