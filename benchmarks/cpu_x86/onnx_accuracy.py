import sys, numpy as np, torch, torch.nn.functional as F, onnxruntime as ort
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from bench import inputs, workloads  # argv[1]: ONNX models dir
from transformers import AutoModel
torch.set_num_threads(16)
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa").float().eval()
so = ort.SessionOptions(); so.intra_op_num_threads = 16
sess = {v: ort.InferenceSession(f"{sys.argv[1]}/{v}.onnx", sess_options=so, providers=["CPUExecutionProvider"]) for v in ("fp32_opt", "int8", "int8_pc")}
for name, rows in (("b1_s16_pad32", 1), ("b8_s32", 8), ("b32_rand_pad1024", 6)):
    seq, lengths = workloads()[name]; ids, mask = inputs(seq, lengths); ids, mask = ids[:rows], mask[:rows]; m = mask.bool()
    with torch.no_grad(): ref = model(input_ids=ids, attention_mask=mask).last_hidden_state
    refm = ref[m]; refp = (ref * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    for v, s in sess.items():
        out = torch.from_numpy(s.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0])
        outp = (out * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        cos = F.cosine_similarity(out[m], refm, dim=-1)
        print(f"{name:18s} {v:9s} tok cos median {cos.median():.5f} min {cos.min():.4f} relL2 {((out[m]-refm).norm()/refm.norm()).item():.2e}  mean-pool cos min {F.cosine_similarity(outp, refp, dim=-1).min():.5f}")
