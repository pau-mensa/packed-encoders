"""Accuracy of the M4 int8 recipe (block Hadamard + GPTQ, per-row int8 activations) on x86, by exact emulation.

Weights are prepared with packed_encoders.cpu.int8's own functions; the integer products run in float64 (exact for
int8 x int8 sums), so a VNNI u8s8 kernel with +128 compensation would produce the same outputs up to the FP32
dequantize. Compares against the FP32 engine on the small workloads and on real text (README paragraphs).
python int8_recipe_accuracy.py THREADS
"""
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench import inputs, workloads  # noqa: E402

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
from transformers import AutoModel, AutoTokenizer  # noqa: E402
import packed_encoders as pe  # noqa: E402
from packed_encoders.cpu.int8 import _calibrate, gptq, rotation  # noqa: E402

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa").float().eval()
pe.pack(model, engine=pe.ModernBertCPU())
eng = pe.get_engine(model)
hidden, width = eng.hidden, model.config.intermediate_size
linears = [(l.attn.Wqkv, l.attn.Wo, l.mlp.Wi, l.mlp.Wo) for l in eng.layers]
weights = [lin.weight for lin in sum(linears, ())]
r_hidden, r_width, r_heads = rotation(hidden, 128), rotation(width, 128), rotation(hidden, 64)
with torch.no_grad():
    grams = _calibrate(eng, weights)
    runtime_rotation = (r_hidden, None, r_hidden, r_width)  # Wo's input arrives rotated (folded into Wv)
    prepared = {}
    for kind, a in enumerate((r_hidden, r_heads, r_hidden, r_width)):
        stack, hessians = [], []
        for index, layer in enumerate(linears):
            w = layer[kind].weight.detach().double()
            if kind == 0:
                w = w.clone()
                w[2 * hidden:] = r_heads.T @ w[2 * hidden:]
            stack.append((w @ a).float())
            hessians.append(a.T @ grams[4 * index + kind] @ a)
        q, scale = gptq(torch.stack(stack), torch.stack(hessians))
        for index, layer in enumerate(linears):
            prepared[id(layer[kind].weight)] = (q[index].double(), scale[index].double(), runtime_rotation[kind])


def int8_linear(x, w, b=None):
    q, ws, rot = prepared[id(w)]
    rows = x.reshape(-1, x.shape[-1]).double()
    if rot is not None:
        rows = rows @ rot
    amax = rows.abs().amax(1, keepdim=True)
    xs = amax / 127
    xq = torch.round(rows * torch.where(amax > 0, 127 / amax, 0))
    y = (xq @ q.T) * xs * ws  # exact integer products, then dequantize
    return y.float().view(*x.shape[:-1], -1)


def compare(name, ids, mask):
    m = mask.bool()
    with torch.inference_mode():
        ref = model(input_ids=ids, attention_mask=mask).last_hidden_state
        core = eng._core
        eng._core = lambda i, l, p: core(i, l, p, linear=int8_linear)
        try:
            out = model(input_ids=ids, attention_mask=mask).last_hidden_state
        finally:
            eng._core = core
    cos = F.cosine_similarity(out[m], ref[m], dim=-1)
    pool = lambda t: (t * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    pc = F.cosine_similarity(pool(out), pool(ref), dim=-1)
    print(f"{name:22s} tok cos median {cos.median():.5f} p5 {cos.quantile(.05):.4f} min {cos.min():.4f} "
          f"relL2 {((out[m] - ref[m]).norm() / ref[m].norm()).item():.2e} | mean-pool cos min {pc.min():.5f} "
          f"median {pc.median():.5f}")


for name in ("b1_s16_pad32", "b1_s32", "b8_s32"):
    seq, lengths = workloads()[name]
    compare(name, *inputs(seq, lengths))
text = open(__file__.rsplit("/", 3)[0] + "/README.md").read()
paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 80][:24]
enc = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")(paras, padding=True, truncation=True,
                                                                  max_length=512, return_tensors="pt")
compare("README (24 paragraphs)", enc["input_ids"], enc["attention_mask"])
