"""Per-stage wall time of the packed CPU engine's FP32 forward (synchronous CPU, so wrappers are exact).

python profile_packed.py THREADS WORKLOAD[,WORKLOAD] [native|mkl|dnn]   (projections; native = the engine's own)
"""
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench import inputs, workloads  # noqa: E402

threads = int(sys.argv[1])
names = sys.argv[2].split(",")
linear_kind = sys.argv[3] if len(sys.argv) > 3 else "native"
torch.set_num_threads(threads)

from transformers import AutoModel  # noqa: E402
import packed_encoders as pe  # noqa: E402

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa").float().eval()
pe.pack(model, engine=pe.ModernBertCPU(threads=threads))
eng = pe.get_engine(model)
acc = defaultdict(float)
calls = defaultdict(int)


def timed(name, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        acc[name] += time.perf_counter() - t
        calls[name] += 1
        return r
    return wrapper


att = eng.attention
for name in ("rope_layout", "local", "global_", "add_norm", "geglu"):
    setattr(att, name, timed(name, getattr(att, name)))
eng.model.embeddings.forward = timed("embeddings", eng.model.embeddings.forward)
eng.layers[0].attn_norm.forward = timed("first_norm", eng.layers[0].attn_norm.forward)

if linear_kind == "dnn":
    def base_linear(x, w, b=None):
        return torch.ops.mkldnn._linear_pointwise(x, w, b, "none", [], "")
elif linear_kind == "native":
    base_linear = eng.attention.linear
else:
    base_linear = F.linear
lin = timed("linear", base_linear)
core = eng._core
eng._core = timed("core", lambda ids, lengths, positions: core(ids, lengths, positions, linear=lin))
eng._packed = timed("packed", eng._packed)

for name in names:
    seq, lengths = workloads()[name]
    ids, mask = inputs(seq, lengths)
    reps = 20 if seq * len(lengths) <= 256 else (2 if seq <= 1024 else 1)
    with torch.inference_mode():
        model(input_ids=ids, attention_mask=mask)  # warm
        acc.clear(); calls.clear()
        t = time.perf_counter()
        for _ in range(reps):
            model(input_ids=ids, attention_mask=mask)
        total = (time.perf_counter() - t) / reps
    per = {k: v / reps * 1e3 for k, v in acc.items()}
    inner = sum(per[k] for k in ("rope_layout", "local", "global_", "add_norm", "geglu", "linear", "embeddings",
                                 "first_norm") if k in per)
    print(f"\n== {name} ({sum(lengths)} live tokens, threads={threads}, linear={linear_kind}) total {total*1e3:.2f} ms")
    for k in ("linear", "global_", "local", "rope_layout", "geglu", "add_norm", "embeddings", "first_norm"):
        if k in per:
            print(f"  {k:12s} {per[k]:9.2f} ms  {per[k] / total / 10:5.1f}%  calls/fwd {calls[k] // reps}")
    print(f"  {'core other':12s} {per['core'] - inner:9.2f} ms   (layout transposes, reshape, glue)")
    print(f"  {'fwd other':12s} {total * 1e3 - per['packed']:9.2f} ms   (_check, masks, index_copy, HF wrapper)")
