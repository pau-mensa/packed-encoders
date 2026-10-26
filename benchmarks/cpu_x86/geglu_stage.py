"""Per-stage timing (GeGLU and each projection kind) with the native GeGLU toggled; python geglu_stage.py THREADS WORKLOAD"""
import sys, time, statistics, torch, torch.nn.functional as F
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from bench import inputs, workloads
threads = int(sys.argv[1]); torch.set_num_threads(threads)
from transformers import AutoModel; import packed_encoders as pe
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa").float().eval()
pe.pack(model, engine=pe.ModernBertCPU(threads=threads)); eng = pe.get_engine(model); att = eng.attention
kind = {}
for l in eng.layers:
    for k, mod in (("Wqkv", l.attn.Wqkv), ("Wo", l.attn.Wo), ("Wi", l.mlp.Wi), ("Wmo", l.mlp.Wo)): kind[id(mod.weight)] = k
acc = defaultdict(float)
def lin(x, w, b=None):
    t = time.perf_counter(); y = F.linear(x, w, b); acc[kind[id(w)]] += time.perf_counter() - t; return y
g = att.geglu
def geglu(h):
    t = time.perf_counter(); y = g(h); acc["geglu"] += time.perf_counter() - t; return y
att.geglu = geglu
core = eng._core; eng._core = lambda i, l, p: core(i, l, p, linear=lin)
name = sys.argv[2]; seq, lengths = workloads()[name]; ids, mask = inputs(seq, lengths)
res = {False: defaultdict(list), True: defaultdict(list)}
reps = 10
with torch.inference_mode():
    for r in range(13):
        for fused in ((False, True) if r % 2 else (True, False)):
            att.fused_geglu = fused; acc.clear()
            t = time.perf_counter()
            for _ in range(reps): model(input_ids=ids, attention_mask=mask)
            total = (time.perf_counter() - t) / reps
            if r:
                res[fused]["total"].append(total)
                for k, v in acc.items(): res[fused][k].append(v / reps)
print(f"{name} {threads}t   torch-gelu -> native-gelu (ms per forward, medians of 12 alternating rounds)")
for k in ("total", "geglu", "Wmo", "Wi", "Wqkv", "Wo"):
    a, b = statistics.median(res[False][k]) * 1e3, statistics.median(res[True][k]) * 1e3
    print(f"  {k:6s} {a:8.3f} -> {b:8.3f}  ({b - a:+.3f})")
