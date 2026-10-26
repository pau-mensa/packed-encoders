"""Run one half of a workload's rows (balanced by tokens) with oneDNN projections; print per-iteration median."""
import json, statistics, sys, time, torch
sys.path.insert(0, sys.argv[1]); from bench import inputs, workloads
name, part, parts, threads, iters = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
torch.set_num_threads(threads)
from transformers import AutoModel; import packed_encoders as pe
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa").float().eval()
pe.pack(model, engine=pe.ModernBertCPU(threads=threads)); eng = pe.get_engine(model); core = eng._core
dnn = lambda x, w, b=None: torch.ops.mkldnn._linear_pointwise(x.view(-1, x.shape[-1]), w, b, "none", [], "").view(*x.shape[:-1], -1)
eng._core = lambda i, l, p: core(i, l, p, linear=dnn)
seq, lengths = workloads()[name]; ids, mask = inputs(seq, lengths)
order = sorted(range(len(lengths)), key=lambda i: -lengths[i]); buckets = [[] for _ in range(parts)]; loads = [0]*parts
for i in order:
    j = loads.index(min(loads)); buckets[j].append(i); loads[j] += lengths[i]
rows = torch.tensor(sorted(buckets[part])); ids, mask = ids[rows], mask[rows]
ts = []
with torch.inference_mode():
    model(input_ids=ids, attention_mask=mask)
    for _ in range(iters):
        t = time.perf_counter(); model(input_ids=ids, attention_mask=mask); ts.append(time.perf_counter() - t)
print(json.dumps({"part": part, "tokens": loads[part], "median_ms": round(statistics.median(ts) * 1e3, 2)}))
