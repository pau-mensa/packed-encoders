import sys, time, torch, torch.nn.functional as F
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
def bench(fn, budget=1.0, max_reps=200):
    fn(); t = time.perf_counter(); fn(); one = time.perf_counter() - t
    ts = []
    for _ in range(max(3, min(max_reps, int(budget / max(one, 1e-6))))):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]
L = 22
shapes = [(2304, 768), (768, 768), (2304, 768), (768, 1152)]
fpt = 2 * sum(n * k for n, k in shapes) * L
fp32 = [[torch.randn(n, k) * 0.02 for n, k in shapes] for _ in range(L)]
dnn = lambda x, w: torch.ops.mkldnn._linear_pointwise(x, w, None, "none", [], "")
from torch.ao.nn.quantized.dynamic import Linear as DQLinear
q8 = []
for ws in fp32:
    row = []
    for w in ws:
        m = DQLinear(w.shape[1], w.shape[0], bias_=False, dtype=torch.qint8)
        m.set_weight_bias(torch.quantize_per_channel(w, w.abs().amax(1) / 127, torch.zeros(w.shape[0], dtype=torch.long), 0, torch.qint8), None)
        row.append(m)
    q8.append(row)
print(f"{'M':>6} | {'dnn plain W':>14} | {'dnn packed@32':>14} | {'dnn packed@M':>14} | {'fbgemm int8 dyn':>16}")
for M in [int(m) for m in (sys.argv[2] if len(sys.argv) > 2 else "16,32,256,4096,32768").split(",")]:
    xs = {k: torch.randn(M, k) for k in (768, 1152)}
    p32 = [[torch.ops.mkldnn._reorder_linear_weight(w, 32) for w in ws] for ws in fp32]
    pM = [[torch.ops.mkldnn._reorder_linear_weight(w, M) for w in ws] for ws in fp32]
    def chain(ws, fn):
        for layer in ws:
            for w in layer: fn(xs[w.shape[1]] if not hasattr(w, 'in_features') else xs[w.in_features], w)
    out = []
    for ws, fn in ((fp32, dnn), (p32, dnn), (pM, dnn), (q8, lambda x, m: m(x))):
        t = bench(lambda: chain(ws, fn)); out.append(f"{t*1e3:7.2f} ({M*fpt/t/1e12:4.2f})")
    print(f"{M:>6} | " + " | ".join(out), flush=True)
