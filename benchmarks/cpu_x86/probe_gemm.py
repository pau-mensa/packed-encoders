"""Hardware ceilings: DRAM read bandwidth and the 88-projection ModernBERT-base GEMM chain per backend.

Weights are 22 distinct layer sets (as in the real model), so small-M timings include streaming
them from DRAM rather than hitting L3 with one reused matrix.
"""
import sys
import time

import torch
import torch.nn.functional as F

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 16
Ms = [int(m) for m in sys.argv[2].split(",")] if len(sys.argv) > 2 else [16, 32, 256, 4096, 32768]
torch.set_num_threads(threads)


def bench(fn, budget=1.0, max_reps=200):
    fn()
    t = time.perf_counter()
    fn()
    one = time.perf_counter() - t
    reps = max(3, min(max_reps, int(budget / max(one, 1e-6))))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


big = torch.ones(512 * 1024 * 1024)
t = bench(lambda: big.sum(), max_reps=10)
print(f"DRAM read {big.numel() * 4 / t / 1e9:6.1f} GB/s (2 GB sum, {threads} threads)")
del big

LAYERS = 22
shapes = [(2304, 768), (768, 768), (2304, 768), (768, 1152)]  # Wqkv, Wo, Wi, Wmo as [N, K]
flops_per_token = 2 * sum(n * k for n, k in shapes) * LAYERS
fp32 = [[torch.randn(n, k) * 0.02 for n, k in shapes] for _ in range(LAYERS)]
bf16 = [[w.bfloat16() for w in ws] for ws in fp32]
print(f"projection weights: fp32 {sum(w.numel() for ws in fp32 for w in ws) * 4 / 1e6:.0f} MB")

print(f"{'M':>6} | {'mkl fp32':>17} | {'onednn fp32':>17} | {'onednn bf16':>17}   (ms for 88 projections, TFLOPS)")
for M in Ms:
    xs = {k: torch.randn(M, k) for k in (768, 1152)}
    xb = {k: v.bfloat16() for k, v in xs.items()}

    def chain(ws, xx, fn=F.linear):
        for layer in ws:
            for w in layer:
                fn(xx[w.shape[1]], w)

    packed = [[torch.ops.mkldnn._reorder_linear_weight(w, M) for w in ws] for ws in fp32]

    def dnn(x, w):
        return torch.ops.mkldnn._linear_pointwise(x, w, None, "none", [], "")

    res = []
    for ws, xx, fn in ((fp32, xs, F.linear), (packed, xs, dnn), (bf16, xb, F.linear)):
        t = bench(lambda: chain(ws, xx, fn))
        res.append(f"{t * 1e3:8.2f} ({M * flops_per_token / t / 1e12:4.2f})")
    print(f"{M:>6} | " + " | ".join(res), flush=True)
