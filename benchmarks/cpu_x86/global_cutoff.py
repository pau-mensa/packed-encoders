"""Native full-window vs SDPA global attention around the 128-token cutoff; python global_cutoff.py THREADS"""
import sys, time, torch
threads = int(sys.argv[1]); torch.set_num_threads(threads)
from packed_encoders.cpu.attention import CPUAttention
a = CPUAttention("auto", 64, 64, threads)
def bench(fn, budget=0.4):
    fn(); t = time.perf_counter(); fn(); one = time.perf_counter() - t; ts = []
    for _ in range(max(5, min(2000, int(budget / max(one, 1e-7))))):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2] * 1e6
with torch.inference_mode():
    for L, n in ((64, 4), (128, 4), (192, 2), (256, 2), (512, 1)):
        lengths = (L,) * n
        q, k, v = [torch.randn(1, 12, L * n, 64) for _ in range(3)]
        b = torch.tensor(lengths, dtype=torch.int32)
        nat = bench(lambda: a._native(q, k, v, lengths, b, L))
        sd = bench(lambda: a._sdpa(q, k, v, lengths, local=False))
        print(f"{threads}t {n} x {L:4d}: native {nat:8.1f} us  sdpa {sd:8.1f} us  sdpa/native {sd / nat:4.2f}")
