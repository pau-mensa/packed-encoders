import sys, time, torch, torch.nn.functional as F
torch.set_num_threads(int(sys.argv[1]))
def bench(fn, reps):
    fn(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts)//2]
with torch.inference_mode():
    for S, B in ((32, 8), (128, 8), (1024, 4), (8192, 1)):
        q, k, v = (torch.randn(B, 12, S, 64) for _ in range(3))
        fl = 4 * B * 12 * S * S * 64
        reps = 30 if S <= 1024 else 3
        t32 = bench(lambda: F.scaled_dot_product_attention(q, k, v), reps)
        qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
        tbf = bench(lambda: F.scaled_dot_product_attention(qb, kb, vb), reps)
        print(f"S={S:5d} B={B}: fp32 {t32*1e3:9.2f} ms ({fl/t32/1e12:4.2f} TFLOPS)  bf16 {tbf*1e3:9.2f} ms ({fl/tbf/1e12:4.2f} TFLOPS)", flush=True)
