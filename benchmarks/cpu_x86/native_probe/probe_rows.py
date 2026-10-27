import ctypes, sys, time, torch, torch.nn.functional as F
threads = int(sys.argv[1]); torch.set_num_threads(threads)
lib = ctypes.CDLL(sys.argv[2]); P, I = ctypes.c_void_p, ctypes.c_int
lib.pe_rope_layout.argtypes = [P]*4 + [I]*3; lib.pe_geglu.argtypes = [P]*2 + [I]*3
lib.pe_add_layer_norm.argtypes = [P]*5 + [I]*2 + [ctypes.c_float, I]
def bench(fn, budget=0.5):
    fn(); t = time.perf_counter(); fn(); one = time.perf_counter() - t; ts = []
    for _ in range(max(5, min(300, int(budget / max(one, 1e-7))))):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts)//2] * 1e3
H, D, W = 12, 64, 1152
print(f"threads={threads}   per-call ms: torch -> native (speedup); per-forward = x22 rope/geglu, x44 add+LN")
with torch.inference_mode():
    for T in (16, 32, 256, 4096, 32768):
        qkv = torch.randn(1, T, 3*H*D); cos, sin = torch.randn(1, 1, T, D), torch.randn(1, 1, T, D)
        def rope_torch():
            q, k, v = (t.transpose(1, 2) for t in qkv.view(1, T, 3, H, D).unbind(2))
            def rot(t):
                a, b = t.chunk(2, -1); return t * cos + torch.cat((-b, a), -1) * sin
            return rot(q), rot(k), v
        out = torch.empty(3, 1, H, T, D)
        rope_native = lambda: lib.pe_rope_layout(qkv.data_ptr(), cos.data_ptr(), sin.data_ptr(), out.data_ptr(), T, H, threads)
        rope_native(); ref = rope_torch()
        err = max((out[i] - ref[i]).abs().max().item() for i in range(3))
        hid = torch.randn(1, T, 2*W); g = torch.empty(1, T, W)
        def geglu_torch():
            a, gate = hid.chunk(2, -1); return F.gelu(a) * gate
        geglu_native = lambda: lib.pe_geglu(hid.data_ptr(), g.data_ptr(), T, W, threads)
        geglu_native(); gerr = (g - geglu_torch()).abs().max().item()
        x0, br = torch.randn(1, T, 768), torch.randn(1, T, 768); ln = torch.nn.LayerNorm(768, bias=False)
        ln.weight.data.uniform_(0.5, 1.5); x = x0.clone(); o = torch.empty_like(x)
        def norm_torch():
            y = x + br; return y, ln(y)
        def norm_native():
            lib.pe_add_layer_norm(x.data_ptr(), br.data_ptr(), ln.weight.data_ptr(), None, o.data_ptr(), T, 768, ln.eps, threads)
        x.copy_(x0); norm_native(); x.copy_(x0); nerr = (o - norm_torch()[1]).abs().max().item()
        r = [bench(rope_torch), bench(rope_native), bench(geglu_torch), bench(geglu_native), bench(norm_torch), bench(norm_native)]
        print(f"T={T:6d} rope {r[0]:8.3f}->{r[1]:7.3f} ({r[0]/r[1]:5.1f}x) | geglu {r[2]:8.3f}->{r[3]:7.3f} ({r[2]/r[3]:5.1f}x) | "
              f"add+LN {r[4]:8.3f}->{r[5]:7.3f} ({r[4]/r[5]:5.1f}x) | fwd saved {22*(r[0]-r[1]) + 22*(r[2]-r[3]) + 44*(r[4]-r[5]):8.2f} ms"
              f" | maxerr {err:.1e} {gerr:.1e} {nerr:.1e}", flush=True)
