"""Do per-item gains add up? Alternating A/B of {projections, row kernels} on/off, one process.

proj: oneDNN prepacked projections (stand-in for any faster projection GEMM; MKL when off).
rows: the x86 build of rows.h (native_probe/rows_x86.c) through the engine's own fused_rope/geglu/norm switches.
Attention stays on the tiled SDPA path in every variant.
python compound.py THREADS ROWS_SO [ROUNDS]
"""
import ctypes
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench import inputs, workloads  # noqa: E402

threads, so = int(sys.argv[1]), sys.argv[2]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 7
torch.set_num_threads(threads)
from transformers import AutoModel  # noqa: E402
import packed_encoders as pe  # noqa: E402

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa").float().eval()
pe.pack(model, engine=pe.ModernBertCPU(threads=threads))
eng = pe.get_engine(model)
att = eng.attention
lib = ctypes.CDLL(so)
P, I = ctypes.c_void_p, ctypes.c_int
lib.pe_rope_layout.argtypes, lib.pe_rope_layout.restype = [P] * 4 + [I] * 3, I
lib.pe_geglu.argtypes, lib.pe_geglu.restype = [P] * 2 + [I] * 3, I
lib.pe_add_layer_norm.argtypes, lib.pe_add_layer_norm.restype = [P] * 5 + [I] * 2 + [ctypes.c_float, I], I
att.library = lib
att.native_global = False
att.local = lambda q, k, v, lengths, boundaries: att._sdpa(q, k, v, lengths, local=True)

weights = [p for l in eng.layers for p in (l.attn.Wqkv.weight, l.attn.Wo.weight, l.mlp.Wi.weight, l.mlp.Wo.weight)]
packed = {id(w): torch.ops.mkldnn._reorder_linear_weight(w, 32) for w in weights}


def dnn(x, w, b=None):
    y = torch.ops.mkldnn._linear_pointwise(x.view(-1, x.shape[-1]), packed[id(w)], b, "none", [], "")
    return y.view(*x.shape[:-1], -1)


core = eng._core
VARIANTS = {"base": (False, False), "proj": (True, False), "rows": (False, True), "proj+rows": (True, True)}


def configure(proj, rows):
    att.fused_rope = att.fused_geglu = att.fused_norm = rows
    lin = dnn if proj else F.linear
    eng._core = lambda i, l, p: core(i, l, p, linear=lin)


for name in ("b1_s16_pad32", "b1_s32", "b8_s32"):
    seq, lengths = workloads()[name]
    ids, mask = inputs(seq, lengths)
    m = mask.bool()
    reps = 10 if threads > 1 or name != "b8_s32" else 3
    times = {v: [] for v in VARIANTS}
    outs = {}
    with torch.inference_mode():
        for r in range(rounds + 1):
            for v in (list(VARIANTS) if r % 2 == 0 else list(VARIANTS)[::-1]):
                configure(*VARIANTS[v])
                t = time.perf_counter()
                for _ in range(reps):
                    out = model(input_ids=ids, attention_mask=mask).last_hidden_state
                if r:
                    times[v].append((time.perf_counter() - t) / reps * 1e3)
                outs[v] = out[m].clone()
    med = {v: statistics.median(t) for v, t in times.items()}
    sp, sr, both = med["base"] - med["proj"], med["base"] - med["rows"], med["base"] - med["proj+rows"]
    rel = max(((outs[v] - outs["base"]).norm() / outs["base"].norm()).item() for v in VARIANTS)
    print(f"{name:13s} {threads}t | base {med['base']:7.2f}  proj {med['proj']:7.2f}  rows {med['rows']:7.2f}  "
          f"proj+rows {med['proj+rows']:7.2f} ms | saved: proj {sp:6.2f} + rows {sr:5.2f} = {sp + sr:6.2f} "
          f"predicted vs {both:6.2f} measured | max relL2 {rel:.1e}", flush=True)
