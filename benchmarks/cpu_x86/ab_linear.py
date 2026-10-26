"""Alternating A/B of the projection backend inside the packed engine's FP32 core (one process).

Variants: mkl (F.linear, today), dnn (oneDNN FP32, original weights), dnn_packed (oneDNN prepacked),
bf16 (oneDNN BF16 weights+activations, FP32 out), int8 (fbgemm dynamic: per-tensor activations,
per-channel max-abs weights; a speed proxy, NOT the GPTQ/Hadamard recipe).
python ab_linear.py THREADS WORKLOADS VARIANTS ROUNDS
"""
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench import inputs, workloads  # noqa: E402

threads, names = int(sys.argv[1]), sys.argv[2].split(",")
variants = sys.argv[3].split(",")
rounds = int(sys.argv[4])
torch.set_num_threads(threads)
from transformers import AutoModel  # noqa: E402
import packed_encoders as pe  # noqa: E402

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa").float().eval()
pe.pack(model, engine=pe.ModernBertCPU(threads=threads))
eng = pe.get_engine(model)
weights = [p for layer in eng.layers for p in (layer.attn.Wqkv.weight, layer.attn.Wo.weight,
                                                layer.mlp.Wi.weight, layer.mlp.Wo.weight)]


def dnn(x, w, b=None):
    return torch.ops.mkldnn._linear_pointwise(x.view(-1, x.shape[-1]), w, b, "none", [], "").view(*x.shape[:-1], -1)


packed = {id(w): torch.ops.mkldnn._reorder_linear_weight(w, 256) for w in weights}
bf = {id(w): w.bfloat16() for w in weights}
from torch.ao.nn.quantized.dynamic import Linear as DQLinear  # noqa: E402
q8 = {}
for w in weights:
    m = DQLinear(w.shape[1], w.shape[0], bias_=False, dtype=torch.qint8)
    scale = w.abs().amax(1).double() / 127
    m.set_weight_bias(torch.quantize_per_channel(w.detach(), scale, torch.zeros(w.shape[0], dtype=torch.long), 0,
                                                 torch.qint8), None)
    q8[id(w)] = m
LINEARS = {
    "mkl": F.linear,
    "dnn": dnn,
    "dnn_packed": lambda x, w, b=None: dnn(x, packed[id(w)], b),
    "bf16": lambda x, w, b=None: F.linear(x.bfloat16(), bf[id(w)]).float(),
    "int8": lambda x, w, b=None: q8[id(w)](x),
}
core = eng._core

for name in names:
    seq, lengths = workloads()[name]
    ids, mask = inputs(seq, lengths)
    m = mask.bool()
    reps = 10 if seq * len(lengths) <= 256 else 1
    times = {v: [] for v in variants}
    outs = {}
    with torch.inference_mode():
        for r in range(rounds + 1):
            for v in (variants if r % 2 == 0 else variants[::-1]):
                lin = LINEARS[v]
                eng._core = lambda i, l, p: core(i, l, p, linear=lin)
                t = time.perf_counter()
                for _ in range(reps):
                    out = model(input_ids=ids, attention_mask=mask).last_hidden_state
                dt = (time.perf_counter() - t) / reps
                if r:  # round 0 is warmup
                    times[v].append(dt)
                outs[v] = out[m]
    ref = outs[variants[0]]
    base = statistics.median(times[variants[0]])
    print(f"\n== {name} ({sum(lengths)} live tokens, {threads} threads, {rounds} alternating rounds x {reps})")
    for v in variants:
        med = statistics.median(times[v])
        cos = F.cosine_similarity(outs[v], ref, dim=-1)
        rel = ((outs[v] - ref).norm() / ref.norm()).item()
        print(f"  {v:11s} {med * 1e3:10.2f} ms  {base / med:5.2f}x   vs {variants[0]}: median cos {cos.median():.6f} "
              f"min cos {cos.min():.4f} relL2 {rel:.2e}")
