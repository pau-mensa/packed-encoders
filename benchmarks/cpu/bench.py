"""Stock HF SDPA vs the packed CPU engine on ModernBERT-base, FP32, one process.

    python benchmarks/cpu/bench.py --threads 8 --out results/t8.json [--profile] [--only b1_s32,b8_s32]

Workloads follow the CPU engine's published ones (random token ids, seed 0); b1_s1024 and
b1_s4096 add single long documents, b32_s1024 a full batch. Per workload: median ms over
--trials after one warm call, parity vs HF (token cosine min over real tokens), and with
--profile the engine's time per op (wall clock around each CPUAttention method and the
projections).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

MODEL = "answerdotai/ModernBERT-base"


def workloads(vocab):
    g = torch.Generator().manual_seed(0)
    out = {}

    def make(name, b, s, lengths=None):
        ids = torch.randint(5, vocab, (b, s), generator=g)
        lengths = lengths if lengths is not None else [s] * b
        mask = (torch.arange(s)[None] < torch.tensor(lengths)[:, None]).long()
        out[name] = (ids, mask)

    make("b1_s16_pad32", 1, 32, [16])
    make("b1_s32", 1, 32)
    make("b8_s32", 8, 32)
    make("b1_s1024", 1, 1024)
    make("b32_rand_pad1024", 32, 1024, torch.randint(1, 1025, (32,), generator=g).tolist())
    make("b1_s4096", 1, 4096)
    make("b32_s1024", 32, 1024)  # appended last so the earlier workloads keep their ids
    return out


def timed(fn, trials):
    fn()
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(samples), 100 * (max(samples) - min(samples)) / statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--skip-hf", default="", help="comma list of workloads to skip HF on")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    from transformers import AutoModel
    import packed_encoders as pe

    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()
    loads = workloads(model.config.vocab_size)
    if a.only:
        loads = {k: v for k, v in loads.items() if k in a.only.split(",")}
    skip_hf = set(filter(None, a.skip_hf.split(",")))
    res = {"threads": a.threads, "torch_threads": torch.get_num_threads(),
           "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"), "rows": {}}

    hf = {}
    with torch.inference_mode():
        for name, (ids, mask) in loads.items():
            if name in skip_hf:
                continue
            ms, spread = timed(lambda: model(input_ids=ids, attention_mask=mask), a.trials)
            hf[name] = model(input_ids=ids, attention_mask=mask).last_hidden_state
            res["rows"].setdefault(name, {})["hf_ms"] = ms
            res["rows"][name]["hf_spread_pct"] = spread
            print(f"[hf ] {name:18s} {ms:9.2f} ms (spread {spread:4.1f}%)", flush=True)

    t0 = time.perf_counter()
    pe.pack(model, engine=pe.ModernBertCPU(threads=a.threads))
    res["pack_s"] = time.perf_counter() - t0
    engine = pe.get_engine(model)
    att = engine.attention
    res["engine"] = {"backend": att.backend, "native_linear": att.native_linear, "fused_rope": att.fused_rope,
                     "native_local": att.native_local, "native_global": att.native_global,
                     "attention_tuning": {str(k): v for k, v in att.attention_tuning.items()},
                     "report": repr(engine.state.validation_report)}
    print("engine", res["engine"], f"pack {res['pack_s']:.1f}s", flush=True)

    prof = defaultdict(float)
    if a.profile:
        def wrap(obj, name, label):
            fn = getattr(obj, name)

            def inner(*args, **kwargs):
                t = time.perf_counter()
                out = fn(*args, **kwargs)
                prof[label] += time.perf_counter() - t
                return out
            setattr(obj, name, inner)
        for name, label in (("linear", "projections"), ("rope_layout", "rope_layout"), ("local", "attn_local"),
                            ("global_", "attn_global"), ("add_norm", "add_norm"), ("geglu", "geglu")):
            wrap(att, name, label)
        core = engine._core

        def core_timed(*args, **kwargs):
            t = time.perf_counter()
            out = core(*args, **kwargs)
            prof["core_total"] += time.perf_counter() - t
            return out
        engine._core = core_timed

    with torch.inference_mode():
        for name, (ids, mask) in loads.items():
            prof.clear()
            ms, spread = timed(lambda: model(input_ids=ids, attention_mask=mask), a.trials)
            row = res["rows"].setdefault(name, {})
            row["engine_ms"], row["engine_spread_pct"] = ms, spread
            if a.profile:
                calls = a.trials + 1
                row["profile_ms"] = {k: 1e3 * v / calls for k, v in prof.items()}
            line = f"[pe ] {name:18s} {ms:9.2f} ms (spread {spread:4.1f}%)"
            if name in hf:
                out = model(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.bool()
                cos = F.cosine_similarity(out[m], hf[name][m], dim=-1)
                row["cos_min"] = float(cos.min())
                row["speedup"] = row["hf_ms"] / ms
                line += f" | x{row['speedup']:.2f} vs HF | cos min {row['cos_min']:.6f}"
            if a.profile:
                p = row["profile_ms"]
                line += " | " + " ".join(f"{k}={v:.1f}" for k, v in sorted(p.items(), key=lambda kv: -kv[1]))
            print(line, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
