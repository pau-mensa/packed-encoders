"""Attention ablation for the packed CPU engine (engine only, FP32).

    python benchmarks/cpu_avx2/attention_ablation.py --threads 8 --mode native|sdpa_attn --tag X --out results/abl_X_t8.json [--dump out.pt]

native: the engine's own attention kernels. sdpa_attn: the engine's private switches route
local and global attention through its tiled PyTorch SDPA path (row kernels unchanged).
--dump saves every workload's output so two runs can be compared bitwise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict

import torch

from bench import MODEL, workloads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--mode", choices=("native", "sdpa_attn"), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--dump", default="")
    ap.add_argument("--only", default="b1_s32,b8_s32,b1_s1024,b32_rand_pad1024")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModel
    import packed_encoders as pe

    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()
    loads = {k: v for k, v in workloads(model.config.vocab_size).items() if k in a.only.split(",")}
    pe.pack(model, engine=pe.ModernBertCPU(threads=a.threads))
    engine = pe.get_engine(model)
    att = engine.attention
    if a.mode == "sdpa_attn":
        att.native_local = att.native_global = att.token_major = False
    prof = defaultdict(float)
    for name, label in (("linear", "proj"), ("local", "local"), ("global_", "global"),
                        ("rope_layout", "rope"), ("add_norm", "norm"), ("geglu", "geglu")):
        fn = getattr(att, name)

        def inner(*args, _fn=fn, _label=label, **kwargs):
            t = time.perf_counter()
            out = _fn(*args, **kwargs)
            prof[_label] += time.perf_counter() - t
            return out
        setattr(att, name, inner)
    rows, dumps = {}, {}
    with torch.inference_mode():
        for name, (ids, mask) in loads.items():
            model(input_ids=ids, attention_mask=mask)
            prof.clear()
            samples = []
            for _ in range(a.trials):
                t = time.perf_counter()
                out = model(input_ids=ids, attention_mask=mask).last_hidden_state
                samples.append(time.perf_counter() - t)
            ms = 1e3 * statistics.median(samples)
            per = {k: 1e3 * v / a.trials for k, v in prof.items()}
            rows[name] = {"ms": ms, "spread_pct": 100 * (max(samples) - min(samples)) / statistics.median(samples),
                          "ops_ms": per}
            dumps[name] = out.clone()
            print(f"[{a.tag:10s} t{a.threads:<2d}] {name:18s} {ms:9.2f} ms | "
                  + " ".join(f"{k}={v:.1f}" for k, v in sorted(per.items(), key=lambda kv: -kv[1])), flush=True)
    json.dump({"tag": a.tag, "threads": a.threads, "mode": a.mode, "rows": rows},
              open(a.out, "w"), indent=1)
    if a.dump:
        torch.save(dumps, a.dump)
    print("ABLATE DONE", flush=True)


if __name__ == "__main__":
    main()
