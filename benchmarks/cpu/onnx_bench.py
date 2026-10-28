"""ONNX Runtime baselines on the same workloads as bench.py (same ids, seed 0), FP32 and int8.

    python benchmarks/cpu/onnx_bench.py --threads 8 --out results/o_t8_short.json \
        --models fp32_opt,int8_w8a8 --only b1_s32,b8_s32 --spin 1,0 --trials 20

Models come from onnx_export.py. Protocol follows the CPU engine's published ONNX baseline:
intra-op threads = --threads, sequential execution, ORT_ENABLE_ALL, and the faster of intra-op
spinning on/off per cell. Feeds per workload:
  batch    the padded batch as bench.py feeds HF and the engine (short queries, single documents)
  rowpad   one row per run at the full padded width (the published "ONNX runs one row at a time")
  rowtrim  one row per run trimmed to its real length (the best a careful ONNX user gets)
Accuracy: token cosine and relL2 against stock HF FP32 over real tokens of the same input.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bench import MODEL, workloads

LONG = {"b32_s1024", "b32_rand_pad1024"}


def session(path, threads, spin):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning", str(spin))
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def feeds(ids, mask, mode):
    ids, mask = ids.numpy().astype(np.int64), mask.numpy().astype(np.int64)
    if mode == "batch":
        return [(ids, mask)]
    out = []
    for i in range(ids.shape[0]):
        n = int(mask[i].sum())
        w = ids.shape[1] if mode == "rowpad" else n
        out.append((ids[i:i + 1, :w], mask[i:i + 1, :w]))
    return out


def run(sess, chunks):
    return [sess.run(None, {"input_ids": i, "attention_mask": m})[0] for i, m in chunks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="fp32_opt,int8_w8a8")
    ap.add_argument("--only", required=True)
    ap.add_argument("--spin", default="1,0")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--dir", default=str(Path(__file__).parent / "onnx_models"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()
    loads = {k: v for k, v in workloads(model.config.vocab_size).items() if k in a.only.split(",")}
    with torch.inference_mode():
        ref = {k: model(input_ids=i, attention_mask=m).last_hidden_state for k, (i, m) in loads.items()}
    del model
    res = {"threads": a.threads, "rows": {}}
    for name in a.models.split(","):
        for spin in map(int, a.spin.split(",")):
            sess = session(f"{a.dir}/{name}.onnx", a.threads, spin)
            for wl, (ids, mask) in loads.items():
                modes = ("rowpad", "rowtrim") if wl in LONG else ("batch",)
                if wl == "b1_s16_pad32":
                    modes = ("batch", "rowtrim")
                if wl == "b32_s1024":
                    modes = ("rowpad",)
                for mode in modes:
                    chunks = feeds(ids, mask, mode)
                    out = run(sess, chunks)
                    samples = []
                    for _ in range(a.trials):
                        t = time.perf_counter()
                        run(sess, chunks)
                        samples.append(time.perf_counter() - t)
                    ms = 1e3 * statistics.median(samples)
                    m = mask.bool()
                    if mode == "batch":
                        got = torch.from_numpy(out[0])[m]
                    else:  # rows are right-padded, so real tokens come first and row order matches ref[m]
                        got = torch.cat([torch.from_numpy(o[0, :int(c[1].sum())]) for o, c in zip(out, chunks)])
                    want = ref[wl][m]
                    cos = F.cosine_similarity(got, want, dim=-1)
                    rel = float((got - want).norm() / want.norm())
                    key = f"{name}|{wl}|{mode}|spin{spin}"
                    res["rows"][key] = {"ms": ms, "spread_pct": 100 * (max(samples) - min(samples)) / statistics.median(samples),
                                        "cos_min": float(cos.min()), "cos_median": float(cos.median()), "relL2": rel}
                    print(f"[ort t{a.threads:<2d}] {name:8s} spin{spin} {wl:18s} {mode:7s} {ms:10.2f} ms "
                          f"| cos min {float(cos.min()):.5f} med {float(cos.median()):.5f} relL2 {rel:.2e}", flush=True)
            del sess
    json.dump(res, open(a.out, "w"), indent=1)
    print("ORT DONE", flush=True)


if __name__ == "__main__":
    main()
