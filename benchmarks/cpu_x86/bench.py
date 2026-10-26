"""ModernBERT-base CPU latency: packed engine vs ONNX Runtime, one system per process.

python bench.py packed  --threads 16 [--workloads a,b] [--precision fp32|int8] [--backend auto|sdpa|native]
python bench.py onnx    --threads 16 --model MODELS/int8.onnx [--microbatch 1]
Prints one JSON line per workload. Pin with taskset (e.g. -c 0-7 for 8 threads on CCD0, -c 0-15 for 16):
unpinned runs are bimodal on this 2-CCD part. Compare variants with alternating A/B in one process
(ab_linear.py) rather than across processes; absolute numbers drift ~10-25% with machine state.
"""
import argparse
import json
import os
import statistics
import time

import numpy as np
import torch

VOCAB_LO, VOCAB_HI = 100, 50000


def workloads():
    g = torch.Generator().manual_seed(0)
    r1024 = torch.randint(1, 1025, (32,), generator=g).tolist()
    r8192 = torch.randint(1, 8193, (32,), generator=g).tolist()
    return {
        "b1_s16_pad32": (32, [16]),
        "b1_s32": (32, [32]),
        "b8_s32": (32, [32] * 8),
        "b32_s1024": (1024, [1024] * 32),
        "b32_s8192": (8192, [8192] * 32),
        "b32_rand_pad1024": (1024, r1024),
        "b32_rand_pad8192": (8192, r8192),
    }


# (warmup, samples) per workload; heavy ones are minutes per sample on the baselines.
PLAN = {"b1_s16_pad32": (10, 60), "b1_s32": (10, 60), "b8_s32": (5, 40), "b32_s1024": (1, 5),
        "b32_s8192": (0, 2), "b32_rand_pad1024": (1, 5), "b32_rand_pad8192": (0, 2)}


def inputs(seq, lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(VOCAB_LO, VOCAB_HI, (len(lengths), seq), generator=g)
    mask = (torch.arange(seq)[None] < torch.tensor(lengths)[:, None]).long()
    ids = ids * mask  # pad id 0 (ModernBERT pad is 50283; ids under the mask are ignored)
    return ids, mask


def timeit(fn, warmup, samples, deadline):
    for _ in range(warmup):
        fn()
    times, start = [], time.perf_counter()
    for _ in range(samples):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
        if time.perf_counter() - start > deadline and len(times) >= 1:
            break
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("system", choices=["packed", "onnx", "hf"])
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--workloads", default=",".join(workloads()))
    p.add_argument("--precision", default="fp32")
    p.add_argument("--backend", default="auto")
    p.add_argument("--model", default=None)
    p.add_argument("--microbatch", type=int, default=0, help="ONNX rows per run; 0 = by token budget")
    p.add_argument("--samples", type=int, default=0)
    p.add_argument("--warmup", type=int, default=-1)
    p.add_argument("--deadline", type=float, default=120.0)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.system == "onnx":
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = a.threads, 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(a.model, sess_options=so, providers=["CPUExecutionProvider"])

        def make(ids, mask):
            seq = ids.shape[1]
            rows = a.microbatch or max(1, min(len(ids), 32768 // seq if seq <= 1024 else 1))
            chunks = [(ids[i:i + rows].numpy(), mask[i:i + rows].numpy()) for i in range(0, len(ids), rows)]

            def run():
                return [sess.run(None, {"input_ids": x, "attention_mask": m})[0] for x, m in chunks]
            return run, rows
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa").float().eval()
        if a.system == "packed":
            import warnings
            import packed_encoders as pe
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                t = time.perf_counter()
                pe.pack(model, engine=pe.ModernBertCPU(threads=a.threads), attention_backend=a.backend,
                        precision=a.precision)
                pack_s = time.perf_counter() - t
            eng = pe.get_engine(model)
            print(json.dumps({"pack_s": round(pack_s, 2), "backend": eng.state.attention_backend,
                              "precision": eng.state.precision, "warnings": [str(w.message) for w in caught]}))

        def make(ids, mask):
            def run():
                with torch.inference_mode():
                    return model(input_ids=ids, attention_mask=mask).last_hidden_state
            return run, len(ids)

    names = a.workloads.split(",")
    for name in names:
        seq, lengths = workloads()[name]
        ids, mask = inputs(seq, lengths)
        run, rows = make(ids, mask)
        warmup, samples = PLAN[name]
        warmup = a.warmup if a.warmup >= 0 else warmup
        times = timeit(run, warmup, a.samples or samples, a.deadline)
        ms = [t * 1e3 for t in times]
        rss = int(open("/proc/self/status").read().split("VmHWM:")[1].split()[0]) // 1024
        print(json.dumps({"system": a.system, "tag": a.tag or a.model or a.precision, "threads": a.threads,
                          "workload": name, "live_tokens": sum(lengths), "padded_tokens": len(lengths) * seq,
                          "rows_per_run": rows, "n": len(ms), "median_ms": round(statistics.median(ms), 2),
                          "min_ms": round(min(ms), 2), "max_ms": round(max(ms), 2), "peak_rss_mb": rss}), flush=True)


if __name__ == "__main__":
    main()
