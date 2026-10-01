"""What a user actually gets: plain Sentence Transformers, then the speed options the model loader
already offers, then `pe.pack` — end to end through `model.encode`, one fresh process per rung.

    python benchmarks/practical_ladder.py --family topk --model topk-io/topk-embed-v1-xsmall \\
        --data sample.json --out ladder.json
    python benchmarks/practical_ladder.py --family st --model Alibaba-NLP/gte-modernbert-base \\
        --data sample.json --out ladder.json

Rungs (each builds on the loader, not on the previous rung's process):

  bare              SentenceTransformer / MultiVectorEncoder(model, device="cuda") — nothing else
  bf16              + model_kwargs={"torch_dtype": torch.bfloat16}
  bf16_fa2          + attn_implementation="flash_attention_2" (needs flash-attn; a model's own
                      attention code may ignore it — the rung shows whether it does anything)
  bf16_fa2_compile  + torch.compile(backbone forward, mode="max-autotune", dynamic=True), as the
                      packed-encoders inference showcase compiles its stock baseline
  pack              bf16 + pe.pack(model) (topk: architecture defaults; ModernBERT: the showcase's
                      flash attention, CUDA graphs on the query endpoint)
  pack_default      bf16 + pe.pack(model, validate=...) with the library defaults, packed once

`--data` is JSON with "queries" and "documents" lists. Every rung encodes the same texts at each
`--batches` size: one cold pass (load-time compile / graph capture land here and are reported),
then `--trials` timed passes, median. Tokenisation, collation, H2D and output conversion are all
inside the timer — this is the number a user sees, not the forward alone. Parity: the first
`--parity-n` outputs of every rung against the `bare` rung (per token for multi-vector models).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

RUNGS = {
    "bare": {"kwargs": {}},
    "bf16": {"kwargs": {"torch_dtype": "bfloat16"}},
    "bf16_fa2": {"kwargs": {"torch_dtype": "bfloat16", "attn_implementation": "flash_attention_2"}},
    "bf16_fa2_compile": {"kwargs": {"torch_dtype": "bfloat16", "attn_implementation": "flash_attention_2"},
                         "compile": True},
    "pack": {"kwargs": {"torch_dtype": "bfloat16"}, "pack": True},
    "pack_default": {"kwargs": {"torch_dtype": "bfloat16"}, "pack": "default"},
}
TASKS = ("query", "document")


def _compat_shims() -> None:
    """topk's remote code targets transformers 5.9; restore the two names later versions moved."""
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    except ImportError:
        return
    if "block_type" in inspect.getsource(mq.Qwen3_5DecoderLayer.__init__):
        mq.Qwen3_5DecoderLayer.layer_type = property(lambda self: self.block_type)
    if not hasattr(mq.Qwen3_5GatedDeltaNet, "chunk_gated_delta_rule"):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        mq.Qwen3_5GatedDeltaNet.chunk_gated_delta_rule = staticmethod(chunk_gated_delta_rule)


# ---------------------------------------------------------------------------- worker (one rung)


CATEGORIES = {"attention": r"flash|fmha|attention|flex|varlen",
              "gemm": r"gemm|xmma|cutlass|splitk|s16816|s1688|cublas|ampere_|sm80_|sm89_|sm90_|nvjet",
              "gdn": r"chunk|solve_tril|recompute|kkt|wy_|cumsum|l2norm|gdn|gated_delta|fwd_h|fwd_o|recurrent",
              "conv": r"conv",
              "norm_act": r"norm|swiglu|geglu|gelu|silu|sigmoid|swish|rms|rope",
              "copy_index": r"elementwise|vectorized|unrolled|copy|cat|fill|index|reduce|mul|add|where|pad|triton_"}


def _profile_encode(run, n_batches: int) -> dict:
    """One encode pass under the profiler: wall time, GPU-busy union, kernels per batch, categories."""
    import re
    import time

    import torch
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t = time.perf_counter()
        run()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
    agg, spans, nk = {c: 0.0 for c in [*CATEGORIES, "other"]}, [], 0
    for e in prof.events():
        if e.device_type.name != "CUDA" or e.device_time <= 0 or getattr(e, "is_user_annotation", False):
            continue
        nk += 1
        spans.append((e.time_range.start, e.time_range.end))
        agg[next((c for c, rx in CATEGORIES.items() if re.search(rx, e.name.lower())), "other")] += e.device_time / 1e3
    busy, end = 0.0, float("-inf")
    for s0, e0 in sorted(spans):
        if e0 > end:
            busy += e0 - max(s0, end)
            end = e0
    return {"wall_ms": wall * 1e3, "gpu_busy_ms": busy / 1e3, "gpu_busy_frac": busy / 1e3 / (wall * 1e3),
            "kernels_per_batch": nk / max(n_batches, 1), "by_category_ms": agg}


def _worker(a: argparse.Namespace) -> None:
    import torch

    spec = RUNGS[a.rung]
    kwargs = {k: getattr(torch, v) if k == "torch_dtype" else v for k, v in spec["kwargs"].items()}
    data = json.load(open(a.data))
    texts = {"query": data["queries"][: a.n_queries], "document": data["documents"][: a.n_docs]}
    res = {"rung": a.rung, "model_kwargs": spec["kwargs"], "gpu": torch.cuda.get_device_name(0),
           "pack_validated": not a.no_validate,
           "torch": str(torch.__version__), "endpoints": {}}

    t0 = time.perf_counter()
    if a.family == "topk":
        _compat_shims()
        from sentence_transformers.multi_vector_encoder import MultiVectorEncoder

        model = MultiVectorEncoder(a.model, model_kwargs=kwargs, trust_remote_code=True, device="cuda").eval()

        def encode(task, xs, bs):
            return model.encode(xs, task=task, batch_size=bs, convert_to_numpy=False, show_progress_bar=False)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(a.model, model_kwargs=kwargs, device="cuda").eval()

        def encode(task, xs, bs):
            model.max_seq_length = a.query_length if task == "query" else a.document_length
            return model.encode(xs, batch_size=bs, convert_to_tensor=True, show_progress_bar=False)
    net = model[0].auto_model
    res["loaded_dtype"] = str(next(net.parameters()).dtype)
    res["attn_implementation"] = getattr(net.config, "_attn_implementation", None)
    if spec.get("compile"):
        net.forward = torch.compile(net.forward, mode="max-autotune", dynamic=True)
    res["load_s"] = time.perf_counter() - t0

    import packed_encoders as pe

    packed = []

    def pack_for(task, bs):
        """ModernBERT graphs are per endpoint (queries only, as the showcase); topk packs once."""
        if a.family == "topk" or spec["pack"] == "default":
            if not packed:
                t = time.perf_counter()
                pe.pack(model) if a.family == "topk" else pe.pack(model, validate=not a.no_validate)
                res["pack_s"] = time.perf_counter() - t
                packed.append(True)
            return
        pe.unpack(model)
        graph = pe.GraphConfig(pad_to=32, max_seq=a.query_length, max_batch=bs) if task == "query" else False
        t = time.perf_counter()
        pe.pack(model, attention_backend="flash", cuda_graph=graph, validate=not a.no_validate)
        res.setdefault("pack_s", {})[f"{task}_b{bs}"] = time.perf_counter() - t

    parity = {}
    for bs in a.batches:
        for task in TASKS:
            key = f"{task}_b{bs}"
            try:
                if spec.get("pack"):
                    pack_for(task, bs)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                out = encode(task, texts[task], bs)
                torch.cuda.synchronize()
                cold = time.perf_counter() - t
                trials = []
                for _ in range(a.trials):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    encode(task, texts[task], bs)
                    torch.cuda.synchronize()
                    trials.append(time.perf_counter() - t)
                med = statistics.median(trials)
                res["endpoints"][key] = {"items_per_s": len(texts[task]) / med, "pass_s": med, "trials_s": trials,
                                         "cold_pass_s": cold, "n": len(texts[task]),
                                         "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
                if a.profile:
                    P = res["endpoints"][key]["profile"] = _profile_encode(
                        lambda: encode(task, texts[task], bs), -(-len(texts[task]) // bs))
                    tot = sum(P["by_category_ms"].values()) or 1.0
                    print(f"[{a.rung:16s}] {key:12s} PROFILE wall {P['wall_ms']:.0f} ms, GPU busy {P['gpu_busy_ms']:.0f} ms "
                          f"({P['gpu_busy_frac'] * 100:.0f}%), {P['kernels_per_batch']:.0f} kernels/batch | "
                          + " ".join(f"{c} {v / tot * 100:.0f}%" for c, v in sorted(P["by_category_ms"].items(),
                                                                                 key=lambda kv: -kv[1]) if v),
                          flush=True)
                if task not in parity:
                    parity[task] = [x.float().cpu() for x in out[: a.parity_n]]
                print(f"[{a.rung:16s}] {key:12s} {len(texts[task]) / med:9.1f} items/s  cold {cold:6.1f}s  "
                      f"peak {res['endpoints'][key]['peak_reserved_gb']:.2f} GB", flush=True)
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
                res["endpoints"][key] = {"error": msg}
                print(f"[{a.rung:16s}] {key:12s} FAILED {msg}", flush=True)
    torch.save(parity, a.parity_file)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


# ---------------------------------------------------------------------------- orchestrator


def _parity(x: list, ref: list) -> dict:
    import torch
    import torch.nn.functional as F

    cos = []
    for a, b in zip(x, ref):
        if a.shape != b.shape:
            return {"error": f"shape {tuple(a.shape)} vs reference {tuple(b.shape)}"}
        cos.append(F.cosine_similarity(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1))
    c = torch.cat(cos)
    return {"cos_mean": c.mean().item(), "cos_min": c.min().item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("topk", "st"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rungs", default=",".join(RUNGS))
    ap.add_argument("--no-validate", action="store_true",
                    help="pack without pack()'s gate (recorded in the result); parity vs the fp32 bare rung on real "
                         "text is still measured")
    ap.add_argument("--batches", default="8,128")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--n-queries", type=int, default=256)
    ap.add_argument("--n-docs", type=int, default=512)
    ap.add_argument("--query-length", type=int, default=128, help="st family: max_seq_length for queries")
    ap.add_argument("--document-length", type=int, default=2048, help="st family: max_seq_length for documents")
    ap.add_argument("--parity-n", type=int, default=64)
    ap.add_argument("--profile", action="store_true",
                    help="per rung and endpoint, profile one encode pass: wall vs GPU-busy time (host-bound or "
                         "not), kernels per batch, time by kernel category")
    ap.add_argument("--timeout", type=int, default=2400, help="seconds per rung")
    ap.add_argument("--rung", help=argparse.SUPPRESS)
    ap.add_argument("--parity-file", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.batches = [int(x) for x in str(a.batches).split(",")]
    if a.rung:
        return _worker(a)

    import torch

    work = Path(a.out).with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    summary = {"family": a.family, "model": a.model, "batches": a.batches, "rungs": {}}
    for rung in a.rungs.split(","):
        rung_out, pfile = work / f"{rung}.json", work / f"{rung}_parity.pt"
        cmd = [sys.executable, "-u", __file__, "--family", a.family, "--model", a.model, "--data", a.data,
               "--batches", ",".join(map(str, a.batches)), "--trials", str(a.trials), "--n-queries", str(a.n_queries),
               "--n-docs", str(a.n_docs), "--query-length", str(a.query_length), "--document-length",
               str(a.document_length), "--parity-n", str(a.parity_n), "--rung", rung, "--out", str(rung_out),
               "--parity-file", str(pfile), *(["--profile"] if a.profile else []),
               *(["--no-validate"] if a.no_validate else [])]
        t = time.perf_counter()
        try:
            p = subprocess.run(cmd, timeout=a.timeout, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        r = json.loads(rung_out.read_text()) if rung_out.exists() else {}
        r["returncode"], r["wall_s"] = rc, time.perf_counter() - t
        summary["rungs"][rung] = r
        if pfile.exists():
            summary["rungs"][rung]["_parity"] = torch.load(pfile)

    ref = summary["rungs"].get("bare", {}).get("_parity", {})
    for rung, r in summary["rungs"].items():
        mine = r.pop("_parity", {})
        r["parity_vs_bare"] = {t: _parity(mine[t], ref[t]) for t in mine if t in ref}
    Path(a.out).write_text(json.dumps(summary, indent=1, default=str))

    rungs = list(summary["rungs"])
    print(f"\n== {a.model} | items/s (x vs bare | x vs best off-the-shelf)")
    for key in [f"{t}_b{b}" for b in a.batches for t in TASKS]:
        cells, base, best = [], None, 0.0
        for rung in rungs:
            e = summary["rungs"][rung].get("endpoints", {}).get(key, {})
            v = e.get("items_per_s")
            if rung == "bare":
                base = v
            if v and not RUNGS[rung].get("pack"):
                best = max(best, v)
            cells.append(f"{rung}={v:.1f}" if v else f"{rung}=FAIL")
        pk = summary["rungs"].get("pack", {}).get("endpoints", {}).get(key, {}).get("items_per_s")
        tail = f"  | pack {pk / base:.2f}x vs bare, {pk / best:.2f}x vs best" if pk and base and best else ""
        print(f"LADDER {key:12s} " + "  ".join(cells) + tail, flush=True)
    for rung, r in summary["rungs"].items():
        print(f"PARITY {rung:16s} {r.get('parity_vs_bare')}", flush=True)
    print("LADDER DONE", flush=True)


if __name__ == "__main__":
    main()
