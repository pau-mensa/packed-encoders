"""topk-embed-v1 (Qwen3.5 hybrid) under packed-encoders: speed and parity vs the model as shipped.

    python benchmarks/qwen35_topk_bench.py --model topk-io/topk-embed-v1-xsmall --data sample.json --out r.json

`--data` is JSON with "queries" and "documents" lists of strings. Two levels, bf16:

- forward: the model forward SentenceTransformer calls (`TopkEmbedModel.forward`) on
  pre-tokenised batches of `--batch` in data order — the packed-encoders protocol
  (tokenisation excluded; H2D of the features, forward and output inside the timer;
  one warmup pass that also captures graphs; `--trials` passes; median).
- encode: `model.encode(texts, batch_size=--batch)` end to end — tokenisation, collation,
  H2D, forward, numpy output — i.e. what a user sees.

Variants on one model object: stock (before pack), eager (packed, graphs off), graphs
(packed default). Parity per real token vs stock: cosine mean / min and sign-flip fraction.
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import time

import torch
import torch.nn.functional as F


def compat_shims() -> None:
    """transformers >= 5.10 renamed Qwen3_5DecoderLayer.layer_type and dropped the fla binding the
    topk remote code calls; restore both (no effect on the pinned 5.9.0). Needed by the *stock* model
    only — packed-encoders calls fla directly."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    if "block_type" in inspect.getsource(mq.Qwen3_5DecoderLayer.__init__):
        mq.Qwen3_5DecoderLayer.layer_type = property(lambda self: self.block_type)
    if not hasattr(mq.Qwen3_5GatedDeltaNet, "chunk_gated_delta_rule"):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        mq.Qwen3_5GatedDeltaNet.chunk_gated_delta_rule = staticmethod(chunk_gated_delta_rule)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="topk-io/topk-embed-v1-xsmall")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--encode-trials", type=int, default=3)
    ap.add_argument("--n-queries", type=int, default=256)
    ap.add_argument("--n-docs", type=int, default=512)
    ap.add_argument("--levels", default="forward,encode")
    ap.add_argument("--no-validate", action="store_true",
                    help="pack without the hard gate (for stacks where the model's own forward cannot run, e.g. "
                         "topk on torch 2.8); parity vs stock is still reported where stock ran")
    ap.add_argument("--fusion-ab", action="store_true",
                    help="after the graphs variant, run (fusions, small GEMMs) off/off, on/off, on/on with graphs "
                         "recaptured, so the A/B shares one process, host and GPU")
    ap.add_argument("--profile", action="store_true",
                    help="after the graphs variant, profile one forward pass per task: GPU busy, kernels per "
                         "batch, time by kernel category, top kernels")
    ap.add_argument("--recurrent-sweep", default="",
                    help="comma list of GatedDeltaNet recurrent-kernel cutoffs (tokens) to time as extra graph "
                         "variants, e.g. 0,32,128,256 (0 = chunked kernel only); forward level, graphs recaptured")
    a = ap.parse_args()

    compat_shims()
    import packed_encoders as pe
    from sentence_transformers.multi_vector_encoder import MultiVectorEncoder

    data = json.load(open(a.data))
    texts = {"query": data["queries"][: a.n_queries], "document": data["documents"][: a.n_docs]}
    model = MultiVectorEncoder(a.model, model_kwargs={"torch_dtype": torch.bfloat16}, trust_remote_code=True,
                               device="cuda").eval()
    st, net = model[0], model[0].auto_model
    dev = torch.device("cuda")
    res = {"model": a.model, "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(),
           "torch": torch.__version__, "batch": a.batch, "levels": {}}
    import transformers, fla

    res["transformers"], res["fla"] = transformers.__version__, fla.__version__
    print(f"{a.model} | {res['gpu']} sm_{res['capability'][0]}{res['capability'][1]} | torch {torch.__version__} "
          f"transformers {transformers.__version__} fla {fla.__version__}", flush=True)

    keys = [p for p in inspect.signature(net.forward).parameters if p not in ("pixel_values", "image_scatter_index",
                                                                            "vision_inputs", "return_dict")]
    batches = {}
    for task, tx in texts.items():
        bs = []
        for s in range(0, len(tx), a.batch):
            f = st.preprocess(tx[s:s + a.batch], task=task)
            bs.append({k: f[k].pin_memory() for k in keys})
        batches[task] = bs
        ntok = sum(int(b["attention_mask"].sum()) for b in bs)
        print(f"== {task}: {len(tx)} texts, {ntok} tokens (mean {ntok / len(tx):.1f})", flush=True)

    @torch.inference_mode()                                  # as SentenceTransformer.encode runs it
    def one_pass(task, collect=False):
        outs = []
        for b in batches[task]:
            f = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            o = net(**f)
            if collect:
                outs.append(o[f["attention_mask"]].float())
        torch.cuda.synchronize()
        return torch.cat(outs) if collect else None

    def time_forward(task):
        t0 = time.perf_counter()
        ref = one_pass(task, collect=True)                  # warmup (+ graph capture) and outputs
        cold = time.perf_counter() - t0
        trials = []
        for _ in range(a.trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_pass(task)
            trials.append(time.perf_counter() - t0)
        return ref, {"pass_s": statistics.median(trials), "trials_s": trials, "cold_pass_s": cold,
                     "items_per_s": len(texts[task]) / statistics.median(trials),
                     "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}

    def time_encode(task):
        model.encode(texts[task][: 4 * a.batch], batch_size=a.batch, task=task, show_progress_bar=False)
        trials = []
        for _ in range(a.encode_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.encode(texts[task], batch_size=a.batch, task=task, show_progress_bar=False)
            torch.cuda.synchronize()
            trials.append(time.perf_counter() - t0)
        return {"pass_s": statistics.median(trials), "items_per_s": len(texts[task]) / statistics.median(trials)}

    # First match wins. Attention precedes gemm: FlashAttention's kernel names carry `cutlass::` types.
    categories = {"attention": r"flash|fmha|attention|flex|varlen",
                  "gemm": r"gemm|xmma|cutlass|splitk|s16816|s1688|cublas|ampere_|sm80_|sm89_|sm90_|nvjet",
                  "fla_gdn": r"chunk|solve_tril|recompute|kkt|wy_|cumsum|l2norm|gdn|gated_delta|fwd_h|fwd_o|recurrent",
                  "conv1d": r"conv",
                  "norm_act": r"norm|swiglu|silu|sigmoid|swish|rms|rope",
                  "copy_index_other": r"elementwise|vectorized|unrolled|copy|cat|fill|index|reduce|mul|add|where|triton_"}

    def profile_task(task):
        import re

        from torch.profiler import ProfilerActivity, profile

        one_pass(task)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            one_pass(task)
        agg, spans, top, cnt, nk = {c: 0.0 for c in [*categories, "other"]}, [], {}, {}, 0
        for e in prof.events():
            if e.device_type.name != "CUDA" or e.device_time <= 0 or getattr(e, "is_user_annotation", False):
                continue
            nk += 1
            ms, name = e.device_time / 1e3, e.name.lower()
            spans.append((e.time_range.start, e.time_range.end))
            top[e.name[:90]] = top.get(e.name[:90], 0.0) + ms
            cnt[e.name[:90]] = cnt.get(e.name[:90], 0) + 1
            agg[next((c for c, rx in categories.items() if re.search(rx, name)), "other")] += ms
        busy, end = 0.0, float("-inf")
        for s0, e0 in sorted(spans):
            if e0 > end:
                busy += e0 - max(s0, end)
                end = e0
        nb = len(batches[task])
        return {"gpu_busy_ms": busy / 1e3, "kernels_per_batch": nk / nb, "by_category_ms": agg,
                "top_kernels_ms": dict(sorted(top.items(), key=lambda kv: -kv[1])[:15]),
                "kernels": {n: {"per_batch": c / nb, "ms": top[n]} for n, c in sorted(cnt.items(), key=lambda kv: -kv[1])}}

    def parity(x, ref):
        cos = F.cosine_similarity(x, ref, dim=-1)
        return {"cos_mean": cos.mean().item(), "cos_min": cos.min().item(),
                "sign_flip_frac": ((x > 0) != (ref > 0)).float().mean().item()}

    levels = a.levels.split(",")
    outs: dict[str, dict] = {"query": {}, "document": {}}

    def run_variant(name):
        for task in ("document", "query"):
            R = res["levels"].setdefault("forward", {}).setdefault(task, {})
            if "forward" in levels:
                try:
                    o, R[name] = time_forward(task)
                    outs[task][name] = o
                    msg = f"{R[name]['items_per_s']:8.1f} items/s  pass {R[name]['pass_s'] * 1e3:8.1f} ms"
                    if name != "stock" and "stock" in outs[task]:
                        R[name]["parity_vs_stock"] = p = parity(o, outs[task]["stock"])
                        msg += f"  | vs stock cos mean {p['cos_mean']:.6f} min {p['cos_min']:.5f} flips {p['sign_flip_frac']:.2e}"
                    if name != "unfused" and "unfused" in outs[task]:
                        d = (o - outs[task]["unfused"]).abs().max().item()
                        R[name]["parity_vs_unfused"] = p = {**parity(o, outs[task]["unfused"]), "max_abs_diff": d}
                        msg += f"  | vs unfused cos min {p['cos_min']:.6f} max|diff| {d:.2e}"
                    print(f"[forward {task:8s}] {name:7s} {msg}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    R[name] = {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"}
                    print(f"[forward {task:8s}] {name:7s} FAILED {R[name]['error']}", flush=True)
            if "encode" in levels:
                E = res["levels"].setdefault("encode", {}).setdefault(task, {})
                try:
                    E[name] = time_encode(task)
                    print(f"[encode  {task:8s}] {name:7s} {E[name]['items_per_s']:8.1f} items/s", flush=True)
                except Exception as exc:  # noqa: BLE001
                    E[name] = {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"}
                    print(f"[encode  {task:8s}] {name:7s} FAILED {E[name]['error']}", flush=True)

    run_variant("stock")
    t0 = time.perf_counter()
    pe.pack(model, validate=not a.no_validate)
    from packed_encoders.state import ATTR

    st_state = getattr(net, ATTR)
    rep = st_state.report
    res["pack"] = {"seconds": time.perf_counter() - t0, "report": rep.__dict__ if rep else None}
    if rep is None:
        eng = st_state.engine
        print(f"pack (NOT validated): {res['pack']['seconds']:.1f}s | attention {eng.attention.name} | gdn "
              f"{eng.gdn.errors} recurrent <= {eng.gdn.recurrent_max_len}", flush=True)
    else:
        print(f"pack: {res['pack']['seconds']:.1f}s | attention {rep.attention_backend} (err {rep.attention_max_abs_err:.1e}, "
              f"rejected {rep.attention_rejected}) | qk kernel err {rep.qk_kernel_max_abs_err:.1e} | gdn rel err "
              f"{ {k: f'{v:.1e}' for k, v in rep.gdn_kernel_rel_err.items()} } rejected {rep.gdn_rejected} recurrent <= "
              f"{rep.gdn_recurrent_max_len} | fused {rep.fused} {({k: f'{v:.1e}' for k, v in rep.fusion_errors.items()})} "
              f"rejected {rep.fusion_rejected} | validate eager cos {rep.eager_cos_mean:.6f}/{rep.eager_cos_min:.5f} graph "
              f"{rep.graph_cos_mean:.6f}/{rep.graph_cos_min:.5f}", flush=True)
    with pe.no_cuda_graph(model):
        run_variant("eager")
    run_variant("graphs")
    res["graphs_captured"] = st_state.runner.num_graphs if st_state.runner else 0
    eng = st_state.engine
    if a.fusion_ab:
        # one process, host and GPU: fusions off, off again, on; graphs recaptured each time
        from packed_encoders.runtime.graphs import PaddedGraphConfig

        saved_levels, levels = levels, ["forward"]
        # unfused_repeat: a second capture of the same engine, so "vs unfused" has a measured floor
        steps = [("unfused", False), ("unfused_repeat", False), ("fused", True)]
        can_fuse = eng.fused                                        # what the probes allowed
        for name, fz in steps:
            if fz and not can_fuse:
                print(f"[ab] skip {name}: the fusion probes rejected it", flush=True)
                continue
            eng.fused = fz
            pe.set_cuda_graph(model, True, config=PaddedGraphConfig())
            run_variant(name)
            if a.profile and name in ("unfused", "fused"):
                for task in ("document", "query"):
                    P = res.setdefault(f"profile_{name}", {})[task] = profile_task(task)
                    print(f"[profile {task:8s}] {name.upper()} GPU busy {P['gpu_busy_ms']:.1f} ms/pass, "
                          f"{P['kernels_per_batch']:.0f} kernels/batch", flush=True)
        eng.fused = can_fuse
        pe.set_cuda_graph(model, True, config=PaddedGraphConfig())
        levels = saved_levels
    if a.profile:
        for task in ("document", "query"):
            P = res.setdefault("profile", {})[task] = profile_task(task)
            tot = sum(P["by_category_ms"].values()) or 1.0
            print(f"[profile {task:8s}] GPU busy {P['gpu_busy_ms']:.1f} ms/pass, {P['kernels_per_batch']:.0f} kernels/batch | "
                  + " ".join(f"{c} {v / tot * 100:.0f}%" for c, v in sorted(P["by_category_ms"].items(), key=lambda kv: -kv[1])
                             if v), flush=True)
            for name, K in P["kernels"].items():            # every kernel: launches per batch and total time
                print(f"[profile {task:8s}]   {K['per_batch']:6.1f}/batch {K['ms']:8.2f} ms  {name}", flush=True)
    if a.recurrent_sweep:
        from packed_encoders.runtime.graphs import PaddedGraphConfig

        default_cut = st_state.engine.gdn.recurrent_max_len
        levels = ["forward"]                        # run_variant reads this: sweep variants are forward-only
        for cut in (int(x) for x in a.recurrent_sweep.split(",")):     # the cutoff only governs the eager path
            st_state.engine.gdn.recurrent_max_len = cut
            with pe.no_cuda_graph(model):
                run_variant(f"eager_r{cut}")
        st_state.engine.gdn.recurrent_max_len = default_cut
        pe.set_cuda_graph(model, True, config=PaddedGraphConfig())
    for lvl, L in res["levels"].items():
        for task, V in L.items():
            s, g = V.get("stock", {}).get("items_per_s"), V.get("graphs", {}).get("items_per_s")
            if s and g:
                print(f"SPEEDUP {lvl:7s} {task:8s} graphs/stock {g / s:5.2f}x  ({g:.1f} vs {s:.1f} items/s)", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print("QWEN35 BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
