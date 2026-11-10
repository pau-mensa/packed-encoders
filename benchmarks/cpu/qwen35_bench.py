"""Pretrained Qwen3.5 CPU vs ONNX FP32/INT8, with identical inputs and HF accuracy gates.

    python benchmarks/cpu/qwen35_bench.py --model benchmarks/data_cache/qwen35-0.8b \
        --threads 8 --out benchmarks/results/qwen35/t8.json

Requires onnx and onnxruntime in addition to the CPU runtime. ONNX models are independent
ordinary-op graphs, with a chunked delta rule, explicit sequence width and dynamic batch.
ORT uses CPUExecutionProvider, sequential execution, all graph optimizations, and both
spinning settings. Each row records every trial; compare against the faster spinning mode.
No tokenization, model preparation or export is timed. Run thread counts in fresh processes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

if __package__:
    from .qwen35_onnx import export, quantize, reuse_quantized_weights
else:
    from qwen35_onnx import export, quantize, reuse_quantized_weights


def load_model(path):
    """Load HF text weights, or a local topk text backbone without its CUDA wrapper."""
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    config = Path(path) / "config.json"
    if config.is_file() and json.loads(config.read_text()).get("model_type") == "topk_embed":
        from safetensors import safe_open
        cfg = Qwen3_5TextConfig(**json.loads(config.read_text())["text_config"])
        with torch.device("meta"):
            model = Qwen3_5TextModel(cfg)
        prefix = "model.language_model."
        weights = {}
        for shard in sorted(Path(path).glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        weights[key[len(prefix):]] = f.get_tensor(key).float()
        model.load_state_dict(weights, strict=True, assign=True)
        # Rotary frequency buffers are nonpersistent and absent from the checkpoint.
        model.rotary_emb = type(model.rotary_emb)(cfg, device="cpu")
        model.set_attn_implementation("sdpa")
    else:
        model = Qwen3_5TextModel.from_pretrained(path, dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    return model.eval()


def provenance():
    root = Path(__file__).resolve().parents[2]
    paths = [*root.joinpath("packed_encoders/cpu").glob("*.[ch]"),
             *root.joinpath("packed_encoders/cpu").glob("*.py"),
             root / "packed_encoders/arch/qwen3_5/cpu.py", Path(__file__),
             Path(__file__).with_name("qwen35_onnx.py")]
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    cpu = Path("/proc/cpuinfo")
    info = cpu.read_text().split("\n\n")[0] if cpu.exists() else ""
    cpu_fields = {"vendor_id", "model name", "cpu family", "model", "stepping", "microcode", "cpu cores", "siblings", "flags"}
    cpu_info = {key.strip(): value.strip() for line in info.splitlines() if ":" in line
                for key, value in [line.split(":", 1)] if key.strip() in cpu_fields}
    return {"commit": git("rev-parse", "HEAD"), "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
            "sources": sources, "cpu": cpu_info, "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "environment": {k: os.environ.get(k) for k in ("OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES", "MKL_NUM_THREADS")}}


def workloads(vocab):
    rng = torch.Generator().manual_seed(0)
    result = {}
    for name, lengths, width in (
        ("b1_s32", (32,), 32), ("b8_s32", (32,) * 8, 32),
        ("b1_s128", (128,), 128), ("b8_pad128", (128, 96, 64, 32, 16, 8, 4, 1), 128),
        ("b1_s512", (512,), 512),
        ("b1_s1024", (1024,), 1024),
        ("b8_pad1024", (1024, 768, 512, 256, 128, 64, 32, 8), 1024),
    ):
        ids = torch.randint(5, min(vocab, 248000), (len(lengths), width), generator=rng)
        mask = (torch.arange(width)[None] < torch.tensor(lengths)[:, None]).long()
        result[name] = ids, mask
    return result


def accuracy(got, want):
    got, want = got.float(), want.float()
    cos = F.cosine_similarity(got, want, dim=-1)
    return {"cos_min": float(cos.min()), "cos_median": float(cos.median()),
            "relative_l2": float((got - want).norm() / want.norm()), "max_abs": float((got - want).abs().max())}


def measure(fn, warmup, trials):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"ms": statistics.median(samples), "trials_ms": samples,
            "spread_pct": (max(samples) - min(samples)) / statistics.median(samples) * 100}


def session(path, threads, spin):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", str(spin))
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir", default="benchmarks/data_cache/qwen35_onnx")
    ap.add_argument("--only", default="b1_s32,b8_s32,b1_s128,b8_pad128")
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--spin", default="1,0")
    ap.add_argument("--onnx-int8", default="int8_dynamic,int8_w8a8")
    ap.add_argument("--engine-precision", default="fp32,int8", help="comma-separated fp32/int8; use fp32 for the initial baseline")
    ap.add_argument("--no-prepack-weights", action="store_true", help="use PyTorch FP32 projections for ablations")
    ap.add_argument("--stage", choices=("all", "engine", "export", "onnx"), default="all")
    ap.add_argument("--time-hf", action="store_true", help="also time stock HF (always used as the accuracy reference)")
    ap.add_argument("--rowtrim", action="store_true", help="also measure the sum of independently warmed ONNX row medians")
    a = ap.parse_args()
    if a.threads <= 0 or a.trials <= 0 or a.warmup < 0:
        ap.error("threads/trials must be positive; warmup must be nonnegative")
    precisions = a.engine_precision.split(",")
    variants = tuple(filter(None, a.onnx_int8.split(",")))
    if not precisions or set(precisions) - {"fp32", "int8"}:
        ap.error("--engine-precision must contain fp32 and/or int8")
    if set(variants) - {"int8_dynamic", "int8_w8a8"}:
        ap.error("unknown --onnx-int8 variant")
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    import packed_encoders as pe
    import transformers
    import onnxruntime as ort
    model = load_model(a.model)
    loads = workloads(model.config.vocab_size)
    unknown = set(a.only.split(",")) - loads.keys()
    if unknown:
        ap.error(f"unknown workloads {sorted(unknown)}")
    loads = {k: loads[k] for k in a.only.split(",")}
    widths = sorted({ids.shape[1] for ids, _ in loads.values()})
    directory = Path(a.dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Refuse accidentally reusing weights from another model/config.
    identity = json.loads(json.dumps({"model": a.model, "config": model.config.to_dict(), "onnx_schema": 1}))
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("ONNX directory belongs to another model/config; choose a new --dir")
    manifest.write_text(json.dumps(identity, indent=2))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": a.model, "threads": a.threads, "torch": torch.__version__, "transformers": transformers.__version__,
              "onnxruntime": ort.__version__, "platform": platform.platform(), "processor": platform.processor(),
              "warmup": a.warmup, "trials": a.trials, "rows": {},
              "prepack_weights": not a.no_prepack_weights,
              "provenance": provenance(), "causal": getattr(model.config, "is_causal", True),
              "workloads": {name: {"shape": list(ids.shape), "lengths": mask.sum(1).tolist()}
                            for name, (ids, mask) in loads.items()}}
    if out.exists():
        previous = json.loads(out.read_text())
        if any(previous.get(k) != result[k] for k in ("model", "threads", "torch", "transformers", "onnxruntime",
                                                      "warmup", "trials", "workloads", "causal", "provenance", "prepack_weights")):
            raise ValueError("output belongs to another configuration; choose a new --out")
        result["rows"].update(previous["rows"])
    def record(key, stats):
        result["rows"][key] = stats
        out.write_text(json.dumps(result, indent=2))
        print(f"{key:45s} {stats['ms']:10.2f} ms | cos {stats['cos_min']:.6f} "
              f"med {stats['cos_median']:.6f} relL2 {stats['relative_l2']:.3e}", flush=True)
    def gate(metrics, precision):
        if not all(np.isfinite(list(metrics.values()))):
            raise ValueError(f"nonfinite output: {metrics}")
        if precision == "fp32":
            ok = metrics["cos_min"] >= .99999 and metrics["relative_l2"] <= 2e-4
        else:
            ok = metrics["cos_median"] >= .99 and metrics["relative_l2"] <= .25
        if not ok:
            raise ValueError(f"{precision} output failed HF parity: {metrics}")
    refs = {}
    with torch.no_grad():
        for name, (ids, mask) in loads.items():
            fn = lambda: model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            refs[name] = fn()[mask.bool()].clone()
            print(f"HF reference ready: {name} ({int(mask.sum())} real tokens)", flush=True)
            if a.time_hf and a.stage in ("all", "engine"):
                record(f"hf_fp32|{name}", {**measure(fn, a.warmup, a.trials), **accuracy(refs[name], refs[name])})
        if a.stage in ("all", "engine"):
            for precision in precisions:
                start = time.perf_counter()
                pe.pack(model, engine=pe.Qwen35CPU(threads=a.threads, prepack_weights=not a.no_prepack_weights), precision=precision)
                backend = pe.get_engine(model).state.projection_backend
                print(f"prepared engine {precision} in {time.perf_counter() - start:.2f}s", flush=True)
                for name, (ids, mask) in loads.items():
                    fn = lambda: model(input_ids=ids, attention_mask=mask).last_hidden_state
                    metrics = accuracy(fn()[mask.bool()], refs[name])
                    gate(metrics, precision)
                    record(f"engine_{precision}|{name}", {**measure(fn, a.warmup, a.trials), **metrics,
                                                         "projection_backend": backend})
                pe.unpack(model)
                gc.collect()
        if a.stage in ("all", "export"):
            for width in widths:
                path = directory / f"fp32_s{width}.onnx"
                if not path.exists():
                    export(model, directory, width)
                # Check FP32 export against HF on every workload before quantizing.
                sess = session(path, a.threads, 0)
                for name, (ids, mask) in loads.items():
                    if ids.shape[1] == width:
                        got = torch.from_numpy(sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0])
                        metrics = accuracy(got[mask.bool()], refs[name])
                        gate(metrics, "fp32")
                        print(f"export parity s{width} {name}: {metrics}", flush=True)
                del sess
                gc.collect()
                for variant in variants:
                    dest = directory / f"{variant}_s{width}.onnx"
                    if not dest.exists():
                        quantize(path, dest, variant)
                        print(f"built {dest}", flush=True)
                    gc.collect()
            if a.rowtrim:
                for _, (ids, mask) in loads.items():
                    width = ids.shape[1]
                    for length in set(mask.sum(1).tolist()) - {width}:
                        path = directory / f"fp32_s{length}.onnx"
                        if not path.exists():
                            export(model, directory, length)
                        for variant in variants:
                            dest = directory / f"{variant}_s{length}.onnx"
                            reuse_quantized_weights(directory / f"fp32_s{width}.onnx",
                                                    directory / f"{variant}_s{width}.onnx", path, dest)
    del model
    gc.collect()
    if a.stage in ("all", "onnx"):
        for variant in ("fp32", *variants):
            for width in widths:
                for spin in map(int, a.spin.split(",")):
                    sess = session(directory / f"{variant}_s{width}.onnx", a.threads, spin)
                    for name, (ids, mask) in loads.items():
                        if ids.shape[1] != width:
                            continue
                        feed = {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
                        fn = lambda: sess.run(None, feed)[0]
                        metrics = accuracy(torch.from_numpy(fn())[mask.bool()], refs[name])
                        gate(metrics, "fp32" if variant == "fp32" else "int8")
                        record(f"onnx_{variant}|{name}|spin{spin}", {**measure(fn, a.warmup, a.trials), **metrics})
                    del sess
                    gc.collect()
        if a.rowtrim:
            for variant in ("fp32", *variants):
                for name, (ids, mask) in loads.items():
                    lengths = mask.sum(1).tolist()
                    if min(lengths) == ids.shape[1]:
                        continue
                    stats, outputs = [], []
                    for row, length in enumerate(lengths):
                        best = None
                        for spin in map(int, a.spin.split(",")):
                            sess = session(directory / f"{variant}_s{length}.onnx", a.threads, spin)
                            feed = {"input_ids": ids[row:row + 1, :length].numpy(),
                                    "attention_mask": np.ones((1, length), dtype=np.int64)}
                            fn = lambda: sess.run(None, feed)[0]
                            got = torch.from_numpy(fn())[0]
                            timing = measure(fn, a.warmup, a.trials)
                            if best is None or timing["ms"] < best[0]["ms"]:
                                best = timing, got
                            del sess
                            gc.collect()
                        stats.append(best[0])
                        outputs.append(best[1])
                        print(f"rowtrim {variant} {name} row {row + 1}/{len(lengths)} "
                              f"length {length}: {best[0]['ms']:.2f} ms", flush=True)
                    metrics = accuracy(torch.cat(outputs), refs[name])
                    gate(metrics, "fp32" if variant == "fp32" else "int8")
                    # This deliberately favors ORT: each row is independently warmed,
                    # with its own best spinning mode. Session changes are not timed.
                    record(f"onnx_{variant}|{name}|rowtrim_lower_bound",
                           {"ms": sum(s["ms"] for s in stats), "rows": stats, **metrics,
                            "protocol": "sum of independently warmed row medians, best spin per row"})
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
