"""Tables from results/*.json, every column: python benchmarks/cpu_avx2/summarize.py > results/summary.md

Times are medians in ms; ratios are speedups of the engine (above 1 = the engine is faster).
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
ORDER = ["b1_s16_pad32", "b1_s32", "b8_s32", "b1_s1024", "b1_s4096", "b32_s1024", "b32_rand_pad1024"]
NAN = float("nan")


def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(str(RESULTS / pattern))) if os.path.getsize(p)]


def num(v, digits=1):
    return "—" if v is None or v != v else f"{v:,.{digits}f}"


def ratio(a, b):
    return "—" if a is None or b is None else f"{a / b:.2f}x"


def threads(t):
    return f"{t} thread" + ("s" if t != 1 else "")


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def sweep():
    runs = sorted(load("sweep_t*.json"), key=lambda d: d["threads"])
    print("## Engine vs HF across thread counts (`sweep_t*.json`)\n")
    rows = []
    for wl in ORDER:
        cells = [d["rows"].get(wl) for d in runs]
        if not any(cells):
            continue
        rows.append([wl] + [f"{num(c['hf_ms'])} / {num(c['engine_ms'])} ({ratio(c['hf_ms'], c['engine_ms'])})"
                            if c else "—" for c in cells])
    table(["workload"] + [f"{threads(d['threads'])}: HF / engine" for d in runs], rows)
    print("Engine op profile at each thread count (ms per call):\n")
    ops = ["projections", "attn_local", "attn_global", "rope_layout", "add_norm", "geglu", "core_total"]
    rows = []
    for d in runs:
        for wl in ORDER:
            p = d["rows"].get(wl, {}).get("profile_ms")
            if p:
                rows.append([f"{d['threads']}t {wl}"] + [num(p.get(op)) for op in ops])
    table(["threads / workload"] + ops, rows)


def omp():
    runs = load("omp_t*.json")
    if not runs:
        return
    print("## OpenMP wait policy above and at the core count (`omp_t*_*.json`)\n")
    rows = []
    for d in sorted(runs, key=lambda d: (d["threads"], d.get("omp_wait_policy") or "")):
        for wl in ("b1_s32", "b8_s32"):
            r = d["rows"][wl]
            rows.append([f"{d['threads']}", d.get("omp_wait_policy") or "default", wl, num(r["hf_ms"]),
                         num(r["engine_ms"]), ratio(r["hf_ms"], r["engine_ms"]),
                         num(r["profile_ms"]["projections"])])
    table(["threads", "OMP_WAIT_POLICY", "workload", "HF", "engine", "engine vs HF", "engine projections"], rows)


def ablation():
    native = {d["threads"]: d for d in load("abl_native_t*.json")}
    sdpa = {d["threads"]: d for d in load("abl_sdpa_t*.json")}
    print("## Native x86 attention vs the engine's SDPA fallback (`abl_*_t*.json`)\n")
    rows = []
    for t in sorted(native):
        for wl, r in native[t]["rows"].items():
            s = sdpa[t]["rows"][wl]
            attn = lambda x: x["ops_ms"].get("local", 0) + x["ops_ms"].get("global", 0)
            rows.append([f"{t}", wl, num(r["ms"]), num(s["ms"]), num(attn(r)), num(attn(s)),
                         f"{attn(s) / attn(r):.2f}x"])
    table(["threads", "workload", "forward native", "forward SDPA", "attention native", "attention SDPA",
           "native attention speedup"], rows)


def best(rows, model, wl, mode):
    cells = [v for k, v in rows.items() if k.startswith(f"{model}|{wl}|{mode}|")]
    return min(cells, key=lambda v: v["ms"]) if cells else None


def ort_rows(threads):
    out = {}
    for d in load(f"ort*_t{threads}_*.json"):
        out.update(d["rows"])
    return out


def onnx():
    pe = {}
    for d in load("hf_engine_t*.json"):
        for wl, r in d["rows"].items():
            pe[(d["threads"], wl)] = r
    print("## HF, engine and ONNX Runtime FP32 (`hf_engine_*.json`, `ort_t*.json`)\n")
    print("ONNX: `batch` feeds the padded batch; large batches run one row at a time, at the padded width "
          "(as published) or trimmed to each row's length.\n")
    for t in sorted({k[0] for k in pe}):
        rows, ort = [], ort_rows(t)
        for wl in ORDER:
            r = pe.get((t, wl))
            if not r:
                continue
            padded = "rowpad" if wl.startswith("b32") else "batch"
            f, ft = best(ort, "fp32_opt", wl, padded), best(ort, "fp32_opt", wl, "rowtrim")
            e = r["engine_ms"]
            rows.append([wl, num(r.get("hf_ms")), num(f and f["ms"]), num(ft and ft["ms"]), num(e),
                         ratio(r.get("hf_ms"), e), ratio(f and f["ms"], e), ratio(ft and ft["ms"], e)])
        print(f"### {threads(t)}\n")
        table(["workload", "HF", "ONNX FP32", "ONNX FP32 trimmed", "engine FP32", "vs HF", "vs ONNX",
               "vs ONNX trimmed"], rows)
    print("## ONNX Runtime int8 variants (`ort*_t*.json`; relL2 on these random-ID inputs)\n")
    models = ["int8", "int8_pc", "int8_lin", "int8_lin_u8", "int8_rr", "int8_proj", "int8_proj_pc", "int8_w8",
              "int8_w8a8"]
    for t in (1, 8, 16):
        ort = ort_rows(t)
        rows = []
        for model in models:
            for wl in ORDER:
                for mode in ("batch", "rowpad", "rowtrim"):
                    c = best(ort, model, wl, mode)
                    if c:
                        e = pe.get((t, wl), {}).get("engine_ms")
                        rows.append([model, wl, mode, num(c["ms"]), ratio(c["ms"], e),
                                     f"{c['relL2']:.3f}", f"{c['cos_median']:.4f}"])
        if rows:
            print(f"### {threads(t)}\n")
            table(["model", "workload", "feed", "ms", "engine FP32 vs it", "relL2", "token cos median"], rows)
    print("## Real-text accuracy vs HF FP32 (`acc_*.json`; README paragraphs, 29–512 tokens, each run alone)\n")
    rows = []
    for d in load("acc_*.json"):
        for name, r in d["rows"].items():
            rows.append([name, f"{r['relL2']:.3g}", f"{r['cos_median']:.5f}", f"{r['cos_p5']:.4f}",
                         f"{r['cos_min']:.4f}", f"{r['pooled_cos_min']:.5f}", str(r["tokens_below_0.5"]),
                         f"{d['paragraphs']} / {d['tokens']}"])
    table(["model", "relL2", "token cos median", "p5", "min", "mean-pooled cos min", "tokens < 0.5",
           "paragraphs / tokens"], rows)


if __name__ == "__main__":
    print("# CPU engine on an AVX2-only x86 CPU: raw-result tables\n")
    print("Generated by `summarize.py` from `results/*.json`. See `README.md` for the machine and protocol.\n")
    sweep()
    omp()
    ablation()
    onnx()
