"""Compare pack-time attention sampling on an x86 host; no model download needed.

    python benchmarks/cpu_review/attention_calibration.py --threads 1 --out calibration.json

--isa x86-64-v3 exercises AVX2 code on a newer CPU; it does not reproduce Zen 3.
Native compilation is excluded from calibration timings. Methods alternate order.
baseline reproduces the original 240-call sweep; adaptive runs the current tuner.
single is an ablation with one call per sample, including for fast kernels.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch

from packed_encoders.cpu import attention


def sample(att, heads, max_seqlen, max_tokens, method):
    """Reference sampling; retain every shape and three measured rounds."""
    rng = torch.Generator().manual_seed(173)
    limit = min(max_seqlen, max_tokens)
    with torch.inference_mode():
        for length in sorted({min(n, limit) for n in (32, 128, 512)}):
            for count in sorted({1, min(8, max_tokens // length)}):
                lengths = (length,) * count
                boundaries = torch.tensor(lengths, dtype=torch.int32, device="cpu")
                q, k, v = [torch.randn(1, heads, length * count, 64, generator=rng,
                                      dtype=torch.float32, device="cpu")
                           for _ in range(3)]
                for local in (False, True):
                    if not local and length > 128:
                        continue
                    half = att.half if local else length
                    calls = {
                        "native": lambda: att._native(q, k, v, lengths, boundaries, half, True),
                        "sdpa": lambda: att._layout(att._sdpa(q, k, v, lengths, local=local), True),
                    }
                    repeats = 3 if method == "baseline" else 1
                    samples = {name: [] for name in calls}
                    for call in calls.values():
                        for _ in range(repeats):
                            call()
                    for round_ in range(3):
                        for name in reversed(calls) if round_ % 2 == 0 else calls:
                            start = time.perf_counter_ns()
                            for _ in range(repeats):
                                calls[name]()
                            samples[name].append((time.perf_counter_ns() - start) / repeats / 1e6)
                    medians = {name: statistics.median(values) for name, values in samples.items()}
                    att.attention_tuning[local, length, count] = {
                        "selected": "native" if medians["native"] < .95 * medians["sdpa"] else "sdpa",
                        "median_ms": medians,
                    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--max-seqlen", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--isa", choices=("native", "x86-64-v3"), default="native")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    shared_library = attention.shared_library

    def build(name, commands, **kwargs):
        commands = [[f"-march={args.isa}" if word == "-march=native" else word for word in command]
                    for command in commands]
        return shared_library(name, commands, **kwargs)

    with patch.object(attention, "shared_library", build):
        attention._x86_library.cache_clear()
        library, _ = attention._x86_library()
    rows = []
    methods = ("baseline", "single", "adaptive")
    for round_ in range(args.rounds):
        for method in methods if round_ % 2 == 0 else methods[::-1]:
            att = attention.CPUAttention("auto", 64, 64, args.threads)
            assert att.library is library
            counts = {name: 0 for name in ("native", "sdpa")}
            for name in counts:
                call = getattr(att, f"_{name}")

                def counted(*values, _call=call, _name=name, **kwargs):
                    counts[_name] += 1
                    return _call(*values, **kwargs)

                setattr(att, f"_{name}", counted)
            start = time.perf_counter()
            if method == "adaptive":
                att.tune_attention(args.heads, args.max_seqlen, args.max_tokens)
            else:
                sample(att, args.heads, args.max_seqlen, args.max_tokens, method)
            elapsed = time.perf_counter() - start
            row = {"round": round_, "method": method, "seconds": elapsed, "calls": counts,
                   "buckets": {str(key): value for key, value in att.attention_tuning.items()}}
            rows.append(row)
            print(json.dumps(row), flush=True)
            att.close()
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    report = {"cpu": cpu, "torch": torch.__version__, "threads": args.threads,
              "affinity": sorted(os.sched_getaffinity(0)), "isa": args.isa,
              "heads": args.heads, "max_seqlen": args.max_seqlen, "max_tokens": args.max_tokens,
              "OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY"), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({method: statistics.median(row["seconds"] for row in rows if row["method"] == method)
                      for method in methods}), flush=True)


if __name__ == "__main__":
    main()
