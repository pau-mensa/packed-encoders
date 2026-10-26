# ModernBERT CPU packing on the Ryzen 9 9950X

Measured 2026-10-20. This experiment measures `pe.pack()` only. It does not measure
request latency or throughput. The comparison is `8df113d` (immediately before the
packing improvements) against `7e0c8c1` (current HEAD).

## Method

- Real `answerdotai/ModernBERT-base` weights, cached revision
  `8949b909ec900327062f0ebf497f51aef5e6f0c8`, FP32 on CPU in evaluation mode.
- Three fresh processes per version, precision, and thread count; source order
  alternates before/after, after/before, before/after. Each process packs once.
- One thread pinned to CPU 0; eight threads pinned to CPUs 0–7, the physical cores
  of one CCD. Both PyTorch and the engine use that thread count; interop threads = 1.
- Time starts after imports, model loading, and a GC. It includes engine creation,
  native library compilation, calibration/quantization for INT8, installation, and
  the default numerical validation. No separate inference workload runs afterward.
- Explicit native attention, default engine options, no weight rehoming or GEMM tuning.
- Memory is parent-process RSS. The pre-pack RSS is approximately 837 MiB. Peak is
  the process high-water mark when packing exceeds the prior high-water mark, with
  a 5 ms sampling fallback otherwise. Compiler child processes are excluded.
- Post-pack RSS is measured after a GC, without allocator trimming. It includes
  resident model pages, prepared state, scratch, and allocator-retained memory.
- Python 3.12.12, PyTorch 2.11.0+cu128 (MKL), Transformers 5.3.0, GCC 16.2.1;
  CPU governor `powersave` with boost enabled. No GPU work is involved.

## Results

Medians of three fresh processes. GiB = 1,024 MiB.

| Precision | Threads | Pack before → after | Time reduction | Peak RSS before → after | Peak reduction |
|---|---:|---:|---:|---:|---:|
| INT8 | 1 | 33.39 → 19.13 s | 42.7% | 3.55 → 1.70 GiB | 52.2% |
| INT8 | 8 | 8.94 → 5.21 s | 41.7% | 3.59 → 1.86 GiB | 48.2% |
| FP32 | 1 | 2.47 → 2.46 s | 0.1% | 1.26 → 1.26 GiB | 0.0% |
| FP32 | 8 | 1.96 → 1.98 s | -1.1% | 1.25 → 1.25 GiB | -0.1% |

The 1.1% FP32 time increase at eight threads is too small to infer a regression
from three runs; this path has no substantive x86 packing change.

| INT8 memory | 1 thread, before → after | 8 threads, before → after |
|---|---:|---:|
| Peak increase above pre-pack RSS | 2803 → 904 MiB | 2840 → 1068 MiB |
| RSS after pack + GC | 2575 → 1580 MiB | 2614 → 1727 MiB |

The peak increase above pre-pack RSS drops **67.7% at one thread** and **62.4%
at eight threads**. Post-pack resident memory drops about 39% / 34%.

INT8 timing ranges across the three samples:

| Threads | Before | After |
|---|---:|---:|
| 1 | 33.366–33.390 s | 19.127–19.144 s |
| 8 | 8.819–8.963 s | 5.206–5.243 s |

## Interpretation

The INT8 changes improve both packing time and peak resident memory on x86. FP32
packing is essentially unchanged. Prepared INT8 weights, scales, and x86 corrections
occupy **106.21875 MiB in both revisions**; the original FP32 parameters remain.
The memory reduction is in packing intermediates and allocator retention.

The changes in this comparison are:

1. `ae8281d`: native GPTQ column updates and per-layer quantization, with concurrent
   inverse-Hessian factors, replacing the large batched Python implementation.
2. `a419464`: block-structured rotations replace dense rotation-matrix products.
3. `5b84aa6`: calibration hands off each layer's FP32 Gram matrices as it completes,
   replacing all-layer FP64 accumulation.
4. `7e0c8c1`: overlapping projection/attention compilation and reducing default
   calibration from 16 × 384 to 16 × 256 tokens (6,144 to 4,096). Its Accelerate
   header trimming applies only to Apple.

These are measurements of the combined shipped changes. They include the smaller
calibration workload and do not attribute a separate speedup to each commit.

All packs passed their built-in validation and selected the requested native backend
and precision without warnings. This is a packing check, not a retrieval-quality
evaluation of the reduced calibration length.

The focused GPTQ and rotation tests also passed: **12 passed in 1.74 s**
(`tests/test_cpu_int8_gptq.py` and `tests/test_cpu_int8_rotation.py`, eight threads
pinned to CPUs 0–7). No production implementation was changed for this experiment.

## Reproduction and artifacts

The [harness](packing_bench.py) accepts independent source directories so the live
checkout does not have to be modified. Extract each revision's `packed_encoders/`
directory with `git archive`, then run:

```bash
.venv/bin/python benchmarks/cpu_x86/packing_bench.py \
  --sources before=/tmp/modernbert-packing-20261020/before \
            after=/tmp/modernbert-packing-20261020/after \
  --thread-counts 1 --rounds 3 \
  --output benchmarks/cpu_x86/results/packing_20261020_1t.jsonl
```

Repeat with `--thread-counts 8` and the corresponding output path.

- [One-thread raw results](results/packing_20261020_1t.jsonl)
- [Eight-thread raw results](results/packing_20261020_8t.jsonl)
- [Aggregate results](results/packing_20261020_summary.json)
- [Environment and full revisions](results/packing_20261020_environment.json)
