# INT8 dequantization and normalization fusion

Measured on 2026-10-18, Ryzen 9 9950X, GCC 16.2.1. Baseline: `46df86f`, including
cache-aware GEMM, fixed-size Hadamard rotation, and the short-attention optimization.

## Retained change

`packed_encoders/cpu/int8_x86.c` fuses dequantization into the first residual-add /
LayerNorm statistics pass. It removes a temporary FP32 row write and reread at both
normalizations in each layer. Two explicit 512-bit vectors maintain the same 32
partial sums as the shared `pe_add_norm_row` implementation. The mean/variance
reduction tree, explicit variance FMAs, normalization, and optional bias order are
unchanged. The final model normalization also uses the fused helper.

No change to the activation rotation/quantization algorithm, scratch allocation,
OpenMP scheduling, GEMM, preparation, Python API, FP32 inference, or Apple code.
There is no new ISA requirement beyond the existing AVX512-VNNI path.

## Baseline profile

Approximate stage wall times from the baseline instrumented inside the OpenMP
region. Thread zero times each entire worksharing stage, including its barrier.
Twenty warm model forwards; profiling overhead is included. Norm and GeGLU columns
include their subsequent rotation and quantization. Attention and Python glue are
in the total, but not in these native counters.

| Workload, 1 thread | Total ms | GEMM | Norm | GeGLU | RoPE | Standalone quantization |
|---|---:|---:|---:|---:|---:|---:|
| b1_s16_pad32 | 4.384 | 3.452 | 0.191 | 0.166 | 0.085 | 0.042 |
| b1_s32 | 7.450 | 5.872 | 0.361 | 0.321 | 0.167 | 0.080 |
| b8_s32 | 52.289 | 41.096 | 2.870 | 2.542 | 1.482 | 0.631 |

`results/int8_rows_profile_1t.jsonl` contains the counters. The instrumented baseline
is preserved as `native_probe/int8_rows_profile.c`.

## Full-model results

| Workload | Threads | Before ms | After ms | Time reduction |
|---|---:|---:|---:|---:|
| b1_s16_pad32 | 1 | 4.237 | 4.207 | 0.70% |
| b1_s32 | 1 | 7.559 | 7.483 | 1.01% |
| b8_s32 | 1 | 52.777 | 52.233 | 1.03% |
| b1_s16_pad32 | 8 | 1.988 | 1.978 | 0.47% |
| b1_s32 | 8 | 2.318 | 2.306 | 0.51% |
| b8_s32 | 8 | 8.467 | 8.393 | 0.87% |

The two small eight-thread differences are modest relative to run-to-run noise.
The batched case improved in the exploratory runs as well as this final comparison.

Same prepared ModernBERT-base INT8 weights and buffers within each process; only
`q8.library` changes. Fifteen measured alternating A/B rounds after warmup. Each
variant gets an untimed warm forward before its timed repetitions. Twenty forwards
per round, except b8_s32 at one thread uses five. One thread pinned to CPU 2; eight
threads pinned to CPUs 0-7. The two processes run serially. Output tensors are
compared as int32 views after every round, including padded positions.

These are incremental savings over the committed engine, not a new ONNX comparison.
Raw rounds: `results/int8_rows_{1,8}t.jsonl`.

## Rejected experiments

- **Scalar-loop dequantization + norm fusion:** slower in the isolated row probe;
  using explicit vectors for the 32 ordered partial sums made the fusion worthwhile.
- **Dequantization + GeGLU:** no consistent full-model improvement. Both an
  auto-vectorized scalar loop and explicit 512-bit polynomial arithmetic were tried.
- **Dequantization + RoPE:** consistently slower for the batched one-thread case.
  Both head-major traversal and projection-major traversal with hoisted trig loads
  were tried. Removing the intermediate row alone did not compensate for the
  changed load/store pattern and generated code.
- **Normalization directly into Hadamard registers:** output bits matched, but it
  did not improve on the smaller norm-only fusion. Kept the existing rotation code.
- **SIMD horizontal maximum for quantization:** replaced the serial 16-lane scan
  with `_mm512_reduce_max_ps`. Small/noisy full-model differences and no consistent
  gain on top of norm fusion; kept the original reduction.

The direct dequantization + normalization microprobe (without rotation/quantization)
was approximately 9-12% faster at the production hidden width. This is not the full
normalization-stage or model saving. Full-model results determined what was retained.

Exploratory raw results are `int8_rows_initial_1t.jsonl` (7 rounds),
`int8_rows_vector_1t.jsonl` (7), `int8_rows_emit_1t.jsonl` (9),
`int8_rows_candidates_8t.jsonl` (15), and `int8_rows_max_ablation_1t.jsonl` (15).
In those files `normv` is the retained vector norm fusion, `norm_emit` also feeds
normalization directly into rotation, `hmax` changes only the maximum reduction,
and `normv_max` combines vector norm fusion and the maximum reduction. In the last
file only, `after` is that combined candidate, which was subsequently rejected;
use the final `int8_rows_{1,8}t.jsonl` for the retained patch.

## Validation

- **124 passed, 96 skipped**, 24 existing NVML warnings. GPU tests skipped on this
  CPU host. Full output: `results/int8_rows_tests.txt`.
- New tests compare against separate dequantization and the existing shared
  residual/LayerNorm body, using int32 views for exact FP32 bit comparisons.
- 260 combinations: widths 64/128/192/768/1152, row counts 1/7/16/32/256, optional
  bias, two epsilons, final FP32 outputs or 64/128-element rotation and quantization.
  Includes zero/constant rows and varying dequantization magnitudes. Residuals,
  normalized outputs, rotated values, scales, and quantized bytes are checked.
- All six full-model benchmark cases preserve output bits on every round.
- Existing tests cover emulated INT8 projections, thread-count parity, and microbatches.
- `git diff --check` clean. Nothing committed.

The baseline source snapshot `native_probe/int8_rows_baseline.c` is byte-identical
to `46df86f:packed_encoders/cpu/int8_x86.c`. SHA256:
`e8b0de078c3bc5dabebaec780c13130989fcdbb1fd1761f842ff7c795c6a1a80`.

## Reproduction

Run from the repository root, serially:

```bash
B=benchmarks/cpu_x86
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 taskset -c 2 .venv/bin/python $B/int8_rows.py --threads 1
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 taskset -c 0-7 .venv/bin/python $B/int8_rows.py --threads 8
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 taskset -c 2 .venv/bin/python $B/int8_rows.py --threads 1 --profile $B/native_probe/int8_rows_profile.c
.venv/bin/python -m pytest tests -q
```

`--candidate FILE.c` adds another private benchmark variant; no production ablation
switch is needed. `--micro-only` checks/measures the existing quantization entry point.
