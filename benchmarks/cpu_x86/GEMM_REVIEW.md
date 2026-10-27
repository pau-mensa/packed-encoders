# FP32 GEMM review — 2026-10-17

Reviewed the uncommitted ModernBERT CPU integration, `gemm_x86.c`, and `memory.py`.
The revised kernel preserves the original weights, in-place weight updates, FP32 arithmetic,
and the exact per-output accumulation/reduction order. No commits were made.

The original kernel was already quite efficient. The remaining useful changes were compiler
specialization and a different tile for the MLP output projection:

- Specialize K=768/1152 and 0/1/2 prefetch streams. The original assembly repeatedly tested
  the prefetch count, ran a nested loop, and formed variable-stride addresses inside the K loop.
- Unroll four K steps, preserving each accumulator's FMA order.
- Use 4 activation rows × 6 weight rows at K=1152, retaining 3×8 elsewhere. The active
  weights plus activations occupy 45 KiB instead of 49.5 KiB, fitting Zen 5's 48 KiB L1.
  Both tiles use 24 accumulators. Disassembly of the revised hot functions has no vector
  stack stores. The reduction tree remains identical, including at tile boundaries.
- Keep generic strides and masked tails. Only the hot widths are specialized.

A smaller 3×6 tile alone did not help the large batch. Unrolling eight steps was worse at
8 threads. Specialization alone saved less than the final combination. The cleaned-up
implementation was also compared directly with the winning probe; it was at least as fast.

## Measured results

Ryzen 9 9950X; 1 thread pinned to CPU 2, 8 threads pinned to CPUs 0–7. PyTorch 2.11.0+cu128,
Transformers 5.3.0, GCC 16. Only b1_s16_pad32, b1_s32, and b8_s32 were timed.

Original/revised are alternating A/B in the same process, on the same model and weight
storage: 15 rounds after warmup, 10 forwards per round (3 for b8_s32 at 1 thread).
All output tensors matched bit-for-bit in every round. Medians in milliseconds:

| Workload | Threads | Original | Revised | Reduction |
|---|---:|---:|---:|---:|
| b1_s16_pad32 | 1 | 19.63 | 16.58 | 15.5% |
| b1_s32 | 1 | 30.08 | 27.12 | 9.8% |
| b8_s32 | 1 | 203.35 | 195.64 | 3.8% |
| b1_s16_pad32 | 8 | 8.68 | 8.72 | within noise |
| b1_s32 | 8 | 9.02 | 9.01 | within noise |
| b8_s32 | 8 | 30.50 | 29.52 | 3.2% |

Raw rounds: `results/review_gemm_1t.jsonl`, `results/review_gemm_8t.jsonl`.

With `rehome_weights=True`, another 15-round A/B measured:

| Workload | Threads | Original | Revised |
|---|---:|---:|---:|
| b1_s16_pad32 | 1 | 18.23 | 16.10 |
| b1_s32 | 1 | 28.18 | 25.96 |
| b8_s32 | 1 | 199.75 | 195.36 |
| b1_s16_pad32 | 8 | 8.43 | 8.44 |
| b1_s32 | 8 | 8.79 | 8.80 |
| b8_s32 | 8 | 29.11 | 28.78 |

Raw rounds: `results/review_gemm_rehome_{1,8}t.jsonl`.

## ONNX comparison and the remaining uncertainty

Regenerated FP32 ONNX using the existing export/optimization recipe, including disabled
SkipLayerNorm fusion. ORT **1.31.0**, matching the original survey, was used for the reported
timings. An initial check accidentally used cached ORT 1.29; that result was excluded.
The raw export's parity check against HF had maximum absolute error 2.84e-5.

Fresh ORT 1.31 process medians were 30.42 / 30.53 / 192.21 ms at 1 thread and approximately
11.17 / 11.17 / 37.33 ms at 8 threads. Raw data: `results/review_onnx*.jsonl`.

Because the last 1-thread comparison remained close, ORT and both GEMM variants were also
alternated in one process at 1 thread, with rehomed weights, for 15 rounds:

| Workload | Original + rehome | Revised + rehome | ONNX FP32 |
|---|---:|---:|---:|
| b1_s16_pad32 | 18.36 | 16.22 | 30.23 |
| b1_s32 | 28.29 | 25.96 | 30.25 |
| b8_s32 | 193.37 | 188.47 | 191.69 |

Raw rounds: `results/review_same_process_1t.jsonl`. The live-token relative L2 difference
against ONNX was 2.1e-6 to 4.7e-6. Original/revised GEMM remained bit-identical.
A fresh-process, 9-round repeat measured b8_s32 at 194.41 / 189.50 / 191.66 ms
(original / revised / ONNX), confirming the narrow win in a second matched comparison.
Raw rounds: `results/review_same_process_repeat_1t.jsonl`.

The b1_s32 tie is resolved. b8_s32 is improved consistently, and wins narrowly in this
same-process comparison, but **a robust all-process win is not established**: rehomed b8_s32
ranged from about 188 to 195 ms between processes. The original kernel also shifted in those
processes. This is not a numerical discrepancy or a regression caused by cleaning up the
probe. Allocation/huge-page placement and machine state remain candidates; this review has
not isolated the cause. Do not substitute the best observed 188 ms for every deployment.

## Other review fixes

The unguarded AVX-512 intrinsics made the whole shared-library build fail on AVX2 CPUs.
An existing vector-ABI warning in the attention file was also fatal under `-Werror` on that
target. The build now omits only the unsupported GEMM entry point and suppresses that warning
for the internal vector helpers. Python detects the symbol, retains native row/attention
kernels, and uses `F.linear` for projections. This was compiled with `-march=x86-64-v3` and
executed/validated on the present host; no older CPU or Intel performance claim is made.

`rehome_weights` previously promised that resident memory would not grow. `madvise` is best
effort and the code intentionally ignores failures; other platforms only copy. The docstrings
now state that limitation, and page alignment uses the OS page size instead of hard-coded 4096.
The option remains opt-in. Aliases obtained before rehoming still refer to the old storage,
as documented by the original change.

## Validation and reproduction

- CPU suite: **30 passed**, including the new AVX2 build/fallback test.
- Repository `tests/`: **103 passed, 96 skipped** (CUDA unavailable).
- Expanded GEMM coverage: both tiles, specialized strides, N/K/M tails, M=16/32/256,
  bias, live weights, and bitwise equality across 1/3/8 threads and individual rows.
- `git diff --check`: clean.
- Bare `pytest` also collected the separate `flash-maxsim` checkout, whose CUDA-only tests
  failed during collection without a GPU. The repository suite above was run explicitly.

```bash
taskset -c 2 .venv/bin/python benchmarks/cpu_x86/review_gemm.py --threads 1
taskset -c 0-7 .venv/bin/python benchmarks/cpu_x86/review_gemm.py --threads 8
# Repeat either command with --rehome.
# With onnxruntime==1.31.0 installed and the existing FP32 export available:
taskset -c 2 .venv/bin/python benchmarks/cpu_x86/review_gemm.py \
  --threads 1 --rehome --onnx /path/to/fp32_opt.onnx
```

`native_probe/fp32_review_baseline.c` preserves the exact uncommitted kernel at review start,
so the comparison remains reproducible without a git commit. `review_gemm.py` compiles it
separately and alternates its entry point with the current engine's entry point. No process-
global threading policy or precision contract was changed in the engine.
