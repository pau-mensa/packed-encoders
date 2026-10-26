# Fixed-size x86 Hadamard kernels

Measured on 2026-10-18, Ryzen 9 9950X. Reference: `a309910`, with its exact
`int8_x86.c` preserved in `native_probe/int8_rotation_baseline.c`.

## Change

The runtime activation rotation now uses named vector registers and unrolled
butterflies for the two supported block sizes, 64 and 128. Dispatch occurs before
the loop over blocks, so each loop calls a constant-size specialization. The
indexed `v16 r[8]` array and runtime butterfly loops are gone.

This changes only the x86 int8 inference path. The Hadamard signs, butterfly order
for each output, normalization, row scales, rounding and packed weight format are
unchanged. GPTQ/calibration and the folded attention-output rotation are untouched.
No new option is needed: `precision="int8"` uses the improved kernel automatically
on the existing AVX512-VNNI path.

Assembly inspection with GCC 16.2.1 and the production flags found 107 static vector
instructions referencing the stack in the old `fwht_block`, plus eight in the
standalone quantization entry point. The new outlined rotation/quantization helper
and its standalone entry point both have zero vector stack references. These are
static assembly counts, not hardware performance-counter measurements.

## Full-model results

ModernBERT-base, the same prepared int8 weights and buffers for both variants.
Each process alternates the saved and current native libraries, with 15 measured
rounds after warmup. CPU 2 for one thread; CPUs 0-7 for eight threads. All outputs,
including padded positions, matched bitwise on every round.

| Workload | Threads | Before (ms) | After (ms) | Reduction |
|---|---:|---:|---:|---:|
| b1_s16_pad32 | 1 | 4.374 | 4.270 | 2.4% |
| b1_s32 | 1 | 7.916 | 7.697 | 2.8% |
| b8_s32 | 1 | 55.603 | 54.120 | 2.7% |
| b1_s16_pad32 | 8 | 2.019 | 1.998 | 1.0% |
| b1_s32 | 8 | 2.375 | 2.353 | 0.9% |
| b8_s32 | 8 | 8.727 | 8.529 | 2.3% |

The two small eight-thread improvements are modest; this is one matched process
per thread count. The larger one-thread saving is consistent across the rounds:
about 1.48 ms for b8_s32. No fresh ONNX comparison was needed for this ablation.

## Isolated rotate + quantize

The serial exported entry point, M=256, repeated on the same input. These timings
include rotation, amax, scaling and conversion; they are not full-model savings.

| K | Rotation block | Before (us) | After (us) | Reduction |
|---:|---:|---:|---:|---:|
| 768 | 64 | 45.9 | 26.5 | 42.3% |
| 768 | 128 | 48.0 | 28.2 | 41.4% |
| 1152 | 64 | 67.1 | 38.0 | 43.3% |
| 1152 | 128 | 69.9 | 40.2 | 42.5% |

The no-rotation control was unchanged (about 27.1/39.1 us at K=768/1152).
The benchmark also checks M=16/32, both block sizes, and the no-rotation path.
Quantized bytes, scale bits and the final rotated scratch row match the saved
kernel exactly.

## Validation and reproduction

Repository suite: **113 passed, 96 skipped** (GPU-dependent tests skipped).

The added regression test uses an independent ordered FP32 butterfly reference
and checks exact rotated values, scale bits and quantized bytes. It covers M=1,
7, 16, 32 and 256; one-block and model-width rows; both rotation sizes and no
rotation; zeros, outliers, and magnitudes spanning 1e-15 through 1e15. The existing
full-model tests also exercise both 64- and 128-element rotations.

```bash
HF_HUB_OFFLINE=1 taskset -c 2 .venv/bin/python benchmarks/cpu_x86/rotation.py --threads 1 --rounds 15 --micro
HF_HUB_OFFLINE=1 taskset -c 0-7 .venv/bin/python benchmarks/cpu_x86/rotation.py --threads 8 --rounds 15
.venv/bin/python -m pytest tests -q
```

Raw rounds: `results/rotation_1t.jsonl` (also includes the microbenchmark) and
`results/rotation_8t.jsonl`. Test output: `results/rotation_tests.txt`.
