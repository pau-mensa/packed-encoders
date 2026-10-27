Add an experimental padding-free ModernBERT CPU runtime for Apple Silicon and
Linux x86-64. `pe.pack(model)` selects it for FP32 CPU models. Exact FP32 is the
default; `precision="int8"` enables Hadamard-rotated GPTQ projections on SME2 or
AVX512-VNNI hardware.

## Runtime behavior

- Live tokens remain packed through the encoder; batches split at sequence
  boundaries under `max_tokens`. HF task heads, float masks and int32 token IDs
  are supported. Training, gradient-enabled calls, extra HF outputs and irregular
  masks fall back to the original HF forward. `pe.unpack()` restores it.
- Linux x86 `auto` attention measures native kernels and tiled SDPA at pack time
  using bounded local/global and single/batched shapes. Explicit backends bypass
  calibration. Native row kernels and supported projection kernels stay enabled
  when SDPA wins. The int8 path honors these decisions.
- Default native workers are capped to physical cores in Linux affinity when
  topology is available; explicit `threads=` overrides the cap. PyTorch/BLAS pools
  remain application-controlled. Set those before packing.
- AVX-512 FP32 GEMM uses original weights; AVX2 uses `F.linear`. Apple SME2 FP32
  projections use prepared weight copies by default; `prepack_weights=False`
  skips those copies. Optional `tune_gemm` and `rehome_weights` remain available.
- Native compilation is cached across processes in a private directory. Keys
  include sources/headers, compiler/ISA, build flags, OS and OpenMP runtime.
  Publication is locked and checksummed. An unavailable cache falls back to a
  temporary build; `PACKED_ENCODERS_CPU_CACHE_DIR=""` disables disk reuse.
- Prepared heads use separate GEMM tuning choices from the backbone. Owned CPU
  projection and CUDA `F.linear` outputs avoid redundant copies; custom heads
  and potentially borrowed kernel outputs retain defensive copies.
- `pe.validate()` honors CPU `seq_lens` and `cos_threshold`. Transformers 5
  `layer_types` supplies the global-attention schedule when the legacy field is
  absent.

## Packaging

GPU dependencies move to the `cuda` extra. GPU users must install
`packed-encoders[cuda]`; `fa2`, `fa4` and `qwen3_5` include it. A CPU install does
not require Triton or CuteDSL. On Linux:

```bash
uv pip install --no-sources --torch-backend cpu -e .
```

## Benchmarks

ModernBERT-base, random token IDs, with identical inputs across systems. Baselines
are **HF SDPA**, **ORT FP32** and **ORT MatMulNBits W8A8**. ORT runs short batches
together and long batches one row at a time, trimming every row to its live length.
The single padded short query is also trimmed for ORT.

Ryzen 9 9950X, 8 threads pinned to one CCD, milliseconds:

| Workload | HF SDPA | ORT FP32 | Engine FP32 | ORT W8A8 | Engine int8 |
|---|---:|---:|---:|---:|---:|
| 1 × 16 live tokens, padded to 32 | 17.40 | 9.92 | 8.58 | 4.39 | 2.37 |
| 1 × 32 | 17.13 | 11.61 | 8.95 | 5.85 | 2.34 |
| 8 × 32 | 74.86 | 37.48 | 29.83 | 30.31 | 8.38 |

16 threads pinned to physical cores across both CCDs, seconds:

| Workload | HF SDPA | ORT FP32 trimmed | Engine FP32 | ORT W8A8 trimmed | Engine int8 |
|---|---:|---:|---:|---:|---:|
| 32 × 1024 | 9.77 | 16.07 | 7.43 | 15.97 | 2.18 |
| 32 × random lengths ≤1024 | 9.64 | 6.68 | 3.75 | 6.51 | 1.11 |

These are one-round fresh-process measurements: seven samples for short queries,
three for 1024-token batches. Short W8A8 takes the faster spinning setting per row;
long ORT uses spinning. All systems use the same inputs. Engine precision is
checked so a fallback cannot be mislabeled int8. Raw samples, environments, model
hashes and reproduction commands are in `benchmarks/cpu_review/`.

Actual 32-row batches at 8192 tokens, 16 threads, seconds:

| Workload | HF SDPA | ORT FP32 trimmed | Engine FP32 | ORT W8A8 trimmed | Engine int8 |
|---|---:|---:|---:|---:|---:|
| 32 × 8192 | 156.02 | 849.82 | 103.12 | 883.54 | 53.90 |
| 32 × random lengths ≤8192 | 160.31 | 218.66 | 37.81 | 224.52 | 17.29 |

These 8192-token results are **single timed complete batches**, with one ORT row
warmed first and no complete-batch warmup. They are not extrapolations. The
random-length inputs use seed 8192.
HF runs padded batches; ORT trims every row. On this ragged workload, the
engine FP32 lead is 4.2x over HF and 5.8x over trimmed ORT.

On the same saved real-text corpus (24 paragraphs, 3,466 tokens):

| System | relL2 vs HF FP32 | Token cosine p5 | Pooled cosine min | Tokens below 0.5 cosine |
|---|---:|---:|---:|---:|
| Engine FP32 | 3.86e-6 | 1.000000 | 1.000000 | 0 |
| Engine GPTQ int8 | 0.0497 | 0.999195 | 0.999592 | 3 |
| ORT MatMulNBits W8A8 | 0.0691 | 0.998514 | 0.999388 | 7 |

Both int8 methods preserve most token directions while retaining low-cosine
outliers. These are numerical checks, not retrieval-quality evaluations.

Fresh-process native-cache measurements (one observation each, 8 threads): FP32
packing was 1.72 s with a cold cache and 0.31 s warm; int8 was 4.93 s cold and
3.52 s warm. Warm processes invoked no native compiler. Quantization and attention
calibration still run per model.

The thread sweep covers the actual `threads=None` default: with PyTorch at 32
threads, capping native workers to 16 reduced 8 × 32 latency from 375 ms to 81 ms
on this Ryzen. Setting both pools to 16 gave 59 ms. Samples are noisy when
oversubscribed; the cap does not replace application-level thread configuration.

## Validation

- Local CPU/kernel/shared suites: **176 passed, 8 skipped**. Native AVX-512 GEMM,
  VNNI int8 and an AVX2-compiled attention target executed. Skips are Apple-only.
- RTX 5090 CUDA epilogues: **6 passed**, including real graph replay and output
  lifetime checks after redundant copies were removed.
- Cache tests cover key invalidation, corruption, failed builds, unsafe-directory
  fallback and concurrent fresh-process publication.
- The native hardware checker requires the selected ISA before running all CPU
  suites and writes JSON/JUnit artifacts. The M4 launcher also measures cold/warm
  packing and paired HF/engine timings at 1 and 8 threads.

The native validation results above are from a Ryzen 9 9950X running Linux.
Executing an AVX2 compiler target on that host verifies the code path; it does
not measure Intel hardware. The Apple-specific tests were skipped on this host.

User guides: `docs/modernbert-cpu.md`, `docs/modernbert-cpu-sme.md`, and
`docs/epilogues.md`. Native shapes remain restricted to the documented ModernBERT
envelope. Int8 requires SME2 or AVX512-VNNI; AVX-VNNI-only CPUs fall back to FP32.
