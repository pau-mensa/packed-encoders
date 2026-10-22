# CPU engine on an AVX2-only x86 CPU

Measurements of the ModernBERT CPU engine on an x86 CPU without AVX-512, against stock
HF and ONNX Runtime FP32 and int8, using the published x86 workloads and ONNX protocol.
The scripts, raw JSON and generated tables are all here; [`results/summary.md`](results/summary.md)
prints every column of every file.

Without AVX-512 the engine's FP32 projections fall back to `F.linear`; its row and
attention kernels still run natively. The engine's int8 path needs AVX512-VNNI or SME2,
so it is not measured here.

## Machine and versions

- Intel Core i9-13900K: 8 performance and 16 efficiency cores, 32 hardware threads; flags
  `fma`, `avx2`, `avx_vnni`, no AVX-512. WSL2 (kernel 5.15.167.4) reports 16 cores x 2
  threads, so PyTorch's default is 16 threads. Runs are unpinned, with no other heavy work.
- Python 3.12.7, torch 2.11.0 (CPU only; OpenMP 4.5), transformers 5.9.0, numpy 2.5.2, gcc 13.3.0.
- onnxruntime 1.31.0, onnx 1.23.2, onnxscript 0.7.2.
- `answerdotai/ModernBERT-base`, FP32.
- Engine at e88cc5a for `sweep_*`, `abl_*` and `hf_engine_*`, and at fe9c5df for `omp_*`.
  The two commits in between do not change the timed path for these inputs (int64 IDs,
  integer masks).

## Protocol

- Workloads (`bench.py`): random token IDs, seed 0. `b1_s16_pad32` is one 16-token query
  padded to 32; `b32_rand_pad1024` is 32 rows of random lengths 1–1024 padded to 1024. The
  others are full rows.
- Timing: median wall clock over `--trials` after one warm call, in one process with
  `torch.set_num_threads(t)` and `ModernBertCPU(threads=t)`. HF is stock SDPA. Each row
  records the engine's token cosine minimum against HF over real tokens.
- ONNX (`onnx_export.py`, `onnx_bench.py`): dynamo export at opset 18, then the
  onnxruntime transformers optimizer. Its SkipLayerNorm fusion is disabled, because it
  indexes a LayerNorm bias that ModernBERT does not have. Sessions use `intra_op = t`,
  sequential execution and `ORT_ENABLE_ALL`; each cell keeps the faster of spinning on and
  off. Short queries and single documents feed the padded batch. Large batches run one row
  at a time, either at the padded width (`rowpad`, as published) or trimmed to each row's
  length (`rowtrim`).
- int8 variants: seven `quantize_dynamic` configurations and two weight-only MatMulNBits
  8-bit models, listed in the `onnx_export.py` docstring.
- Real-text accuracy (`onnx_accuracy.py`): the repository README's paragraphs of 29–512
  tokens (67 paragraphs, 6,970 tokens), each run alone, against HF FP32.

## Findings

Times are in ms; ratios are speedups of the engine (above 1 means the engine is faster).

**Short queries, FP32.** The engine runs at 0.86–1.09x of ONNX FP32. For the padded
query this compares against trimmed ONNX, since the engine skips padding by construction.

| threads | b1_s16_pad32 vs ONNX trimmed | b1_s32 | b8_s32 |
|---|---:|---:|---:|
| 1 | 0.93x | 0.89x | 0.89x |
| 8 | 0.88x | 0.92x | 1.09x |
| 16 | 1.05x | 0.93x | 0.86x |

**Long inputs, FP32.** ORT does not fuse ModernBERT's attention, so ONNX FP32 is slower
than HF here: 2.4x slower at 1,024 tokens and 4.8–5.1x at 4,096. That makes "vs ONNX"
ratios grow with length.

| 16 threads | HF | ONNX FP32 | ONNX FP32 trimmed | engine | vs HF | vs ONNX | vs ONNX trimmed |
|---|---:|---:|---:|---:|---:|---:|---:|
| b1_s1024 | 545 | 1,322 | — | 461 | 1.18x | 2.87x | — |
| b1_s4096 | 3,416 | 17,299 | — | 2,142 | 1.59x | 8.08x | — |
| b32_s1024 | 19,327 | 42,019 | — | 14,968 | 1.29x | 2.81x | — |
| b32_rand_pad1024 | 19,925 | 42,670 | 16,781 | 8,119 | 2.45x | 5.26x | 2.07x |

At 8 threads the ratios are similar: 1.12–1.48x vs HF on full rows, 2.31x vs HF and 1.92x
vs trimmed ONNX on random lengths.

**Threads above the core count.** On short queries the engine runs at 0.75–1.36x of HF at
4–16 threads. Beyond the 16 reported cores it falls to 0.61–0.76x of HF at 24 threads and 0.16–0.21x at
32. Long inputs are unaffected (1.28–2.72x at 32 threads). The lost time sits in the
projections between the engine's OpenMP kernels: 27.7 ms at 16 threads becomes 182 ms at 32
for `b1_s32`, while HF's identical `F.linear` calls stay flat. The wait policy points at
spinning workers: `ACTIVE` makes it 3–5x worse, and `PASSIVE` helps but does not recover it.

| b1_s32 / b8_s32 | HF | engine |
|---|---:|---:|
| 16 threads, default | 30.1 / 95.2 | 31.3 / 120.0 |
| 32 threads, default | 34.9 / 104.2 | 156.9 / 580.1 |
| 32 threads, `ACTIVE` | 37.6 / 98.0 | 760.0 / 1,920.0 |
| 32 threads, `PASSIVE` | 103.9 / 285.2 | 73.7 / 208.5 |

`PASSIVE` at 32 threads leaves the engine 2.4x and 1.7x slower than at 16 threads, and
makes HF about 3x slower. Capping the engine at the core count the OS reports (16 here,
PyTorch's default) avoids the regression; the wait policy does not. With efficiency cores
the reported count is the safer cap: 24 threads, the real core count, already regresses.

**Native attention vs the engine's SDPA fallback** (`attention_ablation.py`, same row
kernels; attention ms only). Native attention is slower at 1 and 8 threads and faster
only at 16, mostly on short queries.

| threads | b1_s32 | b8_s32 | b1_s1024 | b32_rand_pad1024 |
|---|---:|---:|---:|---:|
| 1 | 0.35x | 0.22x | 0.40x | — |
| 8 | 0.80x | 0.93x | 0.51x | 0.67x |
| 16 | 1.29x | 1.61x | 0.82x | 1.06x |

**ONNX int8.** Dynamic int8 (`quantize_dynamic`) does not hold up on ModernBERT in any of
seven configurations. Weight-only MatMulNBits 8-bit does. Results on real text against HF FP32:

| model | relL2 | token cos median | tokens below 0.5 (of 6,970) |
|---|---:|---:|---:|
| ONNX FP32 (optimized) | 1.05e-05 | 1.00000 | 0 |
| dynamic int8, 7 configurations | 0.495–0.566 | 0.855–0.894 | 135–149 |
| `int8_w8` (weights int8, FP32 compute) | 0.0466 | 0.99965 | 7 |
| `int8_w8a8` (weights int8, `accuracy_level=4`) | 0.0671 | 0.99904 | 12 |

- `int8_w8` dequantizes into FP32 compute and is slower than ONNX FP32.
- `int8_w8a8` is 1.5–2.4x faster than the engine's FP32 on `b1_s32` and `b8_s32` at 1–16
  threads (`b1_s32` at 16 threads: 13.2 vs 31.9 ms), and 2.4–2.9x on the padded query with
  trimmed rows.
- On large batches the engine's FP32 leads `int8_w8a8`: 15.0 vs 38.0 s for `b32_s1024`,
  and 8.1 vs 14.4 s (trimmed rows) for `b32_rand_pad1024`, both at 16 threads.
- `int8_w8a8` is therefore the ONNX int8 baseline for the engine's rotated GPTQ int8. That
  comparison needs an AVX512-VNNI or SME2 machine.

In the dynamic configurations, the attention and rotary MatMuls stay FP32: only the 88
projections are quantized (plus the embedding Gather in the default). Restricting
quantization to the projections, per-channel weights, uint8 weights or `reduce_range`
does not change the picture. On the benchmark's random IDs the same models range from 0.10
to 0.78 relL2 depending on the workload, so random-ID accuracy does not stand in for real text.

## Reproduce

```bash
PE_PYTHON=/path/to/python ORT_PYTHON=/path/to/python benchmarks/cpu_avx2/run.sh
```

`run.sh` reruns every file in `results/` with its original arguments, about 2.5 hours on
this machine, then regenerates `summary.md`. Sections can be run on their own, in order:
`sweep omp ablation onnx_export accuracy onnx summary`. `ORT_PYTHON` needs onnxruntime,
onnx and onnxscript next to the same torch and transformers; one interpreter can serve
both. The exported models (about 3.1 GB) go to the ignored `onnx_models/`.

## Files

| File | Contents |
|---|---|
| `bench.py` | HF vs engine, FP32, with an optional per-op engine profile |
| `attention_ablation.py` | engine with native attention vs its SDPA fallback |
| `onnx_export.py` | FP32 export, optimizer pass and the nine int8 variants |
| `onnx_bench.py` | ONNX Runtime timing and accuracy on `bench.py`'s inputs |
| `onnx_accuracy.py` | real-text accuracy of the ONNX models |
| `summarize.py` | `results/summary.md` from the JSON |
| `run.sh` | everything above, in order |
| `results/sweep_t{1,4,8,16,24,32}.json` | engine vs HF across thread counts, with profile |
| `results/omp_t{16,32}_{default,active,passive}.json` | `OMP_WAIT_POLICY` reruns |
| `results/abl_{native,sdpa}_t{1,8,16}.json` | attention ablation |
| `results/hf_engine_t*.json` | HF and engine, paired with the ONNX runs |
| `results/ort*_t*.json` | ONNX FP32 and int8 timing (`ort_dyn`: dynamic variants, `ort_proj`: projections only, `ort_w8`: weight-only) |
| `results/acc_*.json` | real-text accuracy |
