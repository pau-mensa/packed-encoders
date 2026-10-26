# ModernBERT CPU engine on x86: initial 16-thread survey incl. b32 workloads (2026-10-14)

Superseded for planning by `BASELINE.md` (focus: the three small workloads at 1 and 8 threads).

Branch `feat/modernbert-cpu-runtime` at `4b4151e`, unmodified. Nothing here changes the engine; every
"projected" number below is an estimate from an isolated probe, not a measured implementation.

## Machine and software

- AMD Ryzen 9 9950X (Zen 5): 16 cores / 32 threads, 2 CCDs (CPUs 0-7 + SMT 16-23, CPUs 8-15 + SMT 24-31),
  32 MB L3 per CCD. AVX-512 (full width), AVX512-VNNI, AVX512-BF16. No AMX, no AVX-VNNI-INT8.
- DRAM read bandwidth: 72 GB/s (16 threads). Governor `powersave` / EPP `balance_performance` (amd-pstate, untouched).
- torch 2.11.0+cu128 (MKL 2024.2, oneDNN 3.10.2, GNU OpenMP), transformers 5.3.0, onnxruntime 1.31.0,
  onnx 1.23.2, onnxscript 0.7.2. Model `answerdotai/ModernBERT-base`, FP32.

## Workloads

Random token IDs in [100, 50000), seed 1. "Random" lengths are Uniform[1, S] with seed 0 (see `bench.py`).

| name | rows x padded length | live tokens |
|---|---|---:|
| `b1_s16_pad32` | 1 x 32 | 16 |
| `b1_s32` | 1 x 32 | 32 |
| `b8_s32` | 8 x 32 | 256 |
| `b32_s1024` | 32 x 1024 | 32,768 |
| `b32_s8192` | 32 x 8192 | 262,144 |
| `b32_rand_pad1024` | 32 x 1024 | 17,054 |
| `b32_rand_pad8192` | 32 x 8192 | 115,988 |

## ONNX baselines

`onnx/export.py`: `torch.onnx.export(dynamo=True, opset 18)`, dynamic batch and sequence, HF `sdpa`
attention. Max abs diff to HF FP32 is 1e-4 (relL2 2e-6). `onnx/variants.py` then runs the ORT transformers
optimizer (`model_type=bert`, SkipLayerNorm fusion off because it crashes on bias-free LayerNorm). It fuses
only GELU (22). LayerNorm is already native, and no attention or RoPE pattern matches. So ONNX computes
dense S x S attention with masks on all 22 layers, including the 14 sliding-window ones. int8 is ORT's
standard `quantize_dynamic` (QInt8 weights, 88 MatMulInteger, attention stays FP32). Sessions use
`ORT_ENABLE_ALL`, `inter_op=1`.

**ONNX int8 accuracy is not usable.** Against HF FP32 on 24 real-text paragraphs (this README, 3,275
tokens), relL2 is 0.55, median token cosine 0.87, and mean-pooled cosine median 0.944 (min 0.898).
Per-channel weights give about the same (`onnx_accuracy_text.py`). That fails the engine's own int8 pack
gate (median cosine >= 0.99, relL2 <= 0.25). The M4 int8 recipe (Hadamard + GPTQ) measured relL2 3-8e-2.
ONNX int8 is therefore a speed reference only.

Tuning that was applied to give ONNX its best case:
- Threads: 8 threads pinned to one CCD is fastest for the small workloads. ORT is very sensitive to
  cross-CCD sync: 8 threads split 4+4 across CCDs ran b1_s32 in 59.7 ms, against 11.1 ms on one CCD.
  32 threads is worse everywhere.
- Micro-batch: one row per run is fastest for long sequences (b32_s1024 15.7 s, against 18.8 s at 32 rows).
  It is also required at 8192, where one sequence peaks at 27 GB RSS.

## Baseline (current engine vs best ONNX config)

Medians. Small workloads: 3 interleaved fresh-process rounds of 60/60/40 samples (`results/small_interleaved.jsonl`).
1024: 3-5 samples. 8192 packed: 2 samples. ONNX 8192: one 8192-token sequence x 32 (3 samples, 26.2-26.6 s
FP32 and 23.9-24.6 s int8). Rows are padded to 8192 and run one at a time, so all 32 runs are identical.
A full-run confirmation is listed at the end.

| workload | packed FP32 (today) | ONNX FP32 | ONNX int8 (inaccurate) | packed vs ONNX FP32 |
|---|---:|---:|---:|---:|
| b1_s16_pad32 | 13.8 ms (8t, CCD0) | 11.1 ms (8t, CCD0) | 5.0 ms (8t, CCD0) | 0.81x |
| b1_s32 | 17.4 ms (8t, CCD0) | 11.0 ms (8t, CCD0) | 5.1 ms (8t, CCD0) | 0.63x |
| b8_s32 | 66.9 ms (16t) | 37.4 ms (8t, CCD0) | 18.6 ms (8t, CCD0) | 0.56x |
| b32_s1024 | 8.94 s (16t) | 16.4 s (8t or 16t, mb1) | 14.0 s (mb1) | 1.8x |
| b32_rand_pad1024 | 4.32 s (16t) | 16.5 s (16t, mb1) | 14.5 s (mb1) | 3.8x |
| b32_s8192 | 100.0 s (16t) | ~848 s (32 x 26.5 s) | ~771 s (32 x 24.1 s) | ~8.5x |
| b32_rand_pad8192 | 39.3 s (16t) | ~848 s (padded: same as above) | ~771 s | ~22x |

On x86 the engine falls back to tiled PyTorch SDPA and unfused RoPE, GeGLU and add+LayerNorm. `precision="int8"`
warns and runs FP32 (no SME2). It already wins wherever padding or the S x S attention dominate. It loses
1.2-1.8x on the short, unpadded workloads, where ONNX's dense attention costs nothing.

## Where the time goes today (16 threads, `profile_packed.py`)

| stage | b1_s32 | b8_s32 | b32_s1024 | rand_pad8192 |
|---|---:|---:|---:|---:|
| projections (`F.linear` -> MKL) | 73% | 67% | 55% | 43% |
| RoPE + head-major layout (PyTorch) | 8% | 10% | 16.5% | 13% |
| local attention (Python-tiled SDPA) | 4% | 7% | 8% | 6% |
| global attention (per-sequence SDPA) | 1.5% | 3% | 7% | 27% |
| GeGLU + residual add + LayerNorm | 8% | 8% | 12.5% | 10% |
| Python glue, `_check`, masks, `index_copy` | 5.5% | 4% | <1% | <1% |

## Probe findings

1. **MKL runs at roughly 1.5 TFLOPS FP32 on Zen 5. oneDNN runs at 3.4-4.0 TFLOPS** over the 88-projection
   chain (`probe_gemm.py`, 22 distinct layer weight sets so small M streams from DRAM). oneDNN BF16 reaches
   6.5-7.2 TFLOPS. oneDNN on the original row-major weights is within 0-25% of prepacked weights, so the
   engine's storage and in-place-update contract survives.
2. **Small-M latency is bounded by DRAM.** 441 MB of FP32 projection weights at 72 GB/s is at least 6.1 ms
   per forward (BF16 ~3 ms, int8 ~1.5 ms). Each CCD holds 32 MB of L3, so nothing stays resident between
   requests. FP32 cannot reach ONNX int8's 5.0 ms at b1.
3. **The two CCDs matter.** For batched inputs, two 8-thread processes pinned one per CCD, each taking half
   the sequences, beat one 16-thread process: b8_s32 50.4 -> 32.7 ms (1.54x), rand_pad1024 3.08 -> 2.11 s
   (1.46x), b32_s1024 6.48 -> 5.34 s (1.21x). All three runs use oneDNN projections (`shard.py`). Unpinned
   16-thread runs are bimodal (b1_s32 16-24 ms); SMT (32 threads) is worse everywhere.
4. **The existing row kernels port cleanly.** `native_probe/rows_x86.c` is `rows.h` with NEON swapped for
   portable 16-lane C, built with `gcc -O3 -march=native -fopenmp` and linked to torch's own `libgomp`
   (it needs only `GOMP_parallel@GOMP_4.0`, so it shares the hot OpenMP pool):

   | live tokens | RoPE+layout | GeGLU | add+LN | saved per forward (16t) |
   |---:|---:|---:|---:|---:|
   | 32 | 5.6x | 1.3x | 4.0x | ~1.0 ms |
   | 256 | 70x | 3.4x | 6.3x | ~14 ms |
   | 32,768 | 4.5x | 2.3x | 2.1x | ~1.7 s |

   Numerics: RoPE/layout is bitwise identical, GeGLU within 3e-6, add+LN within 1.4e-6 of PyTorch.
   clang 22 ignores `-fopenmp=libgomp` codegen, so the Linux build should use gcc (`cc`).
5. **Torch's CPU flash SDPA runs at about 1.5-1.66 TFLOPS FP32 at 16 threads**, under half the GEMM rate.
   Its BF16 path is no faster on this CPU (`probe_sdpa.py`). Global attention is about 35% of b32_s8192 and
   27% of rand_pad8192.
6. **Precision proxies** (projections only, end to end, `ab_linear.py`). BF16 (naive cast): 2.0x vs MKL at b1
   and 1.4-1.7x at larger sizes, relL2 2-4e-2, median token cosine 0.9998, a few tokens flipped (min cosine
   0.34). That is the same accuracy class as the M4 int8 recipe. fbgemm int8 (naive per-tensor activations):
   3.1x at b1, but relL2 0.2-0.6, so the Hadamard + GPTQ recipe is required, as on the M4.
7. `libtorch_cpu.so` exports only MKL `sgemm_` and `cblas_gemm_bf16bf16f32`, not oneDNN's C GEMM. Native
   attention on x86 therefore needs its own AVX-512 micro-kernels (Accelerate's `cblas_sgemm` and `vvexpf`
   have no counterpart to borrow).

## Ranked to-do list

Ordered by expected gain per unit of effort, with exact FP32 items first. Each item should land as an
isolated change with a private ablation switch, as the M4 kernels do, and be measured by alternating A/B
in one process.

| # | item | evidence | expected effect | effort | exact? |
|---|---|---|---|---|---|
| 1 | **Projections through oneDNN** (`torch.ops.mkldnn._linear_pointwise` on the original weights; vendor- or calibration-gated, since MKL may still win on Intel) | measured end to end: 1.41x b1_s16, 1.53x b1_s32, 1.30x b8_s32, 1.41x rand_pad1024 at 16t (up to 1.72x at 8t) | about -30% on everything | small (Python) | yes (relL2 8e-6) |
| 2 | **Port the native row kernels to Linux/x86**: RoPE+layout, GeGLU, add+LN; gcc + OpenMP on torch's libgomp in place of GCD | kernel probe: ~1 ms (b1), ~14 ms (b8_s32), ~1.7 s per 32K tokens | -8% (b1) to -25% (1024s) after item 1 | moderate | yes (same contract as on the M4) |
| 3 | **CCD-aware execution**: shard sequences across CCDs, each with a pinned team; send single short requests to one CCD | two-process probe: 1.21-1.54x on batched workloads | b8_s32 ~-35%, rand_pad1024 ~-30% | moderate-high; conflicts with "packing never changes process-global threading" | yes |
| 4 | **Native local attention + short global (<=128) on x86**: AVX-512 flash-style kernel for head_dim 64, replacing the Python-tiled SDPA loop | local plus short global is 5.5% (b1), 10% (b8_s32), 15% (b32_s1024) of today's time, and its share grows after items 1-2 | b8_s32 ~-5 ms, 32K tokens ~-0.5 s | moderate-high (no BLAS to call) | yes |
| 5 | **int8 on x86**: AVX512-VNNI (u8s8 + compensation) port of `int8.c`'s fused row-group pipeline, reusing the Hadamard + GPTQ preparation | fbgemm proxy: 88 projections at M=16 drop from 7.8 to 2.45 ms; FP32's floor is 6.1 ms | the only route to ONNX int8's 5.0 / 18.6 ms at b1 / b8_s32, at the M4 recipe's accuracy instead of relL2 0.55 | high | no (approximate) |
| 6 | **Native long global attention** (FP32 flash; the item-4 kernel with a full window, as on the M4) | SDPA runs at 1.5 TFLOPS against ~3.5 for GEMM; global is 27-35% of the 8192 workloads | ~-20-25% on the 8192 workloads after items 1-2, if it reaches 2.5-3 TFLOPS | high | yes |
| 7 | **BF16 precision option** (oneDNN AVX512-BF16 projections; later BF16 attention tiles) | 2x GEMM throughput at large M, half the weight bytes; relL2 2-4e-2 | ~1.4x over item 1 on projections | low to start (casts eat part of the gain); product decision | no |
| 8 | **Small-call overhead**: prepacked oneDNN weights with a version check (as int8 does), plus less glue in `forward`/`_check` | prepacked is 0-25% faster than plain at small M; glue is ~0.9 ms per call | ~-1 to -1.5 ms at b1 | low | yes |

Rough FP32 projection after items 1, 2 and 4 (3 adds 1.2-1.5x on batched inputs, 6 trims the 8192 rows):

| workload | today | after 1+2+4 (est.) | ONNX FP32 |
|---|---:|---:|---:|
| b1_s16_pad32 | 13.8 ms | ~10.5-11 ms | 11.1 ms |
| b1_s32 | 17.4 ms | ~11.5-12 ms | 11.0 ms |
| b8_s32 | 66.9 ms | ~29 ms (~20 with 3) | 37.4 ms |
| b32_s1024 | 8.9 s | ~3.8 s | 16.4 s |
| b32_rand_pad1024 | 4.3 s | ~1.9 s | 16.5 s |
| b32_s8192 | 100 s | ~60 s (~44 with 6) | ~848 s |
| b32_rand_pad8192 | 39.3 s | ~22 s (~17 with 6) | ~848 s |

Not worth pursuing: SMT / 32 threads, torch BF16 SDPA, and more ORT graph fusion (only GELU matches).

## Methodology and reproduction

Absolute timings drift 10-25% between processes on this desktop (boost clocks, temperature 58-80 C,
scheduler placement), so:
- per-item deltas are alternating A/B samples inside one process (`ab_linear.py` is the template);
- baselines are interleaved fresh-process rounds;
- every run is pinned (`taskset -c 0-7` for 8 threads on CCD0, `-c 0-15` for 16).

```bash
# ONNX models (overlay env; the project venv stays untouched)
uv run --no-sync --with onnxruntime --with onnx --with onnxscript python benchmarks/cpu_x86/onnx/export.py MODELS sdpa
uv run --no-sync --with onnxruntime --with onnx --with onnxscript --with sympy python benchmarks/cpu_x86/onnx/variants.py MODELS
# baselines
taskset -c 0-7 .venv/bin/python benchmarks/cpu_x86/bench.py packed --threads 8 --workloads b1_s16_pad32,b1_s32,b8_s32
taskset -c 0-7 uv run --no-sync --with onnxruntime python benchmarks/cpu_x86/bench.py onnx --threads 8 --model MODELS/fp32_opt.onnx --workloads b1_s16_pad32,b1_s32,b8_s32
taskset -c 0-15 uv run --no-sync --with onnxruntime python benchmarks/cpu_x86/bench.py onnx --threads 16 --microbatch 1 --model MODELS/int8.onnx --workloads b32_s1024
# probes
.venv/bin/python benchmarks/cpu_x86/profile_packed.py 16 b8_s32,b32_s1024
.venv/bin/python benchmarks/cpu_x86/ab_linear.py 16 b1_s32,b8_s32 mkl,dnn,dnn_packed,bf16,int8 5
.venv/bin/python benchmarks/cpu_x86/probe_gemm.py 16
gcc -O3 -march=native -fopenmp -ffp-contract=off -fno-trapping-math -fno-math-errno -shared -fPIC benchmarks/cpu_x86/native_probe/rows_x86.c \
    -o /tmp/rows_x86.so -L.venv/lib/python3.12/site-packages/torch/lib -l:libgomp.so.1 \
    -Wl,-rpath,$PWD/.venv/lib/python3.12/site-packages/torch/lib
.venv/bin/python benchmarks/cpu_x86/native_probe/probe_rows.py 16 /tmp/rows_x86.so
```

Raw JSON lines are in `results/`.
