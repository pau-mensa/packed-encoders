# ModernBERT CPU engine on x86: handoff (updated 2026-10-18)

These are historical optimization notes. Their ONNX int8 numbers use dynamic
quantization and the long random-length baseline retains padding. For the revised
HF / trimmed ONNX / MatMulNBits W8A8 comparison, paired real-text accuracy and
current runtime changes, see the [PR review follow-up](../cpu_review/README.md).

Scheduling follow-up: [SCHEDULING.md](SCHEDULING.md) records the native scheduling
experiments and the retained per-forward attention-call preparation change.

INT8 row follow-up: [INT8_ROWS.md](INT8_ROWS.md) records dequantization/LayerNorm fusion,
rejected fusion candidates, exactness checks, and six-case timings.

Short attention follow-up: [SHORT_ATTENTION.md](SHORT_ATTENTION.md) records the completed
16/32-token optimization (item 6), baseline parity checks, and FP32/int8 timings.

Int8 rotation follow-up: [INT8_ROTATION.md](INT8_ROTATION.md) records the completed
fixed-size Hadamard optimization (item 4b), exactness checks, and six-case timings.

Runtime cache selection: [GEMM_CACHE_TUNING.md](GEMM_CACHE_TUNING.md) describes L1-aware
tile dispatch, optional pack-time tuning, and forced 32/48 KiB comparisons.

Follow-up: [GEMM_REVIEW.md](GEMM_REVIEW.md) records the subsequent FP32 kernel review,
the additional tuning, AVX2 fallback fix, tests, and new measurements. The tables below describe
the implementation before that follow-up.

Start here when continuing this work. Companion files in this directory:
- `BASELINE.md`: detailed measurements, the ranked list with per-item logs, and the planning tables.
- `SURVEY_16T.md`: the first 16-thread survey including the b32 workloads, hardware notes and the ONNX
  export recipe.

## Goal and scope

The CPU engine on branch `feat/modernbert-cpu-runtime` was tuned on an Apple M4: SME2 int8, NEON, Accelerate
and GCD. This work makes it fast on x86. The baseline is ONNX Runtime, FP32 and int8.

- **Scope (set by the user):** only the workloads where the engine loses to ONNX: `b1_s16_pad32`, `b1_s32`
  and `b8_s32`.
- **Thread counts:** 1 thread (`taskset -c 2`; "people usually run this under 1 thread") and 8 threads pinned
  to one CCD (`taskset -c 0-7`).
- **Out of scope:** the b32 workloads, which the engine already wins by 1.8-22x (`SURVEY_16T.md`).
- **Working agreement:**
  - Probe and rank before building.
  - One isolated item at a time, behind a private ablation switch, measured by alternating A/B in one
    process, with its gain logged in `BASELINE.md`.
  - The user reviews and commits; the assistant never commits.
  - Apple is validated by the user on the M4 once the list is finished. Nothing here can run on Apple.

## Status

| # | item | state | measured gain (1t / 8t, ms, b1_s16 / b1_s32 / b8_s32) |
|---|---|---|---|
| 1 | Native row kernels on x86 (RoPE+layout, GeGLU, add+LN) | **committed** `0a71260` | -0.57/-0.90/-7.87 and -0.49/-0.49/-4.55 |
| 2 | Native short attention on x86 (local window; global <= 128) | **committed** `9d7dd9c` | -0.86/-0.76/-5.44 and -0.66/-0.82/-5.02 |
| 3 | Glue trim (token-major context, resolved plan, no per-call `_check`, prefix-mask fast path) | **committed** `802db38` | -0.84/-1.15/-3.43 and -0.80/-0.85/-3.08 |
| 4 | int8 on x86 (AVX512-VNNI) | **committed** `802db38` | totals 4.32/7.86/55.5 and 2.00/2.40/8.71 ms (1.2-2.5x faster than ONNX int8) |
| 4b | int8 row-stage tuning (`fwht_block` stack traffic) | **committed** `c371aa9` | See [INT8_ROTATION.md](INT8_ROTATION.md) |
| 5 | Native FP32 projection GEMM on the original weights (FP32 only), plus opt-in `rehome_weights` | **committed** `08eb9a3`, cache tuning `a309910`; 5.2 shelved | -20.08/-33.26/-198.27 and -3.29/-6.26/-34.25 (2.0x at 1t); totals 19.66/30.40/203.78 and 8.43/8.87/29.92 |
| 6 | Attention kernel tuning (follow-up to 2) | **committed** `46df86f` | See [SHORT_ATTENTION.md](SHORT_ATTENTION.md) |
| 7 | INT8 dequantization + residual/LayerNorm fusion | **committed** `822c9b8` | See [INT8_ROWS.md](INT8_ROWS.md) |
| 8 | Scheduling investigation / INT8 attention-call preparation | **committed** `3409909` | See [SCHEDULING.md](SCHEDULING.md) |

Gains add in milliseconds; they do not multiply (`compound.py`). So the landing order does not change an
item's measured saving, but each later item is a larger share of a smaller total.

### Numbers to beat

Final numbers at `3409909` (items 1-8), measured 2026-10-18. Medians in ms, 3 interleaved rounds of fresh
processes (`results/final_small_1t_8t.jsonl`). The first column is the engine when this work started
(`results/small_1t_8t.jsonl`, 2026-10-14).

| | engine FP32 at start | HF SDPA | engine FP32 | ONNX FP32 | engine int8 | ONNX int8 |
|---|---:|---:|---:|---:|---:|---:|
| b1_s16_pad32, 1t | 41.5 | 65.56 | **17.46** | 30.25 | **4.09** | 10.71 |
| b1_s32, 1t | 66.5 | 66.37 | **27.97** | 30.22 | **7.36** | 10.69 |
| b8_s32, 1t | 417.0 | 425.22 | 192.75 | **191.12** | **52.31** | 67.06 |
| b1_s16_pad32, 8t | 13.7 | 17.74 | **8.55** | 11.06 | **1.94** | 5.07 |
| b1_s32, 8t | 17.2 | 17.55 | **8.88** | 11.07 | **2.26** | 5.12 |
| b8_s32, 8t | 75.2 | 77.28 | **29.05** | 37.20 | **8.25** | 18.95 |

- **int8 wins everywhere in scope**, 1.28-2.62x over ONNX int8, at relL2 6.2e-2 against ONNX int8's 0.55 on
  real text (median token cos 0.99957 vs 0.868; `results/final_accuracy_text.txt`).
- **Exact FP32 wins five of six against ONNX FP32** (1.08-1.73x); b8_s32 1t is within 1%.
- **Large batches** (16t, both CCDs, one process each; `results/final_long_16t.jsonl`): FP32 8.04 / 3.57 /
  97.4 / 37.1 s and int8 2.22 / 1.05 / 50.6 / 19.0 s for b32_s1024 / b32_rand_pad1024 / b32_s8192 /
  b32_rand_pad8192. ONNX (mb1) at 1024: FP32 17.5 / 17.4 s, int8 15.5 / 15.3 s.
- **FP32 cannot match ONNX int8.** At 1t it is compute-bound: b1_s32 needs at least 19.4 ms even at FP32
  peak. At 8t it is DRAM-bound: reading the FP32 weights takes at least 7.6 ms. The fair pairings are FP32
  against ONNX FP32 and int8 against ONNX int8.

## Machine facts that drive the decisions

**Hardware.**
- AMD Ryzen 9 9950X (Zen 5): 16C/32T, 2 CCDs (CPUs 0-7 + SMT 16-23, CPUs 8-15 + SMT 24-31), 32 MB L3 per CCD.
- AVX-512 full width, AVX512-VNNI, AVX512-BF16. No AMX, no AVX-VNNI-INT8.

**Peaks** (`native_probe/isa_peak.c`, sustained on all 8 cores of CCD0):

| | per core | CCD0 |
|---|---:|---:|
| FP32 FMA | 363 GFLOPS | 2.84 TFLOPS |
| BF16 dot | 674 GFLOPS | 5.2 TFLOPS |
| int8 VNNI | 1391 GOPS | 11.3 TOPS |
| DRAM read | 61 GB/s | 58 GB/s (72 GB/s with both CCDs) |

The projection weights are 441 MB in FP32 and 110 MB in int8, so nothing stays cached between calls.

**MKL is about 2.3x slower than oneDNN here.** On this AMD CPU MKL doesn't select its AVX-512 kernels: the
88-GEMM chain runs at roughly 1.5 TFLOPS on MKL and 3.4-4.0 TFLOPS on oneDNN (16t). That is why the engine
loses at 1t.
- `torch.ops.mkldnn._linear_pointwise(x, w, b, "none", [], "")` on the original weights: 1.35-2.0x over MKL
  end to end at 1t, 1.1-1.7x at 8t. oneDNN reorders the weights on every call, which hurts at 1t.
- Prepacked (`torch.ops.mkldnn._reorder_linear_weight(w, 32)`): 2.1-2.3x at 1t. The user rejected it as a
  deliverable because of the +441 MB copy. It stays the acceptance bar for item 5 and its fallback.

**Cross-CCD synchronization is very expensive.**
- ORT with 8 threads split 4+4 across the CCDs ran b1_s32 in 59.7 ms, against 11.1 ms on one CCD.
- Unpinned 16-thread runs are bimodal.
- 32 threads (SMT) is worse everywhere.
- Always pin.

**Measurement drift.** Absolute timings drift 10-25% between processes (boost and temperature, 58-80 C;
governor `powersave`/EPP `balance_performance`, untouched). So:
- Never compare across processes.
- Use `ab_switches.py`: alternating A/B, 15 rounds.

**Toolchain gotchas.**
- **Use gcc.** Clang 22 accepts `-fopenmp=libgomp` but does no OpenMP codegen.
- **Link the OpenMP runtime torch already loaded**, found in `/proc/self/maps` (`torch/lib/libgomp.so.1`,
  GOMP <= 5.0), instead of `-fopenmp`'s `-lgomp`. Then the native kernels and torch share one thread pool.
- **The vectorized erf (GeGLU) needs `-fno-trapping-math -fno-math-errno`.** Torch builds with the same
  flags. Without them GCC leaves the erf scalar and GeGLU is 4-7x slower than torch.
- **`rows.c` must build with `-ffp-contract=off`.** GCC defaults to `fast`, and RoPE must stay bitwise
  PyTorch's. The attention file builds with `-ffp-contract=fast` so it gets FMAs.
- **No BLAS to borrow.** `libtorch_cpu.so` exports only MKL's `sgemm_` and `cblas_gemm_bf16bf16f32`, and
  no oneDNN C GEMM. Native kernels need their own micro-kernels.

## Code map (what exists now)

**`packed_encoders/cpu/attention.py`**
- `_x86_library()` compiles `rows.c` (contraction off) and `local_attention_x86.c` (contraction on) with
  `gcc -O3 -march=native -fopenmp -fno-trapping-math -fno-math-errno -Wall -Wextra -Werror`, about 0.3 s,
  into a process-private temp directory. It links them against torch's OpenMP runtime.
- `_library()` is Apple's Accelerate build. Both share `_signatures()`.
- `CPUAttention` takes the native path on Apple arm64 and on Linux x86_64 when head_dim is 64 and
  local_attention is 128. `rows` holds the row-kernel handle; `library` is the attention handle, set inside
  that envelope.
- Private switches: `fused_rope`, `fused_geglu`, `fused_norm`, `native_local`, `native_global`, `token_major`,
  `native_linear`.
- `linear(x, w, b)` (item 5): `pe_linear` on x86 for contiguous FP32 activations, else `F.linear` (Apple always).
  Results borrow one engine-owned buffer per output width (`linear_buffers`), so requests never fault in
  fresh GEMM outputs.
- `local`/`global_` take `token_major=`: native attention then stores [tokens, heads * 64] (both C kernels
  have the flag); SDPA and the switch-off path transpose in Python.
- `ResolvedNorm`: a norm resolved once, with aliases of a plain LayerNorm's parameters for native code.
- `attention_backend="sdpa"` is the pure-PyTorch reference on both platforms. Without gcc, packing warns and
  uses SDPA and PyTorch ops.

**`packed_encoders/cpu/rows.h`**
- Per-row bodies shared by Apple and x86.
- The NEON add+LN is guarded by `__ARM_NEON`. The portable version replicates its 32 partial sums and
  reduction tree, so it should give the same bits as the M4 (unverified on Apple).

**`packed_encoders/cpu/rows.c`** (x86 row entry points)
- Same names, signatures and 32-row tasks as `local_attention.c`.
- OpenMP `parallel for ... if(workers > 1)`.

**`packed_encoders/cpu/local_attention_x86.c`** (`pe_local_attention`, Apple contract)
- GCC vector extensions (`v16`), OpenMP `schedule(dynamic,1)` over (head, sequence) tasks.
- Per 64-row tile:
  1. Transpose the key window.
  2. Q.K^T in 4x64 register blocks over each row block's visible columns.
  3. Masked softmax with a Cephes vector exp.
  4. P.V in 4x64 blocks, 1/sum folded in.
- Scratch per worker: `(tile + 64) x padded columns`, sized in `_native()`.

**`packed_encoders/arch/modernbert_cpu.py`** (item 3)
- `self.plan`: per layer, resolved at pack time: the attention kind, detached aliases of the projection
  weights/biases and `ResolvedNorm`s. `_core` reads only the plan.
- **Contract:** no per-call structure check (`_check()` only rejects a closed engine; training is read from the
  root module). In-place value updates are read through the aliases; any other change after packing needs a
  repack and is not detected, but never reaches native code as a stale pointer
  (`test_structural_changes_are_not_tracked`).
- `forward`: `_prefixes()` fast path for unpadded/right-padded masks (private switch `fast_masks`), else the
  general `_compacted()`; HF delegation for holey/empty/non-0/1 masks is unchanged.

**`packed_encoders/cpu/memory.py`** (item 5): `rehome(parameters)`, used by `ModernBertCPU(rehome_weights=True)`.
It moves the projection weights into fresh memory before the plan takes its aliases (`MADV_HUGEPAGE` on the
new range, `MADV_PAGEOUT` on the file-backed pages left behind, so RSS stays flat). Opt-in by the user's choice.

**`packed_encoders/cpu/gemm_x86.c`** (item 5, `pe_linear`)
- y = x w^T (+ b) on the original row-major [N, K] weights: no reorder, no copy.
- Dot-product tiles of 3 activation rows x 8 weight rows, **named accumulators**, horizontal reduce; masked
  K and N tails; the next 8-row block is prefetched during the current one, spread over its tiles.
- OpenMP over 8-row weight blocks. Batch-invariant: one accumulation order and reduction tree per output, so
  results do not depend on M, tile position or thread count (`test_native_linear`).

**`packed_encoders/cpu/int8_x86.c`** (item 4; `int8.c`'s entry points and structs)
- u8 = q + 128 activation rows x [N/16][K/4][16][4] weights with `vpdpbusd`, minus Python's per-channel
  `128 * sum_k w` (`_Segment` trailing fields); then Apple's dequantization.
- 8-row x 3-vector register tiles with **named accumulators** (an array made GCC spill; 2.5x slower).
- One OpenMP team per call, stage by stage with barriers; GEMM tasks are (panel, 64-row chunk) pairs, so b1
  at 8t uses every thread. Reads the token-major context. Built like `rows.c` (gcc, contraction off).

**`packed_encoders/cpu/int8.py`**: `vnni_available()`, the x86 gcc build, `pack_weights(q, lanes)`, the
corrections; the segments point into the engine's norm aliases. Unsupported machines warn and run FP32.

**`tests/test_modernbert_cpu.py`**
- `_native_kernels()` and `int8_kernels` gate on what this machine builds (SME2 or AVX512-VNNI).
- On this machine the whole CPU suite runs: 29 passed, 0 skipped. `native_linear` gates the GEMM tests;
  `test_rehome_weights` runs everywhere.
- `test_int8_matches_its_emulation` now checks exactness against the kernel's own FP32 quantization (< 1e-6)
  and fidelity to the float64 scheme (< 5e-3); see `BASELINE.md`, item 4.

**Apple-only, unchanged except as listed below:** `int8.c` and `local_attention.c`.

## Next

Items 1-8 are complete and committed. Item 8's measurements and rejected scheduling
candidates are in [SCHEDULING.md](SCHEDULING.md).
Remaining candidates, in priority order:

1. **INT8 GEMM:** refresh the profile and investigate data reuse/instruction scheduling.
   Before item 7, GEMM accounted for about 41 of 52 ms on b8_s32 at one thread.
2. **GPTQ/preparation rotation:** structured Hadamard transforms in place of dense rotation
   matrices could reduce preparation time and memory. This does not improve inference;
   changed reduction order requires quantized-weight and model-accuracy validation.

**FP32 large-M kernel (5.2) remains shelved by the user:** per-call panel packing plus a
broadcast kernel reached 89.5% vs 86% of peak in the original M=256, one-thread probe
(`native_probe/fp32_packed.c`). Its proposed size-dependent dispatch gave up batch invariance.

## Tried, measured, decided against (for this scope)

**BF16 projections** (oneDNN, naive cast).
- Speed: 2.6-3.6x over MKL at 1t (1.4-2.1x at 16t).
- Accuracy: relL2 1.7-4.4e-2, with some bistable tokens dropping to cos 0.34-0.91.
- Decided against: slower than int8 at a similar accuracy, and AVX512-BF16 is rarer than VNNI.

**Naive int8** (fbgemm dynamic, per-tensor activations): relL2 0.17-0.6. The recipe is what makes int8 usable.

**ONNX Runtime.**
- The transformers optimizer fuses only GELU. Its SkipLayerNorm fusion crashes on bias-free LayerNorm, so it
  is disabled in `onnx/variants.py`.
- Attention is dense S x S with masks on all 22 layers.
- Its best settings: 8 threads on one CCD for small inputs; one row per run for long ones (one 8192 sequence
  peaks at 27 GB RSS).

**CCD sharding** (two pinned 8-thread workers, sequences split).
- 1.21-1.54x on batched 16-thread inputs.
- Irrelevant for single requests and pinned 8t runs.
- It would also break "packing never changes process-global threading".

**Long-sequence flash attention.** Torch CPU SDPA reaches only about 1.5 TFLOPS, but that matters only for b32.

**Isolated kernel probes overestimate in-engine savings.** Rows on b8_s32 were 16 ms in isolation and 6.9 ms
in the engine. Native GeGLU makes its own stage 2x faster but slows the following GEMMs by about 0.5% (net
-0.2 to -0.4 ms). Always confirm in the engine.

**Global native-attention cutoff.** The M4's <= 128 tokens holds on x86: native is 1.5-2.8x faster through
192 and slower from 256 (`global_cutoff.py`). The 256 cliff is a 1 KB power-of-two stride in the transposed
window. It is unreachable under the current policy (local windows <= 192 columns), but pad `ld` if global
ever goes past 192.

## Open decisions and loose ends

- **Threading contract.** "Packing never changes process-global threading" blocks the engine from pinning or
  sizing its own teams. CCD-aware execution needs the user's call.
- **Apple checks at the end** (user's plan):
  - the `rows.h` NEON guard;
  - the `_signatures()` refactor;
  - `_native()` now allocating `(tile + 64) x padded columns` of scratch on Apple too (harmless, slightly more);
  - the new `native_local` switch;
  - whether the portable and NEON add+LN agree bitwise;
  - item 3: `local_attention.c`'s `token_major` flag (cblas `ldc = heads * 64`; the new test asserts it equals
    the transposed head-major result exactly), and the FP32 path now asking for token-major context;
  - item 3: `int8.py` segments now point into the engine's `ResolvedNorm` aliases;
  - item 4: `int8.py` changes on the Apple path (`library()` restructure, `prepared` 3-tuples with a `None`
    correction, the longer ctypes `_Segment`, whose trailing fields Apple's struct ignores);
  - item 4: `test_int8_matches_its_emulation`'s new assertions (exact against the kernel-quantize emulation,
    which reads Apple's packed SME layout back);
  - item 5: Apple's FP32 projections now go through `CPUAttention.linear`, which falls through to `F.linear`
    (`native_linear` is x86-only): same results, one extra Python call per projection;
  - item 5: `rehome_weights=True` on macOS only copies (the madvise steps are Linux-only), so there RSS grows
    by the copy and the speed effect is unmeasured; `test_rehome_weights` should still pass.
- **Docs.** `docs/modernbert-cpu.md` now has the x86 section (gcc requirement, native backend, native FP32
  GEMM, VNNI int8, `rehome_weights`/`tune_gemm`, measured numbers, the item-3 contract). It was rebuilt from
  `docs/modernbert_cpu.md`, a garbled email paste of the Apple version; the doc links research notes under
  `benchmarks/docs/modernbert-cpu-*.md` that exist only on the M4 machine.
- **No Intel machine** has been measured. MKL may be fine on Intel, and the int8 path needs a non-AVX-512
  (AVX-VNNI 256) story: today such CPUs warn and run FP32.

## Files in this directory

| file | purpose |
|---|---|
| `bench.py` | latency for packed / ONNX / HF, one system per process, JSON lines; the 7 workloads and the sampling plan |
| `summarize.py` | median-of-process-medians table for interleaved runs |
| `ab_switches.py` | **per-item A/B**: toggles private switches (engine or `CPUAttention`), alternating rounds, relL2 vs reference; `name=@REF[:precision[:rehome]]` runs a git ref (`.` = working tree) on its own model copy (`rehome`: packed with `rehome_weights=True`) |
| `packed_accuracy_text.py` | engine FP32/int8 accuracy on real text vs HF FP32 (the metrics of `onnx_accuracy_text.py`) |
| `ab_linear.py` | projection-backend A/B (MKL, oneDNN plain/prepacked, BF16, fbgemm int8) with accuracy |
| `profile_packed.py` | per-stage wall time of the engine forward (synchronous, so wrappers are exact); projections `native` (default), `mkl` or `dnn` |
| `compound.py` | showed that gains add in ms (pre-item-1 probe; its `.so` argument is now redundant) |
| `geglu_stage.py`, `global_cutoff.py` | GeGLU interaction attribution; native-vs-SDPA global cutoff |
| `int8_recipe_accuracy.py` | exact emulation of the M4 int8 recipe on x86 (accuracy only) |
| `onnx_accuracy.py`, `onnx_accuracy_text.py` | ONNX FP32/int8 accuracy vs HF FP32 (random IDs; README text) |
| `probe_gemm.py`, `probe_gemm2.py`, `probe_sdpa.py` | GEMM-chain and SDPA throughput ceilings per backend |
| `native_probe/isa_peak.c`, `native_probe/rows_x86.c` + `probe_rows.py` | per-core instruction peaks; the pre-item-1 row-kernel probe |
| `native_probe/vnni_gemm.c` | item 4.0: the VNNI GEMM chain per register-block shape (note: array accumulators; see item 4's codegen finding) |
| `native_probe/fp32_gemm.c` + `gen_fp32_tiles.py` | item 5.0: the FP32 NT GEMM chain per tile shape, prefetch and packing options (`python gen_fp32_tiles.py > fp32_tiles.h` first) |
| `native_probe/fp32_packed.c` | item 5.2 probe: per-call weight-panel packing + 6 x 64 broadcast kernel (large M) |
| `shard.py`, `one8192.py` | CCD-sharding probe; one 8192-token ONNX sequence |
| `onnx/export.py`, `onnx/variants.py` | ONNX export (dynamo, opset 18) and the optimized/int8 variants |
| `results/` | raw JSON/text behind every table (`small_1t_8t.jsonl`: the starting baseline; `int8_vs_onnx.jsonl`: int8; `fp32_vs_onnx.jsonl`: FP32 after item 5; `final_*`: the PR numbers at `3409909`) |

ONNX models are in `.cache/onnx/modernbert-base/` (gitignored, 1.4 GB). Regenerate with:

```bash
M=.cache/onnx/modernbert-base
CUDA_VISIBLE_DEVICES= uv run --no-sync --with onnxruntime --with onnx --with onnxscript \
    python benchmarks/cpu_x86/onnx/export.py $M sdpa
CUDA_VISIBLE_DEVICES= uv run --no-sync --with onnxruntime --with onnx --with onnxscript --with sympy \
    python benchmarks/cpu_x86/onnx/variants.py $M
```

Everyday commands (from the repo root; `CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1` keeps the GPU and network
out of it):

```bash
B=benchmarks/cpu_x86; M=.cache/onnx/modernbert-base
# per-item A/B through the engine's private switches (1t, then 8t on CCD0)
taskset -c 2   .venv/bin/python $B/ab_switches.py 1 15 off=native_local:0,native_global:0 on=native_local:1,native_global:1
taskset -c 0-7 .venv/bin/python $B/ab_switches.py 8 15 off=native_local:0,native_global:0 on=native_local:1,native_global:1
# a whole item against a git ref, each on its own model copy; int8 against FP32
taskset -c 2   .venv/bin/python $B/ab_switches.py 1 31 head=@HEAD tree=@.
taskset -c 2   .venv/bin/python $B/ab_switches.py 1 15 fp32=@. int8=@.:int8
# FP32 projections: MKL against the native GEMM; weights in HF's mmap against fresh memory
taskset -c 2   .venv/bin/python $B/ab_switches.py 1 15 mkl=native_linear:0 gemm=native_linear:1
taskset -c 2   .venv/bin/python $B/ab_switches.py 1 15 mmap=@. rehome=@.:fp32:rehome
# where the time goes
taskset -c 2   .venv/bin/python $B/profile_packed.py 1 b1_s16_pad32,b1_s32,b8_s32
# accuracy on real text (FP32 and int8 vs HF FP32)
taskset -c 0-7 .venv/bin/python $B/packed_accuracy_text.py README.md 8
# baselines (repeat interleaved for 3 rounds, then summarize)
taskset -c 2   .venv/bin/python $B/bench.py packed --threads 1 --precision int8 --workloads b1_s16_pad32,b1_s32,b8_s32
taskset -c 2   uv run --no-sync --with onnxruntime python $B/bench.py onnx --threads 1 --model $M/int8.onnx \
    --workloads b1_s16_pad32,b1_s32,b8_s32
.venv/bin/python $B/summarize.py $B/results/int8_vs_onnx.jsonl
# tests
.venv/bin/python -m pytest tests/test_modernbert_cpu.py -q
```
