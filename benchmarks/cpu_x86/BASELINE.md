# ModernBERT CPU engine on x86: small-workload baseline and plan (2026-10-14)

Scope: the three workloads where the engine loses to ONNX Runtime, at **1 thread** (pinned to core 2) and
**8 threads** (pinned to CCD0, CPUs 0-7). The b32 workloads, which the engine already wins by 1.8-22x, are in
`SURVEY_16T.md` together with the hardware description and the ONNX export recipe.

Branch `feat/modernbert-cpu-runtime` at `4b4151e`, unmodified. "Measured" means an end-to-end alternating A/B
inside one process. "Est." means a kernel-level probe added to a measured end-to-end number.

Machine: AMD Ryzen 9 9950X (Zen 5, 2 CCDs x 8 cores, 32 MB L3 each), torch 2.11 (MKL 2024.2, oneDNN 3.10.2,
GNU OpenMP), onnxruntime 1.31, `answerdotai/ModernBERT-base` FP32.

| workload | rows x padded length | live tokens |
|---|---|---:|
| `b1_s16_pad32` | 1 x 32 | 16 |
| `b1_s32` | 1 x 32 | 32 |
| `b8_s32` | 8 x 32 | 256 |

## Baseline

Median of 3 interleaved fresh-process rounds, each the median of 60/60/40 samples (`results/small_1t_8t.jsonl`).
Round-to-round spread is under 3% for every cell.

| workload | packed FP32 (today) | ONNX FP32 | ONNX int8 |
|---|---:|---:|---:|
| b1_s16_pad32, 1 thread | 41.5 ms | 30.1 ms | 10.6 ms |
| b1_s32, 1 thread | 66.5 ms | 30.1 ms | 10.7 ms |
| b8_s32, 1 thread | 417.0 ms | 191.6 ms | 66.9 ms |
| b1_s16_pad32, 8 threads | 13.7 ms | 11.1 ms | 5.0 ms |
| b1_s32, 8 threads | 17.2 ms | 10.9 ms | 5.0 ms |
| b8_s32, 8 threads | 75.2 ms | 37.2 ms | 18.6 ms |

ONNX int8 is ORT's standard `quantize_dynamic`. Its outputs are far from FP32: on 24 README paragraphs, relL2
is 0.55, median token cosine 0.868 and mean-pooled cosine as low as 0.898. That fails the engine's int8 pack
check. It is a speed target, not an accuracy-matched one.

## Where the time goes (`profile_packed.py`, MKL projections)

| stage | b1_s16 1t | b1_s32 1t | b8_s32 1t | b1_s16 8t | b1_s32 8t | b8_s32 8t |
|---|---:|---:|---:|---:|---:|---:|
| 88 projections (MKL) | 38.3 ms (92%) | 62.2 ms (93%) | 391.3 ms (94%) | 10.8 ms (76%) | 13.8 ms (78%) | 60.1 ms (82%) |
| RoPE + layout | 0.71 | 0.97 | 5.87 | 0.79 | 1.02 | 3.37 |
| local + global attention (SDPA) | 1.00 | 1.68 | 9.59 | 0.96 | 1.12 | 4.88 |
| GeGLU + add/LN | 0.67 | 1.00 | 6.48 | 0.61 | 0.72 | 2.44 |
| glue (`_check`, masks, layout copies) | 0.91 | 0.94 | 1.88 | 0.98 | 0.99 | 2.27 |

## Hardware ceilings

| | 1 core | CCD0 (8 cores) |
|---|---:|---:|
| FP32 FMA (`isa_peak.c`) | 363 GFLOPS | 2.84 TFLOPS |
| BF16 dot, FP32 accumulate | 674 GFLOPS | 5.2 TFLOPS |
| int8 VNNI (u8 x s8) | 1,391 GOPS | 11.3 TOPS |
| DRAM read | 61 GB/s | 58 GB/s |

The 88 projection weights are 441 MB in FP32 and 110 MB in int8; nothing stays in L3 between calls.

- At 1 thread every workload is compute-bound. FP32 lower bounds: 9.7 / 19.4 / 155 ms. int8 lower bounds:
  2.5 / 5.1 / 41 ms.
- At 8 threads the b1 FP32 forward is bandwidth-bound: at least 7.6 ms (int8 at least 1.9 ms).

Projection chain alone, 88 GEMMs with 22 distinct weight sets (`probe_gemm.py`, `probe_gemm2.py`):

| M (tokens) | MKL | oneDNN, plain weights | oneDNN, prepacked | oneDNN BF16 | fbgemm int8 |
|---:|---:|---:|---:|---:|---:|
| 16, 1t | 38.1 ms | 29.1 | 16.5 (58% of peak) | 12.7 | 5.6 |
| 32, 1t | 61.7 | 39.3 | 25.4 (77%) | 18.3 | 8.6 |
| 256, 1t | 389 | 177 | 170 (91%) | 90.4 | 51.8 |
| 16, 8t | 10.8 | 10.3 | 8.2 (bandwidth) | 4.8 | 2.5 |
| 32, 8t | 13.7 | 11.0 | 8.4 (bandwidth) | 5.5 | 2.9 |
| 256, 8t | 57.2 | 29.7 | 26.5 (74%) | 15.4 | 10.0 |

MKL does not select its fast kernels on this AMD CPU. oneDNN on unpacked weights reorders them on every
call, which costs about 10 ms at 1 thread.

## Ranked to-do list (execution order)

Each item lands isolated, behind a private ablation switch, and is measured by alternating A/B against the
previous state. Gains add in milliseconds (see below), so the landing order does not change any item's
measured saving.

Order: the shared items come first because they serve both precisions. In int8 mode the GEMMs shrink about
3x, so these stages become a larger share. Today's non-GEMM time is about 35% of a projected int8 b1_s32 at 1t.
The FP32-only GEMM item is decoupled.

| # | item | effect | exact? | effort | status |
|---|---|---|---|---|---|
| 1 | **Native row kernels on Linux/x86.** RoPE+layout, GeGLU, add+LN: `rows.c` + portable `rows.h`, gcc + OpenMP on torch's runtime. | **measured**, see the item log: 1t -0.57/-0.90/-7.87 ms, 8t -0.49/-0.49/-4.55 ms | yes | moderate | **done** (`0a71260`) |
| 2 | **Native short attention.** AVX-512 flash-style kernel, head_dim 64, local window and global <= 128, replacing the Python SDPA loops. | **measured**, see the item log: 1t -0.86/-0.76/-5.44 ms, 8t -0.66/-0.82/-5.02 ms (est. was -0.7/-1.2/-6.6 and -0.7/-0.8/-3.5) | yes | moderate-high | **done** (`9d7dd9c`) |
| 3 | **Glue trim.** Token-major attention output, resolved per-layer plan, no per-call structure check, prefix-mask fast path. | **measured**, see the item log: 1t -0.84/-1.15/-3.43 ms, 8t -0.80/-0.85/-3.08 ms (est. was ~-0.5 b1, ~-0.8 b8_s32) | yes (bit-identical) | low | **done** (`802db38`) |
| 4 | **int8 on x86 (AVX512-VNNI).** u8s8 `vpdpbusd` with +128 compensation (bit-identical integer results), porting `int8.c`'s pipeline and reusing the Hadamard + GPTQ prep unchanged. | **measured**, see the item log: 1t 4.46/7.87/55.6 ms, 8t 2.13/2.39/8.85 ms (in-process; 7.2-8.8x over FP32 at 1t). README text: median token cos 0.99957, mean-pooled >= 0.99953, relL2 5.9e-2 (ONNX int8 0.868 / 0.898 / 0.55). (est. was 1t ~4-6/~7.5-9.5/~58-67, 8t ~3/~3.5/~13-15) | no (passes the int8 gate) | high | **done** (`802db38`) |
| 5 | **Custom AVX-512 FP32 projection GEMM on the original [N, K] weights** (FP32 only). Dot-product formulation, no per-call reorder or copy. | **measured** (5.1), see the item log: 1t -20.08/-33.26/-198.27 ms (2.0x), 8t -3.29/-6.26/-34.25 ms; totals 1t 19.66/30.40/203.78, 8t 8.43/8.87/29.92 (bar was 1t 20.3/30.3/203, 8t 11.2/11.8/40.7). vs ONNX FP32: wins 4 of 6, b1_s32 1t ties, b8_s32 1t 4% behind. | FP32 rounding (relL2 vs HF 4e-6) | high | **5.1 and opt-in `rehome_weights` (1t -1.41/-2.14/-6.62 ms) done, uncommitted**; 5.2 shelved |

Optional FP32 stopgap until item 5: oneDNN on the original weights (`_linear_pointwise`, no prepack, no extra
memory) measured 1.35-2.0x over MKL at 1t and 1.1-1.7x at 8t. It is a few lines, and item 5 supersedes it.

Dropped for this scope:
- BF16: slower than int8 at a similar accuracy class (relL2 2-4e-2), and AVX512-BF16 is on fewer CPUs than VNNI.
- CCD sharding: no gain for single requests or pinned 8-thread runs.
- Long-sequence flash attention: only matters for b32.
- SMT.

## Item log

### Item 1: native row kernels on Linux/x86 (done, `0a71260`)

**Change.**
- `packed_encoders/cpu/rows.c` (new): the RoPE/layout, GeGLU and add+LN entry points with the same 32-row
  tasks as `local_attention.c`, threaded by OpenMP.
- `rows.h`: the NEON add+LN is guarded by `__ARM_NEON`. The new portable version keeps NEON's 32 partial sums
  and reduction tree, so it is meant to give the same bits as the M4.
- `attention.py`: on Linux x86-64, `rows.c` is built at pack time with
  `gcc -O3 -march=native -ffp-contract=off -fno-trapping-math -fno-math-errno -fopenmp`, about 0.13 s, in a
  process-private directory. It is linked against the OpenMP runtime torch already loaded (found in
  `/proc/self/maps`), so there is one thread pool. The existing `fused_rope/geglu/norm` switches drive it.
  `attention_backend="sdpa"` stays the pure-PyTorch path. Without gcc, packing warns and uses PyTorch ops.
- Apple sources and build are unchanged; the guard preprocesses to the same code on ARM.

**Correctness.** The three fused-kernel tests (Apple-only before) now run on Linux: RoPE bitwise,
GeGLU <= 2e-5, add+LN <= 5e-7 abs / 1e-6 rel vs FP64. A new `auto` case runs the full-model parity and
owned-output test. CPU suite: 16 passed / 8 skipped (was 12 / 11); full suite 89 passed. Pack validation on
ModernBERT-base: min cos 0.9999998, relL2 2.9e-6.

**Measured** (`ab_switches.py`, 15 alternating rounds, real engine, MKL projections,
`results/item_rows_ab.txt`):

| | off | on | saved | relL2 vs off |
|---|---:|---:|---:|---:|
| b1_s16_pad32, 1t | 43.50 | 42.93 | -0.57 ms (1.3%) | 1.5e-6 |
| b1_s32, 1t | 67.48 | 66.58 | -0.90 ms (1.3%) | 5.2e-6 |
| b8_s32, 1t | 419.15 | 411.28 | -7.87 ms (1.9%) | 3.6e-6 |
| b1_s16_pad32, 8t | 13.66 | 13.17 | -0.49 ms (3.6%) | 1.5e-6 |
| b1_s32, 8t | 17.22 | 16.73 | -0.49 ms (2.8%) | 5.2e-6 |
| b8_s32, 8t | 74.00 | 69.45 | -4.55 ms (6.1%) | 3.6e-6 |

**Attribution** (one switch at a time):
- RoPE: -0.33 to -3.2 ms.
- add+LN: up to -3.7 ms on b8_s32.
- GeGLU: small. Its own stage is 2x faster (b8_s32 1t: 3.23 -> 1.27 ms), but the following GEMMs come out
  0.5-0.9 ms slower (about 0.5% of their time), for a net -0.2 to -0.4 ms.

Percentages are of today's MKL-bound totals. After the GEMM work the same milliseconds are 2-3x larger shares.

### Item 2: native short attention on Linux/x86 (done, `9d7dd9c`)

**Change.**
- `packed_encoders/cpu/local_attention_x86.c` (new): `pe_local_attention` with the Apple contract. It is
  plain C with GCC vector extensions (AVX-512 here, AVX2 halves elsewhere), OpenMP over (head, sequence) tasks.
  Per 64-row query tile:
  1. Transpose the key window.
  2. Q.K^T in 4-row x 64-key register blocks, limited to the columns each row block can see.
  3. Masked softmax with a Cephes-style vector exp (about 1 ulp).
  4. P.V in 4-row x 64-dim blocks, with 1/sum folded into the output.
- Built into the same gcc library as `rows.c`, but with contraction on (FMAs); `rows.c` still never contracts.
- `attention.py`: Linux x86-64 now gets `backend="native"` under Apple's envelope (head_dim 64,
  local_attention 128), including global attention up to 128 tokens. A new private switch `native_local`
  sits next to `native_global`; Apple behaviour is unchanged. The scratch is sized for the transposed window
  as well (`(tile + 64) x padded columns` per worker).

**Correctness.** Within 7e-6 of tiled SDPA on the test shapes: ragged local lengths across the tile and window
edges, and global up to 128 at 1 and 3 threads. Every native-attention test now runs on Linux too: CPU suite
21 passed / 3 skipped (SME2 only), full suite 94 passed.

**Measured** (`ab_switches.py`, 15 alternating rounds, reference = item 1 state with MKL projections,
`results/item_attention_ab.txt`):

| | SDPA | native | saved | local only | global only | relL2 |
|---|---:|---:|---:|---:|---:|---:|
| b1_s16_pad32, 1t | 41.67 | 40.82 | -0.86 ms | -0.73 | -0.48 | 1.5e-6 |
| b1_s32, 1t | 64.38 | 63.62 | -0.76 ms | -0.53 | -0.09 | 4.2e-6 |
| b8_s32, 1t | 413.01 | 407.57 | -5.44 ms | -3.68 | -1.68 | 3.1e-6 |
| b1_s16_pad32, 8t | 13.48 | 12.82 | -0.66 ms | -0.40 | -0.12 | 1.5e-6 |
| b1_s32, 8t | 17.14 | 16.32 | -0.82 ms | -0.56 | -0.05 | 4.2e-6 |
| b8_s32, 8t | 70.54 | 65.52 | -5.02 ms | -3.65 | -2.18 | 3.1e-6 |

**Notes.**
- Remaining attention cost (`profile_packed.py`): 0.74 / 5.38 ms for b1_s32 / b8_s32 at 1t, 0.28 / 0.94 ms at 8t.
- The kernel runs at about 124 GFLOPS on one core (34% of peak). Python and ctypes overhead is only 1-2 us per call.
- Tuning is a follow-up worth about 0.3 ms (b1_s32) and 2 ms (b8_s32) at 1t. It matters once int8 shrinks
  the GEMMs.
- Global cutoff on x86 (`_native` vs `_sdpa`, full window): native is 1.5-2.8x faster up to 192 tokens and
  slower from 256, at 1t and 8t. The M4's <= 128 cutoff therefore holds with margin.
- The 256 cliff comes from a 1 KB power-of-two stride in the transposed window. It cannot occur under the
  current policy: local windows are at most 192 columns.

### Item 3: glue trim (done, `802db38`)

**Probe first** (`profile_packed.py` and in-situ timers on HEAD `9d7dd9c`). Non-GEMM time on b1_s32 1t was 2.8 ms.
- `_check()` cost 0.29 ms per call inside a real forward, against 0.087 ms in a hot loop: after 441 MB of weights it
  walked ~200 module dicts and 134 parameters on cold caches.
- The rest of `forward()` (mask checks, contiguity, `nonzero`, `index_copy_`) cost ~0.23 ms at b1.
- The 22 `context.transpose(1, 2).reshape_as(x)` copies cost 0.06 ms at b1 and 1.42 ms at b8_s32 1t.
- `_core`'s module lookups ~65 us and `add_norm`'s per-call checks ~35 us per forward (hot).

**Change** (four parts; 3a and 3b are private switches on the engine and its `CPUAttention`).
- 3a `token_major`: `pe_local_attention` takes a `token_major` flag (x86 and Apple) and stores the context as
  [tokens, heads * 64], which Wo reads directly. `local`/`global_` take `token_major=`; SDPA and the switch-off
  path transpose in Python as before.
- 3b `fast_masks`: unpadded and right-padded masks (every row a nonempty prefix) skip the contiguity check,
  `nonzero` and `index_copy_`; unpadded output is a view of the engine's owned output. Anything else takes the
  unchanged general path (`_compacted`), which still delegates holey/empty/non-0/1 masks to HF.
- 3c **no per-call structure check** (the user approved removing it). `_check()` only rejects a closed engine;
  training is read from the root module. The structure/hook/config signatures are gone.
- 3d resolved plan: at pack time the engine keeps, per layer, detached aliases of the projection weights and
  `ResolvedNorm`s (aliases of plain LayerNorm parameters). Aliases share storage, so in-place updates are
  still read, and they keep the storage alive: a later `.data` swap, dtype/device move or replaced module
  never reaches native code as a stale pointer. The int8 segments now point into these aliases too.

**Contract change.** Structural changes after packing (replaced parameters or modules, `.data` swaps,
device/dtype moves, internal hooks, config edits) still require a repack but are no longer detected; the
engine keeps computing with what it resolved. `test_structural_changes_are_not_tracked` pins this (bitwise
unchanged output after each change); `test_mask_paths` checks fast vs general mask paths bitwise and the HF
delegations. CPU suite: 23 passed (non-int8).

**Measured** (`ab_switches.py`, 31 alternating rounds, `results/item_glue_ab.txt` and `item_glue_ab_8t.txt`).
`head` is the package at `9d7dd9c` on its own model copy (`name=@REF`, new in the harness); `head2` is a second
copy of it (A/A). Outputs are bit-identical to HEAD (relL2 0) in every variant.

| | HEAD | A/A | item 3 | saved | 3d plan | 3a | 3b | 3c |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b1_s16_pad32, 1t | 39.43 | -0.03 | 38.59 | **-0.84 ms** (2.1%) | -0.29 | -0.15 | -0.07 | -0.37 |
| b1_s32, 1t | 63.80 | +0.02 | 62.65 | **-1.15 ms** (1.8%) | -0.61 | -0.11 | +0.07 | -0.27 |
| b8_s32, 1t | 402.68 | +0.11 | 399.25 | **-3.43 ms** (0.9%) | -2.22 | -0.74 | -0.03 | -0.19 |
| b1_s16_pad32, 8t | 12.43 | -0.02 | 11.62 | **-0.80 ms** (6.5%) | 3c+3d: -0.59 | -0.17 | -0.04 | |
| b1_s32, 8t | 15.88 | -0.04 | 15.03 | **-0.85 ms** (5.4%) | 3c+3d: -0.58 | -0.19 | -0.08 | |
| b8_s32, 8t | 64.65 | -0.03 | 61.57 | **-3.08 ms** (4.8%) | 3c+3d: -2.00 | -0.47 | -0.61 | |

Attribution: 1t, each switch on top of the plan alone; 8t, each switch off from the full item (the `checks`
switch was removed before the 8t run, so 3c and 3d are combined there).

**Notes.**
- **Model-copy placement bias is small.** Control (1t, 15 rounds): the working tree on its own copy (`tree=@.`)
  vs HEAD gives -2.94 / -1.14 ms on b8_s32 / b1_s32, against -3.40 / -1.15 ms on the main copy. At most
  ~0.5 ms of b8_s32 1t is placement.
- **3d at b8 is page faults, not Python.** HEAD takes 480-1440 minor page faults per b8_s32 forward; the working
  tree takes 0 (`getrusage`). HEAD kept the 2.4 MB Wi output alive into the next layer, which made glibc return
  and re-fault large blocks; the plan loop frees it right after GeGLU. Detached aliases vs live Parameters make
  no difference (-0.06 ms). A follow-up could make this robust with engine-owned GEMM outputs.
- After item 3 the b1 non-GEMM time is about 1.7 ms at 1t; the GEMMs are now 96-97% at 1t and ~90% at 8t.

### Item 4: int8 on x86 with AVX512-VNNI (done, `802db38`)

**4.0 probe first** (`native_probe/vnni_gemm.c`): u8 x s8 `vpdpbusd` on weights packed [N/16][K/4][16][4] (one
zmm = 16 channels x 4 k), activations broadcast 4 bytes at a time, 22 distinct layers (110 MB). 88-GEMM chain,
median of 21, pure GEMM (fbgemm's numbers include its dynamic activation quantization):

| M | rows x vectors | 1t | % of 1-core peak | 8t (channel split) | fbgemm 1t / 8t |
|---:|---|---:|---:|---:|---:|
| 16 | 8 x 3 | 4.02 ms | 63% | 1.61 ms | 5.6 / 2.5 |
| 32 | 8 x 3 | 6.56 ms | 77% | 1.81 ms | 8.6 / 2.9 |
| 256 | 8 x 3 | 42.0 ms | 97% | 6.26 ms | 51.8 / 10.0 |

Other shapes: 4x6 and 4x4 tied with 8x3 at 1t; 6x4 was 1.1-1.8x slower; 8x2 was 0.1 ms better at 8t but 0.8 ms
worse on b1_s32 1t. Chosen: 8 rows x 3 vectors.

**Change.**
- `packed_encoders/cpu/int8_x86.c` (new), `int8.c`'s entry points and structs:
  - Activations are rotated and quantized exactly as on Apple, then stored row-major as u8 = q + 128.
  - The products carry 128 * sum_k w[n][k]; Python precomputes it per channel (`correction`, appended to
    `_Segment`), and dequantization subtracts it, so the int32 sums equal s8 x s8 exactly. Dequantization
    then does Apple's float operations.
  - One OpenMP team per call runs the stages with a barrier after each. GEMM tasks are (3-vector channel
    panel, 64-row chunk) pairs, panel-major, so b1 at 8t spreads over all threads (Apple's 32-row groups
    would leave one). Row stages split rows. Built like `rows.c` (gcc, contraction off, torch's OpenMP).
  - Segments read the token-major context directly (item 3a); Apple keeps its head-major gather.
  - The FWHT uses the same butterflies in the same order as the NEON one (vector-extension lane shuffles).
- `int8.py`: `vnni_available()` (avx512f + avx512bw + avx512_vnni) next to `sme2_available()`; the x86 library
  build; `pack_weights(q, lanes)`; per-channel corrections; the trailing `_Segment` fields; token-major
  context on x86. Unsupported machines keep warning and running FP32.

**Codegen finding.** The first engine build ran the GEMM chain 2.5x slower than the probe (103.8 vs 42 ms at
M=256): GCC kept a stack copy of every accumulator inside the k loop (411 zmm spills). The probe's own tile
did the same once compiled as a standalone function; it was fast only because of how it happened to be
inlined. Named accumulator variables instead of an `__m512i acc[8][3]` array give a clean loop (8 broadcasts +
24 `vpdpbusd` per k step) in any context. The engine's chain is now 3.63 / 6.16 / 41.8 ms (M = 16/32/256, 1t).
`fwht_block` still has stack traffic (its `v16 r[8]`); see the follow-ups.

**Correctness.**
- Kernels: the x86 GEMM equals s8 x s8 products exactly (M = 7..200, narrower last panels); rotate+quantize
  meets the float64 reference within 1 LSB.
- Full model: the engine is **bit-identical** (relL2 0) to an emulation that quantizes with the kernel's own
  FP32 rotate+quantize, at 7-200 tokens. Against the float64 emulation it was 9e-8 at some lengths and ~1e-3
  at others: FP32 rounding moves a few of the ~10^5 quantized values per forward across a .5 boundary
  (each flip ~1e-3 of output; the deviating tokens at L=128 were exactly one token's local window).
- So `test_int8_matches_its_emulation`'s old `< 1e-5` against float64 held only when nothing flipped (the M4
  presumably passes by luck of its FP32 arithmetic). It now asserts `< 1e-6` against the kernel-quantize
  emulation (exact pipeline) and `< 5e-3` against float64 (scheme fidelity). M4 checklist item.
- CPU suite on x86 now runs the int8 tests: 26 passed, 0 skipped.

**Accuracy** (`packed_accuracy_text.py README.md 8`, 24 paragraphs, 3,275 tokens, vs HF FP32):

| | token cos median | p5 | min | relL2 | mean-pooled cos min / median |
|---|---:|---:|---:|---:|---:|
| engine FP32 | 1.00000 | 1.0000 | 1.0000 | 1.9e-6 | 1.00000 / 1.00000 |
| engine int8 (x86) | 0.99957 | 0.9992 | 0.1224 | 5.9e-2 | 0.99953 / 0.99984 |
| ONNX int8 (`onnx_accuracy_text.py`) | 0.868 | | | 0.55 | 0.898 |

The min is one bistable token, as on the M4. Pack gate on random IDs: median cos 0.99910, relL2 4.5e-2.

**Measured, in process** (`ab_switches.py ... fp32=@. int8=@.:int8`, 15 rounds, `results/item_int8_ab.txt`):

| | FP32 (items 1-3) | int8 | speedup | relL2 vs FP32 (random IDs) |
|---|---:|---:|---:|---:|
| b1_s16_pad32, 1t | 39.37 | 4.46 | 8.8x | 4.1e-2 |
| b1_s32, 1t | 63.51 | 7.87 | 8.1x | 3.6e-2 |
| b8_s32, 1t | 399.16 | 55.60 | 7.2x | 9.3e-2 |
| b1_s16_pad32, 8t | 11.75 | 2.13 | 5.5x | 4.1e-2 |
| b1_s32, 8t | 15.08 | 2.39 | 6.3x | 3.6e-2 |
| b8_s32, 8t | 60.85 | 8.85 | 6.9x | 9.3e-2 |

**Measured against ONNX Runtime** (3 interleaved rounds of fresh processes, median of process medians,
`results/int8_vs_onnx.jsonl`; round-to-round spread at most 5%, on packed int8 b1 8t):

| | engine int8 | ONNX int8 | int8 speedup | engine FP32 | ONNX FP32 |
|---|---:|---:|---:|---:|---:|
| b1_s16_pad32, 1t | **4.32** | 10.45 | **2.42x** | 38.26 | 29.73 |
| b1_s32, 1t | **7.86** | 10.42 | **1.33x** | 62.37 | 29.74 |
| b8_s32, 1t | **55.51** | 66.51 | **1.20x** | 398.52 | 190.48 |
| b1_s16_pad32, 8t | **2.00** | 4.96 | **2.48x** | 11.68 | 11.09 |
| b1_s32, 8t | **2.40** | 4.96 | **2.07x** | 15.00 | 11.07 |
| b8_s32, 8t | **8.71** | 18.40 | **2.11x** | 60.79 | 37.14 |

int8 beats ONNX int8 in every in-scope cell, at ~10x lower error (relL2 5.9e-2 vs 0.55 on real text). Exact
FP32 still trails ONNX FP32; that is item 5.

**Where the int8 time goes** (b8_s32 1t, 55.6 ms): GEMMs ~42 ms, attention 4.5 ms, and ~9 ms for the row
stages (dequantize, add+LN, GeGLU, rotate+quantize, RoPE) plus glue. Pack time: 31.5 s at 1t, 7.4 s at 8t (calibration + GPTQ).
Memory: +110 MB of engine-owned int8 weights.

**Follow-ups, by size.**
- Row stages: `fwht_block`'s stack traffic (rotate+quantize alone measured 4.65 ms for 88 inputs at M=256,
  1t); named vectors as in the GEMM. Worth up to a few ms on b8_s32 1t.
- Attention tuning (item 6) is now ~8% of b8_s32 1t.
- An AVX-VNNI (256-bit) path for Intel client CPUs without AVX-512; untested on Intel.

### Item 5: native FP32 projection GEMM on x86 (5.1 and `rehome_weights` done, uncommitted; 5.2 shelved)

**5.0 probe** (`native_probe/fp32_gemm.c`, tiles generated by `gen_fp32_tiles.py`): dot-product (NT) tiles of
R activation rows x C weight rows on the original [N, K] weights, named accumulators (no stack spills in any
shape), horizontal reduce at the end. Same 88-GEMM chain on 22 distinct layers as before (441 MB, anonymous
memory), median of 21, ms:

| tile (R x C), next-block prefetch | M=16, 1t | M=32, 1t | M=256, 1t | M=16, 8t | M=32, 8t | M=256, 8t |
|---|---:|---:|---:|---:|---:|---:|
| **3 x 8** (chosen) | **16.63** | **25.64** | **181.6 (86%)** | 7.60 | 7.55 | **25.09** |
| 4 x 6 | 17.30 | 26.67 | 186.3 | **7.45** | **7.48** | 25.27 |
| 4 x 7 | 16.87 | 27.25 | 188.6 | | | |
| 5 x 5 | 17.89 | 28.83 | 191.1 | | | |
| 8 x 3 (int8's shape) | 22.54 | 40.09 | 249.9 | | | |
| oneDNN prepacked (bar, earlier session) | 16.5 | 25.4 | 170 (91%) | 8.2 | 8.4 | 26.5 |

- The next-block prefetch saves 1.2-2.5 ms at M=16/32. The 8t b1 chain runs at DRAM speed (59 GB/s).
- Did not help: M-chunking; prefetching activations (the next tile's rows: -12 points; this tile's, ahead:
  -4); T1/T2 prefetch hints (worse; NTA equals T0).
- **Why M=256 stops at ~86%.** A hot tile reaches 92-94% (3 x 8 at K=1152 only 87.5%: x + w exceed L1). One
  L1 weight block against 256 activation rows from L2 reaches 87-90% at K=768 and 82-84% at K=1152. Output
  writes cost ~2% (8-column blocks write half lines). Packing each tile's activations into one stream gained
  ~1 ms at M=32 and nothing at M=256. So it is activations streaming from L2 and L1 capacity, not the kernel.

**5.1 change.**
- `packed_encoders/cpu/gemm_x86.c` (new): `pe_linear(x, w, bias, y, M, N, K, workers)`, the 3 x 8 tile with
  K tails (masked loads), N tails (masked stores), bias, and the next-block prefetch. OpenMP splits 8-row
  weight blocks (static). **Batch-invariant**: every output is the same bits whatever M, the tile it lands
  in or the thread count (one accumulation order, one reduction tree per output). Tested.
- `attention.py`: built into the x86 library (contraction fast). `CPUAttention.linear` with the private
  switch `native_linear` (x86 only; Apple keeps Accelerate through `F.linear`). Activations that are not
  contiguous FP32 take `F.linear`, so native code never reads past them. Results borrow one engine-owned
  buffer per output width.
- `modernbert_cpu.py`: `_core` projects with `cpu.linear` by default (int8 calibration still passes its own
  `F.linear` recorder).
- **Engine-owned outputs** (the item-3 robustness follow-up): `F.linear` allocated 88 fresh outputs per forward,
  which faulted erratically under glibc (0 to 4,872 minor faults per b8_s32 1t forward depending on the
  process, ~10 ms). Now 0 in every process.

**Correctness and accuracy.** Per GEMM, relL2 2.8e-7 against float64. README text, FP32 engine against HF FP32:
relL2 4.0e-6, token cos median and min 1.00000 (with MKL: 1.9e-6; both are FP32 rounding). New tests
`test_native_linear` (tails in every dimension, bias, thread and batch invariance bitwise, fallbacks, borrowed
buffers) and `test_native_linear_reads_live_weights`. CPU suite 28 passed; full suite 193 passed.

**Measured, in process** (`ab_switches.py ... mkl=native_linear:0 gemm=native_linear:1`, 15 rounds,
`results/item_gemm_ab.txt`; outputs relL2 1.7-6.3e-6 from MKL's):

| | MKL | native GEMM | saving |
|---|---:|---:|---:|
| b1_s16_pad32, 1t | 39.74 | **19.66** | -20.08 (2.02x) |
| b1_s32, 1t | 63.66 | **30.40** | -33.26 (2.09x) |
| b8_s32, 1t | 402.06 | **203.78** | -198.27 (1.97x) |
| b1_s16_pad32, 8t | 11.72 | **8.43** | -3.29 (1.39x) |
| b1_s32, 8t | 15.12 | **8.87** | -6.26 (1.71x) |
| b8_s32, 8t | 64.17 | **29.92** | -34.25 (2.14x) |

This beats the bar (prepacked oneDNN in the engine: 1t 20.3/30.3/203, 8t 11.2/11.8/40.7) everywhere except a
tie on b1_s32 and b8_s32 at 1t, with no extra memory.

**Where FP32 time goes now** (b8_s32 1t, 201 ms, stage timers): GEMMs 191 ms (Wqkv + Wi at 83.5% of peak,
Wo 83%, Wmo at K=1152 73%), attention 4.8, add+LN 1.7, RoPE 1.65, GeGLU 1.5.

**Measured against ONNX Runtime FP32** (3 interleaved rounds of fresh processes, `results/fp32_vs_onnx.jsonl`):

| | engine FP32 | ONNX FP32 | |
|---|---:|---:|---|
| b1_s16_pad32, 1t | **19.48** | 30.34 | 1.56x |
| b1_s32, 1t | 30.23 | 30.31 | tie |
| b8_s32, 1t | 199.51 | **191.24** | 4% behind |
| b1_s16_pad32, 8t | **8.47** | 11.18 | 1.32x |
| b1_s32, 8t | **8.87** | 10.95 | 1.23x |
| b8_s32, 8t | **29.78** | 37.29 | 1.25x |

**`rehome_weights` (opt-in, the user's call).** `from_pretrained` leaves the weights in a private mmap of the
safetensors file: 4 KB page-cache pages, no THP. The probe and ONNX Runtime read anonymous memory.
- **Measured first** with a plain `param.data = clone` in the harness: 1t -1.44 / -2.45 / -2.39 ms, 8t ~0.
  With THP disabled for the process (`prctl`), b1 still gained -0.89 / -1.93. So about two thirds comes from
  leaving the file mapping and one third from huge pages. But RSS grew by the full 431 MB: the old file
  pages stay mapped.
- **Change.** `ModernBertCPU(rehome_weights=False)` and `packed_encoders/cpu/memory.py`. When set, packing
  moves the 88 projection weights (and biases) into fresh memory before the plan takes its aliases: same
  Parameter objects and values, new storage, `MADV_HUGEPAGE` on the new range (so THP `madvise` systems get
  huge pages too). Then `MADV_PAGEOUT` on the whole pages of the old range that lie in file mappings: it
  never changes contents, so anything still holding the old storage reads the same values (tested on a
  private `torch.from_file` mapping).
- **Memory, ModernBERT-base:** anon +433 MB, file -413 MB, so RSS is flat (+20 MB).
- **Contract** (docstring): in-place updates and `state_dict` keep working; references to the old storage
  taken before packing stop aliasing; unpacking does not move weights back.
- **Measured, in process** (`ab_switches.py ... mmap=@. rehome=@.:fp32:rehome`, 15 rounds,
  `results/item_gemm_rehome_ab.txt`; outputs bit-identical):

| | weights in HF's mmap | `rehome_weights=True` | saving |
|---|---:|---:|---:|
| b1_s16_pad32, 1t | 19.64 | **18.23** | -1.41 |
| b1_s32, 1t | 30.02 | **27.88** | -2.14 |
| b8_s32, 1t | 200.81 | **194.19** | -6.62 |
| b1_s16_pad32, 8t | 8.49 | 8.47 | -0.02 |
| b1_s32, 8t | 8.92 | 8.83 | -0.09 |
| b8_s32, 8t | 29.33 | 29.11 | -0.23 |

With it, b1_s32 1t wins (~27.9 vs ONNX 30.3); b8_s32 1t stays ~1.5% behind (~194 vs 191).

**5.2, a large-M kernel** (`native_probe/fp32_packed.c`, **shelved by the user** for reevaluation). It is
GotoBLAS-style: each 64-row weight panel is packed per call into a thread-private [K][64] buffer, with a
6 x 64 broadcast kernel writing full output lines.
- The kernel alone runs at **96.9%** (160.5 ms at M=256). Packing alone is 11.7-13.5 ms, and at first it
  did not overlap at all (181 ms).
- Packing 16 rows at a time: 176 ms. Then software-pipelined packing (panel p+1 packed in slices between
  panel p's tiles, prefetches spread through the kernel's k loop): **173.7 ms (89.5%)**, against 181-185
  for 3 x 8 in the same session.
- Next-panel L2 prefetch, K blocking and K-outer loop orders did not help.
- It loses at small M (M=32: 32.3 vs 25.6) and at 8t (M=256: 29.6 vs 25.1; 12 panels over 8 threads). So
  it would serve only 1t with M >= ~128, for an estimated -8 to -10 ms on b8_s32.
- It would also give up batch invariance across the M cutoff (sequential-k sums instead of the NT tree).

## Planning notes

### Do the gains compound?

In milliseconds they add, because the items replace disjoint stages. `compound.py` toggles faster projections
(prepacked oneDNN as item 5's stand-in) and native row kernels (item 1, probe build) in one alternating A/B, 7 rounds:

| | today | projections | rows | both | sum of the two savings | measured saving |
|---|---:|---:|---:|---:|---:|---:|
| b1_s16, 1t | 41.58 | 20.26 | 40.20 | 19.47 | 22.69 | 22.11 |
| b1_s32, 1t | 65.64 | 30.34 | 64.86 | 29.42 | 36.08 | 36.23 |
| b8_s32, 1t | 420.74 | 202.77 | 413.83 | 194.68 | 224.87 | 226.06 |
| b1_s16, 8t | 14.02 | 11.19 | 13.24 | 10.54 | 3.61 | 3.48 |
| b1_s32, 8t | 17.34 | 11.75 | 16.40 | 10.81 | 6.52 | 6.52 |
| b8_s32, 8t | 74.56 | 40.69 | 69.16 | 35.92 | 39.26 | 38.64 |

The sum and the measurement agree within 0.6 ms (3% of the saving), and outputs stay within relL2 1.1e-5 of
today's. Ratios do not multiply: each later item is a larger fraction of a smaller total. int8 (item 4) does not
stack on item 5, since it replaces the projections as a separate precision mode. Items 1-3 carry over
into int8 mode, where they are a larger share because the GEMMs shrink about 3x.

The in-engine row-kernel savings on b8_s32 (6.9 ms at 1t, 5.4 ms at 8t) are well below the isolated-kernel
estimate (16 / 9.4 ms). Only in-engine numbers are used below.

### Projection (FP32, exact)

| | today | 5 (bar) + 1, measured | + 2 + 3 (est.) | vs ONNX FP32 | with item 5's target (est.) | ONNX FP32 | ONNX int8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| b1_s16, 1t | 41.6 | 19.5 | ~18.3 | 1.6x | ~13.3 (2.3x) | 30.1 | 10.6 |
| b1_s32, 1t | 65.6 | 29.4 | ~27.7 | 1.09x | ~25.2 (1.19x) | 30.1 | 10.7 |
| b8_s32, 1t | 420.7 | 194.7 | ~187.3 | 1.02x | same | 191.6 | 66.9 |
| b1_s16, 8t | 14.0 | 10.5 | ~9.3 | 1.19x | same (DRAM-bound) | 11.1 | 5.0 |
| b1_s32, 8t | 17.3 | 10.8 | ~9.5 | 1.15x | same (DRAM-bound) | 10.9 | 5.0 |
| b8_s32, 8t | 74.6 | 35.9 | ~31.6 | 1.18x | same | 37.2 | 18.6 |

Exact FP32 ends 1.0-1.2x ahead of ONNX FP32 on the unpadded inputs; both do the same FLOPs near peak.
b1_s16 gains more because packing halves its work. Beating ONNX int8 needs item 4.

## Reproduction

```bash
B=benchmarks/cpu_x86; MODELS=<dir from onnx/export.py + onnx/variants.py, see SURVEY_16T.md>
taskset -c 2   .venv/bin/python $B/bench.py packed --threads 1 --workloads b1_s16_pad32,b1_s32,b8_s32
taskset -c 0-7 uv run --no-sync --with onnxruntime python $B/bench.py onnx --threads 8 --model $MODELS/int8.onnx \
    --workloads b1_s16_pad32,b1_s32,b8_s32
.venv/bin/python $B/summarize.py $B/results/small_1t_8t.jsonl
taskset -c 2   .venv/bin/python $B/ab_linear.py 1 b1_s16_pad32,b1_s32,b8_s32 mkl,dnn,dnn_packed,bf16,int8 5
taskset -c 2   .venv/bin/python $B/profile_packed.py 1 b1_s16_pad32,b1_s32,b8_s32
taskset -c 0-7 .venv/bin/python $B/int8_recipe_accuracy.py 8
gcc -O3 -march=native -fopenmp -ffp-contract=off -fno-trapping-math -fno-math-errno -shared -fPIC \
    $B/native_probe/rows_x86.c -o /tmp/rows_x86.so -L.venv/lib/python3.12/site-packages/torch/lib \
    -l:libgomp.so.1 -Wl,-rpath,$PWD/.venv/lib/python3.12/site-packages/torch/lib
taskset -c 2   .venv/bin/python $B/compound.py 1 /tmp/rows_x86.so 7
taskset -c 2   .venv/bin/python $B/ab_switches.py 1 15 off=fused_rope:0,fused_geglu:0,fused_norm:0 on=fused_rope:1,fused_geglu:1,fused_norm:1
gcc -O2 -march=native $B/native_probe/isa_peak.c -o /tmp/isa_peak && taskset -c 2 /tmp/isa_peak
```
