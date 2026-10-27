# Runtime FP32 GEMM tile selection

Follow-up to [GEMM_REVIEW.md](GEMM_REVIEW.md), measured on 2026-10-18. The reference
is `08eb9a3` (`native_probe/fp32_cache_baseline.c`, an exact source snapshot).

## Usage and behavior

The default now detects L1 data-cache capacity once when preparing the CPU engine
and selects a register tile from its estimated working set. No new argument is
needed. Optional measurement during packing can refine that choice:

```python
pe.pack(model, engine=pe.ModernBertCPU(threads=1, tune_gemm=True))
# Also works with threads=8 and/or rehome_weights=True.
```

`packed_encoders/cpu/gemm.py` reads Linux sysfs for every CPU in the calling
thread's affinity mask and takes the smallest L1D capacity. Missing/incomplete
information selects the smallest candidate. It does not change affinity or
process-global thread settings. Configure affinity before packing; repack after
moving to a different CPU set. Existing OpenMP worker placement must also be
consistent with that CPU set.

Candidate register tiles are 3x8, 4x6, 3x6, 3x4 and 2x4. The selector picks the
first candidate satisfying `4 * K * (min(M, rows) + columns) <= L1_bytes * 15/16`.
If none fit, it uses the smallest working set. A cache miss affects speed, not
correctness. This estimate leaves some headroom but cannot model associativity,
prefetch traffic, or all microarchitecture differences.

There are no hardcoded 32/48 KiB cache classes. For the relevant M >= 16:

| Detected L1D | K=768 tile / estimated working set | K=1152 tile / estimated working set |
|---|---|---|
| 32 KiB | 4x6 / 30 KiB | 2x4 / 27 KiB |
| 48 KiB | 3x8 / 33 KiB | 4x6 / 45 KiB |
| 64 KiB | 3x8 / 33 KiB | 3x8 / 49.5 KiB |

Tile shapes are compiled constants; cache capacities are runtime inputs. All
tiles retain the same FP32 accumulation and reduction order, use live weights,
and support partial rows, columns and K. The AVX2 fallback remains `F.linear`.

With `tune_gemm=True`, packing measures all five candidates at 16, 32 and 256
live tokens (clipped to `max_tokens`) using the configured native thread count.
Each sweep visits all actual model projections of a given `(N, K)`, avoiding a
benchmark of one repeatedly hot matrix. After warmup, three alternating timing
rounds are reduced by median. A candidate must beat the default by 3% to replace
it. The proposed combination must then beat the default by 3% in a second,
alternating measurement that visits projections in model order. That final
check caught a grouped-timing winner that lost with rehomed weights. Measurement
may select a tile exceeding the cache estimate. Tuning is best effort: even the
projection chain omits attention, normalization, and activation work.

Selections are engine-local, indexed by `(M, N, K)`; other lengths use the cache
heuristic without timing during requests. Tuning keeps no weight copies, does
not modify weights or global RNG state, and is skipped for actual int8 execution
and backends without the native GEMM. Unpacking clears choices and measurements.
The private `attention.linear_tuning` report records candidates and timings.

## Full-model measurements

Ryzen 9 9950X, actual 48 KiB L1D. CPU 2 for 1 thread, CPUs 0-7 for 8 threads.
Seven alternating rounds, four variants on identical model weights in one
process; timings below are medians in ms. Every full-model output matched the
reference bitwise. `before` uses the reference C kernel through an adapter with
the same ctypes ABI as the new kernel.

| Workload | Threads | Before | Forced 32 KiB | Detected 48 KiB | Tuned |
|---|---:|---:|---:|---:|---:|
| b1_s16_pad32 | 1 | 17.09 | 17.01 | 16.66 | 16.62 |
| b1_s32 | 1 | 28.04 | 28.66 | 27.28 | 27.27 |
| b8_s32 | 1 | 197.75 | 211.46 | 197.55 | 197.67 |
| b1_s16_pad32 | 8 | 9.00 | 9.13 | 8.98 | 8.92 |
| b1_s32 | 8 | 9.08 | 9.09 | 9.13 | 9.20 |
| b8_s32 | 8 | 29.25 | 30.27 | 28.66 | 28.67 |

The default 48 KiB path improves the 1-thread single-batch cases by 2.5-2.7%,
with the 1-thread batch-of-eight case essentially unchanged. At 8 threads the
changes range from 0.6% slower to 2.0% faster. It preserves the previous tile
choices; the differences include generated-code changes from the added
specializations and dispatching once per GEMM to loops with constant tile
dimensions, avoiding runtime division per weight block.

The final tuner retained the default tiles for all nine shapes in all four
processes. Differences between the detected and tuned columns are timing
variation and their Python lookup paths, not different native tiles. Earlier
grouped-only tuning showed a gain for regular weights but a loss for rehomed
weights; the final model-order check avoids accepting such weak candidates.
Full packing including tuning took 7.57 s at 1 thread and 3.21 s at 8 threads
without rehoming. The default cache heuristic does not incur measurement cost.

With `rehome_weights=True`, independently alternating on identical rehomed
weights within each process:

| Workload | Threads | Before | Forced 32 KiB | Detected 48 KiB | Tuned |
|---|---:|---:|---:|---:|---:|
| b1_s16_pad32 | 1 | 16.58 | 16.62 | 16.73 | 16.79 |
| b1_s32 | 1 | 26.12 | 27.52 | 26.08 | 26.06 |
| b8_s32 | 1 | 190.66 | 199.27 | 190.42 | 190.46 |
| b1_s16_pad32 | 8 | 9.00 | 9.01 | 9.12 | 9.11 |
| b1_s32 | 8 | 9.37 | 9.32 | 9.32 | 9.32 |
| b8_s32 | 8 | 28.45 | 29.50 | 28.27 | 28.25 |

The default path's changes with rehoming range from 1.3% slower to 0.7% faster.
The purpose of runtime selection is portability; these runs do not establish
a universal speed improvement from cache awareness or optional tuning.

**Forcing 32 KiB tests selection and execution, not a physical cache limit.**
Its performance here cannot establish its speed or optimality on a real 32 KiB
CPU. Different execution width, cache associativity, and bandwidth still matter.
Likewise, historical ONNX timings are not a fresh paired comparison with these
runs.

Raw rounds, selected tiles, and pack-time measurements are in
`results/cache_gemm_1t.jsonl`, `results/cache_gemm_8t.jsonl`, and their `_rehome`
counterparts. Files ending `_initial.jsonl` preserve the exploratory runs before
the final dispatch and model-order check.

## Validation

`pytest tests -q`: **110 passed, 96 skipped** (GPU-dependent tests skipped).
The new coverage includes mixed-affinity cache discovery, missing cache data,
capacities beyond 32/48 KiB, every candidate at 1/8 threads, partial M/N/K,
bitwise batch invariance, invalid native tile arguments, tuning with a non-FP32
application default dtype, unchanged live weights/RNG/thread settings, and
model-order acceptance/rejection of simulated timing winners. Full-model packing,
unpacking, the non-native backend, and an actual AVX2-targeted build also pass.
The complete benchmark matrix asserts bitwise full-model parity for every variant.

```bash
HF_HUB_OFFLINE=1 taskset -c 2 .venv/bin/python benchmarks/cpu_x86/cache_gemm.py --threads 1 --rounds 7
HF_HUB_OFFLINE=1 taskset -c 0-7 .venv/bin/python benchmarks/cpu_x86/cache_gemm.py --threads 8 --rounds 7
# Add --rehome to test rehome_weights=True.
.venv/bin/python -m pytest tests/test_cpu_gemm.py tests/test_modernbert_cpu.py -q
```
