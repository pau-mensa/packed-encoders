# Scheduling and attention-call preparation

Measured on 2026-10-18, Ryzen 9 9950X, GCC 16.2.1. Reference: committed `822c9b8`
(including dequantization/LayerNorm fusion). The retained change is Python call
preparation; the native C kernels and OpenMP schedules are unchanged.

## What changed

The INT8 pipeline overwrites the same Q/K/V storage each layer. It now creates the
three tensor views once per forward. On the existing x86 native-attention path
(maximum sequence length <= 128), it also prepares local/global attention argument
tuples once and reuses them across layers. Previously each layer repeated tensor
view creation, contiguous checks, shape queries, buffer checks, and pointer extraction.

`CPUAttention._native_args` centralizes the existing scratch/output allocation and
argument construction. The INT8 caller prepares the global window first, ensuring
scratch is large enough for either window. Both calls then share stable buffers.
Pointers are local to one forward, with their owning tensors kept alive. They are
rebuilt after reservation for each new input size; no cross-forward pointer cache.

Long sequences, SDPA, Apple, and the head-major layout fallback retain their normal
attention dispatch. Q/K/V view reuse also applies on those paths. The general FP32
native-attention wrapper uses the shared preparation helper, so FP32 was benchmarked
too. No new option, ISA requirement, process-global setting, or thread policy.

## Final full-model results

**INT8**

| Workload | Threads | Before ms | After ms | Time reduction |
|---|---:|---:|---:|---:|
| b1_s16_pad32 | 1 | 4.167 | 4.101 | 1.59% |
| b1_s32 | 1 | 7.410 | 7.352 | 0.78% |
| b8_s32 | 1 | 52.226 | 52.154 | 0.14% |
| b1_s16_pad32 | 8 | 1.956 | 1.897 | 3.01% |
| b1_s32 | 8 | 2.272 | 2.220 | 2.31% |
| b8_s32 | 8 | 8.297 | 8.242 | 0.66% |

**FP32**

| Workload | Threads | Before ms | After ms | Time reduction |
|---|---:|---:|---:|---:|
| b1_s16_pad32 | 1 | 17.216 | 17.218 | -0.01% |
| b1_s32 | 1 | 27.476 | 27.507 | -0.11% |
| b8_s32 | 1 | 194.155 | 194.138 | 0.01% |
| b1_s16_pad32 | 8 | 8.529 | 8.526 | 0.03% |
| b1_s32 | 8 | 8.881 | 8.873 | 0.09% |
| b8_s32 | 8 | 28.906 | 28.918 | -0.04% |

The INT8 saving is about 50-70 us per forward, most useful for the small eight-thread
cases. The 0.14% batched one-thread difference is small relative to run-to-run noise.
FP32 is effectively flat (all six differences within 0.12%).

Fifteen measured alternating rounds after warmup, using the same prepared weights
and buffers in each process. Each variant receives an untimed warm forward before
its timed repetitions. Twenty forwards per round, except b8_s32 at one thread uses
five. Processes run serially: CPU 2 for one thread, CPUs 0-7 for eight threads.
Every round compares complete outputs as int32 views, including padded positions.

The reference Python methods are saved in `native_probe/scheduling_python_baseline.py`;
the two C snapshots are the exact sources from `822c9b8`. Default `scheduling.py`
switches both reference Python methods and native handles for a complete matched
comparison. Explicit candidate options keep the baseline Python path fixed to isolate
the candidate. No new ONNX benchmark was needed for this incremental comparison.

## Native scheduling experiments

1. **Remove the last INT8 worksharing barrier:** GCC already does this. Baseline and
   explicit `nowait` assembly differ only in their `.file` directive. A second variant
   also rewrote the final-layer normalization barrier; no consistent model benefit.
2. **Static attention work distribution:** helped isolated 16-token attention
   (2.96 -> 2.41 us), but regressed 32-token and batched attention (4.85 -> 5.10 us,
   21.24 -> 23.06 us). Kept dynamic scheduling.
3. **Split query work within each head:** twelve heads divide unevenly over eight
   workers. Two/four query tasks per head improve balance but repeat the key transpose.
   The best dynamic two-way variant improved isolated 32-token attention
   (4.85 -> 4.40 us), without a consistent full-model win. No splitting retained.
4. **Runtime overhead calibration:** a synthetic team with nine trivial row stages
   costs roughly 2.1-2.3 us at eight threads; explicit final `nowait` is unchanged.
   This estimates scheduling cost for that synthetic workload, not the model's
   load-imbalance or barrier-wait time.

The native candidate model run used eleven alternating rounds. All variants matched
baseline outputs. Raw results: `scheduling_attention_probe_8t.jsonl`,
`scheduling_native_probe_8t.jsonl`, and `scheduling_overhead_8t.jsonl` under `results/`.
The reproducible source generator is `native_probe/scheduling_candidates.py`.

## Python preparation probe

Fifteen alternating rounds, before implementing the shared helper:

| INT8, eight threads | Baseline ms | Reuse Q/K/V views only | Prepare arguments too |
|---|---:|---:|---:|
| b1_s16_pad32 | 1.962 | 1.946 | 1.911 |
| b1_s32 | 2.293 | 2.272 | 2.240 |
| b8_s32 | 8.371 | 8.362 | 8.320 |

This identified about 50 us per forward outside the arithmetic kernels. The final
implementation uses the shared attention helper instead of duplicating its allocation
logic. Raw rounds: `results/scheduling_python_probe_8t.jsonl`.

## Validation

- **126 passed, 96 skipped**, with 26 NVML warnings on this CPU host. Full output:
  `results/scheduling_tests.txt`.
- New one/eight-thread model tests cover changing token counts, mixed lengths,
  16/32/128/129-token sequences, reused larger allocations, and head-major fallback.
  They compare against the independent INT8 emulation and check repeated-input bits.
- The 128-token case exercises different local/global windows sharing scratch;
  129 tokens exercises the SDPA global fallback before returning to the native path.
- All twelve FP32/INT8 benchmark cells match baseline output bits on every round.
- Existing tests cover AVX2 attention, SDPA, owned outputs, native attention masks,
  INT8 weight-update guards, thread-count parity, and microbatches.
- `git diff --check` clean. No commit made.

## Reproduction

Run from the repository root, serially:

```bash
B=benchmarks/cpu_x86
for precision in int8 fp32; do
  OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 taskset -c 2 .venv/bin/python $B/scheduling.py --threads 1 --precision "$precision"
  OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 taskset -c 0-7 .venv/bin/python $B/scheduling.py --threads 8 --precision "$precision"
done
.venv/bin/python -m pytest tests -q

# Rejected native probes and the initial Python ablation:
.venv/bin/python $B/native_probe/scheduling_candidates.py /tmp/pe-scheduling
OPENBLAS_NUM_THREADS=1 taskset -c 0-7 .venv/bin/python $B/scheduling.py --threads 8 --micro-only --attention-candidate /tmp/pe-scheduling/static.c --attention-candidate /tmp/pe-scheduling/split2dynamic.c
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 taskset -c 0-7 .venv/bin/python $B/scheduling.py --threads 8 --rounds 11 --attention-candidate /tmp/pe-scheduling/split2dynamic.c --q8-candidate /tmp/pe-scheduling/nowait.c --q8-candidate /tmp/pe-scheduling/nowait_last.c
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 taskset -c 0-7 .venv/bin/python $B/scheduling.py --threads 8 --python-probe $B/native_probe/scheduling_python_probe.py
OPENBLAS_NUM_THREADS=1 taskset -c 0-7 .venv/bin/python $B/native_probe/scheduling_overhead.py 8
```
