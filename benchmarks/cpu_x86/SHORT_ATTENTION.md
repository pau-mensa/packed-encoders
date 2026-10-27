# Short x86 attention follow-up

Baseline: `c371aa9` (the committed Hadamard optimization). Production changes are in
`packed_encoders/cpu/local_attention_x86.c`; no Python API or preparation changes.

## Implementation

- Name the sixteen P×V accumulators so GCC does not materialize the accumulator array
  on the stack before multiplying by the inverse softmax sum. The baseline already
  kept these accumulators in registers inside the multiply loop: this removes traffic
  around that loop, not a spill on every multiply.
- Dispatch full-window sequences of exactly 16 or 32 tokens to a constant-size helper.
  Dispatch requires `tile >= length` and `half >= length - 1`. Other sizes and masked
  windows retain the general path; short and general sequences can share a batch.
- Inline softmax so the helper's constant bounds eliminate the masks and loop overhead.
  Keep the Cephes exponential, ordered sum, FMA dot products, and final normalization.
- Transpose keys in 4×4 vector blocks, replacing scalar loads/stores. This uses the
  existing GCC vector-extension approach, including on AVX2.

Forced four-row softmax unrolling was rejected: a matched microbenchmark increased
32-token attention from 18.35 to 20.75 µs and batched attention from 144.22 to 162.88 µs.
`results/short_attention_unroll_probe.jsonl` records this experiment: `after` is the
rejected forced-unroll version, `candidate` the retained version, `before` the baseline.

## Measurement method

Ryzen 9 9950X, GCC 16.2.1. Run processes serially, pinning one thread to CPU 2 and eight
threads to CPUs 0–7. Every row is an in-process A/B: same model, same prepared weights,
same buffers, only the attention library changes. Native GEMM and Hadamard tuning
are present on both sides. FP32 uses the default weight storage, without rehoming.

Fifteen measured rounds alternate A/B order, after a warm-up round. Each variant gets
an untimed forward before each round; measure twenty forwards per round except
one-thread b8_s32, which uses five. The microbenchmark runs 200 calls per round over
12 heads and checks outputs after every round. Results below are medians of round means.

The baseline C snapshot is `native_probe/attention_baseline.c`, identical to the source
at `c371aa9`, SHA-256 `c09458ee4682775f45cfe71035214ca959ca03451699bd12010b59c267b31923`.
`short_attention.py` compiles it with production optimization flags and links the same
OpenMP runtime already loaded by PyTorch. It switches only `attention.library`;
row kernels and projections continue using the current library.

## Results

Whole-model latency (milliseconds; positive reduction means faster):

| Precision | Workload | Threads | Before | After | Time reduction |
|---|---|---:|---:|---:|---:|
| fp32 | b1_s16_pad32 | 1 | 18.365 | 18.262 | +0.56% |
| fp32 | b1_s32 | 1 | 30.556 | 30.417 | +0.46% |
| fp32 | b8_s32 | 1 | 202.443 | 201.282 | +0.57% |
| fp32 | b1_s16_pad32 | 8 | 8.677 | 8.683 | -0.06% |
| fp32 | b1_s32 | 8 | 9.041 | 9.027 | +0.15% |
| fp32 | b8_s32 | 8 | 29.550 | 29.373 | +0.60% |
| int8 | b1_s16_pad32 | 1 | 4.390 | 4.320 | +1.59% |
| int8 | b1_s32 | 1 | 7.758 | 7.601 | +2.02% |
| int8 | b8_s32 | 1 | 54.190 | 52.913 | +2.36% |
| int8 | b1_s16_pad32 | 8 | 1.973 | 1.961 | +0.59% |
| int8 | b1_s32 | 8 | 2.334 | 2.307 | +1.16% |
| int8 | b8_s32 | 8 | 8.452 | 8.301 | +1.79% |

Attention alone (microseconds, including the ctypes call):

| Workload | Threads | Before | After | Time reduction |
|---|---:|---:|---:|---:|
| b1_s16 | 1 | 10.71 | 7.46 | 30.3% |
| b1_s32 | 1 | 25.59 | 18.41 | 28.1% |
| b8_s32 | 1 | 200.35 | 143.75 | 28.2% |
| b1_s16 | 8 | 3.20 | 2.65 | 17.1% |
| b1_s32 | 8 | 5.76 | 4.40 | 23.7% |
| b8_s32 | 8 | 28.35 | 20.95 | 26.1% |

The main benefit is int8: 1.6–2.4% less whole-model time at one thread and about
1.8% on b8_s32 at eight threads. FP32 saves about 0.5–0.6% at one thread and on
b8_s32 at eight threads. The two small FP32 eight-thread cases are effectively flat;
their measured differences are too small to claim a win or regression. The 0.6%
int8 eight-thread s16 change is also small relative to round-to-round noise.

These are one matched process per precision/thread count, not cross-process baseline
comparisons. Raw rounds are in `results/short_attention_{fp32,int8}_{1,8}t.jsonl`.


## Validation

- All twelve whole-model precision/workload/thread cells matched the baseline exactly
  after every measured round, including padded outputs.
- 960 kernel comparisons matched the baseline: 240 combinations each for native
  AVX-512 and forced `-march=x86-64-v3` builds, at one and eight threads. This includes
  16/32-token sequences alone and mixed with lengths 1, 3, 4, 7, 15, 17, 31, 33, 65,
  129 and 200; ten window widths; four tile sizes; both output layouts; uniform and
  sharp softmax rows. See `results/short_attention_correctness.jsonl`.
- New regression tests compare the specialized path to the general path and SDPA,
  test dispatch boundaries, and require exact results across threading and batching.
  Arbitrary *masked* windows can change their softmax sum alignment when the tile size
  changes; those cross-tile comparisons use tolerance. Full-window short sequences
  must match exactly. This is existing behavior, not a change made here.
- The existing AVX2 build test now explicitly exercises both short sequence lengths
  and compares them against the general path.
- `pytest tests -q`: **121 passed, 96 skipped** (GPU-dependent tests skipped; 24 NVML
  warnings). Raw output: `results/short_attention_tests.txt`. `git diff --check` clean.
- AVX2 was compiled and executed on this AVX-512 host. This verifies compatibility and
  numerical parity; it does not establish performance on another processor.

The transpose uses `__builtin_shuffle`, already used by the project, so it does not
introduce a newer GCC builtin requirement. Replacing the initial prototype's
`__builtin_shufflevector` with this form produced byte-identical generated assembly.

## Reproduction

Run these serially from the repository root (the model must already be cached):

```bash
B=benchmarks/cpu_x86
OPENBLAS_NUM_THREADS=1 taskset -c 2 .venv/bin/python $B/short_attention.py --threads 1 --micro --precision int8
OPENBLAS_NUM_THREADS=1 taskset -c 0-7 .venv/bin/python $B/short_attention.py --threads 8 --micro --precision int8
OPENBLAS_NUM_THREADS=1 taskset -c 2 .venv/bin/python $B/short_attention.py --threads 1 --precision fp32
OPENBLAS_NUM_THREADS=1 taskset -c 0-7 .venv/bin/python $B/short_attention.py --threads 8 --precision fp32
# Baseline parity for generic and specialized paths, both ISAs and thread counts:
for isa in native x86-64-v3; do
  taskset -c 2 .venv/bin/python $B/short_attention.py --threads 1 --isa "$isa" --check-only
  taskset -c 0-7 .venv/bin/python $B/short_attention.py --threads 8 --isa "$isa" --check-only
done
.venv/bin/python -m pytest tests -q
```
