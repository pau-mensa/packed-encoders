# Attention calibration sampling

The tuner now warms each backend once and retains three measured samples per
backend and shape. A warmup estimates how many calls fit in a 1 ms sample, capped
at three. Slow kernels run once per sample; fast kernels can still amortize timer
and Python overhead. This reduces repeated expensive calls while measuring the
same ten shape buckets and retaining the 5% native-win margin.

The 1 ms target controls repetitions, not a deadline. A single expensive call
can exceed it, and the tuner completes every bucket.

## Local measurements

AMD Ryzen 9 9950X (Zen 5), PyTorch 2.11.0+cu128, 12 heads. Each profile runs in
a fresh process with CPU affinity set before startup. One-thread profiles use
CPU 2; eight-thread profiles use CPUs 0–7. Five sweeps per method alternate
method order. Native compilation is excluded; input generation and allocation
are included. `OMP_WAIT_POLICY` was unset.

| Kernel build | Threads | Original median | Adaptive median | Reduction |
| --- | ---: | ---: | ---: | ---: |
| AVX2 (`x86-64-v3`) | 1 | 3.769 s | 1.282 s | 66.0% |
| AVX2 (`x86-64-v3`) | 8 | 0.574 s | 0.218 s | 62.0% |
| Native (AVX-512) | 1 | 0.674 s | 0.253 s | 62.4% |
| Native (AVX-512) | 8 | 0.155 s | 0.081 s | 47.9% |

The adaptive tuner matched the original backend choice in all 200 paired
bucket comparisons (four profiles × five sweeps × ten buckets). AVX2 selected
SDPA everywhere; the native build selected native attention everywhere. These
profiles do not exercise a close crossover. An ablation using one call per
sample even for fast kernels was slightly faster; adaptive sampling retains
more repetitions for the fast kernels.

Raw timings, call counts and per-bucket medians:

- [AVX2, one thread](results/calibration_zen5_avx2_t1.json)
- [AVX2, eight threads](results/calibration_zen5_avx2_t8.json)
- [Native, one thread](results/calibration_zen5_native_t1.json)
- [Native, eight threads](results/calibration_zen5_native_t8.json)

These are local Zen 5 measurements. Compiling AVX2 on this CPU does not reproduce
the EPYC Zen 3 hardware or its PyTorch dispatch.

## Modal Zen 3 measurement

The run on 2026-10-27 used Modal's EU Function pool, with a request and hard limit
of one physical CPU, 4 GiB memory, PyTorch 2.11.0+cpu, Transformers 5.3.0 and GCC
12.2.0. Affinity exposed two logical CPUs; topology files were hidden. The CPU
brand string was `unknown`, but `/proc/cpuinfo` reported `AuthenticAMD`, family
25 (19h), model 1, AVX2 and no AVX-512: the Milan/Zen 3 server generation.

Five sweeps per method alternated order, with the same ten buckets and 12 heads
as the local experiment. Native compilation was excluded. Median calibration
time fell from **9.583 s to 3.245 s**, a **66.1% reduction**. The one-call ablation
took 3.231 s. All 50 paired bucket decisions matched; both the original and
adaptive tuner selected SDPA everywhere. In the first sweep, the original made
120 calls per backend; adaptive made 40 native and 52 SDPA calls.

The targeted CPU suites completed with **55 passed, 11 skipped**. The skips
require native AVX-512/VNNI projections unavailable on this AVX2 host. The tests
include full-model FP32 parity, automatic attention dispatch and native short
attention. The Function app was stopped after the run.

- [Environment and app ID](results/modal_zen3_cpu1/environment.json)
- [Raw timings, call counts and bucket decisions](results/modal_zen3_cpu1/t1.json)
- [CPU tests](results/modal_zen3_cpu1/tests.log)

Modal placement does not guarantee a CPU generation. The initial default
Sandbox and a later eight-CPU EU Function reported Intel CPUs, so they were
rejected for this comparison. No eight-thread Zen 3 result is claimed.

## Reproduce, including inside the EPYC container

To launch the verified configuration from an authenticated Modal workspace:

```bash
uv run --no-project --python 3.12 --with modal==1.6.2 python \
  benchmarks/cpu_review/modal_attention_calibration.py \
  --function --region eu --cpu 1 --threads 1 --rounds 5 --require-zen3 \
  --out hardware-results/modal-calibration
```

The launcher records the actual CPU and rejects a different generation before
benchmarking. `--cpu` sets both the request and hard limit; the client must use
Python 3.12 to match the serialized Function image.

Use the container's installed Python with this checkout on its import path.
The benchmark needs PyTorch and gcc; it does not download model weights.

```bash
PYTHONPATH=. python benchmarks/cpu_review/attention_calibration.py \
  --threads 1 --rounds 5 --out calibration-epyc-t1.json
PYTHONPATH=. python benchmarks/cpu_review/attention_calibration.py \
  --threads 8 --rounds 5 --out calibration-epyc-t8.json
```

Use `--isa native` (the default) on the actual EPYC host. To reproduce the local
AVX2 experiment on newer hardware, add `--isa x86-64-v3`. Set CPU affinity before
launching Python; the report records it and the actual CPU model. `baseline`
reproduces the original 240-call sweep, `adaptive` invokes the current production
tuner, and `single` runs the one-call ablation.

Validation: **66 passed** across topology, attention policy, ModernBERT CPU and
short native-attention suites, including full-model parity with AVX2 builds.
