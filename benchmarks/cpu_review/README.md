# PR #15 review follow-up

This follows [Joe's review](https://github.com/pau-mensa/packed-encoders/pull/15#issuecomment-6332147296)
and [his correction of the int8 baseline](https://github.com/pau-mensa/packed-encoders/pull/15#issuecomment-6340277903).
The task-head, float-mask, int32-ID, validation and Transformers config fixes are
already in `699873d` and `fe9c5df`; Joe's AVX2 measurements are in `7e4eef0`.

## Runtime changes

- Linux x86 automatic attention selection now measures native attention and SDPA
  at packing. Local/global and single/batched cases are measured separately, at
  bounded 32/128/512-token shapes. Native must win by 5%; otherwise SDPA runs.
  Requests use the nearest measured shape without timing or extending a cache.
  Native row and projection kernels remain enabled. Explicit backends and Apple's
  existing policy are preserved. The int8 prepared-attention path honors the same
  decisions, including mixed native/SDPA choices.
- Default native workers are capped to physical cores within Linux CPU affinity,
  when topology is available. Explicit `threads=` remains an override. This does
  not resize PyTorch/BLAS pools or set `OMP_WAIT_POLICY`; applications should
  configure those before packing. Joe's WSL2 topology reports 16 cores, so this
  uses 16 rather than assuming the CPU's advertised hybrid-core count of 24.
- Head GEMM tuning has its own tile dictionary. Equal backbone/head shapes no
  longer overwrite the backbone's model-order tuning result.
- CPU projections produce owned outputs directly where possible; PyTorch outputs
  no longer receive a redundant copy. The built-in token projection avoids a second
  epilogue output clone. Custom heads retain defensive output copies, and native
  SME scratch is still copied before reuse.
- The local hardware checker runs every CPU suite. ISA-specific tests still skip
  when a host lacks AVX-512/VNNI/SME2; passing portable tests is not evidence for
  those paths.
- Native builds are reused across fresh processes through a private, checksummed
  cache keyed by sources, compiler/ISA, flags, OS and OpenMP runtime. Compilation
  is locked across processes and published atomically. See the [cache controls](../../docs/modernbert-cpu.md#installation-and-use).

The [CPU runtime](../../docs/modernbert-cpu.md),
[SME projection guide](../../docs/modernbert-cpu-sme.md) and
[epilogue guide](../../docs/epilogues.md) supply the missing user-facing docs.
Historical x86 scripts and raw results are under [cpu_x86](../cpu_x86/README.md).

## Benchmark interpretation

The original PR's large-batch ratios are relative to ONNX, which computes dense
attention at every layer in this export. HF SDPA is a stronger long-input baseline.
When running ONNX one row at a time, trim each row to its live length. Joe's
[AVX2 tables](../cpu_avx2/results/summary.md) show both effects explicitly: at 16
threads, full-row engine speedups are 1.18–1.59x over HF versus 2.87–8.08x over ONNX.
Trimming the random-length ONNX rows reduces the reported 5.26x to 2.07x.

`quantize_dynamic` does not represent all ORT int8 methods. Joe's MatMulNBits W8
and W8A8 real-text relL2 values are 0.0466 and 0.0671, respectively. Compare the
engine's rotated GPTQ int8 against **MatMulNBits W8A8**, not just the inaccurate
dynamic-int8 variants. The ONNX timing harness now defaults to this baseline.

## Reproduce the follow-up measurements

The [results table](results/summary.md) is generated from the raw JSON files by
`summarize.py`; `run.sh` regenerates it after all profiles finish.

`compare.py` uses the tracked `cpu_avx2/bench.py` workloads, including its exact
random-length mask. Each system gets the same IDs and masks. HF is stock SDPA;
engine precision is checked so an int8-to-FP32 fallback cannot be labeled int8.
ORT runs full short batches, except for trimming the single padded query. Long
batches run one trimmed row per call. Both latency and parity against HF FP32 are
recorded, along with every timing sample, package versions and CPU affinity.

Install ONNX dependencies alongside the existing Torch/Transformers environment,
then export FP32 and MatMulNBits W8A8:

```bash
python benchmarks/cpu_avx2/onnx_export.py --dir /tmp/modernbert-review-onnx --only int8_w8a8
ORT_DIR=/tmp/modernbert-review-onnx bash benchmarks/cpu_review/run.sh
```

`PE_PYTHON` selects the interpreter. `SHORT_CPUS` and `LONG_CPUS` select affinity;
the defaults, `0-7` and `0-15`, match the Ryzen 9 9950X used here. Set them for your
machine. This is Linux-specific because it uses `taskset`. The suite measures 8-thread
short queries and 16-thread 32×1024 batches, with one warmup then median of seven
and three samples respectively. Each profile runs in a fresh process. W8A8 short
queries run with ORT spinning on and off. Long ORT runs use spinning on.

The measurements in `results/` use the original PR's cached optimized FP32 ONNX
export under `.cache/onnx/modernbert-base/` and a new W8A8 model derived from that
same graph using `cpu_avx2/onnx_export.py::weight_only`. MatMulNBits uses symmetric
8-bit weights, 128-element blocks and `accuracy_level=4`. A fresh export with a
different Transformers version or optimizer can change performance. The local
versions are recorded in each JSON file. These are follow-up measurements in
one round of fresh processes, rather than a rerun of the original three-round
publication protocol. Random-ID parity measures numerical drift, not retrieval
quality; it does not replace paired real-text or downstream evaluations.

## Remaining work

- Repeat the automatic policy measurements on Joe's actual AVX2-only Intel system.
  Executing an AVX2 compiler target on this Ryzen verifies that code path, but does
  not reproduce Intel's cache, scheduler or hybrid cores.
- Recheck on an SME2 Mac. This Linux machine cannot validate native Apple kernels.
- This repository has no configured native self-hosted runners. The local checker
  below runs without registering one.

## Native hardware handoff

Run from a checkout containing this follow-up. Install on each machine, then run
the command for its actual ISA:

```bash
uv venv --python 3.12
uv pip install --no-sources --torch-backend cpu -e . pytest
# Joe's AVX2-only Intel host:
.venv/bin/python benchmarks/cpu_review/hardware_check.py --expect avx2 --out hardware-results/intel.json
# Your M4 Mac (Xcode Command Line Tools with SME2 support):
.venv/bin/python benchmarks/cpu_review/hardware_check.py --expect sme2 --out hardware-results/m4.json
# On the M4, also verify real cold/warm packing for FP32 and SME2 int8:
.venv/bin/python benchmarks/cpu_review/packing.py --out hardware-results/m4-packing.json
```

The checker requires the expected native libraries before running tests. It writes
JSON with every passed/skipped test and JUnit XML. Return both artifacts. Neither
an AVX2-target build on this Ryzen nor a skipped SME2 suite constitutes those runs.

For a complete M4 run, use the launcher below instead of the two individual M4
commands above. It runs the SME2 tests, cold/warm packing measurements, and paired
HF SDPA / engine FP32 / engine int8 timings at 1 and 8 threads. The workloads are
the padded 16-token query, 1 × 32, 8 × 32 and 1 × 1024, with one warmup and seven
timed samples. Every profile runs in a fresh process. No ONNX installation is
required. Use native arm64 Python and Xcode Command Line Tools with SME2 support.

```bash
# Match the Transformers version used for the recorded Linux measurements:
uv pip install 'transformers==5.3.0'
bash benchmarks/cpu_review/run_m4.sh
tar -czf hardware-results/m4.tar.gz -C hardware-results m4
```

The launcher stops on test failures or an int8 fallback. `PE_PYTHON` selects the
interpreter and `RESULT_DIR` changes the output directory (default
`hardware-results/m4`). The JSON files retain timings and numerical parity against
HF. Return the archive, including the hardware check's JUnit XML.

For Intel performance, also run the thread sweep and attention ablation:

```bash
.venv/bin/python benchmarks/cpu_review/thread_defaults.py --out hardware-results/intel-threads
for mode in auto native sdpa_attn; do
  .venv/bin/python benchmarks/cpu_avx2/attention_ablation.py --threads 1 --mode "$mode" \
    --tag "$mode" --only b1_s32,b8_s32,b1_s1024 --out "hardware-results/intel-${mode}-t1.json"
  .venv/bin/python benchmarks/cpu_avx2/attention_ablation.py --threads 8 --mode "$mode" \
    --tag "$mode" --only b1_s32,b8_s32,b1_s1024 --out "hardware-results/intel-${mode}-t8.json"
done
```

## Additional checks

`thread_defaults.py` compares uncapped explicit workers, the capped default, fully
physical-core PyTorch/native settings, and passive OpenMP in fresh processes.
The [local results](results/thread_defaults/summary.md) retain every timing sample.
The cap improves oversubscribed native work; it does not resize PyTorch pools.

`packing.py` measures disabled, cold and warm disk caches in fresh processes for
FP32 and int8, and fails if a warm process invokes the compiler. Results are in
[`packing_cache.json`](results/packing_cache.json).

`accuracy.py` runs engine FP32, GPTQ int8 and ORT W8A8 against the same HF FP32
outputs on the saved [`corpus.json`](corpus.json). It records the corpus hash, exact
token lengths, token cosine percentiles and mean-pooled cosine. Reproduce with:

```bash
.venv/bin/python benchmarks/cpu_review/accuracy.py --ort-model "$ORT_DIR/int8_w8a8.onnx" \
  --out results/accuracy_text.json
```

The [paired real-text results](results/accuracy_text.json) assess numerical drift;
they do not establish retrieval-task quality. On 24 paragraphs / 3,466 tokens,
engine FP32 has relL2 3.86e-6; GPTQ int8 has 0.0497 and W8A8 has 0.0691. Their
minimum pooled cosines are 0.999592 and 0.999388 respectively. GPTQ has 3 tokens
below 0.5 cosine and W8A8 has 7, so neither method eliminates low-cosine outliers.

To measure full 8192-token batches on a machine with enough memory:

```bash
ORT_DIR=/path/to/onnx/models bash benchmarks/cpu_review/run_long.sh
```

The default is one timed complete batch per workload/system, with one ORT row
warmed first and no complete-batch warmup. Increase `LONG_TRIALS` for repeated
samples. `LONG_PROFILES="ort_fp32 ort_w8a8"` can rerun selected systems without
repeating the others. The full and random-length inputs use a separate fixed seed (8192), so
these are a new paired comparison, not a rerun of the historical random corpus.
HF runs the padded batch; ONNX runs all 32 trimmed rows. Oracle tensors for long
workloads can be reused from `.cache/cpu-review-oracles/`, keyed by model config
(including checkpoint revision), versions, thread count and exact inputs/masks.
This cache only avoids repeating untimed reference forwards. The completed [results table](results/summary.md#8192-token-batches-seconds)
contains actual measurements for all five systems on both workloads. The long-run
wrapper was edited during execution and its final summary invocation failed after
all measurements had been saved; the summary was regenerated separately. Both
launchers then passed fresh-shell argument checks, including result paths with spaces.

## Validation on this machine

- CPU runtime, all CPU kernel suites, shared epilogues, registry and config:
  **176 passed, 8 skipped**. Skips are Apple-specific on this Linux host. This
  includes executing an AVX2-compiled attention library, AVX-512 GEMM, VNNI int8,
  mixed attention dispatch, independent head tuning and output lifetime checks.
- CUDA epilogues on RTX 5090: **6 passed**, including actual CUDA graph replay
  after the built-in head's final output copy was removed.
- Python parsing, benchmark shell syntax, local guide links and `git diff --check`
  passed. Model hashes and benchmark setup are recorded in
  [metadata.json](results/metadata.json).

The hardware-gated CPU command was:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python benchmarks/cpu_review/hardware_check.py \
  --expect vnni --out benchmarks/cpu_review/results/hardware_vnni.json
```

CUDA check: `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q tests/test_cuda_epilogues.py`.
