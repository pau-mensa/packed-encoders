# ModernBERT CPU benchmarks

Run these commands from the repository root. The scripts compare
`answerdotai/ModernBERT-base` on CPU using identical inputs for HF SDPA, the packed
FP32/int8 engine, and ONNX Runtime FP32/MatMulNBits W8A8. Model downloads are
required on the first run. Native int8 requires SME2 or AVX512-VNNI; the scripts
fail if an int8 run falls back to FP32.

Install the CPU runtime and benchmark dependencies in your environment:

```bash
uv pip install --no-sources --torch-backend cpu -e . pytest onnxruntime onnx onnxscript
```

Use `.venv/bin/python` below, or your environment's Python. Launchers accept
`PE_PYTHON`. Generated JSON, tables and validation reports belong in the ignored
`benchmarks/results/cpu/` or `hardware-results/` directories. ONNX exports and
reference tensors are local artifacts as well.

## Paired performance comparisons

```bash
.venv/bin/python benchmarks/cpu/onnx_export.py --dir .cache/modernbert-onnx --only int8_w8a8
ORT_DIR=.cache/modernbert-onnx bash benchmarks/cpu/run.sh
```

The Linux launcher uses `taskset`. Set `SHORT_CPUS` (default `0-7`) and `LONG_CPUS`
(default `0-15`) for your host. It measures short queries with 8 threads and
32-row, 1024-token batches with 16 threads, using one warmup and medians of seven
and three samples respectively. Each profile runs in a fresh process. Short W8A8
comparisons keep the faster spinning setting per workload; long runs use spinning.
ORT trims the single padded query and runs long batches one trimmed row at a time.
HF runs padded batches. Results include individual samples, numerical parity,
package versions and affinity. `RESULT_DIR` overrides the output directory.

For a single HF/engine comparison, without ONNX installed:

```bash
.venv/bin/python benchmarks/cpu/compare.py engine --threads 8 --only b1_s32,b8_s32 --out benchmarks/results/cpu/engine.json
```

`bench.py` provides paired HF/engine timing with optional operation profiling;
`onnx_bench.py` provides individual ONNX batch, padded-row and trimmed-row modes.
Both accept `--help`. Create their output directory before running them.

For complete 32-row batches at 8192 tokens, on a host with enough memory:

```bash
ORT_DIR=.cache/modernbert-onnx bash benchmarks/cpu/run_long.sh
```

This defaults to one timed complete batch, no complete-batch warmup, and one ORT
row warmup. Set `LONG_TRIALS` for repeated samples and `LONG_PROFILES` to select
systems. Run `run.sh` first: the summary also requires its short/1024-token files.
These are actual full-batch measurements. Reference tensors are cached under
`.cache/cpu-benchmark-oracles/` to avoid repeated untimed HF forwards.

## Accuracy and packing

```bash
.venv/bin/python benchmarks/cpu/accuracy.py --ort-model .cache/modernbert-onnx/int8_w8a8.onnx --out benchmarks/results/cpu/accuracy_text.json
.venv/bin/python benchmarks/cpu/packing.py --out benchmarks/results/cpu/packing.json
.venv/bin/python benchmarks/cpu/thread_defaults.py --out benchmarks/results/cpu/thread_defaults
```

The accuracy check tokenizes up to 24 paragraphs from `README.md` in memory.
Use `--source /path/to/fixed-text.md` to compare runs on a stable text snapshot.
Results record source and token hashes, token lengths, relative L2 error and
token/pooled cosine statistics. No corpus file is saved. These checks measure
numerical drift; retrieval quality requires a downstream evaluation.

`packing.py` measures disabled, cold and warm native compilation caches in fresh
processes and fails if the warm process recompiles. `thread_defaults.py` compares
default, explicit and physical-core thread settings, plus passive OpenMP.

## Native hardware validation

```bash
.venv/bin/python benchmarks/cpu/hardware_check.py --expect avx2 --out hardware-results/intel.json
.venv/bin/python benchmarks/cpu/hardware_check.py --expect vnni --out hardware-results/vnni.json
.venv/bin/python benchmarks/cpu/hardware_check.py --expect sme2 --out hardware-results/m4.json
```

Choose the command matching your hardware. The checker requires that native ISA
before running the CPU suites, and writes JSON and JUnit XML. An AVX2-target build
on AVX-512 hardware does not qualify as AVX2-only hardware validation.

On an SME2 Mac, `bash benchmarks/cpu/run_m4.sh` combines validation, cold/warm
packing and HF/engine timings at 1 and 8 threads. It needs native arm64 Python
and Xcode Command Line Tools, and does not require ONNX. Outputs default to
`hardware-results/m4/`.

See the [CPU runtime guide](../../docs/modernbert-cpu.md) for supported shapes,
fallback behavior, compiler requirements and thread configuration.
