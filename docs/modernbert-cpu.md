# ModernBERT CPU runtime (experimental)

`pe.pack()` selects the CPU engine automatically for FP32 CPU ModernBERT models.
It packs live tokens through the full encoder, so padding costs nothing, and uses
native kernels where the platform supports them:

| | Apple Silicon (arm64 macOS) | Linux x86-64 | Other CPUs |
|---|---|---|---|
| Built with | Xcode Command Line Tools (`xcrun clang`) | `gcc` with OpenMP | — |
| Local and short global attention | native (Accelerate) | pack-time choice of native and SDPA (AVX-512 or AVX2) | tiled PyTorch SDPA |
| RoPE + QKV layout, GeGLU, residual add + LayerNorm | native | native | PyTorch |
| FP32 projections | PyTorch (Accelerate) | native GEMM (AVX-512), else PyTorch | PyTorch |
| `precision="int8"` | M4 and later (SME2) | AVX512-VNNI | warns, runs FP32 |

Native attention requires 64-dimensional heads and a 128-token local window
(ModernBERT base and large qualify). Native global attention supports sequences of
at most 128 tokens; longer inputs use per-sequence global SDPA. On Linux x86-64,
`attention_backend="auto"` times native attention and SDPA at packing, separately
for local/global attention and single/batched inputs at 32, 128 and 512 tokens
(bounded by the model and token budget). It uses the nearest measured shape at
inference, without timing requests. Native must win by at least 5%; otherwise SDPA
is selected. The native row and projection kernels remain available in either case.
This is a bounded heuristic based on longest sequence and sequence count, rather
than a guarantee of the fastest kernel for every ragged length distribution.
Apple retains its native local and short-global policy. Explicit `"native"` and
`"sdpa"` bypass attention tuning. Repack after changing affinity or PyTorch's threads.

The native GeGLU uses a rational erf that is FP32-accurate but not bitwise PyTorch's. Each
residual add and the LayerNorm after it run as one native pass. The residual stream
stays bitwise PyTorch's, while the normalized values are FP32-accurate but not
bitwise. Numerical checks are in [the CPU tests](../tests/test_modernbert_cpu.py).
RoPE stays bitwise PyTorch's on both platforms.

## Installation and use

From this checkout:

```bash
uv pip install --no-sources --torch-backend cpu -e .
```

No Triton, CuteDSL, ONNX or FlashAttention is required. The first pack compiles the
bundled C sources; fresh processes reuse a private on-disk build cache. Cache keys
include bundled source/header contents, build commands, compiler identity and target ISA
macros, OS, relevant compiler environment and the linked OpenMP runtime contents.
Concurrent processes serialize each build; only complete, checksummed libraries
are published. The cache does not retain model weights or attention timing choices.

The default location is `$XDG_CACHE_HOME/packed-encoders/cpu` (or
`~/.cache/packed-encoders/cpu`) on Linux and
`~/Library/Caches/packed-encoders/cpu` on macOS. Set
`PACKED_ENCODERS_CPU_CACHE_DIR` to another private directory, or to an empty string
to disable disk reuse. Unavailable caches and directories accessible by other
users fall back to process-private temporary builds. Cache directories can be
removed when no processes are packing models; they are rebuilt on demand.
On Linux the native library links the OpenMP runtime PyTorch already loaded, so the
engine and PyTorch share one thread pool. If compilation is unavailable, automatic
selection warns and uses SDPA and PyTorch ops. Explicit `native` selection requires
successful compilation.

```python
import torch
from transformers import AutoModel
import packed_encoders as pe

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base").float().cpu().eval()
pe.pack(model)  # includes a small numerical check against the original HF forward
with torch.inference_mode():
    output = model(input_ids=ids, attention_mask=mask).last_hidden_state
pe.unpack(model)
```

Configure the native workers and the live-token budget explicitly:

```python
pe.pack(model, engine=pe.ModernBertCPU(threads=8, max_tokens=32768),
        attention_backend="native")  # also accepts "auto" and "sdpa"
```

- `threads` sets the native kernels' workers: attention, the row kernels, the x86
  FP32 GEMM and the int8 projections. It defaults to `torch.get_num_threads()` at
  packing, capped to the current Linux affinity mask and to its physical cores
  when the OS exposes topology. Explicit `threads=` overrides that cap. Other matrix
  operations keep your PyTorch/Accelerate thread settings;
  packing never changes process-global threading.
- The engine divides batches at sequence boundaries to respect `max_tokens`. A single
  sequence must fit both this budget and the model's position limit.
- `attention_backend="sdpa"` is the pure-PyTorch reference on every platform.

### x86 options

- **Pin your threads.** On CPUs with several core complexes (e.g. AMD's two-CCD
  parts), cross-complex synchronization is expensive: keep a process on one CCD
  (`taskset -c 0-7`) unless the batch is large. Configure affinity before packing.
- `ModernBertCPU(rehome_weights=True)` moves the projection weights out of
  `from_pretrained`'s safetensors file mapping into fresh memory (same Parameter
  objects and values). FP32 then reads them from huge pages, which saved 3-7% at
  one thread on ModernBERT-base (Ryzen 9 9950X). On Linux the file-backed pages are evicted, so
  resident memory stays flat; on other platforms this only copies. References to the
  old storage taken before packing stop aliasing the model.
- The native FP32 GEMM picks its register tile from the detected L1 data cache size.
  `ModernBertCPU(tune_gemm=True)` additionally times candidate tiles on the model's
  weights at 16, 32 and 256 tokens during packing. Repack after changing CPU affinity.
- The native GEMM keeps one accumulation order per output, so results do not depend
  on batch composition, tile position or thread count. It reads the original weights
  in place: no reordered copy.
- On AVX2-only CPUs, projections use `F.linear`. Automatic attention selection
  measures native attention against SDPA at packing. Oversubscribing native
  OpenMP teams can substantially slow short requests. Set PyTorch's threads
  before packing and avoid exceeding the OS-reported physical cores.
  `OMP_WAIT_POLICY=PASSIVE` alone does not recover the best latency. The
  [CPU benchmark guide](../benchmarks/cpu/README.md) includes a thread comparison.

## int8 precision

```python
pe.pack(model, precision="int8")  # default "fp32": the exact path above
```

`precision="int8"` runs the four projections of every layer as int8 matrix products:
SME2 outer products on Apple M4-class CPUs, u8 x s8 `vpdpbusd` dot products on x86
CPUs with AVX512-VNNI. Attention, RoPE, GeGLU and the norms stay FP32. Each projection
input is rotated (a block Hadamard transform with random signs, carried by the weights
too) and quantized per row; weights are quantized per output channel with GPTQ. There
are no other settings. `threads` also sets the int8 workers.

**ModernBERT-base on an M4 Pro**, 15 alternating samples (median ms):

| Workload | FP32, 8 threads | int8, 8 threads | FP32, 1 thread | int8, 1 thread |
|---|---:|---:|---:|---:|
| 1 query, 16 tokens | 13.42 | 4.06 (−70%) | 12.90 | 3.70 (−71%) |
| 1 query, 32 tokens | 14.84 | 6.83 (−54%) | 15.30 | 6.39 (−58%) |
| 32 queries, up to 32 tokens | 52.07 | 23.55 (−55%) | 103.15 | 69.22 (−33%) |
| 32 queries, up to 128 tokens | 140.11 | 67.17 (−52%) | 303.06 | 222.22 (−27%) |
| 8 documents, up to 512 tokens | 169.12 | 93.36 (−45%) | 354.71 | 274.89 (−23%) |

Long sequences gain less, because global attention (quadratic, FP32 in both modes)
dominates: for one 8,192-token sequence it is 74% of the FP32 time at one thread and
57% at eight. Full-length sequences, no padding (3 samples; 2 for the 1-thread B8 case):

| Workload | FP32, 8 threads | int8, 8 threads | FP32, 1 thread | int8, 1 thread |
|---|---:|---:|---:|---:|
| 1 × 8,192 tokens | 1.64 s | 1.27 s (−23%) | 6.32 s | 5.99 s (−5%) |
| 8 × 8,192 tokens | 13.86 s | 10.51 s (−24%) | 51.93 s | 49.59 s (−4.5%) |

**ModernBERT-base on a Ryzen 9 9950X** (Zen 5), median ms of 3 interleaved fresh-process
rounds; 1 thread pinned to one core, 8 threads pinned to one CCD:

| Workload | FP32, 8 threads | int8, 8 threads | FP32, 1 thread | int8, 1 thread |
|---|---:|---:|---:|---:|
| 1 query, 16 tokens (padded to 32) | 8.55 | 1.94 (−77%) | 17.46 | 4.09 (−77%) |
| 1 query, 32 tokens | 8.88 | 2.26 (−75%) | 27.97 | 7.36 (−74%) |
| 8 queries, 32 tokens | 29.05 | 8.25 (−72%) | 192.75 | 52.31 (−73%) |

For reference, ONNX Runtime (optimized graph, best thread setting) takes 11.1 / 11.1 /
37.2 ms in FP32 and 5.1 / 5.1 / 19.0 ms in dynamic int8 at 8 threads, and 30.3 / 30.2 /
191 ms and 10.7 / 10.7 / 67.1 ms at one thread. Large batches at 16 threads across both
CCDs (one process each, 5 samples at 1,024 tokens and 2 at 8,192):

| Workload | FP32 | int8 |
|---|---:|---:|
| 32 × 1,024 tokens | 8.04 s | 2.22 s (−72%) |
| 32 × up to 1,024 tokens (random lengths) | 3.57 s | 1.05 s (−71%) |
| 32 × 8,192 tokens | 97.4 s | 50.6 s (−48%) |
| 32 × up to 8,192 tokens (random lengths) | 37.1 s | 19.0 s (−49%) |

Historical measurements are recorded in [PR #15](https://github.com/pau-mensa/packed-encoders/pull/15).
The current harness is documented in the [CPU benchmark guide](../benchmarks/cpu/README.md).

These tables describe the historical M4/Zen 5 runs before automatic attention tuning.
The ONNX int8 figures above use `quantize_dynamic`, not MatMulNBits W8A8. They are
not a comparison against the best available ORT int8 configuration. Large-batch
comparisons should include HF SDPA and trim each ONNX row when running rows
individually. On Joe's AVX2 system, full-row speedups were 1.18–1.59x against HF
versus 2.87–8.08x against ONNX; trimming the random-length ONNX rows reduced that
speedup from 5.26x to 2.07x. See [PR #15](https://github.com/pau-mensa/packed-encoders/pull/15) for the benchmark discussion.

int8 is approximate. Against FP32 on the same inputs, relative L2 error is 3e-2 to
8e-2, the median token cosine is about 0.9995 and the mean pooled cosine at least
0.9995, on both platforms. ONNX Runtime's dynamic int8 reaches only relL2 0.55 and a
median token cosine of 0.87 on the same text. This does not generalize to ORT's
weight-only quantizer: Joe measured real-text relL2 0.0466 for MatMulNBits W8 and
0.0671 for W8A8, with median token cosine 0.99965 and 0.99904 respectively. W8A8
is the appropriate int8 baseline for future engine comparisons on VNNI/SME2
hardware. The corpora differ, so these are not paired accuracy measurements.
Some punctuation-like tokens have much larger errors than the median. Check
retrieval quality on your own model before switching.

- Packing calibrates GPTQ on FP32 forward passes over random token IDs, so it needs no
  data and produces the same weights every time. It takes about 6 s at 8 threads and
  10 s at 1 thread on the M4 Pro, and about 9 s at 8 threads and 33 s at 1 thread on
  a Ryzen 9 9950X (FP32 packing takes about 2 s there).
- Prepared int8 weights add about 110 MB; the FP32 parameters stay, because training,
  extra outputs and `pe.unpack()` use them.
- Without SME2 or AVX512-VNNI (Apple M1–M3, AVX2-only or AVX-VNNI-only x86, other
  CPUs) or without a compiler that targets them, packing warns and uses exact FP32;
  `pe.get_engine(model).state.precision` reports which precision runs. Unsupported
  model shapes still raise.
- Training, gradient-enabled calls and extra HF outputs run the original FP32 forward.
- In-place updates of projection weights raise on the next call; unpack and pack
  again. Updates through `.data` bypass PyTorch's version counters and are not detected.
- Changing `precision` requires unpacking first.
- Pack validation bounds the median token cosine (at least 0.99) and the relative L2
  error (at most 0.25) instead of the FP32 bounds.
- Requires 64-dimensional heads, hidden and intermediate sizes divisible by 64,
  bias-free projections and plain LayerNorms (ModernBERT base and large qualify).

## Packed inputs

For already packed inputs:

```python
batch = pe.PackedBatch(flat_ids, host_lengths=(7, 31, 64))
with torch.inference_mode():
    hidden = pe.get_engine(model).forward_packed(batch)  # [102, hidden_size]
```

Alternatively supply CPU int32 `cu_seqlens`. IDs and optional positions must be CPU
int64 tensors. Omitted positions restart at zero for each sequence. Supplying both
length representations checks that they agree.

## Behavior and scope

The optimized path requires evaluation mode with gradients and CPU autocast off.
Training, gradient-enabled calls, extra HF outputs, explicit HF position arguments,
and masks with empty rows or internal holes use the original HF forward. Ordinary
left/right padding is supported; padded output positions are zero. Direct packed
calls require the inference conditions and reject invalid metadata.

Weights retain their original storage and state-dict keys (unless `rehome_weights`
moved them at packing). In FP32, in-place weight and norm updates are read on the
next call. The model's structure is resolved once at packing and not re-checked per
call: replacing parameters or submodules, assigning `.data`, changing device/dtype,
adding hooks on internal modules or editing the config requires unpacking and
repacking; internal custom forward hooks present at packing are unsupported. Until then the engine keeps computing with the tensors it resolved at
packing, so freed or swapped storage never reaches native code. Hooks on the
top-level model still run. Unmerged adapter projections are unsupported. The engine
supports default split-half RoPE and exact GeGLU. CUDA graphs are not implemented.
With `precision="int8"`, in-place projection weight updates require a repack (see
above).

Calls on one engine serialize access to its reusable scratch. Returned model outputs
own their storage and survive subsequent calls. `pe.unpack()` releases the scratch
and restores the original forward.

See the [CPU benchmark guide](../benchmarks/cpu/README.md) for paired HF,
trimmed ONNX and MatMulNBits comparisons and native hardware checks. AVX-512,
VNNI and SME2 tests skip on runners without those instructions; portable CI alone
does not validate those kernels.
