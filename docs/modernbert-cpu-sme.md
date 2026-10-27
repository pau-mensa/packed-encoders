# ModernBERT FP32 projections on Apple SME2

On Apple Silicon with SME2 (M4 and later), `ModernBertCPU(prepack_weights=True)`
prepares FP32 projection weights for the native SME2 GEMM. It is enabled by default.
ModernBERT-base adds about 421 MiB of prepared copies; the original FP32 parameters
remain available to HF and `state_dict`.

The prepared GEMM runs for 2–128 packed tokens with one thread, up to 1,024 with
2–3 threads and up to 4,096 with four or more. Other sizes use PyTorch/Accelerate.
Biases and unsupported projection shapes also use PyTorch. Normal in-place weight
updates refresh the prepared copies. Structural changes require unpacking and
packing again.

With multiple threads, projections, attention and row kernels share an engine-owned
worker pool. Packing does not change PyTorch or Accelerate's process thread settings.
The native library requires Xcode Command Line Tools with SME2 support and 512-bit
streaming vectors; preparation warns and falls back to Accelerate if unavailable.

To skip the copies:

```python
pe.pack(model, engine=pe.ModernBertCPU(prepack_weights=False))
```

See [CPU runtime](modernbert-cpu.md) for installation and benchmark limitations,
[implementation](../packed_encoders/cpu/sme.py) and [tests](../tests/test_cpu_sme.py).
