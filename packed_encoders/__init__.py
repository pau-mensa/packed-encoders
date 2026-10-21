"""packed-encoders — fast encoder runtimes composed and installed by engines.

`pack(model)` locates the backbone inside a HF model, a SentenceTransformer, or a PyLate
ColBERT, and installs a validated fast forward in place, so every framework on top
inherits the speedup with no adapter. Curated engines (`packed_encoders.arch`):

- ModernBERT / Ettin / mmBERT: CuteDSL LayerNorm, RoPE and GeGLU, cuBLAS GEMMs, packed
  attention with per-GPU dispatch; CUDA graphs optional (off by default).
- CPU ModernBERT: FP32 packed inference with native attention and row kernels on
  Apple Silicon (Accelerate, plus a prepared SME2 FP32 GEMM; with it, the native kernels run
  on an engine-owned worker pool) and Linux x86-64 (GCC, plus a native FP32 GEMM on AVX-512), else tiled CPU SDPA;
  opt-in int8 projections on Apple SME2 or x86
  AVX512-VNNI (`precision="int8"`); original-forward fallback for training and
  extra HF outputs.
- Qwen3.5 / topk-embed-v1: hybrid FLA/Triton pieces, graphs on by default,
  original-forward fallback for gradients and image inputs.

    import packed_encoders as pe
    pe.pack(model)                   # architecture defaults
    pe.pack(model, cuda_graph=True)  # bucketed CUDA graphs

Only light modules load here. Each architecture's kernel toolchain (CuteDSL for ModernBERT)
loads when a model of that architecture is packed.
"""

from __future__ import annotations

from packed_encoders.dispatch import get_engine, no_cuda_graph, pack, set_cuda_graph, set_train_cuda_graph, unpack, validate
from packed_encoders.batch import PackedBatch
from packed_encoders.errors import (
    PackedEncodersError,
    UnsupportedTargetError,
    ValidationError,
)

# Architecture-specific public names, loaded on first access.
_LAZY = {
    "ModernBertCPU": ("packed_encoders.arch.modernbert_cpu", "ModernBertCPU"),
    "PaddedGraphConfig": ("packed_encoders.runtime.graphs", "PaddedGraphConfig"),
    "Qwen35Report": ("packed_encoders.arch.qwen3_5", "Qwen35Report"),
    "GraphConfig": ("packed_encoders.graph", "GraphConfig"),
    "TrainGraphConfig": ("packed_encoders.train_graph", "TrainGraphConfig"),
    "ValidationReport": ("packed_encoders.validate", "ValidationReport"),
}


def __getattr__(name: str):
    if name == "epilogues":
        import importlib

        return importlib.import_module("packed_encoders.epilogues")
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'packed_encoders' has no attribute {name!r}")


def _keep_public_functions() -> None:
    """`pack` and `validate` are both public functions and submodules. Importing a submodule
    binds it on the package, so the first lazy import of `packed_encoders.pack` would replace
    `pe.pack` the function with the module. Ignore exactly those rebinds; the submodules stay
    importable (`from packed_encoders.pack import ...` reads sys.modules)."""
    import sys
    import types

    class _Package(types.ModuleType):
        def __setattr__(self, name, value):
            if name in ("pack", "validate") and isinstance(value, types.ModuleType):
                return
            super().__setattr__(name, value)

    sys.modules[__name__].__class__ = _Package


_keep_public_functions()


__all__ = [
    "epilogues",
    "ModernBertCPU",
    "PaddedGraphConfig",
    "Qwen35Report",
    "PackedBatch",
    "get_engine",
    "pack",
    "unpack",
    "validate",
    "ValidationReport",
    "GraphConfig",
    "set_cuda_graph",
    "no_cuda_graph",
    "TrainGraphConfig",
    "set_train_cuda_graph",
    "PackedEncodersError",
    "UnsupportedTargetError",
    "ValidationError",
]
