"""packed-encoders — fast, monkeypatching encoder runtimes, one plugin per architecture.

`pack(model)` locates the backbone inside a HF model, a SentenceTransformer, or a PyLate
ColBERT, and installs a validated fast forward in place, so every framework on top
inherits the speedup with no adapter. Architectures (`packed_encoders.arch`):

- ModernBERT / Ettin / mmBERT: CuteDSL LayerNorm, RoPE and GeGLU, cuBLAS GEMMs, packed
  attention with per-GPU dispatch; CUDA graphs optional (off by default).
- Qwen3.5 hybrid (topk-embed-v1): merged GEMMs, fla GatedDeltaNet with fused gates, a fused
  q/k-norm + RoPE kernel, probed varlen attention; CUDA graphs on by default.

    import packed_encoders as pe
    pe.pack(model)                   # architecture defaults
    pe.pack(model, cuda_graph=True)  # bucketed CUDA graphs

Only light modules load here. Each architecture's kernel toolchain (CuteDSL for ModernBERT,
fla for Qwen3.5) loads when a model of that architecture is packed.
"""

from __future__ import annotations

from packed_encoders.dispatch import no_cuda_graph, pack, set_cuda_graph, unpack, validate
from packed_encoders.errors import (
    PackedEncodersError,
    UnsupportedTargetError,
    ValidationError,
)

# Architecture-specific public names, loaded on first access (ModernBERT's import CuteDSL).
_LAZY = {
    "GraphConfig": ("packed_encoders.graph", "GraphConfig"),
    "TrainGraphConfig": ("packed_encoders.train_graph", "TrainGraphConfig"),
    "set_train_cuda_graph": ("packed_encoders.train_graph", "set_train_cuda_graph"),
    "ValidationReport": ("packed_encoders.validate", "ValidationReport"),
    "PaddedGraphConfig": ("packed_encoders.runtime.graphs", "PaddedGraphConfig"),
}


def __getattr__(name: str):
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
    "pack",
    "unpack",
    "validate",
    "ValidationReport",
    "GraphConfig",
    "PaddedGraphConfig",
    "set_cuda_graph",
    "no_cuda_graph",
    "TrainGraphConfig",
    "set_train_cuda_graph",
    "PackedEncodersError",
    "UnsupportedTargetError",
    "ValidationError",
]
