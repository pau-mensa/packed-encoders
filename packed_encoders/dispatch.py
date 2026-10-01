"""The public entry points, dispatched through the architecture registry.

This module (and everything it imports at load time) stays free of any one architecture's
kernel toolchain: ModernBERT's CuteDSL kernels load when a ModernBERT model is packed, fla
when a Qwen3.5 model is. So `import packed_encoders` works in an environment built for
either — e.g. topk-embed's pinned torch 2.11 stack, where nvidia-cutlass-dsl 4.5.2 would
pull a CUDA 13 torch over the cu128 one.
"""

from __future__ import annotations

from typing import Any

from packed_encoders.locate import find_backbone
from packed_encoders.state import get_state


def pack(
    target: object,
    *,
    cuda_graph: Any = None,
    train_cuda_graph: Any = False,
    cuda_graph_seq_cutoff: int = 64,
    attention_backend: str | None = None,
    validate: bool = True,
) -> object:
    """Install the architecture's fast forward onto the backbone inside `target`, in place,
    and return `target`. See `packed_encoders.pack._pack_modernbert` for the ModernBERT
    options and `packed_encoders.arch.qwen3_5` for Qwen3.5. `cuda_graph=None` means the
    architecture's default (off for ModernBERT, on for Qwen3.5)."""
    arch, encoder = find_backbone(target)
    arch.pack(
        target, encoder, cuda_graph=cuda_graph, train_cuda_graph=train_cuda_graph,
        cuda_graph_seq_cutoff=cuda_graph_seq_cutoff, attention_backend=attention_backend,
        validate=validate,
    )
    return target


def unpack(target: object) -> object:
    """Restore the original forward, reverting `pack()`."""
    arch, encoder = find_backbone(target)
    arch.unpack(encoder)
    return target


def validate(target: object, **kwargs: Any):
    """Run the target architecture's hard gate; return its report or raise ValidationError."""
    arch, encoder = find_backbone(target)
    return arch.validate(encoder, **kwargs)


def set_cuda_graph(model: object, enabled: bool, *, config: Any = None) -> None:
    """Turn the inference graph layer on or off after `pack()`."""
    state = get_state(model)
    if hasattr(state, "set_cuda_graph"):          # non-ModernBERT architectures own their runner
        state.set_cuda_graph(enabled, config)
        return
    from packed_encoders.graph import set_cuda_graph as _modernbert_set_cuda_graph

    _modernbert_set_cuda_graph(model, enabled, config=config)


class _NoCudaGraph:
    def __init__(self, model: object):
        self._state = get_state(model)
        self._previous = self._state.graph_enabled

    def __enter__(self):
        self._previous = self._state.graph_enabled
        self._state.graph_enabled = False
        return self

    def __exit__(self, *exc):
        self._state.graph_enabled = self._previous
        return False


def no_cuda_graph(model: object) -> _NoCudaGraph:
    """Context manager that bypasses captured graphs for a one-off odd shape."""
    return _NoCudaGraph(model)
