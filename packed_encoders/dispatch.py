"""Public lifecycle API, dispatched once at preparation time."""

from __future__ import annotations

from typing import Any

from packed_encoders.engine import OMITTED, Installation
from packed_encoders.errors import PackedEncodersError
from packed_encoders.locate import select_engine
from packed_encoders.state import ATTR, INSTALL_ATTR, find_installation, get_installation


def pack(
    target: object,
    *,
    engine=None,
    cuda_graph: Any = OMITTED,
    train_cuda_graph: Any = OMITTED,
    cuda_graph_seq_cutoff: Any = OMITTED,
    attention_backend: Any = OMITTED,
    precision: Any = OMITTED,
    validate: bool = True,
) -> object:
    """Prepare and install an engine in place; return the original target.

    Omitted options use the engine's defaults. An explicit engine bypasses the
    registry. To switch engines, unpack first; patches are never layered.
    """
    options = {k: v for k, v in (
        ("cuda_graph", cuda_graph), ("train_cuda_graph", train_cuda_graph),
        ("cuda_graph_seq_cutoff", cuda_graph_seq_cutoff),
        ("attention_backend", attention_backend), ("precision", precision),
    ) if v is not OMITTED}
    installed = find_installation(target)
    if installed is not None:
        if engine is not None and engine is not installed.engine:
            raise PackedEncodersError("unpack the model before switching engines")
        installed.packed.configure(options)
        return target

    selected, binding = select_engine(target, engine=engine)
    module = binding.patch_target
    if getattr(module, ATTR, None) is not None:
        raise PackedEncodersError("the target already carries an unowned packed state; unpack it first")
    original = module.forward
    had_forward = "forward" in module.__dict__
    # prepare must unwind its own partial work if it raises (see engine.py).
    prepared = selected.prepare(binding, {**options, "validate": validate})
    installed = Installation(selected, binding, prepared, original, had_forward)
    try:
        binding.adapter.install(binding, prepared)
        setattr(module, INSTALL_ATTR, installed)
    except BaseException:
        installed.restore_forward()
        module.__dict__.pop(INSTALL_ATTR, None)
        module.__dict__.pop(ATTR, None)
        prepared.close(rollback=True)
        raise
    return target


def unpack(target: object) -> object:
    """Restore the recorded forward and release resources, retaining trained weights."""
    installed = find_installation(target)
    if installed is not None:
        installed.packed.close()
        installed.restore_forward()
        module = installed.binding.patch_target
        module.__dict__.pop(ATTR, None)
        module.__dict__.pop(INSTALL_ATTR, None)
    return target


def get_engine(target: object):
    """Return the installed packed engine (provisional extension API)."""
    return get_installation(target).packed


def validate(target: object, *, engine=None, **kwargs: Any):
    """Return the engine-specific report (preserving ModernBERT's public type)."""
    installed = find_installation(target)
    if installed is not None:
        if engine is not None and engine is not installed.engine:
            raise PackedEncodersError("validation must use the installed engine; unpack before switching")
        return installed.packed.validate(**kwargs).details
    selected, binding = select_engine(target, engine=engine)
    return selected.validate(binding, **kwargs).details


def set_cuda_graph(model: object, enabled: bool, *, config: Any = None) -> None:
    get_installation(model).packed.set_cuda_graph(enabled, config)


def set_train_cuda_graph(model: object, enabled: bool, *, config: Any = None) -> None:
    get_installation(model).packed.set_train_cuda_graph(enabled, config)


class _NoCudaGraph:
    def __init__(self, model: object):
        self._engine = get_installation(model).packed

    def __enter__(self):
        self._previous = self._engine.graph_enabled
        self._engine.graph_enabled = False
        return self

    def __exit__(self, *exc):
        self._engine.graph_enabled = self._previous
        return False


def no_cuda_graph(model: object) -> _NoCudaGraph:
    """Temporarily disable inference graphs, restoring the prior setting on exit."""
    return _NoCudaGraph(model)
