"""Find the backbone inside whatever the caller hands `pack()`.

HF `AutoModel`, SentenceTransformers, and PyLate ColBERT all ultimately hold one backbone
module that an engine adapts (`packed_encoders.arch`). This walks the
known wrapper shapes, checking engines at each node, and raises on an unknown
container rather than guessing — a wrong guess would patch the wrong weights.
"""

from __future__ import annotations

from torch import nn

from packed_encoders.config import SUPPORTED_MODEL_TYPES
from packed_encoders.errors import UnsupportedTargetError

# Attribute chains a wrapper uses to reach the backbone, most-specific first.
# `auto_model`: SentenceTransformers Transformer / PyLate. `model`: HF task heads
# (ModernBertForMaskedLM, ...). `modernbert`/`bert`: occasional custom heads.
# `language_model`: the text backbone of a multimodal wrapper, when it is the patch target.
_WRAPPER_ATTRS = ("auto_model", "model", "modernbert", "bert", "encoder", "backbone", "language_model")


def is_modernbert_encoder(module: object) -> bool:
    """True for a `ModernBertModel`-shaped backbone: a supported `model_type` and
    the embeddings / layers / final_norm trio the forward reads."""
    if not isinstance(module, nn.Module):
        return False
    config = getattr(module, "config", None)
    if getattr(config, "model_type", None) not in SUPPORTED_MODEL_TYPES:
        return False
    return (
        hasattr(module, "embeddings")
        and hasattr(module, "layers")
        and hasattr(module, "final_norm")
    )


def walk_targets(target: object, *, _depth: int = 0):
    """Known wrapper paths, most-specific first; no architecture matching."""
    seen = set()

    def walk(node, depth):
        if id(node) in seen:
            return
        seen.add(id(node))
        yield node
        if depth >= 4:
            return
        first = _first_submodule(node)
        if first is not None:
            yield from walk(first, depth + 1)
        for attr in _WRAPPER_ATTRS:
            sub = getattr(node, attr, None)
            if isinstance(sub, nn.Module):
                yield from walk(sub, depth + 1)

    yield from walk(target, _depth)


def find_encoder(target: object, *, _depth: int = 0) -> nn.Module:
    """Prefer the recorded patch target, even if selection rules have changed."""
    from packed_encoders.state import find_installation

    installed = find_installation(target)
    if installed is not None:
        return installed.binding.patch_target
    return find_backbone(target, _depth=_depth)[1]


def select_engine(target: object, *, engine=None, _depth: int = 0):
    from packed_encoders.arch.base import select

    for module in walk_targets(target, _depth=_depth):
        result = select(module, engine=engine)
        if result is not None:
            return result
    raise UnsupportedTargetError(_describe(target))


def find_backbone(target: object, *, _depth: int = 0):
    """Compatibility helper returning (engine, patch target)."""
    engine, binding = select_engine(target, _depth=_depth)
    return engine, binding.patch_target


def _first_submodule(target: object):
    """`target[0]` for indexable containers (ST `SentenceTransformer` is one),
    without assuming the framework is installed."""
    if not hasattr(target, "__getitem__"):
        return None
    try:
        return target[0]
    except (KeyError, IndexError, TypeError):
        return None


def _describe(target: object) -> str:
    from packed_encoders.arch import registered

    names = ", ".join(a.name for a in registered())
    return (
        f"could not locate a supported backbone in {type(target).__name__!r} "
        f"(registered engines: {names}). packed-encoders patches a Hugging Face "
        "backbone, a SentenceTransformer / PyLate model wrapping one, or a task model "
        "exposing it via .auto_model / .model. Pass the backbone directly if it lives "
        "somewhere else."
    )
