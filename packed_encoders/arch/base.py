"""The architecture plugin contract and its registry.

packed-encoders speeds a model up by replacing one module's `forward`, in place, with an
engine that computes the same function faster. *Which* module and *how* are
architecture-specific; graphs, staging, and attention-kernel selection are shared
(`packed_encoders.runtime`). An `Architecture` is the seam between the two:

    match(module)             is this live module one I patch?
    validate(module, ...)     hard gate: return a report or raise ValidationError
    pack(module, **options)   install the fast forward in place
    unpack(module)            restore the original forward

`find_backbone()` walks wrapper objects (SentenceTransformer, PyLate, HF task heads) and
asks every registered architecture at every node, so a new family is one module plus one
`register()` call — `pack()`, `unpack()`, and `validate()` dispatch through here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import nn


@runtime_checkable
class Architecture(Protocol):
    name: str

    def match(self, module: nn.Module) -> bool: ...

    def validate(self, module: nn.Module, **kwargs: Any) -> Any: ...

    def pack(self, target: object, module: nn.Module, **options: Any) -> Any: ...

    def unpack(self, module: nn.Module) -> None: ...


_REGISTRY: list[Architecture] = []


def register(arch: Architecture) -> Architecture:
    if any(a.name == arch.name for a in _REGISTRY):
        raise ValueError(f"architecture {arch.name!r} is already registered")
    _REGISTRY.append(arch)
    return arch


def registered() -> tuple[Architecture, ...]:
    return tuple(_REGISTRY)


def match(module: object) -> Architecture | None:
    if not isinstance(module, nn.Module):
        return None
    for arch in _REGISTRY:
        if arch.match(module):
            return arch
    return None
