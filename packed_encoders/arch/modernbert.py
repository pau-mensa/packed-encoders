"""ModernBERT as a registered architecture — a thin adapter over the original modules
(`forward`, `graph`, `train_graph`, `validate`), whose behavior is unchanged."""

from __future__ import annotations

from typing import Any

from torch import nn

from packed_encoders.locate import is_modernbert_encoder


class ModernBert:
    name = "modernbert"

    def match(self, module: nn.Module) -> bool:
        return is_modernbert_encoder(module)

    def validate(self, module: nn.Module, **kwargs: Any):
        from packed_encoders.validate import _validate_modernbert

        return _validate_modernbert(module, **kwargs)

    def pack(self, target: object, module: nn.Module, **options: Any):
        from packed_encoders.pack import _pack_modernbert

        return _pack_modernbert(target, module, **options)

    def unpack(self, module: nn.Module) -> None:
        from packed_encoders.pack import _unpack_modernbert

        _unpack_modernbert(module)
