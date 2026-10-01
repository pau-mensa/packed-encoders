"""The architecture registry: routing and import hygiene. CPU only, no downloads."""

from __future__ import annotations

import subprocess
import sys

import pytest
from torch import nn

import packed_encoders as pe
from packed_encoders.arch import base
from packed_encoders.errors import UnsupportedTargetError
from packed_encoders.locate import find_backbone
from packed_encoders.state import ATTR


def test_import_does_not_load_cutedsl():
    code = "import sys, packed_encoders; print(any(m.split('.')[0] == 'cutlass' for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_public_functions_survive_submodule_imports():
    import packed_encoders.pack  # noqa: F401 — binds the submodule on the package
    import packed_encoders.validate  # noqa: F401
    from packed_encoders import dispatch

    assert pe.pack is dispatch.pack and pe.validate is dispatch.validate


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _Wrapper(nn.Module):             # the shape of a SentenceTransformer / task model
    def __init__(self):
        super().__init__()
        self.auto_model = _Backbone()


class _State:
    graph_enabled = False

    def set_cuda_graph(self, enabled, config=None):
        self.graph_enabled = enabled


class _Fake:
    name = "fake"

    def __init__(self):
        self.calls = []

    def match(self, module):
        return isinstance(module, _Backbone)

    def validate(self, module, **kwargs):
        self.calls.append(("validate", module))
        return "report"

    def pack(self, target, module, **options):
        self.calls.append(("pack", module))
        setattr(module, ATTR, _State())

    def unpack(self, module):
        self.calls.append(("unpack", module))
        delattr(module, ATTR)


@pytest.fixture
def fake(monkeypatch):
    arch = _Fake()
    monkeypatch.setattr(base, "_REGISTRY", [*base._REGISTRY, arch])
    return arch


def test_calls_route_to_the_matching_architecture(fake):
    model = _Wrapper()
    backbone = model.auto_model
    assert find_backbone(model) == (fake, backbone)
    assert pe.pack(model) is model
    assert pe.validate(model) == "report"
    state = getattr(backbone, ATTR)
    pe.set_cuda_graph(model, True)
    assert state.graph_enabled
    with pe.no_cuda_graph(model):
        assert not state.graph_enabled
    assert state.graph_enabled
    pe.unpack(model)
    assert [c for c, m in fake.calls if m is backbone] == ["pack", "validate", "unpack"]


def test_unsupported_target_names_the_registered_architectures():
    with pytest.raises(UnsupportedTargetError, match="modernbert"):
        find_backbone(nn.Linear(2, 2))
