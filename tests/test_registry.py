"""Engine selection, installation ownership and failure atomicity, without CUDA."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import packed_encoders as pe
from packed_encoders.arch import base
from packed_encoders.engine import Capabilities, ModelBinding, ValidationResult
from packed_encoders.errors import PackedEncodersError, UnsupportedTargetError
from packed_encoders.locate import find_backbone, find_encoder
from packed_encoders.state import ATTR, INSTALL_ATTR, get_installation


def test_import_does_not_load_toolchains():
    code = """
import sys, packed_encoders as pe
assert callable(pe.set_train_cuda_graph)
assert not {'cutlass', 'triton', 'transformers', 'fla'} & {m.split('.')[0] for m in sys.modules}
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_functions_survive_submodule_imports():
    import packed_encoders.pack
    import packed_encoders.validate
    from packed_encoders import dispatch

    assert pe.pack is dispatch.pack and pe.validate is dispatch.validate


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.auto_model = Backbone()


class Adapter:
    name = "fake-adapter"
    fail_install = False

    def bind(self, module):
        return ModelBinding(self, module, module) if isinstance(module, Backbone) else None

    def install(self, binding, engine):
        binding.patch_target.forward = lambda x: engine.original(x) + 1
        setattr(binding.patch_target, ATTR, engine.state)
        if self.fail_install:
            raise RuntimeError("installation failed")


class Prepared:
    capabilities = Capabilities()
    pieces = ()
    graph_enabled = False

    def __init__(self, binding, options):
        self.binding, self.options = binding, options
        self.original = binding.patch_target.forward
        self.state = SimpleNamespace()
        self.closed = []
        self.configurations = []
        self.trained = False

    def forward_packed(self, batch):
        return batch.input_ids

    def validate(self, **kwargs):
        return ValidationResult("fake", "report")

    def configure(self, options):
        self.configurations.append(options)

    def set_cuda_graph(self, enabled, config=None):
        self.graph_enabled = enabled

    def set_train_cuda_graph(self, enabled, config=None):
        raise PackedEncodersError("training capture unsupported")

    def close(self, *, rollback=False):
        self.closed.append(rollback)


class FakeEngine:
    def __init__(self, name="fake"):
        self.name = name
        self.adapters = (Adapter(),)
        self.prepared = []

    def prepare(self, binding, options):
        packed = Prepared(binding, options)
        self.prepared.append(packed)
        return packed

    def validate(self, binding, **kwargs):
        return ValidationResult(self.name, "report")


@pytest.fixture
def fake(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(base, "_REGISTRY", [*base._REGISTRY, engine])
    monkeypatch.setattr(base, "_DEFAULTS", set(base._DEFAULTS))
    return engine


def test_calls_route_to_owner_after_registry_changes(fake, monkeypatch):
    model = Wrapper()
    assert find_backbone(model) == (fake, model.auto_model)
    assert pe.pack(model) is model
    installed = get_installation(model)
    assert installed.engine is fake
    assert installed.binding.adapter is fake.adapters[0]
    monkeypatch.setattr(base, "_REGISTRY", [])
    assert find_encoder(model) is model.auto_model
    assert pe.validate(model) == "report"
    pe.set_cuda_graph(model, True)
    with pe.no_cuda_graph(model):
        assert not installed.packed.graph_enabled
        with pe.no_cuda_graph(model):
            assert not installed.packed.graph_enabled
        assert not installed.packed.graph_enabled
    assert installed.packed.graph_enabled
    with pytest.raises(PackedEncodersError, match="unsupported"):
        pe.set_train_cuda_graph(model, True)
    assert pe.unpack(model) is model
    assert installed.packed.closed == [False]
    assert pe.unpack(model) is model


def test_ambiguity_does_not_depend_on_registration_order(fake, monkeypatch):
    other = FakeEngine("other")
    wrapper = Wrapper()
    for engines in ([fake, other], [other, fake]):
        monkeypatch.setattr(base, "_REGISTRY", engines)
        with pytest.raises(PackedEncodersError, match="ambiguous"):
            pe.pack(wrapper)
    assert not fake.prepared and not other.prepared


def test_curated_default_and_explicit_unregistered_engine(fake, monkeypatch):
    other = FakeEngine("other")
    monkeypatch.setattr(base, "_REGISTRY", [other, fake])
    monkeypatch.setattr(base, "_DEFAULTS", {fake.name})
    model = Wrapper()
    pe.pack(model)
    assert get_installation(model).engine is fake
    pe.unpack(model)
    external = FakeEngine("external")
    pe.pack(model, engine=external)
    assert get_installation(model).engine is external
    assert external not in base.registered()
    pe.unpack(model)


def test_repeated_pack_and_switching(fake):
    model = Wrapper()
    pe.pack(model)
    pe.pack(model, engine=fake, cuda_graph=True)
    assert len(fake.prepared) == 1
    assert fake.prepared[0].configurations == [{"cuda_graph": True}]
    with pytest.raises(PackedEncodersError, match="unpack"):
        pe.pack(model, engine=FakeEngine())
    pe.unpack(model)
    other = FakeEngine()
    pe.pack(model, engine=other)
    assert get_installation(model).engine is other
    pe.unpack(model)


def test_omitted_options_are_not_forwarded(fake):
    model = Wrapper()
    pe.pack(model)
    assert fake.prepared[-1].options == {"validate": True}
    pe.unpack(model)
    pe.pack(model, cuda_graph=None, train_cuda_graph=False, cuda_graph_seq_cutoff=64)
    assert fake.prepared[-1].options == {
        "validate": True, "cuda_graph": None, "train_cuda_graph": False, "cuda_graph_seq_cutoff": 64,
    }
    pe.unpack(model)


@pytest.mark.parametrize("custom_forward", [False, True])
def test_failed_installation_restores_forward_and_releases_resources(fake, custom_forward):
    model = Wrapper()
    target = model.auto_model
    if custom_forward:
        target.forward = lambda x: x + 3
    original = target.forward
    weight = target.proj.weight
    storage = weight.data_ptr()
    before = weight.detach().clone()
    fake.adapters[0].fail_install = True
    with pytest.raises(RuntimeError, match="installation failed"):
        pe.pack(model)
    assert target.forward == original
    assert ("forward" in target.__dict__) == custom_forward
    assert not hasattr(target, ATTR) and not hasattr(target, INSTALL_ATTR)
    assert fake.prepared[-1].closed == [True]
    assert target.proj.weight is weight and weight.data_ptr() == storage
    torch.testing.assert_close(weight, before)
    fake.adapters[0].fail_install = False
    pe.pack(model)
    pe.unpack(model)
    assert target.forward == original


def test_unpack_preserves_parameter_updates(fake):
    model = Backbone()
    original = model.forward
    weight = model.proj.weight
    pe.pack(model)
    with torch.no_grad():
        weight.add_(2)
    updated = weight.clone()
    pe.unpack(model)
    assert model.forward == original and model.proj.weight is weight
    torch.testing.assert_close(weight, updated)


def test_unsupported_target_names_the_registered_engines():
    with pytest.raises(UnsupportedTargetError, match="modernbert"):
        find_backbone(nn.Linear(2, 2))


def test_batch_retains_metadata_without_conversion():
    ids = torch.tensor([1, 2, 3])
    batch = pe.PackedBatch(ids, host_lengths=(1, 2))
    assert batch.input_ids is ids and batch.cu_seqlens is None
    assert batch.position_ids is None and batch.max_seqlen is None


def test_failed_modernbert_preparation_leaves_no_patch(monkeypatch):
    pytest.importorskip("cutlass")
    import importlib
    pack_module = importlib.import_module("packed_encoders.pack")
    from packed_encoders.arch.modernbert import ModernBert
    from packed_encoders.config import ModernBertParams

    model = Backbone()
    model.config = SimpleNamespace(model_type="modernbert")
    model.embeddings, model.layers, model.final_norm = nn.Identity(), nn.ModuleList(), nn.Identity()
    original = model.forward
    params = ModernBertParams(4, 1, 1, 1e-5, 10000, 10000, 128, 3)
    monkeypatch.setattr(ModernBertParams, "from_hf_config", lambda cfg: params)
    monkeypatch.setattr(pack_module, "_require_cuda", lambda m: None)
    states = []

    def first_graph(module, state, *args):
        state.graph_runner = object()
        states.append(state)

    def fail(*args):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(pack_module, "_enable_graphs", first_graph)
    monkeypatch.setattr(pack_module, "_enable_train_graphs", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        pe.pack(model, engine=ModernBert(device="cuda"), attention_backend="sdpa", validate=False,
                cuda_graph=True, train_cuda_graph=True)
    assert model.forward == original
    assert not hasattr(model, ATTR) and not hasattr(model, INSTALL_ATTR)
    assert states[0].graph_runner is None


def test_explicit_options_rejected_before_preparation(fake):
    pytest.importorskip("cutlass")
    from packed_encoders.arch.modernbert import ModernBert

    model = Backbone()
    model.config = SimpleNamespace(model_type="modernbert")
    model.embeddings, model.layers, model.final_norm = nn.Identity(), nn.ModuleList(), nn.Identity()
    with pytest.raises(PackedEncodersError, match="cuda_graph requires"):
        pe.pack(model, engine=ModernBert(device="cuda"), cuda_graph="yes", validate=False)
    assert not hasattr(model, ATTR)


def test_legacy_pack_entry_preserves_sparse_options(fake):
    from packed_encoders.pack import pack

    model = Backbone()
    pack(model, engine=fake)
    assert fake.prepared[-1].options == {"validate": True}
    pe.unpack(model)


def test_topk_binding_separates_patch_target_from_weights():
    from packed_encoders.arch.qwen3_5 import Qwen35Hybrid
    model = nn.Module()
    model.config = SimpleNamespace(model_type="topk_embed")
    model.head = nn.Linear(4, 4, bias=False)
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.config = SimpleNamespace(model_type="qwen3_5_text")
    engine, binding = base.select(model)
    assert isinstance(engine, Qwen35Hybrid)
    assert binding.patch_target is model
    assert binding.weight_source is model.model.language_model
    assert base.select(model.model.language_model) is None  # not a transformers Qwen3.5 class


def test_hf_binding_matches_stock_qwen35_classes_after_topk():
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    from packed_encoders.arch.qwen3_5 import HFQwen35Adapter, TopkEmbedAdapter
    from packed_encoders.locate import select_engine

    class Stock(Qwen3_5Model):  # models built on Qwen3.5 subclass it
        pass

    text = dict(vocab_size=64, hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                num_key_value_heads=1, head_dim=256, linear_num_key_heads=2, linear_num_value_heads=2,
                linear_key_head_dim=16, linear_value_head_dim=16, layer_types=["linear_attention", "full_attention"])
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                  num_position_embeddings=16)
    model = Stock(Qwen3_5Config(text_config=text, vision_config=vision))
    _, binding = base.select(model)
    assert isinstance(binding.adapter, HFQwen35Adapter)
    assert binding.patch_target is model and binding.weight_source is model.language_model
    _, binding = base.select(model.language_model)
    assert binding.patch_target is binding.weight_source is model.language_model
    # topk's wrapper holds a stock Qwen3_5Model: the walk meets the wrapper first, and topk binds it.
    wrapper = nn.Module()
    wrapper.config = SimpleNamespace(model_type="topk_embed")
    wrapper.head = nn.Linear(64, 4, bias=False)
    wrapper.model = model
    _, binding = select_engine(SimpleNamespace(auto_model=wrapper))
    assert isinstance(binding.adapter, TopkEmbedAdapter) and binding.patch_target is wrapper


def test_hf_rows_take_right_or_left_padding_and_refuse_the_rest():
    from packed_encoders.arch.qwen3_5 import _hf_rows

    state = SimpleNamespace(engine=SimpleNamespace(causal=True, cfg=SimpleNamespace()))
    ids = torch.zeros(3, 5, dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    holes, empty = mask.clone(), mask.clone()
    holes[0, 1] = 0
    empty[1] = 0
    text_only = {"use_cache": False, "return_dict": True, "pixel_values": None, "is_causal": True,
                 "mm_token_type_ids": torch.zeros_like(ids)}
    with torch.no_grad():
        assert _hf_rows(state, ids, mask, (), {}) == ([3, 3, 5], [0, 2, 0])
        assert _hf_rows(state, ids, mask.bool(), (), text_only) == ([3, 3, 5], [0, 2, 0])
        assert _hf_rows(state, ids, None, (), {}) == ([5, 5, 5], [0, 0, 0])
        for m, args, kwargs in [(holes, (), {}), (empty, (), {}), (mask[:, :4], (), {}), (mask, (None,), {}),
                                (mask, (), {"use_cache": True}), (mask, (), {"output_hidden_states": True}),
                                (mask, (), {"position_ids": ids}), (mask, (), {"is_causal": False}),
                                (mask, (), {"pixel_values": torch.zeros(1)})]:
            assert _hf_rows(state, ids, m, args, kwargs) is None, (m, args, kwargs)
    assert _hf_rows(state, ids, mask, (), {}) is None  # grad enabled


@pytest.mark.parametrize("option", ["use_cache", "output_hidden_states", "output_attentions"])
def test_hf_rows_resolve_text_config_defaults(option):
    from packed_encoders.arch.qwen3_5 import _hf_rows

    cfg = SimpleNamespace(use_cache=False, output_hidden_states=False, output_attentions=False)
    state = SimpleNamespace(engine=SimpleNamespace(causal=True, cfg=cfg))
    ids = torch.zeros(1, 5, dtype=torch.long)
    with torch.no_grad():
        assert _hf_rows(state, ids, None, (), {}) == ([5], [0])
        setattr(cfg, option, True)  # read current config, including changes after packing
        assert _hf_rows(state, ids, None, (), {}) is None
        assert _hf_rows(state, ids, None, (), {option: True}) is None
        assert _hf_rows(state, ids, None, (), {option: False}) == ([5], [0])
        expected = None if option == "use_cache" else ([5], [0])
        assert _hf_rows(state, ids, None, (), {option: None}) == expected


@pytest.mark.parametrize("options", [
    {"cuda_graph_seq_cutoff": 64}, {"train_cuda_graph": True},
    {"cuda_graph": "yes"}, {"attention_backend": "unknown"},
])
def test_qwen_rejects_options_before_loading_kernels(options):
    from packed_encoders.arch.qwen3_5 import Qwen35Hybrid
    with pytest.raises(PackedEncodersError):
        Qwen35Hybrid().prepare(None, options)


def test_qwen_validation_ids_stop_before_the_special_tokens():
    from packed_encoders.arch.qwen3_5 import _validation_ids_below
    qwen = SimpleNamespace(vocab_size=248320, eos_token_id=248044, pad_token_id=None, image_token_id=248056)
    assert _validation_ids_below(qwen) == 248044
    assert _validation_ids_below(SimpleNamespace(vocab_size=1024, eos_token_id=[2, 1000], bos_token_id=1)) == 1000
    assert _validation_ids_below(SimpleNamespace(vocab_size=1024, eos_token_id=2, pad_token_id=0)) == 1024
    assert _validation_ids_below(SimpleNamespace(vocab_size=1024, eos_token_id=None)) == 1024
