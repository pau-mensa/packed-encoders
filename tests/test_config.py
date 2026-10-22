"""ModernBertParams.from_hf_config across the transformers 4 and 5 config schemas; CPU only."""

from types import SimpleNamespace

import pytest
from transformers import AutoConfig, ModernBertConfig

from packed_encoders.config import ModernBertParams


def test_layer_schedule_from_layer_types(tmp_path):
    # Transformers 5 neither stores nor saves global_attn_every_n_layers unless it is passed.
    built = ModernBertConfig(num_hidden_layers=7)
    if not getattr(built, "layer_types", None):
        pytest.skip("transformers 4 configs have no layer_types")
    built.save_pretrained(tmp_path)
    explicit = ModernBertConfig(num_hidden_layers=7, global_attn_every_n_layers=2)
    for config, every in ((built, 3), (AutoConfig.from_pretrained(tmp_path), 3), (explicit, 2)):
        params = ModernBertParams.from_hf_config(config)
        assert [params.is_global_layer(i) for i in range(7)] == [k == "full_attention" for k in config.layer_types]
        assert params.global_attn_every_n_layers == every


def test_transformers_4_schema_and_unsupported_schedules():
    config = ModernBertConfig(num_hidden_layers=4)
    legacy = SimpleNamespace(**{k: getattr(config, k) for k in (
        "model_type", "hidden_size", "num_attention_heads", "num_hidden_layers", "norm_eps", "local_attention")},
        global_rope_theta=160_000.0, local_rope_theta=10_000.0, global_attn_every_n_layers=2)
    assert ModernBertParams.from_hf_config(legacy).global_attn_every_n_layers == 2
    for kinds in (["sliding_attention", "full_attention"] * 2,
                  ["full_attention", "sliding_attention", "full_attention", "full_attention"],
                  ["full_attention", "linear_attention"] * 2):
        config.layer_types = kinds
        with pytest.raises(ValueError, match="layer_types"):
            ModernBertParams.from_hf_config(config)
