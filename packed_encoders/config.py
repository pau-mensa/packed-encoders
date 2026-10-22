"""The slice of ModernBERT config the fused forward consumes.

Built from a HF config once at `pack()` time, decoupling the forward from HF's
schema churn — notably the transformers 5.x move of the RoPE thetas under
`rope_parameters[{full,sliding}_attention]` and of the layer schedule into
`layer_types`, which `from_hf_config` absorbs.
"""

from __future__ import annotations

from dataclasses import dataclass

# config.model_type values whose weight layout and RoPE convention match the
# validated split-half ModernBERT path. mmBERT checkpoints report "modernbert".
SUPPORTED_MODEL_TYPES = frozenset({"modernbert", "mmbert"})


@dataclass(frozen=True)
class ModernBertParams:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    norm_eps: float
    global_rope_theta: float
    local_rope_theta: float
    local_attention: int
    global_attn_every_n_layers: int

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size={self.hidden_size} is not divisible by "
                f"num_attention_heads={self.num_attention_heads}"
            )
        return self.hidden_size // self.num_attention_heads

    @property
    def scaling(self) -> float:
        return self.head_dim ** -0.5

    @property
    def sliding_half_window(self) -> int:
        return self.local_attention // 2

    def is_global_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.global_attn_every_n_layers == 0

    @classmethod
    def from_hf_config(cls, config) -> "ModernBertParams":
        model_type = getattr(config, "model_type", None)
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                "packed-encoders only supports ModernBERT-architecture "
                f"checkpoints {sorted(SUPPORTED_MODEL_TYPES)}; got "
                f"model_type={model_type!r}"
            )
        global_theta, local_theta = _rope_thetas(config)
        return cls(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            norm_eps=config.norm_eps,
            global_rope_theta=global_theta,
            local_rope_theta=local_theta,
            local_attention=config.local_attention,
            global_attn_every_n_layers=_global_attn_every_n_layers(config),
        )


def _global_attn_every_n_layers(config) -> int:
    """The global-attention period across both ModernBERT config schemas. Transformers 5
    runs the layers from `layer_types` and neither stores nor saves
    `global_attn_every_n_layers` unless it was passed explicitly."""
    kinds = getattr(config, "layer_types", None)
    if not kinds:
        return config.global_attn_every_n_layers
    if set(kinds) <= {"full_attention", "sliding_attention"}:
        for every in range(1, len(kinds) + 1):
            if all((kind == "full_attention") == (i % every == 0) for i, kind in enumerate(kinds)):
                return every
    raise ValueError(
        "packed-encoders needs global attention every n layers, starting at layer 0; "
        f"got layer_types={list(kinds)}"
    )


def _rope_thetas(config) -> tuple[float, float]:
    """(global, local) RoPE theta across both ModernBERT config schemas."""
    global_theta = getattr(config, "global_rope_theta", None)
    local_theta = getattr(config, "local_rope_theta", None)
    if global_theta is None or local_theta is None:
        params = getattr(config, "rope_parameters", None) or {}
        full = params.get("full_attention", {}) if isinstance(params, dict) else {}
        slide = params.get("sliding_attention", {}) if isinstance(params, dict) else {}
        fallback = getattr(config, "rope_theta", 160_000.0)
        global_theta = global_theta or full.get("rope_theta") or fallback
        local_theta = local_theta or slide.get("rope_theta") or fallback
    return float(global_theta), float(local_theta)
