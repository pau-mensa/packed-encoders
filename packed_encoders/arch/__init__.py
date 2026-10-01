"""Registered architectures. Order matters only if two plugins could match one module;
each `match()` is written to be exclusive.

| name        | backbone                                  | entry points patched                |
|-------------|-------------------------------------------|-------------------------------------|
| modernbert  | ModernBERT / Ettin / mmBERT (+ finetunes) | `ModernBertModel.forward`           |
| qwen3_5     | Qwen3.5 hybrid (GatedDeltaNet + gated     | `TopkEmbedModel.forward` (topk),    |
|             | softmax attention): topk-embed-v1,        | `Qwen3_5Model.forward` /            |
|             | pplx-embed-v2-context, stock HF Qwen3.5   | `Qwen3_5TextModel.forward` (hf)     |

An architecture is a backbone, not a model: a new model on a registered backbone is a new entry
inside that plugin (see `arch/qwen3_5`), not a new registration. Adding a backbone: implement the
`Architecture` protocol in `arch/<name>/`, keep `match()` exact, and register it below. Reuse
`packed_encoders.runtime` for graphs, staging, and attention.
"""

from packed_encoders.arch.base import Architecture, match, register, registered
from packed_encoders.arch.modernbert import ModernBert
from packed_encoders.arch.qwen3_5 import Qwen35Hybrid

register(ModernBert())
register(Qwen35Hybrid())

__all__ = ["Architecture", "match", "register", "registered"]
