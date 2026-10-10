"""Weight plumbing shared by the hybrid engines: merged projections that alias the model's own
parameters, the guards that make that aliasing safe, and the embedding rows validation avoids."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from packed_encoders.errors import UnsupportedTargetError


def require_plain(linears: Sequence[nn.Module]) -> None:
    """The engine reads `.weight` and nothing else. A peft tuner layer exposes its *base* weight
    there (BaseTunerLayer.weight), so an unmerged adapter would be silently dropped; refuse it,
    and a bias, which would be dropped too."""
    for lin in linears:
        if hasattr(lin, "lora_A") or hasattr(lin, "base_layer") or getattr(lin, "bias", None) is not None:
            raise UnsupportedTargetError(
                f"{type(lin).__name__} carries a bias or an unmerged LoRA adapter; merge adapters "
                "first (peft: model = model.merge_and_unload()) — packing reads plain dense weights"
            )


def share_rows(linears: Sequence[nn.Linear], registry: list[nn.Linear]) -> Tensor:
    """Concatenate the weights row-wise and re-point every parameter at its slice of the
    result, so the merged GEMM weight and the HF parameters are one storage. Re-pointed
    layers are appended to `registry` for `unshare_rows`."""
    require_plain(linears)
    merged = torch.cat([lin.weight.detach() for lin in linears], 0)
    off = 0
    for lin in linears:
        n = lin.weight.shape[0]
        lin.weight.data = merged[off:off + n]
        registry.append(lin)
        off += n
    return merged


def unshare_rows(registry: list[nn.Linear], *, rollback: bool = False) -> None:
    """Restore independent parameter storage, retaining live values and identities.

    Shared source parameters are rejected before preparation. Restoring independent
    storage therefore restores the accepted aliasing contract too. Addresses can change;
    callers must not retain pre-pack tensor views or external graphs across pack/unpack.
    """
    for lin in registry:
        lin.weight.data = lin.weight.data.clone()
    registry.clear()


def require_independent_parameters(model: nn.Module, family: str) -> None:
    """Reject aliasing before any re-pointing; merged projections must be independent."""
    seen = set()
    for name, param in model.named_parameters(remove_duplicate=False):
        key = (param.device, param.untyped_storage().data_ptr())
        if key in seen:
            raise UnsupportedTargetError(f"{family} requires independent parameter storage; {name} is tied or aliased")
        seen.add(key)


def validation_ids_below(cfg) -> int:
    """Validation's random token ids come from the regular vocabulary, below this. Many tokenizers put their
    special tokens after it, then the embedding's padding rows: ids no tokenizer emits, rows training never
    reached. One of those in a row can take the model's own bf16 forward far from fp32 (Qwen3.5-4B on an
    L40S: cosine 0.87 on that token, the engine 0.97), so a check there measures the reference, not the
    engine. The lowest special token the config names in the embedding's upper half starts that tail."""
    rows = cfg.vocab_size
    named = [i for k, v in vars(cfg).items() if k.endswith("_token_id")
             for i in (v if isinstance(v, (list, tuple)) else (v,)) if isinstance(i, int) and rows // 2 <= i < rows]
    return min(named, default=rows)
