"""GQA attention pieces over an explicitly prepared runtime kernel choice."""
import torch

from packed_encoders.errors import UnsupportedTargetError
from packed_encoders.pieces.base import Contract, Piece

PACKED = Contract("varlen_attention", "q [T,H,D], k,v [T,KV,D], cu32, host lengths -> [T,H,D]",
                  "same-length Q/K segments; GQA; optional causal mask; zero dropout; scale D**-0.5", autograd=False)
PADDED = Contract("segmented_padded_attention", "q [B*S,H,D], k,v [B*S,KV,D], static segments -> [B*S,H,D]",
                  "each row split into real/pad segments; GQA; optional causal mask; zero dropout; scale D**-0.5", autograd=False)


def _check(q, k, v):
    if (q.ndim != 3 or k.ndim != 3 or k.shape != v.shape or q.shape[0] != k.shape[0]
            or q.shape[2] != k.shape[2] or q.shape[1] % k.shape[1]):
        raise UnsupportedTargetError("varlen attention requires matching token/head dimensions and integer GQA ratio")


def check_packed(choice, q, k, v, cu32, max_len, lengths, causal):
    _check(q, k, v)
    if cu32.dtype != torch.int32 or cu32.numel() != len(lengths)+1 or sum(lengths) != q.shape[0] or max_len < max(lengths):
        raise UnsupportedTargetError("varlen attention requires consistent sequence boundaries and host lengths")


def check_padded(choice, q, k, v, static, causal):
    _check(q, k, v)
    if q.shape[0] != static.rows*static.seq or static.lens.shape != (static.rows,):
        raise UnsupportedTargetError("padded attention metadata must match the rectangular token layout")


def reference_packed(choice, q, k, v, cu32, max_len, lengths, causal):
    from packed_encoders.runtime.attention import _reference
    return _reference(q, k, v, list(lengths), causal).to(q.dtype)


def reference_padded(choice, q, k, v, static, causal):
    from packed_encoders.runtime.attention import _sdpa
    return _sdpa()[1](q.float(), k.float(), v.float(), static, causal).to(q.dtype)


def attention_pieces():
    return (
        Piece("selected-varlen-attention", PACKED,
              lambda choice, *args: choice.packed(*args), reference_packed, check_packed),
        Piece("selected-segmented-attention", PADDED,
              lambda choice, *args: choice.padded(*args), reference_padded, check_padded),
    )
