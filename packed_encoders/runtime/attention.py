"""Choose a varlen attention kernel for one head geometry by measuring it.

Whether a kernel serves a shape depends on the GPU, the wheel, and the head dim (FA4 on
sm_90/sm_100, FA2 wheels built for some torch/CUDA pairs, torch's own varlen kernel from
2.10, head_dim 256 support varying across all three). A lookup table goes stale; a probe
does not. Each candidate must import, run, and match fp32 per-segment SDPA on a probe that
includes an empty segment and grouped KV heads, in both layouts the engines use:

- packed: tokens back to back, `cu_seqlens` (eager forward)
- padded: `(rows, S)` rectangles with `[real | pad]` segments per row (CUDA graphs)

The first candidate that passes wins. `sdpa` always passes, so selection never fails; it
only gets slower. Rejections are recorded with their reason for `validate()` reports.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from packed_encoders.errors import ValidationError
from packed_encoders.runtime.graphs import PaddedStatic

PROBE_TOLERANCE = 2e-2  # max abs error vs fp32 SDPA on unit-variance bf16 inputs


@dataclass
class AttentionChoice:
    """`packed(q, k, v, cu32, max_len, lengths, causal)` and `padded(q, k, v, static, causal)`
    both take (tokens, heads, head_dim) q/k/v (k/v may have fewer heads) and return q's shape."""

    name: str
    packed: Callable[..., Tensor]
    padded: Callable[..., Tensor]
    max_abs_err: float = 0.0
    rejected: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------- candidates


def _flash2() -> tuple[Callable, Callable]:
    from flash_attn import flash_attn_varlen_func as f

    def packed(q, k, v, cu32, max_len, lengths, causal):
        return f(q, k, v, cu32, cu32, max_len, max_len, dropout_p=0.0, causal=causal)

    def padded(q, k, v, static, causal):
        return f(q, k, v, static.seg_cu32, static.seg_cu32, static.seq, static.seq, dropout_p=0.0, causal=causal)

    return packed, padded


def _flash4() -> tuple[Callable, Callable]:
    from flash_attn.cute import flash_attn_varlen_func as f

    def call(q, k, v, cu, max_len, causal):
        out = f(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max_len, max_seqlen_k=max_len, causal=causal)
        return out[0] if isinstance(out, (tuple, list)) else out

    return (lambda q, k, v, cu32, max_len, lengths, causal: call(q, k, v, cu32, max_len, causal),
            lambda q, k, v, static, causal: call(q, k, v, static.seg_cu32, static.seq, causal))


def _repeat_kv(q: Tensor, k: Tensor, v: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    r = q.shape[dim] // k.shape[dim]
    return (k, v) if r == 1 else (k.repeat_interleave(r, dim), v.repeat_interleave(r, dim))


def _torch_varlen() -> tuple[Callable, Callable]:
    from torch.nn.attention.varlen import varlen_attn as f  # torch >= 2.10

    has_causal = "is_causal" in inspect.signature(f).parameters

    def call(q, k, v, cu, max_len, causal):
        if causal and not has_causal:
            raise NotImplementedError("this torch varlen_attn has no is_causal")
        k, v = _repeat_kv(q, k, v, 1)
        return f(q, k, v, cu, cu, max_len, max_len, **({"is_causal": causal} if has_causal else {}))

    return (lambda q, k, v, cu32, max_len, lengths, causal: call(q, k, v, cu32, max_len, causal),
            lambda q, k, v, static, causal: call(q, k, v, static.seg_cu32, static.seq, causal))


def _sdpa() -> tuple[Callable, Callable]:
    def packed(q, k, v, cu32, max_len, lengths, causal):
        # Eager only (reads host lengths): one SDPA per sequence.
        k, v = _repeat_kv(q, k, v, 1)
        outs = [
            F.scaled_dot_product_attention(qs.transpose(0, 1), ks.transpose(0, 1), vs.transpose(0, 1),
                                           is_causal=causal).transpose(0, 1)
            for qs, ks, vs in zip(q.split(list(lengths)), k.split(list(lengths)), v.split(list(lengths)))
            if qs.shape[0]
        ]
        return torch.cat(outs) if outs else torch.empty_like(q)

    def padded(q, k, v, static: PaddedStatic, causal):
        # Capture-safe: the [real | pad] segment mask is built on device from `lens`.
        rows, seq = static.rows, static.seq
        k, v = _repeat_kv(q, k, v, 1)
        qb, kb, vb = (t.view(rows, seq, t.shape[1], t.shape[2]).transpose(1, 2) for t in (q, k, v))
        pos = torch.arange(seq, device=q.device)
        segment = pos[None, :] >= static.lens[:, None]                     # False real, True pad
        allowed = segment[:, :, None] == segment[:, None, :]
        if causal:
            allowed = allowed & (pos[:, None] >= pos[None, :])
        out = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=allowed[:, None])
        return out.transpose(1, 2).reshape(rows * seq, q.shape[1], q.shape[2])

    return packed, padded


_CANDIDATES: dict[str, Callable[[], tuple[Callable, Callable]]] = {
    "flash2": _flash2, "flash4": _flash4, "torch_varlen": _torch_varlen, "sdpa": _sdpa,
}


def default_order(device: torch.device) -> tuple[str, ...]:
    major = torch.cuda.get_device_capability(device)[0]
    if major in (9, 10, 11):                        # FA4 (CuteDSL) targets Hopper / datacenter Blackwell
        return ("flash4", "flash2", "torch_varlen", "sdpa")
    return ("flash2", "torch_varlen", "flash4", "sdpa")


# ---------------------------------------------------------------------------- probe


def _reference(q, k, v, lengths, causal):
    k, v = _repeat_kv(q, k, v, 1)
    outs = [
        F.scaled_dot_product_attention(qs.transpose(0, 1).float(), ks.transpose(0, 1).float(),
                                       vs.transpose(0, 1).float(), is_causal=causal).transpose(0, 1)
        for qs, ks, vs in zip(q.split(lengths), k.split(lengths), v.split(lengths)) if qs.shape[0]
    ]
    return torch.cat(outs)


def _probe(packed, padded, *, n_heads, n_kv_heads, head_dim, causal, device, dtype) -> float:
    gen = torch.Generator(device=device).manual_seed(0)

    def rnd(t, h):
        return torch.randn(t, h, head_dim, device=device, dtype=dtype, generator=gen)

    # packed layout, including an empty sequence
    lengths = [5, 64, 0, 37]
    total = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    q, k, v = rnd(total, n_heads), rnd(total, n_kv_heads), rnd(total, n_kv_heads)
    ref = _reference(q, k, v, lengths, causal)
    err = (packed(q, k, v, cu, max(lengths), lengths, causal).float() - ref).abs().max().item()

    # padded layout: rows of S with [real | pad]; only real rows are compared
    rows, seq, lens = 3, 48, [5, 48, 0]
    starts = [i * seq for i in range(rows)]
    seg = [0]
    for s, n in zip(starts, lens):
        seg += [s + n, s + seq]
    static = PaddedStatic(rows=rows, seq=seq, ids=torch.zeros(rows, seq, dtype=torch.long, device=device),
                          seg_cu32=torch.tensor(seg, dtype=torch.int32, device=device),
                          lens=torch.tensor(lens, dtype=torch.int32, device=device))
    q, k, v = rnd(rows * seq, n_heads), rnd(rows * seq, n_kv_heads), rnd(rows * seq, n_kv_heads)
    out = padded(q, k, v, static, causal).float()
    keep = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, lens)]).to(device)
    ref = _reference(q.index_select(0, keep), k.index_select(0, keep), v.index_select(0, keep),
                     [n for n in lens], causal)
    err = max(err, (out.index_select(0, keep) - ref).abs().max().item())
    return err


def select_attention(
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    causal: bool,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    order: Sequence[str] | None = None,
) -> AttentionChoice:
    """Return the first candidate in `order` that imports, runs, and matches fp32 SDPA."""
    rejected: dict[str, str] = {}
    for name in order or default_order(device):
        try:
            packed, padded = _CANDIDATES[name]()
            err = _probe(packed, padded, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
                         causal=causal, device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001 — any import/launch failure just rejects the candidate
            rejected[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
            continue
        if not err < PROBE_TOLERANCE:
            rejected[name] = f"mismatch vs fp32 SDPA: max abs err {err:.3g}"
            continue
        return AttentionChoice(name=name, packed=packed, padded=padded, max_abs_err=err, rejected=rejected)
    raise ValidationError(f"no attention candidate passed: {rejected}")
