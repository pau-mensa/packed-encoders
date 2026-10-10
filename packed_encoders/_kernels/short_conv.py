"""Triton kernel for the gated short convolution of LFM2-style hybrids."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit(do_not_specialize=["N"])
def _short_conv_kernel(X, W, POS, Y, stride_x, N, D: tl.constexpr, KW: tl.constexpr, BT: tl.constexpr,
                       BD: tl.constexpr):
    """BT tokens x BD channels of `C * conv(B * x)`, where X holds B|C|x per token. A tap j tokens
    back is read only when the token's position is >= j, so sequences never mix. Rounds where the
    unfused ops round: B * x, then the conv, then the product with C."""
    i_d, i_t = tl.program_id(0), tl.program_id(1)
    o_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
    o_d = i_d * BD + tl.arange(0, BD)
    m_t, m_d = o_t < N, o_d < D
    pos = tl.load(POS + o_t, mask=m_t, other=0)
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for i in tl.static_range(KW):                          # oldest tap first, as the depthwise conv sums
        j = KW - 1 - i
        m = (m_t & (pos >= j))[:, None] & m_d[None, :]
        row = X + (o_t - j)[:, None] * stride_x + o_d[None, :]
        b = tl.load(row, mask=m, other=0.).to(tl.float32)
        x = tl.load(row + 2 * D, mask=m, other=0.).to(tl.float32)
        bx = (b * x).to(Y.dtype.element_ty).to(tl.float32)
        acc += bx * tl.load(W + o_d * KW + i, mask=m_d, other=0.).to(tl.float32)[None, :]
    conv = acc.to(Y.dtype.element_ty).to(tl.float32)
    mask = m_t[:, None] & m_d[None, :]
    c = tl.load(X + o_t[:, None] * stride_x + D + o_d[None, :], mask=mask, other=0.).to(tl.float32)
    tl.store(Y + o_t[:, None] * D + o_d[None, :], (c * conv).to(Y.dtype.element_ty), mask=mask)


def short_conv(x: Tensor, weight: Tensor, pos: Tensor) -> Tensor:
    """The gated short conv in one launch. x: the in-projection output (N, 3D) laid out B|C|x,
    unit column stride; weight: (D, KW) depthwise taps, oldest first; pos: (N,) position of each
    token in its sequence (row). Returns a contiguous (N, D) = C * causal_conv(B * x)."""
    N, D, KW = x.shape[0], weight.shape[0], weight.shape[1]
    y = torch.empty((N, D), device=x.device, dtype=x.dtype)
    if N:
        BT, BD = (16 if N <= 2048 else 64), 128
        _short_conv_kernel[(triton.cdiv(D, BD), triton.cdiv(N, BT))](
            x, weight, pos, y, x.stride(0), N, D=D, KW=KW, BT=BT, BD=BD, num_warps=4)
    return y
