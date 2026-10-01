"""Triton kernels for the Qwen3.5 hybrid."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _qk_norm_rope_kernel(X, W, COS, SIN, Y, stride_xt, stride_xh, stride_yt, stride_yh, stride_ct, H, eps,
                         D: tl.constexpr, HALF: tl.constexpr):
    """One (token, head) row: Qwen RMSNorm with a (1 + w) weight, rounded to the model dtype, then
    partial RoPE on the first 2 * HALF dims (rotate_half). Rounds where Qwen3_5RMSNorm followed by
    apply_rotary_pos_emb rounds, so it reproduces the two-module result, not an fp32 idealisation."""
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H
    offs = tl.arange(0, D)
    xp = X + t * stride_xt + h * stride_xh
    x = tl.load(xp + offs).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    w = tl.load(W + offs)
    y = (x * r * w).to(Y.dtype.element_ty).to(tl.float32)
    rot = offs < 2 * HALF
    pidx = tl.where(offs < HALF, offs + HALF, offs - HALF)
    xpart = tl.load(xp + pidx, mask=rot, other=0.).to(tl.float32)
    wpart = tl.load(W + pidx, mask=rot, other=0.)
    ypart = (xpart * r * wpart).to(Y.dtype.element_ty).to(tl.float32)
    c = tl.load(COS + t * stride_ct + offs, mask=rot, other=1.).to(tl.float32)
    sn = tl.load(SIN + t * stride_ct + offs, mask=rot, other=0.).to(tl.float32)
    sign = tl.where(offs < HALF, -1.0, 1.0)
    out = tl.where(rot, y * c + sign * ypart * sn, y)
    tl.store(Y + t * stride_yt + h * stride_yh + offs, out.to(Y.dtype.element_ty))


def qk_norm_rope(x: Tensor, w: Tensor, cos: Tensor, sin: Tensor, eps: float) -> Tensor:
    """x: (T, H, D) with unit last stride (row/head strides free, so the q half of the
    interleaved q|gate projection is read in place); w: (D,) fp32 = 1 + norm weight;
    cos/sin: (T, rotary_dim). Returns a contiguous (T, H, D)."""
    T, H, D = x.shape
    y = torch.empty((T, H, D), device=x.device, dtype=x.dtype)
    if T:
        _qk_norm_rope_kernel[(T * H,)](x, w, cos, sin, y, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                                       cos.stride(0), H, eps, D=D, HALF=cos.shape[-1] // 2, num_warps=4)
    return y


def qk_norm_rope_pair(x: Tensor, q_off: int, q_head_stride: int, nq: int, k_off: int, k_head_stride: int, nk: int,
                      w: Tensor, cos: Tensor, sin: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """q and k in one launch. x: the attention projection output (T, width), unit column
    stride; q head h starts at column q_off + h * q_head_stride, k head h at k_off + h *
    k_head_stride. w: (2, D) fp32 = 1 + (q_norm, k_norm) weight. Returns contiguous q (T, nq, D)
    and k (T, nk, D), two views of one buffer."""
    T, D = x.shape[0], w.shape[-1]
    buf = torch.empty(T * (nq + nk) * D, device=x.device, dtype=x.dtype)
    q, k = buf[: T * nq * D].view(T, nq, D), buf[T * nq * D:].view(T, nk, D)
    if T:
        _qk_pair_kernel[(T * (nq + nk),)](x, w, cos, sin, buf, x.stride(0), q_off, q_head_stride, k_off, k_head_stride,
                                          cos.stride(0), T, eps, NQ=nq, NK=nk, D=D, HALF=cos.shape[-1] // 2,
                                          num_warps=4)
    return q, k


@triton.jit(do_not_specialize=["T"])                    # T == 1 would become a constexpr without .to()
def _qk_pair_kernel(X, W, COS, SIN, Y, stride_xt, q_off, q_hs, k_off, k_hs, stride_ct, T, eps,
                    NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr):
    """`_qk_norm_rope_kernel` for one (token, head) of q or of k, chosen by the program id."""
    pid = tl.program_id(0)
    t = pid // (NQ + NK)
    h = pid % (NQ + NK)
    is_q = h < NQ
    col = tl.where(is_q, q_off + h * q_hs, k_off + (h - NQ) * k_hs)
    dst = tl.where(is_q, t.to(tl.int64) * NQ * D + h * D, T.to(tl.int64) * NQ * D + t.to(tl.int64) * NK * D + (h - NQ) * D)
    wrow = tl.where(is_q, 0, D)
    offs = tl.arange(0, D)
    xp = X + t.to(tl.int64) * stride_xt + col
    x = tl.load(xp + offs).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    w = tl.load(W + wrow + offs)
    y = (x * r * w).to(Y.dtype.element_ty).to(tl.float32)
    rot = offs < 2 * HALF
    pidx = tl.where(offs < HALF, offs + HALF, offs - HALF)
    xpart = tl.load(xp + pidx, mask=rot, other=0.).to(tl.float32)
    wpart = tl.load(W + wrow + pidx, mask=rot, other=0.)
    ypart = (xpart * r * wpart).to(Y.dtype.element_ty).to(tl.float32)
    c = tl.load(COS + t * stride_ct + offs, mask=rot, other=1.).to(tl.float32)
    sn = tl.load(SIN + t * stride_ct + offs, mask=rot, other=0.).to(tl.float32)
    sign = tl.where(offs < HALF, -1.0, 1.0)
    out = tl.where(rot, y * c + sign * ypart * sn, y)
    tl.store(Y + dst + offs, out.to(Y.dtype.element_ty))


@triton.jit(do_not_specialize=["N"])
def _conv_split_kernel(X, W, BIAS, POS, Y, G, stride_x, N, G_OFF,
                       KD: tl.constexpr, VD: tl.constexpr, NB: tl.constexpr, NA: tl.constexpr, KW: tl.constexpr,
                       NCB: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr,
                       HAS_BIAS: tl.constexpr, SILU: tl.constexpr):
    """Programs [0, NCB) along axis 0: the depthwise causal conv (+ SiLU) of BD channels of
    q|k|v for BT tokens, stored into q, k or v (back to back in Y). Program NCB: copies the b|a
    gate columns of the same tokens into contiguous b and a (back to back in G). A tap j tokens
    back is read only when the token's position is >= j, so sequences never mix."""
    i_d, i_t = tl.program_id(0), tl.program_id(1)
    o_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
    m_t = o_t < N
    if i_d < NCB:
        o_d = i_d * BD + tl.arange(0, BD)
        pos = tl.load(POS + o_t, mask=m_t, other=0)
        acc = tl.zeros((BT, BD), dtype=tl.float32)
        for i in tl.static_range(KW):                      # oldest tap first, fla's summation order
            j = KW - 1 - i
            m = m_t & (pos >= j)
            x = tl.load(X + (o_t - j)[:, None] * stride_x + o_d[None, :], mask=m[:, None], other=0.).to(tl.float32)
            acc += x * tl.load(W + o_d * KW + (KW - 1 - j)).to(tl.float32)[None, :]
        if HAS_BIAS:
            acc += tl.load(BIAS + o_d).to(tl.float32)[None, :]
        if SILU:
            acc = acc * tl.sigmoid(acc)
        c0 = i_d * BD                                        # BD divides KD and VD: one segment per block
        seg = tl.minimum(c0 // KD, 2)
        n64 = N.to(tl.int64)
        base = tl.where(seg < 2, seg * n64 * KD, 2 * n64 * KD)
        width = tl.where(seg < 2, KD, VD)
        col = o_d - tl.where(seg < 2, seg * KD, 2 * KD)
        tl.store(Y + base + o_t[:, None] * width + col[None, :], acc.to(Y.dtype.element_ty), mask=m_t[:, None])
    else:
        o_g = tl.arange(0, BG)
        g = tl.load(X + o_t[:, None] * stride_x + G_OFF + o_g[None, :], mask=m_t[:, None] & (o_g < NB + NA)[None, :])
        tl.store(G + o_t[:, None] * NB + o_g[None, :], g, mask=m_t[:, None] & (o_g < NB)[None, :])
        tl.store(G + N.to(tl.int64) * NB + o_t[:, None] * NA + (o_g - NB)[None, :], g,
                 mask=m_t[:, None] & ((o_g >= NB) & (o_g < NB + NA))[None, :])


def _pow2_divisor(n: int, cap: int) -> int:
    b = cap
    while b > 1 and n % b:
        b //= 2
    return b


def conv_split(x: Tensor, weight: Tensor, bias: Tensor | None, pos: Tensor, kd: int, vd: int, gate_off: int,
               nb: int, na: int, silu: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The GatedDeltaNet causal conv over q|k|v plus the b|a gate columns, in one launch.

    x: the merged in-projection output (N, width), unit column stride, channels [0, 2kd + vd) =
    q|k|v and [gate_off, gate_off + nb + na) = b|a. weight: (2kd + vd, KW); pos: (N,) position
    of each token in its sequence (row). Returns contiguous q (N, kd), k (N, kd), v (N, vd),
    b (N, nb), a (N, na), with q = silu(conv(x)) exactly as `causal_conv1d(activation='silu')`."""
    N, KW = x.shape[0], weight.shape[1]
    y = torch.empty(N * (2 * kd + vd), device=x.device, dtype=x.dtype)
    g = torch.empty(N * (nb + na), device=x.device, dtype=x.dtype)
    q, k, v = y[: N * kd].view(N, kd), y[N * kd: 2 * N * kd].view(N, kd), y[2 * N * kd:].view(N, vd)
    b, a = g[: N * nb].view(N, nb), g[N * nb:].view(N, na)
    if N:
        BD = _pow2_divisor(math.gcd(kd, vd), 128)
        BT = 16 if N <= 2048 else 64
        ncb = (2 * kd + vd) // BD
        _conv_split_kernel[(ncb + 1, triton.cdiv(N, BT))](
            x, weight, bias, pos, y, g, x.stride(0), N, gate_off, KD=kd, VD=vd, NB=nb, NA=na, KW=KW, NCB=ncb, BT=BT,
            BD=BD, BG=triton.next_power_of_2(nb + na), HAS_BIAS=bias is not None, SILU=silu, num_warps=4)
    return q, k, v, b, a


@triton.jit
def _gated_rms_norm_kernel(X, Z, W, Y, stride_zt, stride_zh, R, eps, H: tl.constexpr, D: tl.constexpr,
                           BR: tl.constexpr):
    """BR (token, head) rows: RMSNorm(x) * w * silu(z) in fp32, rounded once — fla's
    FusedRMSNormGated, which the unfused path runs, reading z in place instead of a copy."""
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    m = r < R
    offs = tl.arange(0, D)
    x = tl.load(X + r[:, None] * D + offs[None, :], mask=m[:, None], other=0.).to(tl.float32)
    rs = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    y = x * rs[:, None] * tl.load(W + offs).to(tl.float32)[None, :]
    zp = Z + (r // H)[:, None] * stride_zt + (r % H)[:, None] * stride_zh + offs[None, :]
    z = tl.load(zp, mask=m[:, None], other=0.).to(tl.float32)
    tl.store(Y + r[:, None] * D + offs[None, :], (y * z * tl.sigmoid(z)).to(Y.dtype.element_ty), mask=m[:, None])


def gated_rms_norm(x: Tensor, z: Tensor, weight: Tensor, eps: float) -> Tensor:
    """x: (N, H, D) contiguous; z: (N, H, D) with unit last stride (read in place from the
    projection); weight: (D,). Returns (N, H * D)."""
    N, H, D = x.shape
    y = torch.empty((N, H * D), device=x.device, dtype=x.dtype)
    R = N * H
    if R:
        BR = 8
        _gated_rms_norm_kernel[(triton.cdiv(R, BR),)](x, z, weight, y, z.stride(0), z.stride(1), R, eps, H=H, D=D,
                                                       BR=BR, num_warps=4)
    return y


@triton.jit
def _sigmoid_gate_kernel(O, G, Y, stride_gt, stride_gh, R, H: tl.constexpr, D: tl.constexpr, BR: tl.constexpr):
    """o * sigmoid(gate) with the model's roundings: sigmoid to the model dtype, then the product."""
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    m = r < R
    offs = tl.arange(0, D)
    o = tl.load(O + r[:, None] * D + offs[None, :], mask=m[:, None], other=0.).to(tl.float32)
    g = tl.load(G + (r // H)[:, None] * stride_gt + (r % H)[:, None] * stride_gh + offs[None, :], mask=m[:, None],
                other=0.).to(tl.float32)
    s = tl.sigmoid(g).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + r[:, None] * D + offs[None, :], (o * s).to(Y.dtype.element_ty), mask=m[:, None])


def sigmoid_gate(o: Tensor, gate: Tensor) -> Tensor:
    """o: (N, H, D) contiguous; gate: (N, H, D) unit last stride. Returns (N, H * D)."""
    N, H, D = o.shape
    o = o.contiguous()
    y = torch.empty((N, H * D), device=o.device, dtype=o.dtype)
    R = N * H
    if R:
        BR = 4
        _sigmoid_gate_kernel[(triton.cdiv(R, BR),)](o, gate, y, gate.stride(0), gate.stride(1), R, H=H, D=D, BR=BR,
                                                     num_warps=4)
    return y
