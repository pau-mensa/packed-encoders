"""Inference pieces for hybrid encoders. Weights and sequence policy are explicit.

References use PyTorch, independently of the selected Triton/FLA implementation.
Imports of optional kernel toolchains occur only in the factories.
"""
import torch
import torch.nn.functional as F

from packed_encoders.errors import UnsupportedTargetError
from packed_encoders.pieces.base import Contract, Piece

RMS = Contract("rms_norm", "[...,D], [D] -> [...,D]", "fp32 norm and scale, round once; scale already includes offset", autograd=False)
ADD_RMS = Contract("add_rms_norm", "x,residual [...,D], [D] -> norm,sum", "sum rounded to input dtype before RMSNorm; owned outputs", autograd=False)
SWIGLU = Contract("swiglu", "g,u [...,I] -> [...,I]", "SiLU(g) * u in fp32, round once", autograd=False)
CONV = Contract("causal_conv_split", "projection [T,C], weights [2K+V,W], positions [T] -> q,k,v,b,a", "depthwise causal conv then optional SiLU; positions restart per sequence; gate copies; outputs may share storage", autograd=False)
GATED_RMS = Contract("gated_rms_norm", "x,z [T,H,D], [D] -> [T,H*D]", "RMSNorm(x) * scale * SiLU(z) in fp32, round once", autograd=False)
QK_ROPE = Contract("rms_partial_rope", "x [T,H,D], scale [D], cos/sin [T,R] -> [T,H,D]", "RMSNorm rounded before split-half rotation of first R dimensions; fp32 tables arithmetic", autograd=False)
QK_PAIR = Contract("rms_partial_rope_pair", "projection [T,C], offsets/strides/heads, scales [2,D], cos/sin [T,R] -> q,k", "Q/K RMSNorm and partial split-half rotation; contiguous outputs may share storage", autograd=False)
GATE = Contract("sigmoid_gate", "x,g [T,H,D] -> [T,H*D]", "sigmoid rounded before multiplication", autograd=False)


def _require(condition, message):
    if not condition:
        raise UnsupportedTargetError(message)


def check_rms(x, w, eps):
    _require(x.ndim in (2, 3) and w.shape == (x.shape[-1],) and eps > 0, "RMSNorm requires [...,D], [D] and positive epsilon")


def ref_rms(x, w, eps):
    y = x.float()
    return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def ref_add_rms(x, residual, w, eps):
    summed = x + residual
    return ref_rms(summed, w, eps), summed


def check_add_rms(x, residual, w, eps):
    check_rms(x, w, eps)
    _require(x.shape == residual.shape and x.dtype == residual.dtype and x.device == residual.device, "RMSNorm residual must match input")


def check_pair(x, y):
    _require(x.shape == y.shape and x.dtype == y.dtype and x.device == y.device, "activation inputs must match")


def ref_swiglu(g, u):
    return (F.silu(g.float()) * u.float()).to(g.dtype)


def ref_conv(x, w, bias, pos, kd, vd, gate_off, nb, na, silu):
    width = 2 * kd + vd
    acc = torch.zeros(x.shape[0], width, device=x.device, dtype=torch.float32)
    idx = torch.arange(x.shape[0], device=x.device)
    for i in range(w.shape[1]):
        lag = w.shape[1] - 1 - i
        values = x[(idx - lag).clamp_min(0), :width].float()
        acc += torch.where((pos >= lag)[:, None], values, 0) * w[:, i].float()
    if bias is not None:
        acc += bias.float()
    if silu:
        acc = F.silu(acc)
    q, k, v = acc.to(x.dtype).split((kd, kd, vd), dim=-1)
    return q.contiguous(), k.contiguous(), v.contiguous(), x[:, gate_off:gate_off+nb].clone(), x[:, gate_off+nb:gate_off+nb+na].clone()


def check_conv(x, w, bias, pos, kd, vd, gate_off, nb, na, silu):
    _require(x.ndim == 2 and x.stride(-1) == 1 and min(kd, vd, nb, na) > 0 and
             w.ndim == 2 and w.shape[0] == 2*kd+vd and w.is_contiguous() and
             pos.shape == (x.shape[0],) and gate_off >= 2*kd+vd and gate_off+nb+na <= x.shape[1],
             "causal conv requires contiguous weights, valid channel slices and one position per token")


def ref_gated_rms(x, z, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float() * F.silu(z.float())).to(x.dtype).flatten(1)


def check_gated_rms(x, z, w, eps):
    check_pair(x, z)
    check_rms(x, w, eps)
    _require(x.ndim == 3 and x.is_contiguous() and z.stride(-1) == 1, "gated RMSNorm requires contiguous [T,H,D] input and unit gate last stride")


def ref_rope(x, w, cos, sin, eps):
    y = ref_rms(x, w, eps).float()
    r = cos.shape[-1]
    yr = y[..., :r]
    rot = torch.cat((-yr[..., r//2:], yr[..., :r//2]), -1)
    return torch.cat((yr * cos[:, None].float() + rot * sin[:, None].float(), y[..., r:]), -1).to(x.dtype)


def check_rope(x, w, cos, sin, eps):
    check_rms(x, w, eps)
    d = x.shape[-1]
    _require(x.ndim == 3 and x.stride(-1) == 1 and d & (d-1) == 0 and
             cos.shape == sin.shape and cos.shape[0] == x.shape[0] and
             0 < cos.shape[1] <= d and cos.shape[1] % 2 == 0 and cos.stride(-1) == sin.stride(-1) == 1,
             "partial RoPE requires power-of-two head dimension, unit last strides and even rotary dimension")


def _qk_views(x, q_off, q_stride, nq, k_off, k_stride, nk, w):
    d = w.shape[-1]
    def view(off, stride, heads):
        return x.as_strided((x.shape[0], heads, d), (x.stride(0), stride, 1), x.storage_offset()+off)
    return view(q_off, q_stride, nq), view(k_off, k_stride, nk)


def ref_rope_pair(x, q_off, q_stride, nq, k_off, k_stride, nk, w, cos, sin, eps):
    q, k = _qk_views(x, q_off, q_stride, nq, k_off, k_stride, nk, w)
    return ref_rope(q, w[0], cos, sin, eps), ref_rope(k, w[1], cos, sin, eps)


def check_rope_pair(x, q_off, q_stride, nq, k_off, k_stride, nk, w, cos, sin, eps):
    _require(x.ndim == 2 and x.stride(-1) == 1 and w.ndim == 2 and w.shape[0] == 2 and w.is_contiguous(), "Q/K pair requires [T,C] projection and contiguous [2,D] scales")
    _require(min(q_off, k_off) >= 0 and min(nq, nk) > 0 and min(q_stride, k_stride) >= w.shape[1] and
             max(q_off+(nq-1)*q_stride+w.shape[1], k_off+(nk-1)*k_stride+w.shape[1]) <= x.shape[1], "Q/K projection slices out of range")
    q, k = _qk_views(x, q_off, q_stride, nq, k_off, k_stride, nk, w)
    check_rope(q, w[0], cos, sin, eps)
    check_rope(k, w[1], cos, sin, eps)


def ref_gate(x, g):
    return (x * torch.sigmoid(g)).flatten(1)


def check_gate(x, g):
    check_pair(x, g)
    _require(x.ndim == 3 and g.stride(-1) == 1, "sigmoid gate requires [T,H,D] with unit gate last stride")


def default_numerical():
    from fla.modules.layernorm import rms_norm
    from fla.modules.activations import swiglu
    from packed_encoders._kernels import hybrid as k
    from packed_encoders.pieces.numerical import LINEAR, reference_linear, check_linear

    def norm(x, w, eps):
        return rms_norm(x, w, None, eps=eps)

    def add_norm(x, residual, w, eps):
        return rms_norm(x, w, None, residual=residual, eps=eps, prenorm=True)

    return dict(
        linear=Piece("torch-linear", LINEAR, F.linear, reference_linear, check_linear),
        rms_norm=Piece("fla-rms-norm", RMS, norm, ref_rms, check_rms),
        add_rms_norm=Piece("fla-add-rms-norm", ADD_RMS, add_norm, ref_add_rms, check_add_rms),
        swiglu=Piece("fla-swiglu", SWIGLU, swiglu, ref_swiglu, check_pair),
        conv_split=Piece("triton-causal-conv-split", CONV, k.conv_split, ref_conv, check_conv),
        gated_rms_norm=Piece("triton-gated-rms-norm", GATED_RMS, k.gated_rms_norm, ref_gated_rms, check_gated_rms),
        qk_norm_rope=Piece("triton-rms-partial-rope", QK_ROPE, k.qk_norm_rope, ref_rope, check_rope),
        qk_norm_rope_pair=Piece("triton-rms-partial-rope-pair", QK_PAIR, k.qk_norm_rope_pair, ref_rope_pair, check_rope_pair),
        sigmoid_gate=Piece("triton-sigmoid-gate", GATE, k.sigmoid_gate, ref_gate, check_gate),
    )

GDN = Contract("gated_delta_rule", "q,k [B,S,H,K], v [B,S,H,V], raw a,b [B,S,H], A_log,dt_bias [H] -> [B,S,H,V]",
               "causal zero-state recurrence; in-kernel L2 Q/K norm, softplus decay and sigmoid beta; optional sequence boundaries", autograd=False)


def reference_delta(q, k, v, g, beta, lengths):
    """Independent fp32 token-by-token oracle; expanded value heads, log decay."""
    q, k = F.normalize(q.float(), dim=-1) * q.shape[-1] ** -0.5, F.normalize(k.float(), dim=-1)
    v, g, beta = v.float(), g.float(), beta.float()
    out, start = torch.empty_like(v), 0
    for n in lengths:
        state = v.new_zeros(v.shape[1], k.shape[-1], v.shape[-1])
        for t in range(start, start+n):
            state = state * g[t].exp()[:, None, None]
            update = (v[t] - torch.einsum("hk,hkv->hv", k[t], state)) * beta[t][:, None]
            state = state + k[t][:, :, None] * update[:, None, :]
            out[t] = torch.einsum("hk,hkv->hv", q[t], state)
        start += n
    return out


def ref_gdn(q, k, v, a, b, a_log, dt_bias, cu=None, cu_cpu=None):
    lengths = cu_cpu.diff().tolist() if cu_cpu is not None else [q.shape[1]] * q.shape[0]
    repeat = v.shape[2] // q.shape[2]
    qq, kk = (t.flatten(0, 1).repeat_interleave(repeat, 1) for t in (q, k))
    decay = -a_log.float().exp() * F.softplus(a.flatten(0, 1).float() + dt_bias.float())
    return reference_delta(qq, kk, v.flatten(0, 1), decay, b.flatten(0, 1).float().sigmoid(), lengths).view_as(v).to(v.dtype)


def check_gdn(q, k, v, a, b, a_log, dt_bias, cu=None, cu_cpu=None):
    _require(q.ndim == k.ndim == v.ndim == 4 and q.shape == k.shape and q.shape[:2] == v.shape[:2]
             and v.shape[2] % q.shape[2] == 0 and a.shape == b.shape == v.shape[:3]
             and a_log.shape == dt_bias.shape == (v.shape[2],), "invalid gated delta rule head geometry or gates")
    _require((cu is None) == (cu_cpu is None), "GDN validation requires paired device/host boundaries")


def gdn_piece(*, recurrent=False):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    fn = fused_recurrent_gated_delta_rule if recurrent else chunk_gated_delta_rule
    def execute(q, k, v, a, b, a_log, dt_bias, cu=None, cu_cpu=None):
        extra = {} if recurrent else {"cu_seqlens_cpu": cu_cpu}
        return fn(q, k, v, g=a, beta=b, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  A_log=a_log, dt_bias=dt_bias, use_beta_sigmoid_in_kernel=True, cu_seqlens=cu, **extra)[0]
    return Piece("fla-recurrent-gdn" if recurrent else "fla-chunk-gdn", GDN, execute, ref_gdn, check_gdn,
                 rtol=0.03, atol=0.03)
