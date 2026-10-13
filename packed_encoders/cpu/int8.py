"""Opt-in int8 projections for the CPU ModernBERT engine (``precision="int8"``).

Apple CPUs with SME2 (M4 and later) only. The four projections of every layer run as
SME int8 outer products; attention, RoPE, GeGLU and the norms stay FP32. Each projection
input is rotated by a block Hadamard transform with random signs, which the weights carry
too, so outliers spread before rounding. The attention-output rotation is folded per head
into Wv and Wo; the others run on the activations. Weights are quantized per output
channel with GPTQ, calibrated on FP32 forward passes over random token IDs, so packing
needs no data and always produces the same weights. Activations are quantized per row.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from operator import attrgetter
from pathlib import Path
import platform
import subprocess
import tempfile

import torch
import torch.nn.functional as F
from torch import nn

from packed_encoders.errors import PackedEncodersError, UnsupportedTargetError

_P, _I, _F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
_version = attrgetter("_version")
# Calibration: 16 rows of up to 384 random tokens (~6K tokens).
CALIBRATION_ROWS, CALIBRATION_LENGTH = 16, 384
TABLES = {"full_attention": 0, "sliding_attention": 1}


class Int8Unavailable(PackedEncodersError):
    """This machine cannot run the int8 kernels (no SME2, or no compiler for them)."""


class _Model(ctypes.Structure):
    _fields_ = [("hidden", _I), ("heads", _I), ("width", _I), ("block_hidden", _I), ("block_width", _I),
                ("signs_hidden", _P), ("signs_width", _P)]


class _Segment(ctypes.Structure):
    _fields_ = [(name, _P) for name in ("wo", "wi", "wmo", "wqkv", "so", "si", "smo", "sqkv",
                                        "mlp_weight", "mlp_bias", "after_weight", "after_bias")] + [
        ("mlp_eps", _F), ("after_eps", _F), ("table", _I)]


class _Buffers(ctypes.Structure):
    _fields_ = [("cos", _P * 2), ("sin", _P * 2), ("rope", _P), ("residual", _P), ("out", _P),
                ("a_hidden", _P), ("a_width", _P), ("c", _P), ("scales", _P), ("scratch", _P),
                ("scratch_floats", ctypes.c_long), ("scratch_workers", _I)]


def sme2_available():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
    status = ctypes.CDLL(None).sysctlbyname(b"hw.optional.arm.FEAT_SME2", ctypes.byref(value), ctypes.byref(size), None, 0)
    return status == 0 and value.value == 1


@lru_cache(maxsize=1)
def library():
    # Process-private directory, as for the attention library. Streaming SME code needs
    # an SME2 target; only this translation unit is built for it.
    directory = tempfile.TemporaryDirectory(prefix="packed-encoders-int8-")
    path = Path(directory.name) / "int8.dylib"
    try:
        subprocess.run(
            ["xcrun", "clang", "-O3", "-mcpu=apple-m4", "-Wall", "-Wextra", "-Werror", "-dynamiclib",
             str(Path(__file__).with_name("int8.c")), "-o", str(path)],
            check=True, capture_output=True, timeout=60,
        )
        lib = ctypes.CDLL(str(path))
    except (OSError, subprocess.SubprocessError) as exc:
        directory.cleanup()
        raise Int8Unavailable("precision='int8' needs the Xcode Command Line Tools with Apple M4 support "
                              "(xcrun clang -mcpu=apple-m4)") from exc
    for name in ("pe_q8_input", "pe_q8_segment"):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [_P] * 4 + [_I] * 2, _I
    lib.pe_q8_gemm.argtypes = [_P] * 3 + [_I] * 3
    lib.pe_q8_quantize.argtypes = [_P, _I, _I, _P, _I, _P, _P, _P]
    return lib, directory


def signs(k, block):
    generator = torch.Generator().manual_seed(1000 * k + block)
    return torch.randint(0, 2, (k,), generator=generator).double() * 2 - 1


def rotation(k, block):
    """Orthogonal R with x·R = H_block(signs·x) / sqrt(block) per block (Sylvester order)."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < block:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)])
    return torch.block_diag(*[h / block ** .5] * (k // block)) * signs(k, block)[:, None]


def pack_weights(q):
    """[..., N, K] int8 -> the SMOPA layout [..., N/64, K/4, 64, 4]."""
    *lead, n, k = q.shape
    return q.reshape(*lead, n // 64, 64, k // 4, 4).transpose(-3, -2).contiguous()


def gptq(weights, hessians, block=128, damp=0.01):
    """Batched GPTQ for [B, N, K] weights and [B, K, K] Hessians: symmetric int8 values and
    per-row scales (max |w| / 127). Column updates run in FP32 across the whole batch."""
    w = weights.float().clone()
    h = hessians.clone()
    k = w.shape[2]
    diagonal = torch.diagonal(h, dim1=1, dim2=2)
    dead = diagonal == 0
    diagonal[dead] = 1
    w.masked_fill_(dead[:, None, :], 0)
    h += damp * diagonal.mean(1)[:, None, None] * torch.eye(k, dtype=h.dtype)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True).float()
    scale = weights.abs().amax(2).float() / 127
    safe = torch.where(scale > 0, scale, 1)[:, :, None]
    scale3 = scale[:, :, None]
    q = torch.empty_like(w)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        w1, hinv1 = w[:, :, i1:i2].clone(), hinv[:, i1:i2, i1:i2]
        errors = torch.empty_like(w1)
        for i in range(i2 - i1):
            column = w1[:, :, i:i + 1]
            quantized = torch.round(column / safe).clamp_(-127, 127)
            q[:, :, i1 + i:i1 + i + 1] = quantized
            error = (column - quantized * scale3) / hinv1[:, i, i][:, None, None]
            w1[:, :, i:] -= error * hinv1[:, i:i + 1, i:]
            errors[:, :, i:i + 1] = error
        w[:, :, i2:] -= errors @ hinv[:, i1:i2, i2:]
    return q.to(torch.int8), scale


def _calibrate(engine, projections):
    """Input Gram matrices per projection from FP32 forwards over random token IDs."""
    grams = {id(weight): torch.zeros(weight.shape[1], weight.shape[1], dtype=torch.float64) for weight in projections}

    def record(x, weight, bias=None):
        rows = x.reshape(-1, weight.shape[1])
        grams[id(weight)] += (rows.T @ rows).double()
        return F.linear(x, weight, bias)

    # One batch regardless of max_tokens, so the prepared weights depend only on the model.
    length = min(CALIBRATION_LENGTH, engine.max_position)
    vocab = engine.model.embeddings.tok_embeddings.num_embeddings
    ids = torch.randint(vocab, (CALIBRATION_ROWS * length,), generator=torch.Generator().manual_seed(0))
    states = [(module, module.training) for module in engine.model.modules()]
    try:
        engine.model.eval()
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            engine._core(ids, (length,) * CALIBRATION_ROWS, torch.arange(length).repeat(CALIBRATION_ROWS), linear=record)
    finally:
        for module, training in states:
            module.training = training
    return [grams[id(weight)] for weight in projections]


class Int8Projections:
    """Prepared int8 weights and scratch for one packed CPU engine."""

    def __init__(self, engine):
        cfg, layers = engine.model.config, engine.layers
        hidden, heads, width = engine.hidden, engine.heads, cfg.intermediate_size
        if not sme2_available():
            raise Int8Unavailable("precision='int8' needs an Apple CPU with SME2 (M4 or later)")
        if engine.dim != 64 or hidden % 64 or width % 64:
            raise UnsupportedTargetError("CPU int8 requires head_dim=64 and hidden/intermediate sizes divisible by 64")
        norms = [layer.mlp_norm for layer in layers] + [layer.attn_norm for layer in layers[1:]] + [engine.model.final_norm]
        if any(type(norm) is not nn.LayerNorm or norm.weight is None or "forward" in norm.__dict__ for norm in norms):
            raise UnsupportedTargetError("CPU int8 requires plain affine LayerNorms after layer 0")
        linears = [(layer.attn.Wqkv, layer.attn.Wo, layer.mlp.Wi, layer.mlp.Wo) for layer in layers]
        if any(linear.bias is not None for linear in sum(linears, ())):
            raise UnsupportedTargetError("CPU int8 requires bias-free projections")
        self.library, _ = library()
        self.engine, self.workers = engine, engine.attention.threads
        self.weights = [linear.weight for linear in sum(linears, ())]
        block_hidden, block_width = (128 if hidden % 128 == 0 else 64), (128 if width % 128 == 0 else 64)
        r_hidden, r_width, r_heads = rotation(hidden, block_hidden), rotation(width, block_width), rotation(hidden, 64)
        grams = _calibrate(engine, self.weights)
        # Runtime inputs are x·A for an orthogonal A, so W·A gives the same products and
        # A^T·H·A is the Gram matrix of the rotated inputs. The context arrives rotated per
        # head (V·R_heads), so Wo's input needs no runtime rotation.
        inputs = (r_hidden, r_heads, r_hidden, r_width)
        self.prepared = []
        for kind, a in enumerate(inputs):
            stack, hessians = [], []
            for index, layer in enumerate(linears):
                w = layer[kind].weight.detach().double()
                if kind == 0:
                    w = w.clone()
                    w[2 * hidden:] = r_heads.T @ w[2 * hidden:]
                stack.append((w @ a).float())
                hessians.append(a.T @ grams[4 * index + kind] @ a)
            q, scale = gptq(torch.stack(stack), torch.stack(hessians))
            self.prepared.append((pack_weights(q), scale.contiguous()))
        self.signs = (signs(hidden, block_hidden).float(), signs(width, block_width).float())
        self.geometry = _Model(hidden, heads, width, block_hidden, block_width,
                            self.signs[0].data_ptr(), self.signs[1].data_ptr())
        (qkv, s_qkv), (o, s_o), (i, s_i), (mo, s_mo) = self.prepared
        kinds = [TABLES[kind] for kind in engine.kinds]
        following = [*(layer.attn_norm for layer in layers[1:]), engine.model.final_norm]
        self.first = _Segment(wqkv=qkv[0].data_ptr(), sqkv=s_qkv[0].data_ptr(), table=kinds[0])
        self.segments = []
        for n, (layer, after) in enumerate(zip(layers, following)):
            last = n + 1 == len(layers)
            norm = layer.mlp_norm
            self.segments.append(_Segment(
                wo=o[n].data_ptr(), wi=i[n].data_ptr(), wmo=mo[n].data_ptr(), wqkv=None if last else qkv[n + 1].data_ptr(),
                so=s_o[n].data_ptr(), si=s_i[n].data_ptr(), smo=s_mo[n].data_ptr(),
                sqkv=None if last else s_qkv[n + 1].data_ptr(),
                mlp_weight=norm.weight.data_ptr(), mlp_bias=None if norm.bias is None else norm.bias.data_ptr(),
                after_weight=after.weight.data_ptr(), after_bias=None if after.bias is None else after.bias.data_ptr(),
                mlp_eps=norm.eps, after_eps=after.eps, table=0 if last else kinds[n + 1]))
        self.versions = list(map(_version, self.weights))
        per_worker = max(3 * hidden, 4 * width)
        self.scratch = torch.empty(self.workers * per_worker)
        self.buffers = _Buffers(scratch=self.scratch.data_ptr(), scratch_floats=per_worker,
                                scratch_workers=self.workers)
        self.capacity, self.stride = 0, max(3 * hidden, 2 * width)

    def _reserve(self, tokens):
        if tokens <= self.capacity:
            return
        blocks = (tokens + 15) // 16 * 16
        cfg = self.geometry
        self.a_hidden = torch.empty(blocks * cfg.hidden, dtype=torch.int8)
        self.a_width = torch.empty(blocks * cfg.width, dtype=torch.int8)
        self.c = torch.empty(tokens * self.stride, dtype=torch.int32)
        self.scales = torch.empty(tokens)
        self.rope = torch.empty(3 * cfg.heads * tokens * 64)
        b = self.buffers
        b.a_hidden, b.a_width, b.c, b.scales = (t.data_ptr() for t in (self.a_hidden, self.a_width, self.c, self.scales))
        b.rope = self.rope.data_ptr()
        self.capacity = tokens

    def core(self, ids, lengths, positions):
        engine = self.engine
        if list(map(_version, self.weights)) != self.versions:
            raise PackedEncodersError("CPU int8 projection weights changed in place since packing; unpack and repack")
        tables = {kind: (cos[positions].contiguous(), sin[positions].contiguous())
                  for kind, (cos, sin) in engine.tables.items()}
        boundaries = torch.tensor(lengths, dtype=torch.int32)
        x = engine.model.embeddings(ids[None])
        normed = engine.layers[0].attn_norm(x).contiguous()
        tokens = x.shape[1]
        self._reserve(tokens)
        out = torch.empty_like(x)
        b = self.buffers
        for kind, index in TABLES.items():
            b.cos[index], b.sin[index] = (t.data_ptr() for t in tables[kind])
        b.residual, b.out = x.data_ptr(), out.data_ptr()
        rope = self.rope[:3 * engine.heads * tokens * 64].view(3, 1, engine.heads, tokens, 64)
        model, buffers = ctypes.addressof(self.geometry), ctypes.addressof(b)
        if self.library.pe_q8_input(model, ctypes.addressof(self.first), buffers, normed.data_ptr(), tokens, self.workers):
            raise PackedEncodersError("native CPU int8 input projection failed")
        for kind, segment in zip(engine.kinds, self.segments):
            q, k, v = rope.unbind(0)
            context = (engine.attention.local(q, k, v, lengths, boundaries) if kind == "sliding_attention"
                       else engine.attention.global_(q, k, v, lengths, boundaries)).contiguous()
            if self.library.pe_q8_segment(model, ctypes.addressof(segment), buffers, context.data_ptr(),
                                          tokens, self.workers):
                raise PackedEncodersError("native CPU int8 layer failed")
        return out[0]

    def close(self):
        self.prepared = self.segments = []
        self.a_hidden = self.a_width = self.c = self.scales = self.rope = self.scratch = None
        self.capacity = 0
