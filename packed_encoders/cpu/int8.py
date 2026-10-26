"""Opt-in int8 projections for the CPU ModernBERT engine (``precision="int8"``).

Apple CPUs with SME2 (M4 and later), and Linux x86-64 CPUs with AVX512-VNNI. The four
projections of every layer run as SME int8 outer products or u8 x s8 VNNI dot products;
attention, RoPE, GeGLU and the norms stay FP32. Each projection
input is rotated by a block Hadamard transform with random signs, which the weights carry
too, so outliers spread before rounding. The attention-output rotation is folded per head
into Wv and Wo; the others run on the activations. Weights are quantized per output
channel with GPTQ, calibrated on FP32 forward passes over random token IDs, so packing
needs no data and always produces the same weights. Activations are quantized per row.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
from functools import lru_cache
from operator import attrgetter
from pathlib import Path
import platform
import subprocess

import torch
import torch.nn.functional as F

from packed_encoders.cpu.build import shared_library
from packed_encoders.cpu.attention import _openmp_runtime
from packed_encoders.cpu.sme import sme2_available
from packed_encoders.errors import PackedEncodersError, UnsupportedTargetError

_P, _I, _F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
_version = attrgetter("_version")
# Calibration: 16 rows of up to 256 random tokens (4K). 16 x 384 (6K) was no more accurate across
# seeds, up to 8K-token documents (benchmarks/docs/modernbert-cpu-reduced-precision.md).
CALIBRATION_ROWS, CALIBRATION_LENGTH = 16, 256
TABLES = {"full_attention": 0, "sliding_attention": 1}


class Int8Unavailable(PackedEncodersError):
    """This machine cannot run the int8 kernels (no SME2/AVX512-VNNI, or no compiler for them)."""


class _Model(ctypes.Structure):
    _fields_ = [("hidden", _I), ("heads", _I), ("width", _I), ("block_hidden", _I), ("block_width", _I),
                ("signs_hidden", _P), ("signs_width", _P)]


class _Segment(ctypes.Structure):
    # The trailing per-channel corrections are x86's (int8_x86.c); Apple's struct ends before them.
    _fields_ = [(name, _P) for name in ("wo", "wi", "wmo", "wqkv", "so", "si", "smo", "sqkv",
                                        "mlp_weight", "mlp_bias", "after_weight", "after_bias")] + [
        ("mlp_eps", _F), ("after_eps", _F), ("table", _I)] + [(name, _P) for name in ("co", "ci", "cmo", "cqkv")]


class _Buffers(ctypes.Structure):
    _fields_ = [("cos", _P * 2), ("sin", _P * 2), ("rope", _P), ("residual", _P), ("out", _P),
                ("a_hidden", _P), ("a_width", _P), ("c", _P), ("scales", _P), ("scratch", _P),
                ("scratch_floats", ctypes.c_long), ("scratch_workers", _I)]


def vnni_available():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return False
    try:
        with open("/proc/cpuinfo") as info:
            flags = next((line.split(":", 1)[1].split() for line in info if line.startswith("flags")), [])
    except OSError:
        return False
    return {"avx512f", "avx512bw", "avx512_vnni"} <= set(flags)


def x86():
    return platform.system() == "Linux" and platform.machine() == "x86_64"


@lru_cache(maxsize=1)
def library():
    directory = None
    here = Path(__file__).parent
    try:
        runtime = _openmp_runtime() if x86() else None
        if x86():
            if runtime is None:
                raise OSError("PyTorch's OpenMP runtime not found")
            name = "int8.so"
            command = ["gcc", "-O3", "-march=native", "-fPIC", "-fopenmp", "-ffp-contract=off", "-fno-trapping-math",
                       "-fno-math-errno", "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas", "-shared",
                       str(here / "int8_x86.c"), runtime, f"-Wl,-rpath,{Path(runtime).parent}", "-o", "{build}/" + name]
        else:
            name = "int8.dylib"
            command = ["xcrun", "clang", "-O3", "-mcpu=apple-m4", "-Wall", "-Wextra", "-Werror", "-dynamiclib",
                       str(here / "int8.c"), "-o", "{build}/" + name]
        path, directory = shared_library(name, [command], runtime=runtime)
        lib = ctypes.CDLL(str(path))
    except (OSError, subprocess.SubprocessError) as exc:
        if directory is not None:
            directory.cleanup()
        raise Int8Unavailable("precision='int8' needs gcc with OpenMP" if x86() else
                              "precision='int8' needs the Xcode Command Line Tools with Apple M4 support "
                              "(xcrun clang -mcpu=apple-m4)") from exc
    for name in ("pe_q8_input", "pe_q8_segment"):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [_P] * 4 + [_I] * 2, _I
    lib.pe_q8_gemm.argtypes = [_P] * 3 + [_I] * 3
    lib.pe_q8_quantize.argtypes = [_P, _I, _I, _P, _I, _P, _P, _P]
    lib.pe_q8_gptq_block.argtypes, lib.pe_q8_gptq_block.restype = [_P] * 5 + [_I] * 5, _I
    return lib, directory


def signs(k, block):
    generator = torch.Generator().manual_seed(1000 * k + block)
    return torch.randint(0, 2, (k,), generator=generator).double() * 2 - 1


def hadamard(block):
    """The orthonormal Sylvester-order Hadamard matrix of size block (float64)."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < block:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)])
    return h / block ** .5


def rotation(k, block):
    """Orthogonal R with x·R = H_block(signs·x) / sqrt(block) per block (Sylvester order)."""
    return torch.block_diag(*[hadamard(block)] * (k // block)) * signs(k, block)[:, None]


def rotate(x, k, block):
    """x·rotation(k, block) for float64 x [..., k], one Hadamard block at a time rather than
    through the dense k x k matrix: the same products, block / k of the work. Pass contiguous
    x: on a transposed view the product takes a copying path that costs more than the product."""
    return ((x * signs(k, block)).unflatten(-1, (k // block, block)) @ hadamard(block)).flatten(-2)


def pack_weights(q, lanes=64):
    """[..., N, K] int8 -> [..., N/lanes, K/4, lanes, 4]: the SMOPA layout (64 lanes) or one
    VNNI vector of 16 channels x 4 k (16 lanes)."""
    *lead, n, k = q.shape
    return q.reshape(*lead, n // lanes, lanes, k // 4, 4).transpose(-3, -2).contiguous()


def inverse_factor(hessian, damp=0.01):
    """GPTQ's upper Cholesky factor of the damped inverse of one [K, K] float64 Hessian, as
    FP32, and the dead (zero-diagonal) columns. One matrix per call: a single factor scales
    poorly across cores (on an M4 Pro, Accelerate's inverse keeps ~7 busy for 15-30 ms), so
    callers run a few side by side."""
    h = hessian.clone()
    diagonal = torch.diagonal(h)
    dead = diagonal == 0
    diagonal[dead] = 1
    h += damp * diagonal.mean() * torch.eye(h.shape[0], dtype=h.dtype)
    return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True).float(), dead


def gptq(library, weight, factor, workers, block=128):
    """GPTQ for one FP32 [N, K] weight, given inverse_factor's results: symmetric int8 values
    and per-row scales (max |w| / 127). The column loop of each block runs natively, row by
    row; the update of the columns after the block is one FP32 product."""
    hinv, dead = factor
    rows = hinv.contiguous()  # the native loop reads factor rows; the product keeps LAPACK's layout
    w = weight.clone(memory_format=torch.contiguous_format)
    n, k = w.shape
    w[:, dead] = 0
    scale = weight.abs().amax(1) / 127
    q = torch.empty(n, k, dtype=torch.int8)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        errors = torch.empty(n, i2 - i1)
        if library.pe_q8_gptq_block(w.data_ptr(), rows.data_ptr(), scale.data_ptr(), q.data_ptr(),
                                    errors.data_ptr(), n, k, i1, i2, workers):
            raise PackedEncodersError("native int8 GPTQ failed")
        w[:, i2:] -= errors @ hinv[i1:i2, i2:]
    return q, scale


def _calibrate(engine, layers, layer_done):
    """An FP32 forward over random token IDs that records the input Gram matrix of every
    projection. Once all of layers[i]'s projections have run, layer_done(i, grams) receives their
    matrices in order, so only one layer's exist at a time. They stay FP32: each projection runs
    once, on the whole batch, so its FP32 product is the float64 sum exactly, in half the memory."""
    slots = {id(weight): (index, kind) for index, weights in enumerate(layers) for kind, weight in enumerate(weights)}
    pending = {}

    def record(x, weight, bias=None):
        if id(weight) not in slots:
            raise PackedEncodersError("int8 calibration ran a projection more than once")
        index, kind = slots.pop(id(weight))
        rows = x.reshape(-1, weight.shape[1])
        grams = pending.setdefault(index, [None] * len(layers[index]))
        grams[kind] = rows.T @ rows
        if all(gram is not None for gram in grams):
            layer_done(index, pending.pop(index))
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
    if slots:
        raise PackedEncodersError("int8 calibration did not run every projection")


class Int8Projections:
    """Prepared int8 weights and scratch for one packed CPU engine."""

    def __init__(self, engine):
        cfg, layers = engine.model.config, engine.layers
        hidden, heads, width = engine.hidden, engine.heads, cfg.intermediate_size
        self.x86 = vnni_available()
        if not (self.x86 or sme2_available()):
            raise Int8Unavailable("precision='int8' needs an Apple CPU with SME2 (M4 or later) "
                                  "or an x86 CPU with AVX512-VNNI")
        if engine.dim != 64 or hidden % 64 or width % 64:
            raise UnsupportedTargetError("CPU int8 requires head_dim=64 and hidden/intermediate sizes divisible by 64")
        # Each layer's MLP norm and the norm after it (the next attention norm, then the final
        # norm), resolved by the engine: the segments point into their aliases, which keep that
        # storage alive whatever later happens to the modules.
        norms = [layer[5:7] for layer in engine.plan]
        if any(norm.weight is None for pair in norms for norm in pair):
            raise UnsupportedTargetError("CPU int8 requires plain affine LayerNorms after layer 0")
        linears = [(layer.attn.Wqkv, layer.attn.Wo, layer.mlp.Wi, layer.mlp.Wo) for layer in layers]
        if any(linear.bias is not None for linear in sum(linears, ())):
            raise UnsupportedTargetError("CPU int8 requires bias-free projections")
        self.library, _ = library()
        self.engine, self.workers = engine, engine.attention.threads
        # The engine's aliases: _core passes these to calibration, and they share the
        # parameters' version counters.
        plan = [tuple(weight for weight, _ in layer[1:5]) for layer in engine.plan]
        self.weights = sum(plan, ())
        block_hidden, block_width = (128 if hidden % 128 == 0 else 64), (128 if width % 128 == 0 else 64)
        # Runtime inputs are x·A for an orthogonal A = rotation(k, block), so W·A gives the same
        # products and A^T·H·A is the Gram matrix of the rotated inputs. The context arrives
        # rotated per head (V·R_heads), so Wo's input needs no runtime rotation.
        inputs = ((hidden, block_hidden), (hidden, 64), (hidden, block_hidden), (width, block_width))
        # Per projection kind: packed weights, scales and (x86) corrections for every layer. x86
        # multiplies u8 = q + 128 activations: its products carry 128 * sum_k w[n][k].
        lanes = 16 if self.x86 else 64
        self.prepared = [(torch.empty(len(layers), n // lanes, k // 4, lanes, 4, dtype=torch.int8),
                          torch.empty(len(layers), n),
                          torch.empty(len(layers), n, dtype=torch.int32) if self.x86 else None)
                         for n, k in (linear.weight.shape for linear in linears[0])]

        def quantize(index, grams):
            # Calibration calls this as soon as layer index has run, so only this layer's Gram
            # matrices, rotated weights and Hessians exist at once; each Gram matrix is released
            # after use.
            weights, hessians = [], []
            for kind, ((k, block), linear) in enumerate(zip(inputs, linears[index])):
                w = linear.weight.detach().double()
                if kind == 0:  # R_heads^T·Wv = (Wv^T·R_heads)^T
                    w = w.clone()
                    w[2 * hidden:] = rotate(w[2 * hidden:].T.contiguous(), hidden, 64).T
                weights.append(rotate(w, k, block).float())
                hessians.append(rotate(rotate(grams[kind].double(), k, block).T.contiguous(), k, block).T)
                grams[kind] = None
            for (packed, scales, corrections), w, factor in zip(self.prepared, weights,
                                                               pool.map(inverse_factor, hessians)):
                q, scales[index] = gptq(self.library, w, factor, self.workers)
                packed[index] = pack_weights(q, lanes)
                if corrections is not None:
                    corrections[index] = 128 * q.sum(-1, dtype=torch.int32)

        # On x86, each factor already uses the configured BLAS workers. One task at a time
        # lowers peak memory without slowing packing on Zen 5/MKL, and still overlaps the
        # next factor with GPTQ. Accelerate benefits from concurrent factors on Apple.
        with ThreadPoolExecutor(1 if self.x86 else min(4, self.workers)) as pool:
            _calibrate(engine, plan, quantize)
        self.signs = (signs(hidden, block_hidden).float(), signs(width, block_width).float())
        self.geometry = _Model(hidden, heads, width, block_hidden, block_width,
                            self.signs[0].data_ptr(), self.signs[1].data_ptr())
        (qkv, s_qkv, c_qkv), (o, s_o, c_o), (i, s_i, c_i), (mo, s_mo, c_mo) = self.prepared
        kinds = [TABLES[kind] for kind in engine.kinds]
        pointer = (lambda t, n: t[n].data_ptr()) if self.x86 else (lambda t, n: None)
        self.first = _Segment(wqkv=qkv[0].data_ptr(), sqkv=s_qkv[0].data_ptr(), table=kinds[0],
                              cqkv=pointer(c_qkv, 0))
        self.segments = []
        for n, (norm, after) in enumerate(norms):
            last = n + 1 == len(layers)
            self.segments.append(_Segment(
                wo=o[n].data_ptr(), wi=i[n].data_ptr(), wmo=mo[n].data_ptr(), wqkv=None if last else qkv[n + 1].data_ptr(),
                so=s_o[n].data_ptr(), si=s_i[n].data_ptr(), smo=s_mo[n].data_ptr(),
                sqkv=None if last else s_qkv[n + 1].data_ptr(),
                mlp_weight=norm.weight.data_ptr(), mlp_bias=None if norm.bias is None else norm.bias.data_ptr(),
                after_weight=after.weight.data_ptr(), after_bias=None if after.bias is None else after.bias.data_ptr(),
                mlp_eps=norm.eps, after_eps=after.eps, table=0 if last else kinds[n + 1],
                co=pointer(c_o, n), ci=pointer(c_i, n), cmo=pointer(c_mo, n),
                cqkv=None if last else pointer(c_qkv, n + 1)))
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
        # Every segment overwrites the same RoPE storage; its views and native call
        # arguments can be reused throughout this forward, including changing windows.
        q, k, v = rope.unbind(0)
        att, longest = engine.attention, max(lengths)
        native_args = None
        if (self.x86 and att.native_local and att.native_global and att.token_major and longest <= 128
                and att._use_native(True, lengths) and att._use_native(False, lengths)):
            # Global attention reserves the largest scratch window. Preparing local
            # attention afterwards cannot invalidate the global call's pointers.
            context, global_args = att._native_args(q, k, v, lengths, boundaries, longest, True)
            _, local_args = att._native_args(q, k, v, lengths, boundaries, att.half, True)
            native_args = {"full_attention": global_args, "sliding_attention": local_args}
            native_attention = att.library.pe_local_attention
            context_ptr = context.data_ptr()
        for kind, segment in zip(engine.kinds, self.segments):
            if native_args is not None:
                status = native_attention(*native_args[kind])
                if status:
                    raise PackedEncodersError(f"native CPU attention failed: {status}")
            else:
                # x86 reads token-major context; Apple gathers heads from head-major.
                attention = att.local if kind == "sliding_attention" else att.global_
                context = attention(q, k, v, lengths, boundaries, token_major=self.x86).contiguous()
                context_ptr = context.data_ptr()
            if self.library.pe_q8_segment(model, ctypes.addressof(segment), buffers, context_ptr,
                                          tokens, self.workers):
                raise PackedEncodersError("native CPU int8 layer failed")
        return out[0]

    def close(self):
        self.prepared = self.segments = []
        self.a_hidden = self.a_width = self.c = self.scales = self.rope = self.scratch = None
        self.capacity = 0
