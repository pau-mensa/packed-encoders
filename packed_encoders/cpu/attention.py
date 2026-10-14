"""Bounded CPU attention with an optional Apple Accelerate implementation, and native row kernels
(RoPE/layout, GeGLU, residual LayerNorm) on Apple and Linux x86-64."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from packed_encoders.errors import PackedEncodersError


@lru_cache(maxsize=1)
def _library():
    # Process-private directory: no shared writable executable cache, stale ABI,
    # or concurrent compiler races. Keep the directory alive with the library.
    directory = tempfile.TemporaryDirectory(prefix="packed-encoders-cpu-")
    path = Path(directory.name) / "attention.dylib"
    try:
        subprocess.run(
            ["xcrun", "clang", "-O3", "-Wall", "-Wextra", "-Werror", "-dynamiclib",
             "-framework", "Accelerate", str(Path(__file__).with_name("local_attention.c")), "-o", str(path)],
            check=True, capture_output=True, timeout=30,
        )
        return _signatures(ctypes.CDLL(str(path))), directory
    except (OSError, subprocess.SubprocessError) as exc:
        directory.cleanup()
        raise PackedEncodersError("Apple CPU attention needs the Xcode Command Line Tools (xcrun clang)") from exc


def _signatures(library):
    fn = library.pe_local_attention
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 6 + [ctypes.c_void_p, ctypes.c_size_t]
    fn.restype = ctypes.c_int
    rope = library.pe_rope_layout
    rope.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
    rope.restype = ctypes.c_int
    geglu = library.pe_geglu
    geglu.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3
    geglu.restype = ctypes.c_int
    norm = library.pe_add_layer_norm
    norm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_float, ctypes.c_int]
    norm.restype = ctypes.c_int
    return library


def _openmp_runtime():
    # The runtime PyTorch already loaded (GNU, Intel or LLVM; all export the GOMP ABI).
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and Path(fields[5].strip()).name.startswith(("libgomp", "libiomp5", "libomp")):
                    return fields[5].strip()
    except OSError:
        pass
    return None


@lru_cache(maxsize=1)
def _x86_library():
    # Process-private, as for the Apple library. GCC compiles the OpenMP pragmas (clang only targets
    # LLVM's runtime); linking PyTorch's loaded runtime instead of -fopenmp's -lgomp keeps one pool.
    # Row kernels never contract (RoPE stays bitwise PyTorch's); attention accumulates with FMAs.
    runtime, compiler = _openmp_runtime(), shutil.which("gcc")
    if runtime is None or compiler is None:
        raise PackedEncodersError("native x86 CPU kernels need gcc and PyTorch's OpenMP runtime")
    directory = tempfile.TemporaryDirectory(prefix="packed-encoders-x86-")
    here, out = Path(__file__).parent, Path(directory.name)
    compile_ = [compiler, "-O3", "-march=native", "-fPIC", "-fopenmp", "-fno-trapping-math", "-fno-math-errno",
                "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas", "-c"]
    sources = (("rows.c", "-ffp-contract=off"), ("local_attention_x86.c", "-ffp-contract=fast"))
    try:
        for name, contract in sources:
            subprocess.run([*compile_, contract, str(here / name), "-o", str(out / f"{name}.o")],
                           check=True, capture_output=True, timeout=60)
        subprocess.run([compiler, "-shared", *(str(out / f"{name}.o") for name, _ in sources), runtime,
                        f"-Wl,-rpath,{Path(runtime).parent}", "-o", str(out / "native.so")],
                       check=True, capture_output=True, timeout=60)
        return _signatures(ctypes.CDLL(str(out / "native.so"))), directory
    except (OSError, subprocess.SubprocessError) as exc:
        directory.cleanup()
        raise PackedEncodersError("native x86 CPU kernels need gcc with OpenMP") from exc


class CPUAttention:
    """Engine-owned scratch. The caller serializes access; returned context is borrowed."""

    def __init__(self, backend, head_dim, half_window, threads, tile=64):
        self.tile, self.half, self.threads = tile, half_window, threads
        apple = platform.system() == "Darwin" and platform.machine() == "arm64"
        x86 = platform.system() == "Linux" and platform.machine() == "x86_64"
        native = (apple or x86) and head_dim == 64 and half_window == 64
        self.library = self.rows = None
        if backend == "native" and not native:
            raise PackedEncodersError("native CPU attention requires Apple Silicon or Linux x86-64, head_dim=64 "
                                      "and local_attention=128")
        # One library carries the row kernels and, inside the envelope, native attention. Apple's
        # loads only inside it; x86's row kernels serve any shape. "sdpa" stays pure PyTorch.
        if backend != "sdpa" and (native or x86):
            try:
                self.rows, _ = _library() if apple else _x86_library()
            except PackedEncodersError:
                if backend == "native":
                    raise
                warnings.warn("Apple attention compiler unavailable; using tiled CPU SDPA" if apple else
                              "native x86 CPU kernels unavailable (needs gcc); using tiled CPU SDPA and PyTorch ops",
                              stacklevel=3)
            self.library = self.rows if native else None
        self.backend = "native" if self.library is not None else "sdpa"
        self.scratch = self.output = None
        self.rope_buffer = self.geglu_buffer = self.norm_buffer = None
        # Private ablation switches keep the original paths available to benchmarks.
        self.fused_rope = self.rows is not None and head_dim == 64
        self.fused_geglu = self.rows is not None
        self.fused_norm = self.rows is not None
        self.native_local = self.library is not None
        # Native full-window attention beat per-sequence and grouped SDPA up to
        # 128 tokens on an Apple M4 at one and eight threads; SDPA won from 256.
        # A Zen 5 agreed (native ahead through 192, SDPA from 256).
        self.native_global = self.library is not None

    def rope_layout(self, qkv, cos, sin, heads, dim):
        """Internal FP32 conversion; native results borrow engine-owned storage.

        QKV is contiguous [1, tokens, 3*heads*dim]; gathered cosine/sine are
        contiguous [1, 1, tokens, dim]. Native engine preparation ensures dim=64.
        The caller holds the engine lock through consumption of the results.
        """
        tokens = qkv.shape[1]
        if self.fused_rope:
            shape = (3, 1, heads, tokens, dim)
            if self.rope_buffer is None or self.rope_buffer.shape != shape:
                self.rope_buffer = torch.empty(shape, dtype=torch.float32)
            status = self.rows.pe_rope_layout(
                qkv.data_ptr(), cos.data_ptr(), sin.data_ptr(), self.rope_buffer.data_ptr(),
                tokens, heads, self.threads,
            )
            if status:
                raise PackedEncodersError("native CPU RoPE/layout failed")
            return self.rope_buffer.unbind(0)
        q, k, v = (t.transpose(1, 2) for t in qkv.view(1, tokens, 3, heads, dim).unbind(2))
        def rotate(t):
            a, b = t.chunk(2, -1)
            return t * cos + torch.cat((-b, a), -1) * sin
        return rotate(q), rotate(k), v

    def geglu(self, hidden):
        """Exact-erf GELU(activation) * gate; native results borrow engine-owned storage.

        Hidden is the contiguous [1, tokens, 2*width] MLP-input projection. The
        native rational erf is not bitwise PyTorch's; both are FP32-accurate.
        The caller holds the engine lock through consumption of the result.
        """
        if not self.fused_geglu:
            a, gate = hidden.chunk(2, -1)
            return F.gelu(a, approximate="none") * gate
        tokens, width = hidden.shape[1], hidden.shape[2] // 2
        if self.geglu_buffer is None or self.geglu_buffer.shape != (1, tokens, width):
            self.geglu_buffer = torch.empty((1, tokens, width), dtype=torch.float32)
        status = self.rows.pe_geglu(hidden.data_ptr(), self.geglu_buffer.data_ptr(), tokens, width, self.threads)
        if status:
            raise PackedEncodersError("native CPU GeGLU failed")
        return self.geglu_buffer

    def add_norm(self, x, branch, norm, *, owned=False):
        """Residual stream ``x + branch`` and ``norm`` of it, as ``(stream, normed)``.

        Both inputs are contiguous [1, tokens, hidden]. The native path updates
        ``x`` in place (bitwise PyTorch's add) and normalizes plain LayerNorms;
        its statistics are not bitwise PyTorch's, but FP32-accurate. Unless
        ``owned``, the normalized result borrows engine-owned storage and the
        caller holds the engine lock through its consumption.
        """
        if (not self.fused_norm or type(norm) is not nn.LayerNorm or norm.weight is None or "forward" in norm.__dict__
                or x.shape[2] % 32):
            x = x + branch
            return x, norm(x)
        shape = x.shape
        if owned:
            out = torch.empty(shape, dtype=torch.float32)
        else:
            if self.norm_buffer is None or self.norm_buffer.shape != shape:
                self.norm_buffer = torch.empty(shape, dtype=torch.float32)
            out = self.norm_buffer
        status = self.rows.pe_add_layer_norm(
            x.data_ptr(), branch.data_ptr(), norm.weight.data_ptr(), None if norm.bias is None else norm.bias.data_ptr(),
            out.data_ptr(), shape[1], shape[2], norm.eps, self.threads,
        )
        if status:
            raise PackedEncodersError("native CPU residual LayerNorm failed")
        return x, out

    def local(self, q, k, v, lengths, boundaries):
        if not self.native_local:
            return self._sdpa(q, k, v, lengths, local=True)
        return self._native(q, k, v, lengths, boundaries, self.half)

    def global_(self, q, k, v, lengths, boundaries):
        if self.native_global and max(lengths) <= 128:
            return self._native(q, k, v, lengths, boundaries, max(lengths))
        return self._sdpa(q, k, v, lengths, local=False)

    def _native(self, q, k, v, lengths, boundaries, half):
        # Keys within `half` positions; at least the longest length is global.
        q, k, v = (t.contiguous() for t in (q, k, v))
        _, heads, tokens, dim = q.shape
        workers = min(self.threads, heads * len(lengths))
        # Scores [tile, columns] per worker; x86 also transposes the key window, padded to 16 keys.
        columns = -(-min(self.tile + 2 * half, max(lengths)) // 16) * 16
        needed = workers * (self.tile + 64) * columns
        if self.scratch is None or self.scratch.numel() < needed:
            self.scratch = torch.empty(needed, dtype=torch.float32)
        if self.output is None or self.output.shape != q.shape:
            self.output = torch.empty_like(q)
        status = self.library.pe_local_attention(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), self.output.data_ptr(), boundaries.data_ptr(),
            tokens, heads, len(lengths), self.tile, half, workers, self.scratch.data_ptr(), self.scratch.numel(),
        )
        if status:
            raise PackedEncodersError(f"native CPU attention failed: {status}")
        return self.output

    def _sdpa(self, q, k, v, lengths, *, local):
        out = torch.empty_like(q)
        offset = 0
        for length in lengths:
            for start in range(0, length, self.tile if local else length):
                end = min(length, start + self.tile) if local else length
                lo, hi = (max(0, start-self.half), min(length, end+self.half)) if local else (0, length)
                mask = None
                if local:
                    mask = (torch.arange(start, end)[:, None] - torch.arange(lo, hi)).abs() <= self.half
                out[:, :, offset+start:offset+end] = F.scaled_dot_product_attention(
                    q[:, :, offset+start:offset+end], k[:, :, offset+lo:offset+hi],
                    v[:, :, offset+lo:offset+hi], attn_mask=mask,
                )
            offset += length
        return out

    def close(self):
        self.output = self.scratch = None
        self.rope_buffer = self.geglu_buffer = self.norm_buffer = None
