"""Bounded CPU attention with an optional Apple Accelerate implementation."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import platform
import subprocess
import tempfile
import warnings

import torch
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
        library = ctypes.CDLL(str(path))
        fn = library.pe_local_attention
        fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_void_p, ctypes.c_size_t]
        fn.restype = ctypes.c_int
        rope = library.pe_rope_layout
        rope.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
        rope.restype = ctypes.c_int
        return library, directory
    except (OSError, subprocess.SubprocessError) as exc:
        directory.cleanup()
        raise PackedEncodersError("Apple CPU attention needs the Xcode Command Line Tools (xcrun clang)") from exc


class CPUAttention:
    """Engine-owned scratch. The caller serializes access; returned context is borrowed."""

    def __init__(self, backend, head_dim, half_window, threads, tile=64):
        self.tile, self.half, self.threads = tile, half_window, threads
        native = platform.system() == "Darwin" and platform.machine() == "arm64" and head_dim == 64 and half_window == 64
        self.library = None
        if backend == "native" and not native:
            raise PackedEncodersError("native CPU attention requires Apple Silicon, head_dim=64 and local_attention=128")
        if native and backend != "sdpa":
            try:
                self.library, _ = _library()
            except PackedEncodersError:
                if backend == "native":
                    raise
                warnings.warn("Apple attention compiler unavailable; using tiled CPU SDPA", stacklevel=3)
        self.backend = "native" if self.library is not None else "sdpa"
        self.scratch = self.output = None
        self.rope_buffer = None
        # Private ablation switch keeps the original path available to benchmarks.
        self.fused_rope = self.library is not None

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
            status = self.library.pe_rope_layout(
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

    def local(self, q, k, v, lengths, boundaries):
        if self.library is None:
            return self._sdpa(q, k, v, lengths, local=True)
        q, k, v = (t.contiguous() for t in (q, k, v))
        _, heads, tokens, dim = q.shape
        workers = min(self.threads, heads * len(lengths))
        needed = workers * self.tile * (self.tile + 128)
        if self.scratch is None or self.scratch.numel() < needed:
            self.scratch = torch.empty(needed, dtype=torch.float32)
        if self.output is None or self.output.shape != q.shape:
            self.output = torch.empty_like(q)
        status = self.library.pe_local_attention(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), self.output.data_ptr(), boundaries.data_ptr(),
            tokens, heads, len(lengths), self.tile, workers, self.scratch.data_ptr(), self.scratch.numel(),
        )
        if status:
            raise PackedEncodersError(f"native CPU attention failed: {status}")
        return self.output

    def global_(self, q, k, v, lengths):
        return self._sdpa(q, k, v, lengths, local=False)

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
        self.rope_buffer = None
