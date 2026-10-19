"""CPU attention/row kernels, x86 FP32 GEMM, and prepared Apple SME2 FP32 projections."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import warnings
import weakref

import torch
from torch import nn
import torch.nn.functional as F

from packed_encoders.errors import PackedEncodersError
from packed_encoders.cpu.gemm import choose_tile, l1_data_cache_bytes, tune_tiles


@lru_cache(maxsize=1)
def _library():
    # Process-private directory: no shared writable executable cache, stale ABI,
    # or concurrent compiler races. Keep the directory alive with the library.
    directory = tempfile.TemporaryDirectory(prefix="packed-encoders-cpu-")
    path, here = Path(directory.name) / "attention.dylib", Path(__file__).parent
    try:
        subprocess.run(
            ["xcrun", "clang", "-O3", "-Wall", "-Wextra", "-Werror", "-dynamiclib", "-framework", "Accelerate",
             str(here / "local_attention.c"), str(here / "pool.c"), "-o", str(path)],
            check=True, capture_output=True, timeout=30,
        )
        library = _signatures(ctypes.CDLL(str(path)))
        library.pe_pool.argtypes, library.pe_pool.restype = [ctypes.c_int] * 3, ctypes.c_void_p
        library.pe_pool_free.argtypes, library.pe_pool_free.restype = [ctypes.c_void_p], None
        library.pe_pool_threads.argtypes, library.pe_pool_threads.restype = [], ctypes.c_int
        return library, directory
    except (OSError, subprocess.SubprocessError) as exc:
        directory.cleanup()
        raise PackedEncodersError("Apple CPU attention needs the Xcode Command Line Tools (xcrun clang)") from exc


def _signatures(library):
    # Every entry point takes the worker count and then a worker pool (Apple; x86 ignores it).
    fn = library.pe_local_attention
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 7 + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
    fn.restype = ctypes.c_int
    rope = library.pe_rope_layout
    rope.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    rope.restype = ctypes.c_int
    geglu = library.pe_geglu
    geglu.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    geglu.restype = ctypes.c_int
    norm = library.pe_add_layer_norm
    norm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
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
    # Row kernels never contract (RoPE stays bitwise PyTorch's); attention and the projection GEMM
    # accumulate with FMAs.
    # The static vector helpers stay within their translation units; their AVX2 psABI warning
    # is harmless and must not turn into a build failure under -Werror.
    runtime, compiler = _openmp_runtime(), shutil.which("gcc")
    if runtime is None or compiler is None:
        raise PackedEncodersError("native x86 CPU kernels need gcc and PyTorch's OpenMP runtime")
    directory = tempfile.TemporaryDirectory(prefix="packed-encoders-x86-")
    here, out = Path(__file__).parent, Path(directory.name)
    compile_ = [compiler, "-O3", "-march=native", "-fPIC", "-fopenmp", "-fno-trapping-math", "-fno-math-errno",
                "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas", "-Wno-psabi", "-c"]
    sources = (("rows.c", "-ffp-contract=off"), ("local_attention_x86.c", "-ffp-contract=fast"),
               ("gemm_x86.c", "-ffp-contract=fast"))
    try:
        for name, contract in sources:
            subprocess.run([*compile_, contract, str(here / name), "-o", str(out / f"{name}.o")],
                           check=True, capture_output=True, timeout=60)
        subprocess.run([compiler, "-shared", *(str(out / f"{name}.o") for name, _ in sources), runtime,
                        f"-Wl,-rpath,{Path(runtime).parent}", "-o", str(out / "native.so")],
                       check=True, capture_output=True, timeout=60)
        library = _signatures(ctypes.CDLL(str(out / "native.so")))
        if hasattr(library, "pe_linear"):
            library.pe_linear.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
            library.pe_linear.restype = ctypes.c_int
            library.pe_linear_tile.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6
            library.pe_linear_tile.restype = ctypes.c_int
        return library, directory
    except (OSError, subprocess.SubprocessError) as exc:
        directory.cleanup()
        raise PackedEncodersError("native x86 CPU kernels need gcc with OpenMP") from exc


class WorkerPool:
    """Native worker threads owned by one engine (pool.h); ``threads`` counts the calling thread."""

    # Idle workers spin this long before parking: parked workers wake on the caller's P cluster
    # (one SME unit). Budgets beyond the performance cores park at once, so spinning threads
    # never take a core from working ones.
    spin_us = 200

    def __init__(self, library, threads):
        from packed_encoders.cpu.sme import _sysctl_int
        spin = self.spin_us if threads <= (_sysctl_int(b"hw.perflevel0.logicalcpu") or 0) else 0
        # Without a second performance level, perflevel1 is absent and no core counts as efficiency.
        self.handle = library.pe_pool(min(threads, 16), spin, _sysctl_int(b"hw.perflevel1.logicalcpu") or 0)
        if not self.handle:
            raise PackedEncodersError("native CPU worker pool unavailable")
        self._release = weakref.finalize(self, library.pe_pool_free, self.handle)

    def close(self):
        if self.handle is not None:
            self._release()
            self.handle = None


class ResolvedNorm:
    """A norm module resolved once for add_norm. A plain affine LayerNorm also keeps aliases of
    its parameters: they share the module's storage, so in-place updates are read, while
    pointers handed to native code stay valid whatever later happens to the module."""

    __slots__ = ("module", "weight", "bias", "eps")

    def __init__(self, norm):
        self.module, self.weight, self.bias, self.eps = norm, None, None, None
        if type(norm) is nn.LayerNorm and norm.weight is not None and "forward" not in norm.__dict__:
            self.weight, self.eps = norm.weight.detach(), norm.eps
            self.bias = None if norm.bias is None else norm.bias.detach()


class CPUAttention:
    """Engine-owned scratch. The caller serializes access; returned context is borrowed."""

    def __init__(self, backend, head_dim, half_window, threads, tile=64):
        self.tile, self.half, self.threads = tile, half_window, threads
        apple = platform.system() == "Darwin" and platform.machine() == "arm64"
        x86 = platform.system() == "Linux" and platform.machine() == "x86_64"
        native = (apple or x86) and head_dim == 64 and half_window == 64
        self.library = self.rows = self.pool = None
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
        self.linear_buffers = {}
        # Private ablation switches keep the original paths available to benchmarks.
        self.fused_rope = self.rows is not None and head_dim == 64
        self.fused_geglu = self.rows is not None
        self.fused_norm = self.rows is not None
        self.native_local = self.library is not None
        # Native full-window attention beat per-sequence and grouped SDPA up to
        # 128 tokens on an Apple M4 at one and eight threads; SDPA won from 256.
        # A Zen 5 agreed (native ahead through 192, SDPA from 256).
        self.native_global = self.library is not None
        # Native attention stores token-major context for callers that ask for it, instead
        # of a head-major result they would transpose into a copy.
        self.token_major = self.library is not None
        # x86 projections: MKL does not pick its AVX-512 kernels on AMD; the native GEMM reads the
        # original weights (unprepared Apple projections keep Accelerate through F.linear).
        self.native_linear = self.rows is not None and x86 and hasattr(self.rows, "pe_linear")
        self.linear_l1_bytes = l1_data_cache_bytes() if self.native_linear else None
        self.linear_tiles, self.linear_tuning = {}, {}
        self.sme = None

    def prepare_sme(self, projections):
        from packed_encoders.cpu.sme import FP32Projections, sme2_available
        if self.rows is None or not sme2_available():
            return
        try:
            # With SME projections, every native kernel runs on one engine-owned pool.
            if self.threads > 1:
                self.pool = WorkerPool(self.rows, self.threads)
            prepared = FP32Projections(projections, self.threads, self.pool)
            self.sme = prepared if prepared.weights else None
        except PackedEncodersError as exc:
            warnings.warn(str(exc), stacklevel=3)
        if self.sme is None and self.pool is not None:
            self.pool.close()
            self.pool = None

    @property
    def _pool_handle(self):
        return None if self.pool is None else self.pool.handle

    def tune_linear(self, projections, max_tokens):
        if self.native_linear:
            choices, report = tune_tiles(self.rows.pe_linear_tile, projections, self.threads,
                                         self.linear_l1_bytes, max_tokens)
            self.linear_tiles.update(choices)
            self.linear_tuning = report

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
                tokens, heads, self.threads, self._pool_handle,
            )
            if status:
                raise PackedEncodersError("native CPU RoPE/layout failed")
            return self.rope_buffer.unbind(0)
        q, k, v = (t.transpose(1, 2) for t in qkv.view(1, tokens, 3, heads, dim).unbind(2))
        def rotate(t):
            a, b = t.chunk(2, -1)
            return t * cos + torch.cat((-b, a), -1) * sin
        return rotate(q), rotate(k), v

    def linear(self, x, weight, bias=None):
        """FP32 x @ weight.T + bias; native results borrow engine-owned storage.

        Weight [N, K] and bias are contiguous FP32 (the engine's parameter aliases). Activations
        that are not contiguous FP32 [..., K] take F.linear, so native code never reads or writes
        past a tensor. One buffer per output width: the caller holds the engine lock and consumes
        each result before the next projection of that width (requests never fault in fresh GEMM
        outputs).
        """
        if self.sme is not None:
            out = self.sme.linear(x, weight, bias, self.threads)
            if out is not None:
                return out
        n, k = weight.shape
        if not self.native_linear or x.dtype != torch.float32 or x.shape[-1] != k or not x.is_contiguous():
            return F.linear(x, weight, bias)
        shape = (*x.shape[:-1], n)
        out = self.linear_buffers.get(n)
        if out is None or out.shape != shape:
            out = self.linear_buffers[n] = torch.empty(shape, dtype=torch.float32)
        m = x.numel() // k
        key = (m, n, k)
        tile = self.linear_tiles.get(key)
        if tile is None:
            # Residency is unchanged once M fills the largest (4-row) tile.
            # Bound the cache across varying request lengths; exact keys hold tuned choices.
            key = (min(m, 4), n, k)
            tile = self.linear_tiles.get(key)
            if tile is None:
                tile = self.linear_tiles[key] = choose_tile(m, k, self.linear_l1_bytes)
        status = self.rows.pe_linear_tile(x.data_ptr(), weight.data_ptr(), None if bias is None else bias.data_ptr(),
                                          out.data_ptr(), m, n, k, self.threads, *tile)
        if status:
            raise PackedEncodersError("native CPU projection failed")
        return out

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
        status = self.rows.pe_geglu(hidden.data_ptr(), self.geglu_buffer.data_ptr(), tokens, width, self.threads,
                                    self._pool_handle)
        if status:
            raise PackedEncodersError("native CPU GeGLU failed")
        return self.geglu_buffer

    def add_norm(self, x, branch, norm, *, owned=False):
        """Residual stream ``x + branch`` and ``norm`` of it, as ``(stream, normed)``.

        Both inputs are contiguous [1, tokens, hidden]; ``norm`` is a module or
        its ResolvedNorm. The native path updates ``x`` in place (bitwise
        PyTorch's add) and normalizes plain LayerNorms; its statistics are not
        bitwise PyTorch's, but FP32-accurate. Unless ``owned``, the normalized
        result borrows engine-owned storage and the caller holds the engine lock
        through its consumption.
        """
        if type(norm) is not ResolvedNorm:
            norm = ResolvedNorm(norm)
        if not self.fused_norm or norm.weight is None or x.shape[2] % 32:
            x = x + branch
            return x, norm.module(x)
        shape = x.shape
        if owned:
            out = torch.empty(shape, dtype=torch.float32)
        else:
            if self.norm_buffer is None or self.norm_buffer.shape != shape:
                self.norm_buffer = torch.empty(shape, dtype=torch.float32)
            out = self.norm_buffer
        status = self.rows.pe_add_layer_norm(
            x.data_ptr(), branch.data_ptr(), norm.weight.data_ptr(), None if norm.bias is None else norm.bias.data_ptr(),
            out.data_ptr(), shape[1], shape[2], norm.eps, self.threads, self._pool_handle,
        )
        if status:
            raise PackedEncodersError("native CPU residual LayerNorm failed")
        return x, out

    def local(self, q, k, v, lengths, boundaries, *, token_major=False):
        """Context as head-major [1, heads, tokens, dim], or [1, tokens, heads * dim] if token_major."""
        if not self.native_local:
            return self._layout(self._sdpa(q, k, v, lengths, local=True), token_major)
        return self._native(q, k, v, lengths, boundaries, self.half, token_major)

    def global_(self, q, k, v, lengths, boundaries, *, token_major=False):
        if self.native_global and max(lengths) <= 128:
            return self._native(q, k, v, lengths, boundaries, max(lengths), token_major)
        return self._layout(self._sdpa(q, k, v, lengths, local=False), token_major)

    @staticmethod
    def _layout(context, token_major):
        return context.transpose(1, 2).flatten(2) if token_major else context

    def _native(self, q, k, v, lengths, boundaries, half, token_major=False):
        # Keys within `half` positions; at least the longest length is global.
        q, k, v = (t.contiguous() for t in (q, k, v))
        out, args = self._native_args(q, k, v, lengths, boundaries, half, token_major)
        status = self.library.pe_local_attention(*args)
        if status:
            raise PackedEncodersError(f"native CPU attention failed: {status}")
        return out if not token_major or self.token_major else self._layout(out, True)

    def _native_args(self, q, k, v, lengths, boundaries, half, token_major=False):
        """Prepare a call on contiguous Q/K/V; pointers borrow the tensors and engine buffers.

        The caller keeps inputs alive and consumes the plan before any buffer reallocation.
        Multiple windows on the same input can share storage if the widest is prepared first.
        """
        _, heads, tokens, dim = q.shape
        workers = min(self.threads, heads * len(lengths))
        # Scores [tile, columns] per worker; x86 also transposes the key window, padded to 16 keys.
        columns = -(-min(self.tile + 2 * half, max(lengths)) // 16) * 16
        needed = workers * (self.tile + 64) * columns
        if self.scratch is None or self.scratch.numel() < needed:
            self.scratch = torch.empty(needed, dtype=torch.float32)
        direct = token_major and self.token_major
        shape = (1, tokens, heads * dim) if direct else q.shape
        if self.output is None or self.output.shape != shape:
            self.output = torch.empty(shape, dtype=torch.float32)
        args = (
            q.data_ptr(), k.data_ptr(), v.data_ptr(), self.output.data_ptr(), boundaries.data_ptr(),
            tokens, heads, len(lengths), self.tile, half, direct, workers, self._pool_handle,
            self.scratch.data_ptr(), self.scratch.numel(),
        )
        return self.output, args

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
        if self.sme is not None:
            self.sme.close()
            self.sme = None
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        self.output = self.scratch = None
        self.rope_buffer = self.geglu_buffer = self.norm_buffer = None
        self.linear_buffers = {}
        self.linear_tiles.clear()
        self.linear_tuning.clear()
