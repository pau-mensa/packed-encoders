"""Native packed GatedDeltaNet; no CUDA, Triton or FLA imports."""
import ctypes
from functools import lru_cache
from pathlib import Path
import platform
import subprocess

import torch
import torch.nn.functional as F

from packed_encoders.cpu.build import shared_library
from packed_encoders.errors import PackedEncodersError


@lru_cache(maxsize=1)
def library():
    here = Path(__file__).parent
    apple = platform.system() == "Darwin"
    runtime = None
    if apple:
        name = "qwen35.dylib"
        command = ["xcrun", "clang", "-O3", "-ffp-contract=off", "-dynamiclib", "-Wall", "-Wextra", "-Werror",
                   str(here / "qwen35.c"), str(here / "pool.c")]
    else:
        from packed_encoders.cpu.attention import _openmp_runtime
        runtime = _openmp_runtime()
        if runtime is None:
            raise PackedEncodersError("native Qwen3.5 needs gcc and PyTorch's OpenMP runtime")
        name = "qwen35.so"
        command = ["gcc", "-O3", "-ffp-contract=off", "-fPIC", "-fopenmp", "-shared", "-Wall", "-Wextra", "-Werror",
                   str(here / "qwen35.c"), runtime, f"-Wl,-rpath,{Path(runtime).parent}", "-lm"]
    try:
        path, directory = shared_library(name, [[*command, "-o", "{build}/" + name]], runtime=runtime)
        lib = ctypes.CDLL(str(path))
    except (OSError, subprocess.SubprocessError) as exc:
        raise PackedEncodersError("native Qwen3.5 CPU kernels need a C compiler (Xcode tools on macOS, gcc on Linux)") from exc
    lib.pe_qwen35_gdn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 7 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    lib.pe_qwen35_gdn.restype = ctypes.c_int
    if apple:
        lib.pe_pool.argtypes, lib.pe_pool.restype = [ctypes.c_int] * 3, ctypes.c_void_p
        lib.pe_pool_free.argtypes, lib.pe_pool_free.restype = [ctypes.c_void_p], None
    return lib, directory


class GatedDeltaNet:
    def __init__(self, threads):
        from packed_encoders.cpu.attention import WorkerPool
        self.library, self._directory = library()
        self.threads = min(threads, 16)
        self.pool = WorkerPool(self.library, self.threads) if platform.system() == "Darwin" and threads > 1 else None
        self.scratch = None

    def __call__(self, projection, layer, cu):
        g = layer.mixer
        nk, nv, hk, hv = g.num_k_heads, g.num_v_heads, g.head_k_dim, g.head_v_dim
        needed = self.threads * (hk * hv + 2 * hk + 2 * hv)
        if self.scratch is None or self.scratch.numel() < needed:
            self.scratch = torch.empty(needed, dtype=torch.float32)
        out = torch.empty(projection.shape[0], nv * hv, dtype=torch.float32)
        status = self.library.pe_qwen35_gdn(
            projection.data_ptr(), g.conv1d.weight.data_ptr(), None if g.conv1d.bias is None else g.conv1d.bias.data_ptr(),
            g.A_log.data_ptr(), g.dt_bias.data_ptr(), g.norm.weight.data_ptr(), cu.data_ptr(), out.data_ptr(),
            self.scratch.data_ptr(), cu.numel() - 1, nk, nv, hk, hv, g.conv_kernel_size, projection.shape[1],
            g.norm.variance_epsilon, self.threads, None if self.pool is None else self.pool.handle)
        if status:
            raise PackedEncodersError(f"native Qwen3.5 GatedDeltaNet failed: {status}")
        return out

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        self.scratch = None


def reference_gdn(projection, layer, lengths):
    """Independent torch recurrence, also used when a native compiler is unavailable."""
    g = layer.mixer
    nk, nv, hk, hv = g.num_k_heads, g.num_v_heads, g.head_k_dim, g.head_v_dim
    kd, vd = nk * hk, nv * hv
    chunks, offset = [], 0
    for length in lengths:
        p = projection[offset:offset + length]
        qkv = F.silu(F.conv1d(p[:, :2 * kd + vd].T[None], g.conv1d.weight, g.conv1d.bias,
                             padding=g.conv_kernel_size - 1, groups=2 * kd + vd)[0, :, :length].T)
        q, k, v = qkv.split((kd, kd, vd), -1)
        q, k = (x.reshape(length, nk, hk).repeat_interleave(nv // nk, 1) for x in (q, k))
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) / hk**0.5
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        v = v.reshape(length, nv, hv)
        z = p[:, 2 * kd + vd:2 * kd + 2 * vd].reshape(length, nv, hv)
        beta = p[:, 2 * kd + 2 * vd:2 * kd + 2 * vd + nv].sigmoid()
        decay = (-g.A_log.exp() * F.softplus(p[:, 2 * kd + 2 * vd + nv:2 * kd + 2 * vd + 2 * nv] + g.dt_bias)).exp()
        state = projection.new_zeros(nv, hk, hv)
        out = []
        for t in range(length):
            state = state * decay[t, :, None, None]
            delta = (v[t] - (state * k[t, :, :, None]).sum(-2)) * beta[t, :, None]
            state = state + k[t, :, :, None] * delta[:, None]
            out.append((state * q[t, :, :, None]).sum(-2))
        chunks.append(g.norm(torch.stack(out), z).reshape(length, vd))
        offset += length
    return torch.cat(chunks)
