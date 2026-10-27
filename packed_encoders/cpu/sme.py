"""Prepared FP32 projections on Apple CPUs with SME2."""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import platform
import subprocess

import torch

from packed_encoders.cpu.build import shared_library
from packed_encoders.errors import PackedEncodersError


def _sysctl_int(name):
    value, size = ctypes.c_int(), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
    status = ctypes.CDLL(None).sysctlbyname(name, ctypes.byref(value), ctypes.byref(size), None, 0)
    return value.value if status == 0 else None


def sme2_available():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    return _sysctl_int(b"hw.optional.arm.FEAT_SME2") == 1


@lru_cache(maxsize=1)
def library():
    directory = None
    try:
        output, directory = shared_library("fp32.dylib", [
            # Every SME2 Mac runs macOS 15 or later; pool.h wakes with os_sync_wake_by_address_all (14.4).
            ["xcrun", "clang", "-O3", "-march=armv8-a+sme2", "-mmacosx-version-min=15.0", "-Wall", "-Wextra",
             "-Werror", "-dynamiclib", str(Path(__file__).with_name("fp32_sme.c")), "-o", "{build}/fp32.dylib"],
        ])
        lib = ctypes.CDLL(str(output))
    except (OSError, subprocess.SubprocessError) as exc:
        if directory is not None:
            directory.cleanup()
        raise PackedEncodersError("FP32 SME GEMM unavailable; using Accelerate "
                                  "(needs Xcode Command Line Tools with SME2 support)") from exc
    lib.pe_sme_fp32_available.argtypes, lib.pe_sme_fp32_available.restype = [], ctypes.c_int
    if not lib.pe_sme_fp32_available():
        if directory is not None:
            directory.cleanup()
        raise PackedEncodersError("FP32 SME needs 512-bit streaming vectors; using Accelerate")
    lib.pe_sme_fp32_linear.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_size_t]
    lib.pe_sme_fp32_linear.restype = ctypes.c_int
    return lib, directory


class FP32Projections:
    """Engine-owned weight copies and reusable buffers; original parameters stay intact. Workers
    come from the engine's pool (attention.WorkerPool), or from GCD without one."""

    # One token favors Accelerate. The upper crossover depends on the thread budget: on the
    # M4 Pro, SME led through 128 rows on one thread, 1,024 on two and 4,096 on four or eight.
    min_rows, max_rows = 2, 128
    worker_limit = 8

    def __init__(self, projections, threads=1, pool=None):
        self.library, _ = library()
        self.max_rows = 128 if threads == 1 else 1024 if threads < 4 else 4096
        self.weights, self.outputs = {}, {}
        self.scratch = None
        self.workers, self.pool = min(threads, self.worker_limit), pool
        for weight, bias in projections:
            if (bias is not None or weight.ndim != 2 or weight.device.type != "cpu"
                    or weight.dtype != torch.float32 or not weight.is_contiguous()):
                continue
            n, k = weight.shape
            if n % 64 or k % 4 or min(n, k) <= 0 or max(n, k) > 2**31 - 1:
                continue
            try:
                version = weight._version
            except RuntimeError:  # Inference tensors have no version counter: keep reading their live storage.
                continue
            self.weights[id(weight)] = [weight, self._pack(weight), version]

    @staticmethod
    def _pack(weight):
        n, k = weight.shape
        return weight.detach().view(n // 64, 64, k).transpose(1, 2).contiguous()

    def linear(self, x, weight, bias, threads):
        """Borrowed native output, or None when this projection should use Accelerate."""
        if bias is not None or x.device.type != "cpu" or x.dtype != torch.float32 or not x.is_contiguous():
            return None
        if (weight.ndim != 2 or weight.device.type != "cpu" or weight.dtype != torch.float32
                or not weight.is_contiguous() or not x.ndim or x.shape[-1] != weight.shape[1]):
            return None
        n, k = weight.shape
        if n < 64 or n % 64 or k < 4 or k % 4 or max(n, k) > 2**31 - 1:
            return None
        m = x.numel() // k
        if not self.min_rows <= m <= self.max_rows:
            return None
        entry = self.weights.get(id(weight))
        if entry is None:
            return None
        original, packed, version = entry
        if original._version != version or packed.shape != (n // 64, k, 64):
            packed = self._pack(original)
            entry[1], entry[2] = packed, original._version
        shape = (*x.shape[:-1], n)
        out = self.outputs.get(n)
        if out is None or out.shape != shape:
            out = self.outputs[n] = torch.empty(shape, dtype=torch.float32, device="cpu")
        needed = ((m + 15) // 16) * 16 * k
        if self.scratch is None or self.scratch.numel() < needed:
            self.scratch = torch.empty(needed, dtype=torch.float32, device="cpu")
        pool = None if self.pool is None else self.pool.handle
        status = self.library.pe_sme_fp32_linear(pool, x.data_ptr(), packed.data_ptr(), out.data_ptr(),
                                                self.scratch.data_ptr(), m, n, k, min(threads, self.workers),
                                                self.scratch.numel())
        if status:
            raise PackedEncodersError(f"native SME2 FP32 projection failed: {status}")
        return out

    def close(self):
        self.pool = None
        self.weights.clear()
        self.outputs.clear()
        self.scratch = None
