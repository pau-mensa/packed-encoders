"""Prepared FP32 projections on Apple CPUs with SME2."""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import platform
import subprocess
import tempfile
import weakref

import torch

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
    directory = tempfile.TemporaryDirectory(prefix="packed-encoders-fp32-sme-")
    output = Path(directory.name) / "fp32.dylib"
    try:
        subprocess.run(
            # Every SME2 Mac runs macOS 15 or later; the pool parks on os_sync_wait_on_address (14.4).
            ["xcrun", "clang", "-O3", "-march=armv8-a+sme2", "-mmacosx-version-min=15.0", "-Wall", "-Wextra",
             "-Werror", "-dynamiclib", str(Path(__file__).with_name("fp32_sme.c")), "-o", str(output)],
            check=True, capture_output=True, timeout=60,
        )
        lib = ctypes.CDLL(str(output))
    except (OSError, subprocess.SubprocessError) as exc:
        directory.cleanup()
        raise PackedEncodersError("FP32 SME GEMM unavailable; using Accelerate "
                                  "(needs Xcode Command Line Tools with SME2 support)") from exc
    lib.pe_sme_fp32_available.argtypes, lib.pe_sme_fp32_available.restype = [], ctypes.c_int
    if not lib.pe_sme_fp32_available():
        directory.cleanup()
        raise PackedEncodersError("FP32 SME needs 512-bit streaming vectors; using Accelerate")
    lib.pe_sme_fp32_linear.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_size_t]
    lib.pe_sme_fp32_linear.restype = ctypes.c_int
    lib.pe_sme_fp32_pool.argtypes, lib.pe_sme_fp32_pool.restype = [ctypes.c_int] * 2, ctypes.c_void_p
    lib.pe_sme_fp32_pool_free.argtypes, lib.pe_sme_fp32_pool_free.restype = [ctypes.c_void_p], None
    lib.pe_sme_fp32_pool_threads.argtypes, lib.pe_sme_fp32_pool_threads.restype = [], ctypes.c_int
    return lib, directory


class FP32Projections:
    """Engine-owned weight copies, reusable buffers and worker pool; original parameters stay intact."""

    # One token favors Accelerate. The upper crossover depends on the thread budget: on the
    # M4 Pro, SME led through 128 rows on one thread, 1,024 on two and 4,096 on four or eight.
    min_rows, max_rows = 2, 128
    worker_limit = 8
    # Idle pool workers spin this long before parking. Parked workers wake on one P cluster,
    # so small budgets spin to keep both SME units; when the spinning workers and the row
    # kernels' GCD workers would not fit on the performance cores together, workers park.
    spin_us = 200

    def __init__(self, projections, threads=1):
        self.library, _ = library()
        self.max_rows = 128 if threads == 1 else 1024 if threads < 4 else 4096
        self.weights, self.outputs = {}, {}
        self.scratch = None
        self.workers = min(threads, self.worker_limit)
        self.pool = None
        if self.workers > 1:
            spin = self.spin_us if 2 * threads - 1 <= (_sysctl_int(b"hw.perflevel0.logicalcpu") or 0) else 0
            self.pool = self.library.pe_sme_fp32_pool(self.workers, spin)
            if not self.pool:
                raise PackedEncodersError("FP32 SME worker pool unavailable")
            self._release = weakref.finalize(self, self.library.pe_sme_fp32_pool_free, self.pool)
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
        status = self.library.pe_sme_fp32_linear(self.pool, x.data_ptr(), packed.data_ptr(), out.data_ptr(),
                                                self.scratch.data_ptr(), m, n, k, min(threads, self.workers),
                                                self.scratch.numel())
        if status:
            raise PackedEncodersError(f"native SME2 FP32 projection failed: {status}")
        return out

    def close(self):
        if self.pool is not None:
            self._release()
            self.pool = None
        self.weights.clear()
        self.outputs.clear()
        self.scratch = None
