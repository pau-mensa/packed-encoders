"""Prepared FP32 projections for short inputs on Apple CPUs with SME2."""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import platform
import subprocess
import tempfile

import torch

from packed_encoders.errors import PackedEncodersError


def sme2_available():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    value, size = ctypes.c_int(), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
    status = ctypes.CDLL(None).sysctlbyname(b"hw.optional.arm.FEAT_SME2", ctypes.byref(value),
                                          ctypes.byref(size), None, 0)
    return status == 0 and value.value == 1


@lru_cache(maxsize=1)
def library():
    directory = tempfile.TemporaryDirectory(prefix="packed-encoders-fp32-sme-")
    output = Path(directory.name) / "fp32.dylib"
    try:
        subprocess.run(
            ["xcrun", "clang", "-O3", "-march=armv8-a+sme2", "-Wall", "-Wextra", "-Werror", "-dynamiclib",
             str(Path(__file__).with_name("fp32_sme.c")), "-o", str(output)],
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
    lib.pe_sme_fp32_linear.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_size_t]
    lib.pe_sme_fp32_linear.restype = ctypes.c_int
    return lib, directory


class FP32Projections:
    """Engine-owned weight copies and reusable buffers; original parameters stay intact."""

    # One token favors Accelerate. Larger crossovers depend on the thread budget;
    # the M4 Pro comparisons favor SME through 128 at 1/8 threads, 64 at 4, 32 at 2.
    min_rows, max_rows = 2, 128
    small_worker_limit, worker_limit = 4, 8

    def __init__(self, projections, threads=1):
        self.library, _ = library()
        self.max_rows = 128 if threads == 1 or threads >= 8 else 64 if threads >= 4 else 32
        self.weights, self.outputs = {}, {}
        self.scratch = None
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
        status = self.library.pe_sme_fp32_linear(x.data_ptr(), packed.data_ptr(), out.data_ptr(),
                                                self.scratch.data_ptr(), m, n, k,
                                                min(threads, self.worker_limit,
                                                    self.small_worker_limit if m <= 16 else self.worker_limit),
                                                self.scratch.numel())
        if status:
            raise PackedEncodersError(f"native SME2 FP32 projection failed: {status}")
        return out

    def close(self):
        self.weights.clear()
        self.outputs.clear()
        self.scratch = None
