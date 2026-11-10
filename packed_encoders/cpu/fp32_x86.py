"""Prepared FP32 output-channel panels for single-core AVX-512 projections."""
import ctypes

import torch

from packed_encoders.cpu.attention import _x86_library
from packed_encoders.errors import PackedEncodersError


class FP32Projections:
    """Private weight copies; ordinary in-place updates refresh them lazily."""

    def __init__(self, projections, threads=1):
        self.library, self._directory = _x86_library()
        if not hasattr(self.library, "pe_linear_packed"):
            raise PackedEncodersError("prepared FP32 projections require AVX-512")
        self.library.pe_linear_packed.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
        self.library.pe_linear_packed.restype = ctypes.c_int
        self.threads = threads
        self.weights, self.outputs = {}, {}
        for weight in projections:
            if (weight.ndim != 2 or weight.dtype != torch.float32 or weight.device.type != "cpu"
                    or not weight.is_contiguous() or min(weight.shape) <= 0 or weight.shape[0] % 64):
                continue
            try:
                version = weight._version
            except RuntimeError:
                continue
            self.weights[id(weight)] = [weight, self._pack(weight), version]

    @staticmethod
    def _pack(weight):
        n, k = weight.shape
        with torch.inference_mode(False), torch.no_grad():
            return weight.detach().view(n // 64, 64, k).transpose(1, 2).contiguous()

    def linear(self, x, weight):
        """Borrowed output, or None for inputs handled by the existing projection path."""
        entry = self.weights.get(id(weight))
        if (entry is None or x.device.type != "cpu" or x.dtype != torch.float32
                or not x.is_contiguous() or not x.ndim or x.shape[-1] != weight.shape[1]):
            return None
        m = x.numel() // weight.shape[1]
        if m < 2:
            return None
        source, packed, version = entry
        if source._version != version:
            packed = self._pack(source)
            entry[1], entry[2] = packed, source._version
        n, k = weight.shape
        shape = (*x.shape[:-1], n)
        out = self.outputs.get(n)
        if out is None or out.shape != shape:
            out = self.outputs[n] = torch.empty(shape, dtype=torch.float32, device="cpu")
        status = self.library.pe_linear_packed(
            x.data_ptr(), packed.data_ptr(), out.data_ptr(), m, n, k, self.threads)
        if status:
            raise PackedEncodersError(f"prepared FP32 projection failed: {status}")
        return out

    @property
    def packed_bytes(self):
        return sum(packed.numel() * packed.element_size() for _, packed, _ in self.weights.values())

    def close(self):
        self.weights.clear()
        self.outputs.clear()
