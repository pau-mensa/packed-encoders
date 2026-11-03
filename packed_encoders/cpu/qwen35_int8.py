"""Per-channel INT8 weights and per-token INT8 activations, with outlier-spreading rotation."""
import ctypes

import torch

from packed_encoders.cpu.int8 import Int8Unavailable, library, pack_weights, rotate, signs, sme2_available, vnni_available
from packed_encoders.errors import PackedEncodersError, UnsupportedTargetError


class Int8Projections:
    def __init__(self, projections, threads, pool=None):
        self.x86 = vnni_available()
        if not (self.x86 or sme2_available()):
            raise Int8Unavailable("Qwen3.5 int8 needs Apple SME2 or Linux x86 AVX512-VNNI")
        self.library, self._directory = library()
        self.library.pe_q8_linear.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        self.library.pe_q8_linear.restype = ctypes.c_int
        self.workers, self.pool = min(threads, 16), pool
        self.prepared, self.signs = {}, {}
        self.projections = tuple(projections)
        self.sources = tuple(p for projection in projections for p in projection.parameters)
        try:
            self.versions = tuple(p._version for p in self.sources)
        except RuntimeError as exc:
            raise UnsupportedTargetError("INT8 requires parameter version counters; pack outside inference_mode()") from exc
        with torch.no_grad():
            for projection in projections:
                weight = projection.weight
                n, k = weight.shape
                if n % 64 or k % 64:
                    raise UnsupportedTargetError("Qwen3.5 int8 requires projection widths divisible by 64")
                block = 128 if k % 128 == 0 else 64
                if k not in self.signs:
                    self.signs[k] = signs(k, block).float()
                # Rotation is orthogonal, shared with activation quantization in the native kernel.
                w = rotate(weight, k, block).float()
                scale = w.abs().amax(1).clamp_min(1e-30) / 127
                q = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
                correction = 128 * q.sum(1, dtype=torch.int32) if self.x86 else None
                self.prepared[id(projection)] = (pack_weights(q, 16 if self.x86 else 64), scale, correction, block)
        self.a = self.scales = self.scratch = self.products = None

    def check_versions(self):
        if tuple(p._version for p in self.sources) != self.versions:
            raise PackedEncodersError("CPU int8 projection weights changed; unpack and repack")

    def linear(self, x, projection):
        w, scale, correction, block = self.prepared[id(projection)]
        m, k = x.shape
        n = projection.shape[0]
        def reserve(tensor, count, dtype):
            return torch.empty(count, dtype=dtype) if tensor is None or tensor.numel() < count else tensor
        self.a = reserve(self.a, ((m + 15) // 16) * 16 * k, torch.int8)
        self.scales = reserve(self.scales, m, torch.float32)
        self.scratch = reserve(self.scratch, self.workers * k, torch.float32)
        self.products = reserve(self.products, m * n if self.x86 else self.workers * 32 * 64, torch.int32)
        out = torch.empty(m, n, dtype=torch.float32)
        status = self.library.pe_q8_linear(
            x.data_ptr(), w.data_ptr(), scale.data_ptr(), None if correction is None else correction.data_ptr(),
            self.signs[k].data_ptr(), out.data_ptr(), self.a.data_ptr(), self.scales.data_ptr(), self.scratch.data_ptr(),
            self.products.data_ptr(), m, n, k, block, self.workers, None if self.pool is None else self.pool.handle)
        if status:
            raise PackedEncodersError(f"native Qwen3.5 int8 projection failed: {status}")
        return out

    def close(self):
        self.prepared.clear()
        self.signs.clear()
        self.projections = self.sources = ()
        self.a = self.scales = self.scratch = self.products = self.pool = None
