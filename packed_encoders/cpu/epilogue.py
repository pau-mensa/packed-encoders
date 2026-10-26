"""Native CPU projection preparation for the shared epilogue composition API."""

import warnings

import torch

from packed_encoders.errors import PackedEncodersError
from packed_encoders.runtime.epilogue import PreparationContext, PreparedLinear


class CPUPreparation(PreparationContext):
    input_dtypes = (torch.float32,)

    def _linear_dtype(self, projection, precision):
        if precision not in (None, "fp32"):
            raise PackedEncodersError("CPU epilogue projections currently support precision='fp32' only")
        return torch.float32

    def _make_linear(self, weight, bias):
        return _CPULinear(self, weight, bias)


class _CPULinear(PreparedLinear):
    def __init__(self, context, weight, bias):
        super().__init__(context, weight, bias)
        self._sme = None
        self._tiles = {}
        runtime = context._runtime
        cpu = runtime.attention
        try:
            if runtime.prepack_weights and cpu.rows is not None and runtime.requested_backend != "sdpa":
                from packed_encoders.cpu.sme import FP32Projections, sme2_available

                if sme2_available():
                    try:
                        self._sme = FP32Projections([(weight, bias)], cpu.threads, cpu.pool)
                    except PackedEncodersError as exc:
                        warnings.warn(str(exc), stacklevel=3)
            if runtime.tune_gemm:
                self._tiles = cpu.tune_linear([(weight, bias)], runtime.max_tokens, shared=False)
        except BaseException:
            self._close()
            raise

    def _execute_owned(self, x):
        cpu = self._context._runtime.attention
        out = None if self._sme is None else self._sme.linear(x, self._weight, self._bias, cpu.threads)
        # F.linear already owns its result; native GEMM writes directly into an
        # owned tensor. Only SME's reusable output still needs a copy here.
        return (cpu.linear(x, self._weight, self._bias, tiles=self._tiles, owned=True)
                if out is None else out.clone())

    def _close(self):
        if self._sme is not None:
            self._sme.close()
            self._sme = None
        self._tiles.clear()
        super()._close()
