"""CUDA operation preparation for the shared epilogue composition API."""

import torch
import torch.nn.functional as F

from packed_encoders.errors import PackedEncodersError
from packed_encoders.pieces.numerical import LINEAR
from packed_encoders.runtime.epilogue import EpilogueSupport, PreparationContext, PreparedLinear


class CUDAEpilogueSupport(EpilogueSupport):
    def _epilogue_context(self):
        return CUDAPreparation(self)


class CUDAPreparation(PreparationContext):
    def __init__(self, runtime):
        super().__init__(runtime)
        if self.device.type != "cuda":
            raise PackedEncodersError("CUDA epilogues require a CUDA backbone")

    def _linear_dtype(self, projection, precision):
        dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        if precision is None:
            dtype = projection.weight.dtype
        elif precision in dtypes:
            dtype = dtypes[precision]
        else:
            raise PackedEncodersError("CUDA epilogue precision must be None, 'fp32', 'bf16' or 'fp16'")
        if dtype not in self.input_dtypes:
            raise PackedEncodersError("CUDA epilogue weights must be FP32, BF16 or FP16")
        return dtype

    def _make_linear(self, weight, bias):
        # Existing pieces declare bias-free BF16 CUDA semantics. Keep other
        # supported head dtypes and biased projections on cuBLAS via F.linear.
        piece = self._runtime.composition.linear
        execute = piece.execute if weight.dtype == torch.bfloat16 and bias is None and piece.contract == LINEAR else None
        return _CUDALinear(self, weight, bias, execute)


class _CUDALinear(PreparedLinear):
    def __init__(self, context, weight, bias, execute):
        super().__init__(context, weight, bias)
        self._linear = execute

    def _execute(self, x):
        # FP32 retriever heads can consume BF16 hidden states without changing
        # either module's parameters. Compute/output dtype follows the head.
        x = x.to(self._weight.dtype)
        if self._linear is None:
            return F.linear(x, self._weight, self._bias)
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        return self._linear(flat, self._weight).reshape(*x.shape[:-1], self._weight.shape[0])

    def _execute_owned(self, x):
        output = self._execute(x)
        # cuBLAS through F.linear allocates its result. Pieces may return scratch.
        return output if self._linear is None else output.clone()

    def _close(self):
        super()._close()
        self._linear = None
