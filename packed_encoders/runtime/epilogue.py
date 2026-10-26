"""Shared inference-head composition, validation and resource ownership.

Engines supply a preparation context and optionally override _forward_epilogue
to apply a head at internal chunk boundaries. Backends supply only linear dtype
policy and execution; neither duplicates the composition or lifecycle machinery.
"""

from __future__ import annotations

from contextlib import nullcontext
import threading

import torch
from torch import nn

from packed_encoders.errors import PackedEncodersError


def require_inference(device):
    if torch.is_grad_enabled() or torch.is_autocast_enabled(device.type):
        raise PackedEncodersError("epilogues require no_grad/inference_mode and no autocast")


class EpilogueSupport:
    """Opt-in engine mixin; initialize at preparation and close before teardown.

    Requires binding, _require_open(), forward_packed() and a backend-specific
    _epilogue_context(). The ordinary engine's forward and capabilities stay intact.
    """

    def _init_epilogues(self, *, lock=None):
        self._epilogue_lock = lock if lock is not None else threading.RLock()
        self._epilogues = []

    def _epilogue_context(self):
        raise NotImplementedError

    def with_epilogue(self, epilogue):
        """Prepare epilogue.prepare(ops) -> forward(hidden, batch) once.

        Return an inference-only packed encoder. Heads preserve token count/order
        and are closed with the backbone. See packed_encoders.epilogues.
        """
        with self._epilogue_lock:
            self._require_open()
            context = self._epilogue_context()
            try:
                forward = epilogue.prepare(context)
                if not callable(forward):
                    raise PackedEncodersError("epilogue.prepare(ops) must return a callable")
                context._prepared = True
                composed = PackedEpilogue(self, context, forward)
                self._epilogues.append(composed)
                return composed
            except BaseException:
                context._close()
                raise

    def _forward_epilogue(self, batch, apply):
        # Preserve existing graph/eager dispatch. The head runs after replay,
        # outside the backbone graph, on the same CUDA stream.
        return apply(self.forward_packed(batch), batch)

    def _close_epilogues(self):
        with self._epilogue_lock:
            for epilogue in tuple(self._epilogues):
                epilogue.close()


class PreparationContext:
    """Backend extension point: resolve projection dtype and build its operation."""

    input_dtypes = (torch.float32, torch.float16, torch.bfloat16)

    def __init__(self, runtime):
        self._runtime = runtime
        self.device = next(runtime.binding.weight_source.parameters()).device
        self._operations = []
        self._closed = self._prepared = False

    def _check(self):
        self._runtime._require_open()
        if self._closed:
            raise PackedEncodersError("this epilogue has been closed")

    def _device_context(self):
        return torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()

    def _linear_dtype(self, projection, precision):
        raise NotImplementedError

    def _make_linear(self, weight, bias):
        raise NotImplementedError

    def prepare_linear(self, projection: nn.Linear, *, precision=None):
        """Prepare a plain linear with owned outputs and live parameter aliases.

        precision=None selects CPU FP32 or CUDA head weight dtype. Explicit
        precision must match the supplied weights; parameters are never converted.
        """
        with self._runtime._epilogue_lock:
            self._check()
            if self._prepared:
                raise PackedEncodersError("prepare_linear is only available during epilogue preparation")
            if (type(projection) is not nn.Linear or "forward" in projection.__dict__
                    or projection._forward_hooks or projection._forward_pre_hooks):
                raise PackedEncodersError("epilogue projections must be plain nn.Linear modules without hooks")
            dtype = self._linear_dtype(projection, precision)
            weight, bias = projection.weight, projection.bias
            if (weight.ndim != 2 or min(weight.shape) <= 0 or max(weight.shape) > 2**31 - 1
                    or (bias is not None and bias.shape != (weight.shape[0],))):
                raise PackedEncodersError("invalid epilogue projection shape")
            for parameter in (weight, bias):
                if parameter is not None and (parameter.device != self.device or parameter.dtype != dtype
                                              or parameter.layout != torch.strided or not parameter.is_contiguous()):
                    raise PackedEncodersError(f"epilogue projections require contiguous {dtype} parameters on {self.device}")
            operation = self._make_linear(weight.detach(), None if bias is None else bias.detach())
            self._operations.append(operation)
            return operation

    def _close(self):
        for operation in self._operations:
            operation._close()
        self._operations.clear()
        self._closed = True


class PreparedLinear:
    """Shared inference, activation and ownership checks around backend math."""

    def __init__(self, context, weight, bias):
        self._context, self._weight, self._bias = context, weight, bias

    def __call__(self, x):
        with self._context._runtime._epilogue_lock:
            self._context._check()
            require_inference(self._context.device)
            if (not isinstance(x, torch.Tensor) or x.device != self._context.device
                    or x.dtype not in self._context.input_dtypes or x.layout != torch.strided
                    or x.ndim < 1 or x.shape[-1] != self._weight.shape[1]
                    or x.numel() // self._weight.shape[1] > 2**31 - 1):
                raise PackedEncodersError("prepared linear requires supported floating activations "
                                          "[..., in_features] on the weights' device")
            with self._context._device_context():
                # Native kernels may reuse scratch; branches and consecutive
                # same-width projections must not overwrite their inputs.
                return self._execute_owned(x)

    def _execute_owned(self, x):
        return self._execute(x).clone()

    def _execute(self, x):
        raise NotImplementedError

    def _close(self):
        self._weight = self._bias = None


class PackedEpilogue:
    """Prepared packed-output view; close() releases only this head's resources."""

    def __init__(self, runtime, context, forward):
        self._runtime, self._context, self._forward = runtime, context, forward

    def forward_packed(self, batch):
        with self._runtime._epilogue_lock:
            self._context._check()
            require_inference(self._context.device)
            if self._runtime.binding.weight_source.training:
                raise PackedEncodersError("epilogues require the backbone in eval mode")
            signature = None

            def apply(hidden, chunk):
                nonlocal signature
                output = self._forward(hidden, chunk)
                if (not isinstance(output, torch.Tensor) or output.ndim != 2
                        or output.shape[0] != chunk.input_ids.numel() or output.shape[1] <= 0
                        or output.device != hidden.device or output.layout != torch.strided
                        or not output.is_floating_point()):
                    raise PackedEncodersError("epilogue must return a floating tensor [tokens, features] on "
                                              "the hidden-state device, preserving token count and order")
                current = (output.shape[1], output.dtype)
                if signature is not None and signature != current:
                    raise PackedEncodersError("epilogue output width and dtype must agree across chunks")
                signature = current
                # Custom heads may return views of graph output or their own scratch.
                return (output if getattr(self._forward, "_packed_encoders_owned_output", False)
                        else output.clone())

            with self._context._device_context():
                return self._runtime._forward_epilogue(batch, apply)

    def close(self):
        with self._runtime._epilogue_lock:
            self._context._close()
            self._forward = None
            if self in self._runtime._epilogues:
                self._runtime._epilogues.remove(self)
