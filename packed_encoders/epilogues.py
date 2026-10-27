"""Token-preserving heads for the provisional CPU/CUDA epilogue extension API.

Pass an object with ``prepare(ops) -> forward(hidden, batch)`` to a packed
engine's ``with_epilogue``. Preparation runs once; forward runs after final
normalization on each CPU chunk or after the CUDA packed forward/graph replay.
It must return a floating tensor [tokens, features] on the hidden-state device
with unchanged token order. CPU chunk metadata is populated; CUDA passes through
the input PackedBatch without materializing missing metadata or reading the GPU.
Callbacks must not mutate input metadata or recursively invoke the engine.

``ops.prepare_linear(nn.Linear, precision=None)`` prepares a projection
and returns an inference-only callable with owned output. Custom code can mix
these operations with PyTorch. Token removal/pooling belongs after the epilogue.
"""

from dataclasses import dataclass

import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TokenProjection:
    """Project each token, optionally L2-normalizing the resulting embedding.

    Head precision is independent of backbone precision: CPU uses FP32; CUDA
    defaults to the projection's weight dtype (FP32, BF16 or FP16), casting input
    activations as needed. Explicit precision must match the head weights.
    The projection must be a plain nn.Linear on the backbone device.
    In-place parameter updates remain visible;
    replacing parameters, moving devices or changing dtype requires a new head.
    """

    projection: nn.Linear
    normalize: bool = True
    precision: str | None = None

    def prepare(self, ops):
        project = ops.prepare_linear(self.projection, precision=self.precision)

        def forward(hidden, batch):
            output = project(hidden)
            return F.normalize(output, p=2, dim=-1) if self.normalize else output

        # Both prepared projections and F.normalize return owned tensors. The
        # shared wrapper need not clone this built-in head's result a second time.
        forward._packed_encoders_owned_output = True
        return forward
