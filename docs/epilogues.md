# Retriever heads on packed encoders

An epilogue applies a token-preserving head after the encoder's final normalization.
Prepare it once from the packed backbone, then supply `PackedBatch` inputs:

```python
import torch
from torch import nn
import packed_encoders as pe

pe.pack(backbone)
projection = nn.Linear(backbone.config.hidden_size, 128, bias=False).to(
    device=next(backbone.parameters()).device,
    dtype=next(backbone.parameters()).dtype,
)
encoder = pe.get_engine(backbone).with_epilogue(
    pe.epilogues.TokenProjection(projection, normalize=True)
)
with torch.inference_mode():
    embeddings = encoder.forward_packed(batch)  # [live tokens, 128]
encoder.close()
```

`TokenProjection` optionally L2-normalizes each token. CPU heads use FP32, including
on an int8 backbone; CUDA heads use their projection weights' dtype. Projections
must be plain `nn.Linear` modules without hooks. In-place weight updates remain
visible; replacing parameters or changing devices or dtype requires a new head.

Custom heads implement `prepare(ops) -> forward(hidden, batch)`. During preparation,
`ops.prepare_linear(projection)` returns an inference-only operation with owned
outputs. It can be combined with PyTorch operations and other prepared linears.
On CPU, the head runs once per token-budget chunk with local sequence metadata;
on CUDA it runs after the packed forward or graph replay and receives the input
batch's metadata. Return a floating tensor `[tokens, features]` preserving token
count and order. Pooling or removing tokens belongs after the epilogue. Do not
mutate input metadata or recursively call the engine from the head.

Calls require an evaluation-mode backbone, disabled gradients and no autocast.
Returned outputs survive subsequent calls, including custom heads that reuse
scratch. Closing a head releases that head's resources; `pe.unpack(backbone)`
closes all attached heads. Backbone forwards remain available while a head exists.

CPU head GEMM tuning is private to each prepared projection, so preparing a head
does not replace backbone tile choices for an equal shape. See the
[extension API](../packed_encoders/epilogues.py),
[CPU tests](../tests/test_cpu_epilogues.py) and
[CUDA tests](../tests/test_cuda_epilogues.py).
