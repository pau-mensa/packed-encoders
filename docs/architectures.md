# Encoder engines

`pe.pack(model)` selects a curated engine, prepares it, and installs an adapter on
its patch target. ModernBERT remains the default for ModernBERT-shaped backbones,
including supported mmBERT checkpoints. Its kernels, layer schedule, attention
selection and inference/training graph runners remain specialized implementations.

The extension contracts in `packed_encoders.engine` are provisional. An engine
can be passed directly without registering it:

```python
from packed_encoders.arch.modernbert import ModernBert
import packed_encoders as pe

pe.pack(model, engine=ModernBert())
packed = pe.get_engine(model)  # the model-bound PackedEngine
hidden = packed.forward_packed(pe.PackedBatch(ids, cu_seqlens, max_seqlen, positions))
pe.unpack(model)
```

`pack` and `unpack` return the original target, including when it is a wrapper.
Existing direct `packed_encoders.forward.packed_forward` callers remain supported.

## Responsibilities

- **Adapter:** locate the patch target and weight source, interpret the public
  forward signature, and install the replacement preserving the model's outputs
  and unsupported-input policy. An engine authors adapter precedence explicitly.
- **Engine:** identify compatible bindings without importing kernel toolchains;
  prepare a model-bound packed engine and perform validation. One engine may support
  several model families, and a family may have multiple engines.
- **Packed engine:** own execution, native state, selected pieces, validation,
  runtime controls and resource cleanup. Its execution must not query a registry.

The registry uses explicitly designated curated defaults. Multiple applicable
engines without a unique default raise an ambiguity error; registration order
never selects a winner. An explicit engine bypasses registry selection.

An installation records the engine object, adapter, patch target, weight source,
packed engine, and original forward. Validation, runtime controls and unpack use
this owner even if registry contents subsequently change. Repeated packing reuses
that packed engine; switching engines requires unpack first. ModernBERT retains its
repeat-pack graph-enabling behavior. Use `set_cuda_graph(False)` to disable graphs;
passing `cuda_graph=False` to a repeat pack does not disable an existing runner.
Changing a prepared attention backend requires unpack and re-preparation.

## Preparation and teardown

An engine must roll back its own mutations if `prepare` raises before returning.
After it returns, the dispatcher restores the original forward and calls
`close(rollback=True)` if installation fails. Adapter installation must not mutate
weights; all weight preparation and rollback belong to the engine. A future engine
that changes storage must restore required aliasing as well as values on failure.

Successful `close()` must retain parameter updates made while packed. ModernBERT
does not replace parameter storage; it releases all inference/training runners on
unpack and restores the original forward attribute, including whether it was an
instance override. A closed packed engine cannot execute again.

ModernBERT retains existing weight semantics: eager calls use live parameters;
in-place optimizer/copy updates retain captured addresses. Parameter replacement,
`load_state_dict(assign=True)`, device/dtype changes, and storage replacement require
unpack before the operation and pack afterward. No generic mutation tracking is
promised. Graph buffers are not safe for concurrent calls on the same engine.
Training capture retains its exact-shape and persistent-gradient requirements.

## Packed input and pieces

`PackedBatch` construction neither copies metadata nor reads device values.
ModernBERT requires flat int64 token IDs, int32 device sequence boundaries, a host
maximum sequence length and int64 device positions. The caller/collator must supply
consistent contiguous nonempty sequences and HF-compatible positions within that
maximum. Optional host lengths are not rebuilt or consumed by ModernBERT.
The result has shape `[real_tokens, hidden]` in input token order, including the
encoder's final norm. Task heads, pooling and embedding normalization remain outside
the encoder. Output buffer lifetime follows the existing engine path; consume or
clone graph outputs before another replay when retaining them.

ModernBERT selects executable operation pieces: LayerNorm, residual-add plus
LayerNorm, dispatched and dense projections, BHSD RoPE, packed-QKV-to-BSHD RoPE,
BHSD and BSHD attention, and GeGLU. `ModernBertPieces` is an immutable selection;
`bind()` checks semantic compatibility and extracts the direct callable for each
slot. The existing layer schedule invokes these calls in eager execution, graph
capture, and training. Replay does not dispatch through Python pieces.

Each reusable `Piece` has an execution callable, a semantic/layout contract, an
independent reference, fixture checks and numerical tolerances. `piece.validate()`
is a forward probe; gradient behavior is covered separately in tests. Default
`pack()` validates selected pieces with engine-owned fixtures before validating
the complete encoder against HF. Reports explicitly identify inactive pieces that
could not be probed. `validate=False` bypasses numerical gates, but still checks
selection contracts. Pieces do not read model configs, own model weights, or
choose sequence/window policy.

For example, another engine can use the very same RoPE object without adopting
ModernBERT's schedule:

```python
from packed_encoders.pieces import split_half_rope

rope = split_half_rope()
cos, sin = rope.prepare(seq_len, head_dim, theta, device, dtype)
# Caller may gather tables for its positions before execution.
q_rotated, k_rotated = rope.execute(q, k, cos, sin)
```

This piece supports full-dimension split-half rotation of equal-shape CUDA bf16
BHSD Q/K, even head dimensions >=16, and shared position tables. It supports Q/K
gradients, but not trainable tables, partial rotation, interleaved pairs or GQA.
Those semantics need a different piece; sharing an operation name is insufficient.
The separate packed-QKV variant extracts and rotates Q/K in one existing kernel,
producing the native BSHD layout. Its V input remains a view, preserving the current
Flash/Triton path without a transpose, copy or extra kernel launch.

To substitute an implementation at preparation time:

```python
from dataclasses import replace
from packed_encoders.arch.modernbert import ModernBert
from packed_encoders.arch.modernbert_pieces import default_pieces

pieces = default_pieces()
alternative = replace(pieces.rope, name="my-split-half-rope", execute=my_rope)
pe.pack(model, engine=ModernBert(pieces=replace(pieces, rope=alternative)))
```

The replacement must implement the declared contract and pass its reference check.
This BHSD substitution affects the BHSD schedule; substitute `rope_qkv` separately
for the inference BSHD path. Selection is fixed for the prepared lifetime; unpack
before changing pieces so captured graphs cannot retain stale implementations.
There is no piece registry or automatic fusion/search. Existing residual/LN and
QKV/RoPE fusions remain pieces, and larger fusions can have their own contracts.

Capabilities distinguish inference, training, inference/training capture and
original-forward fallback. ModernBERT keeps its existing unsupported-call errors;
it does not claim generic HF fallback. `packed.validate()` returns a common
`ValidationResult` with an engine-specific `details` report. `pe.validate()` retains
the native report type for compatibility. Pack-time ModernBERT results are available
on `packed.state.validation_report`; `validate=False` leaves it unset.

Omitted options reach engines as absent keys; explicit requests are preserved.
Engines must reject unsupported requests rather than discard them. ModernBERT's
omitted graph options remain off, its sequence cutoff remains 64, and its attention
backend default remains unchanged. Engine-specific configuration stays with the
engine (including ModernBERT's `GraphConfig` and `TrainGraphConfig`).

## Regression checks

`benchmarks/engine_refactor.py --source CHECKOUT --output RESULT.json` measures a
frozen source tree in the active Python environment and records its actual import
path, dependency versions, model revision, token checksums, raw timing samples,
cold-call latency and peak allocated/reserved memory. Use fresh processes for paired
baseline/candidate runs. The baseline must include the padding fix in PR #3.

The harness covers synthetic query/document padded and direct-packed execution,
Flash/auto/SDPA, graph on/off and long-input fallback, plus eager/captured training.
It isolates encoder execution; it does not establish tokenizer, retrieval, or
end-to-end training-recipe performance. Use the showcase protocols for those claims.

## Qwen3.5 / topk-embed-v1

`Qwen35Hybrid` is the curated engine for the xsmall/small topk wrapper. Its
`TopkEmbedAdapter` binds the wrapper as the patch target and `model.language_model`
as the weight source. The existing dispatcher owns installation, validation,
controls and teardown. Plain HF Qwen3.5 entry points are not installed by this PR.

The adapter preserves topk's padded per-token vectors: projection through its head,
output-dimension slice, optional normalization and zero padding. Grad-enabled calls
and image inputs delegate to the saved original forward. This fallback is separate
from the prepared engine, whose `training` capability is false.

```python
from packed_encoders.arch.qwen3_5 import Qwen35Hybrid

pe.pack(model, engine=Qwen35Hybrid())  # inference graphs on by default
packed = pe.get_engine(model)
with torch.no_grad():
    hidden = packed.forward_packed(pe.PackedBatch(ids, host_lengths=(7, 29, 64)))
```

The direct entry returns `[real_tokens, hidden_size]` after the backbone's final
norm, before the topk head. It requires flat int64 IDs on the model's CUDA device
and positive host lengths summing to the number of IDs. Sequence positions restart
at zero. Other `PackedBatch` metadata is rejected explicitly; the engine does not
silently discard custom positions or transfer device boundaries back to the host.
The topk adapter still reads its wrapper's device boundaries to obtain host lengths.

`Qwen35Pieces` selects 13 executable pieces: projection, RMSNorm, residual RMSNorm,
SwiGLU, causal convolution with gate extraction, gated RMSNorm, single and paired
Q/K RMSNorm with partial RoPE, sigmoid output gate, chunked and recurrent
GatedDeltaNet, and packed and segmented-padded GQA attention. The reusable pieces
live under `packed_encoders.pieces`; they take explicit weights and metadata, not
models. A composition's `bind()` checks contracts once. The layer schedule invokes
the selected callables in eager execution and capture; replay bypasses Python.

A caller can substitute pieces with `dataclasses.replace`, as for ModernBERT.
The engine retains the contribution's attention/GatedDeltaNet probes and fusion
fallback policy. Pack-time validation additionally probes the selected pieces
against independent PyTorch references and checks the complete adapter output
against the saved model forward. `report.pieces` identifies inactive operations.
`validate=False` skips those composition/end-to-end checks; the original kernel
selection probes still run because they determine usable implementations.

Graph configuration uses `PaddedGraphConfig`. Omitted or `None` means graphs on;
`False` means off. On repeat pack, an explicit graph setting applies. Backend
changes require unpacking. Training capture and ModernBERT's sequence-cutoff
option are rejected rather than ignored. `set_train_cuda_graph(False)` is a no-op.

**Graph memory.** The runner releases unused warmup allocations before capture.
If a Qwen hidden-state or topk projection call runs out of GPU memory while a graph
runner is held, the engine drops the runner, disables graphs, warns, and retries
once eagerly. Cleanup runs on the weights' device. An OOM without a runner, or on
the retry, propagates. Graphs stay disabled until explicitly re-enabled with
`pe.set_cuda_graph(model, True, config=PaddedGraphConfig(...))`. Use smaller buckets
or `cuda_graph=False` when graphs consume too much memory; recovery does not make
an oversized eager workload fit. Pack-time graph validation still fails if it
cannot complete; it does not silently skip validation.

Only independent, bias-free dense projections are supported. Unmerged adapters,
MoE and tied/aliased backbone parameter storage are rejected. Aliasing is checked
before any mutation. Prepared projections share their merged storage with the
original parameters; no extra full projection copies are retained. Failed
preparation/installation and successful unpack restore independent storage and
parameter identities. Unpack preserves optimizer updates. Storage addresses can
change: do not retain external tensor views or CUDA graphs across pack/unpack.
Norm scales refresh after normal in-place updates; `.data` writes, parameter
replacement, device/dtype moves and `load_state_dict(assign=True)` require repacking.
The engine pins launches/capture to the weights' device. Runners and FLA's temporary
global tensor-cache setting do not support concurrent calls on the same process.

Install the complete topk environment with
`uv sync --locked --no-dev --extra qwen3_5 --extra fa2`. The lockfile uses
Torch 2.11 / CUDA 12.8 for both engines. See [the installation review](torch-2.11-review.md)
for installation details and the separate PyLate environment requirement.
