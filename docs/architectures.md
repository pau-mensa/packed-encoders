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
CUDA ModernBERT requires flat int64 token IDs, int32 device sequence boundaries, a host
maximum sequence length and int64 device positions. The caller/collator must supply
consistent contiguous nonempty sequences and HF-compatible positions within that
maximum. Optional host lengths are not rebuilt or consumed by CUDA ModernBERT.
The result has shape `[real_tokens, hidden]` in input token order, including the
encoder's final norm. Task heads, pooling and embedding normalization remain outside
the encoder. Output buffer lifetime follows the existing engine path; consume or
clone graph outputs before another replay when retaining them.

CPU ModernBERT also accepts `host_lengths` instead of sequence boundaries and
defaults omitted positions to zero-based indices per sequence.

All built-in packed engines support `runtime.with_epilogue(head)`. Custom heads
implement `prepare(ops) -> forward(hidden, batch)` and can use
`ops.prepare_linear()` for prepared projections. The composed encoder is
inference-only and returns owned outputs. CPU applies the head per internal chunk;
CUDA applies it after the existing packed forward/graph replay. The original
runtime keeps returning hidden states. Unpacking closes its composed encoders too.

`runtime.epilogue` shares preparation lifecycle, linear input validation, output
validation/ownership and cleanup. Backends implement projection dtype policy and
math (`cpu.epilogue`, `runtime.cuda_epilogue`). Engines opt in with initialization
and teardown hooks; CPU additionally supplies the chunk boundary hook. CUDA
BF16 bias-free heads reuse each engine's selected linear piece; other supported
heads use PyTorch/cuBLAS. Heads preserve token count and order and return floating
tensors on the hidden-state device. CPU head precision is FP32; CUDA defaults to
the head weight dtype (BF16, FP16 or FP32), casting activations as needed.

CUDA ModernBERT selects executable operation pieces: LayerNorm, residual-add plus
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
controls and teardown. Stock HF Qwen3.5 models bind through `HFQwen35Adapter`
([below](#stock-hf-qwen35-models)).

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

`Qwen35Pieces` selects 15 executable pieces: projection, RMSNorm, residual RMSNorm,
SwiGLU, causal convolution with gate extraction, gated RMSNorm, single and paired
Q/K RMSNorm with partial RoPE, sigmoid output gate, chunked, recurrent and resumed
GatedDeltaNet, and packed, segmented-padded and prefixed GQA attention. The reusable pieces
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

### Stock HF Qwen3.5 models

`HFQwen35Adapter` binds transformers' own `Qwen3_5TextModel` and `Qwen3_5Model`
(subclasses included) as the patch target and the text model as the weight source. Topk's
adapter is tried first, since its wrapper holds a stock `Qwen3_5Model`. The patched forward
keeps HF's contract: `input_ids` / `attention_mask`, right or left padded, in;
`last_hidden_state` out, with pads as zeros. Attention is causal or bidirectional as
`config.is_causal` says (absent means causal), the same rule HF's masks follow. The model's
own pooling or head then runs unchanged:

```python
model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")   # Qwen3_5Model or a subclass
pe.pack(model, cuda_graph=False)                 # multi-row batches: see "CUDA graphs on large models"
batch = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    hidden = model(**batch).last_hidden_state    # pads are zeros; pool or read out as before
```

A model that wraps the backbone in its own module, such as a decision model with a readout
over the last token, packs the `Qwen3_5Model` it holds; the readout then runs on the returned
hidden states.

Text batches carrying the `mm_token_type_ids` that Qwen3-VL processors always return still
run packed. Calls the engine can't serve run the original forward unchanged: images,
gradients, a KV cache (`past_key_values`, `use_cache=True`), explicit `position_ids` /
`inputs_embeds`, extra outputs (`output_hidden_states`, ...) and masks with holes. The
engine runs in bf16; when a checkpoint ships fp32 weights, check quality against fp32 on
your data.

**CUDA graphs on large models.** Graphs remove launch overhead, but a graph also runs every
row padded to its bucket. On a large model, the overhead saving still pays for single-row
calls, while the padding dominates multi-row batches:

| Model, GPU | Batch | Without graphs → with graphs |
|---|---|---|
| 27B, causal, H100 | 1 row | 6.38 → 7.91 items/s |
| 27B, causal, H100 | 8 rows | unchanged |
| 9B, bidirectional, L40S | 32 queries | roughly halved |

Measured on pplx-decider-v1-27b's backbone, with `PaddedGraphConfig(row_buckets=(1, 8),
max_tokens=8192, max_seq=4096)`, and on pplx-embed-v2-context-9b-preview. Pack large models
with `cuda_graph=False` for padded batches, and keep graphs for single-row traffic.

**Shared prefixes.** A decision model asking several questions about one context sends rows
that repeat the system prompt and the context; a shared system prompt or few-shot block does
the same. On a causal engine that opts in, rows whose first `min_shared_prefix` tokens agree
run their longest common prefix once, and each row continues from it with its own tokens
(every row keeps at least one). Rows sharing a prefix must be in the same batch, in any order.

A mask can't do this on a hybrid backbone. Attention could be masked so that each row sees
only the prefix and itself, but GatedDeltaNet carries context in a recurrent state, and its
short conv reads the previous tokens directly. A row packed after another row would still
see it. So the engine forks the state instead. Each prefix runs from a zero state. Each
continuation starts from its prefix's final GatedDeltaNet state and conv inputs, and attends
to the prefix's keys and values with an end-aligned causal mask. fla's `initial_state` and
the varlen attention kernels already do this; the resume and the end-aligned attention are
probed at pack time like the other kernels. Bidirectional models never share, since a
prefix's states depend on what follows.

```python
pe.pack(model, cuda_graph=False)
pe.get_engine(model).min_shared_prefix = 64      # opt in; 0, the default, runs every row in full
rows = [context + question for context in contexts for question in questions]
batch = tokenizer(rows, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    hidden = model(**batch).last_hidden_state
```

Sharing is off by default because the shared pass runs eagerly and calls GatedDeltaNet twice
per layer, once for the prefixes and once for the continuations. It pays where a forward is
compute bound, as on a large model or long rows, and costs on a small model with short rows.
Speedup at batch 8 with `min_shared_prefix=64` over every row in full, on the faster of graphs
and eager (for the 27B, eager: graphs don't help its 8-row batches):

| Model, GPU | Rows | 2 rows per prefix | 4 | 8 |
|---|---|---|---|---|
| 27B decision model, H100 | 71- to 3,096-token context + a question | 1.8x | 3.0x | 4.2x |
| Qwen3.5-0.8B, L40S | 1,024-token prefix + up to 1,024 own tokens | 1.31x | 1.35x | 1.37x |
| Qwen3.5-0.8B, L40S | 1,024-token prefix + up to 32 | 0.98x | 0.99x | 0.99x |
| Qwen3.5-0.8B, L40S | 128-token prefix + up to 1,024 | 0.66x | 0.58x | 0.58x |
| Qwen3.5-0.8B, L40S | 128-token prefix + up to 32 | 0.12x | 0.13x | 0.13x |

The 27B rows are pplx-decider-v1-27b's backbone under its own batching (flash-attn 2,
`cuda_graph=False`). With 4 rows per context, its probabilities stay as close to stock as
stock's own batch 1 and batch 8 are to each other: max absolute difference 0.021 against a
0.020 floor, and the argmax agrees on 64/64 rows. The Qwen3.5-0.8B rows come from the
benchmark below on the pinned stack (torch varlen attention). There, the shared pass has a
floor of about 85 ms per batch however few tokens it computes, and shared rows match the
model's own forward as closely as unshared rows (per-token cosine 0.99985 to 0.99988 mean).
Measure a model before opting in:

```bash
python benchmarks/qwen35_shared_prefix_bench.py --model Qwen/Qwen3.5-0.8B --out shared.json
```

With graphs on, batches with shared prefixes replay one CUDA graph per plan
(`runtime.shared_graphs`); the measurements above ran the shared pass eagerly. With sharing on, planning reads the
batch's leading tokens once per call, when at least two rows are longer than
`min_shared_prefix`. `validate()` checks the shared pass against the model's own forward even
while sharing is off and reports `shared_cos_mean` / `shared_cos_min`;
`shared_prefix_rejected` says why a model can't share.

## LFM2

`Lfm2Hybrid` packs LFM2 text backbones: gated short convolutions (`C * conv(B * x)`, a
3-tap depthwise causal conv) interleaved with GQA softmax attention with per-head Q/K
RMSNorm and full RoPE. `HFLfm2Adapter` binds transformers' `Lfm2Model` as both patch
target and weight source, whether it is handed over directly, under a task head
(`Lfm2ForCausalLM.model`), or as the `.language_model` of a multimodal wrapper such as
`Lfm2VlModel`. The patched forward keeps HF's contract: `input_ids` *or* `inputs_embeds`
plus `attention_mask`, right or left padded, in; `last_hidden_state` out, pads as zeros.

```python
model = Lfm2VlForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
pe.pack(model)                                   # packs model.model.language_model
with torch.no_grad():
    out = model.model(**batch, use_cache=False)  # use_cache=False, or set it in the config
```

A multimodal wrapper merges its image features into `inputs_embeds` before calling the
backbone, so image batches run packed too. Token-id batches replay CUDA graphs when they fit
a bucket; embedding batches run the eager packed engine; batches whose rows share a prefix run
it once when the caller opts in (see **Shared prefixes** below). Calls the engine can't serve run
the original forward unchanged: gradients, a KV cache (`use_cache` resolves to True by
default in LFM2 configs), explicit `position_ids`, extra outputs and masks with holes.

`Lfm2Pieces` selects 9 pieces: projection, RMSNorm, residual RMSNorm, SwiGLU, the gated
short conv (one Triton launch reading the in-projection in place, positions restarting per
sequence), paired Q/K RMSNorm + RoPE, and packed, segmented-padded and prefixed attention. q|k|v and
w1|w3 are merged GEMMs sharing the HF parameters' storage, as for Qwen3.5. LFM2's norms
scale by `weight` (no `1 +`); fp32 copies refresh after in-place updates (`sync_norms`).
At build time the fused Q/K kernel and the packed conv are checked against the model's own
modules; pack-time validation compares token ids (eager and graphed) and input embeddings
with the model's forward. Only causal backbones are supported: a bidirectional LFM2 also
changes the convolution (centred taps), not just the mask. Conv biases, unmerged adapters
and aliased parameters are rejected before any mutation.
An explicit `config.is_causal=False` is refused even with `validate=False`. Single-row
left-padded calls fall back to HF: its short convolution skips padding masking for one
row, so stripping those pads could change real-token outputs.

Pack-time validation holds eager, input-embedding and graphed output alike to the model's own
forward (per-token cosine mean ≥ 0.99, min ≥ 0.85). Graphs are not compared with eager: a
bucket's padded GEMMs round differently, and over a deep stack two bf16 paths drift apart more
than either drifts from fp32. The gate is loose because it compares two valid bf16 paths, and on
a 30-layer model they separate further than either separates from fp32: on d1-3B the engine is
*closer* to an fp32 forward than the model's own bf16 forward is, on both validation batches
(RTX 4090, flash2; rows 7/129/300/64: stock 0.99805 mean, engine 0.99832; rows 3/21/30: stock 0.99916,
engine 0.99940), while engine against stock bf16 reads 0.99909 and 0.99891. Replayed from a
graph the short batch reads 0.99697/0.91294, the one path that does drift further than stock,
since a bucket pads every row and the GEMMs change shape. An earlier 0.999 mean gate therefore
refused d1-3B on an RTX 4090 and an RTX 5090 while passing on an L40S. The thresholds are set
from what a broken engine scores, not from how close a correct one lands on one device.
Structural errors stay far outside the loosened gate — deliberately breaking one piece at a
time, engine against stock bf16 on those batches:

| Engine | mean cosine, rows 7/129/300/64 | mean cosine, rows 3/21/30 |
|---|---|---|
| Correct | 0.99909 | 0.99891 |
| Convolution positions never restart per row | 0.71970 | 0.55188 |
| RoPE rotation removed | 0.68983 | 0.92181 |
| SwiGLU gate and value swapped | 0.19642 | 0.28816 |
| Norm scales dropped | 0.01841 | 0.02100 |

Measured on LiquidAI/d1-3B's backbone (an LFM2-VL checkpoint, 2048 hidden, 30 layers, 8 of them
attention), L40S, torch 2.11, torch varlen attention (no flash-attn), random rows, ms per batch:

| Batch | Padding | Stock HF (sdpa, padded) | Packed, eager | Packed, graphs | Speedup |
|---|---|---|---|---|---|
| 1 × 128 tokens | 0% | 21.4 | 19.8 | 9.7 | 2.21x |
| 8 rows, 32–512 tokens | 47% | 121.0 | 57.9 | 74.3 | 2.09x |
| 32 rows, 32–512 tokens | 39% | 592.4 | 282.6 | 309.0 | 2.10x |
| 8 × 1,024 tokens | 0% | 266.7 | 230.9 | 230.7 | 1.16x |

```bash
python benchmarks/hf_forward_bench.py --model LiquidAI/d1-3B --out forward.json
```

The benchmark takes any model `pack()` supports. Per-token cosine with stock HF: 0.9991 to
0.9995 mean. The wrapper path (`inputs_embeds`, eager) times as eager. Packing pays where it
removes padding; on equal-length rows this 3B model is GEMM bound, and stock HF already runs
those GEMMs about as fast. Single rows are host bound and vary with the machine: on another
L40S host, 1 × 128 took 12.3 ms stock and 9.5 ms graphed (1.30x). As on Qwen3.5, a graph runs
every row padded to its bucket: graphs win single-row calls and lose mixed-length multi-row
batches.

**Shared prefixes.** As for Qwen3.5, rows whose first `min_shared_prefix` tokens agree run
their longest common prefix once (opt in; 0, the default, runs every row in full). LFM2 has no
recurrent state to resume, so the whole forest of prefixes and continuations is one pass:
a continuation's first two tokens recompute their conv taps over the prefix's last tokens, and
its attention reads the prefix's keys and values with an end-aligned causal mask (the prefixed
varlen kernel, probed at pack time). Through a multimodal wrapper, rows are matched on their
input embeddings instead of token ids: a 64-bit hash of each row's bits groups them, and a plan
is used only if every row it pairs is bit-identical, so a shared image prefix is shared too.

```python
pe.pack(model)
pe.get_engine(model).min_shared_prefix = 64
rows = [state + question for question in questions]       # one state, several questions
batch = tokenizer(rows, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    hidden = model.model(**batch, use_cache=False).last_hidden_state
```

d1-3B, L40S, one state and several 16–40-token questions (the same benchmark run), ms per batch,
speedup over stock HF in parentheses:

| State, questions | Tokens not recomputed | Stock HF | Packed, every row in full | Shared, eager | Shared, graphs | Shared, via the VL wrapper |
|---|---|---|---|---|---|---|
| 256 tokens, 8 | 80% | 77.4 | 64.6 | 28.6 (2.7x) | 17.1 (4.5x) | 29.1 |
| 1,024 tokens, 8 | 85% | 285.1 | 238.4 | 39.6 (7.2x) | 38.3 (7.4x) | 40.2 |
| 2,048 tokens, 16 | 93% | 1,290.1 | 1,002.7 | 81.8 (15.8x) | 81.5 (15.8x) | 85.8 |

The graph columns above describe the requested setting. With the benchmark's default
graph bounds (`max_seq=2048`, `max_tokens=16384`), the 2,048-token state plus questions
exceeds the shared graph bounds and runs eagerly. New benchmark results include an
`execution` entry for each variant, measured in an untimed parity call, and speedups
per variant. The aggregate speedup identifies the fastest variant rather than assuming
the default graph-enabled path was fastest.

As for Qwen3.5, token-id batches replay the shared pass as one CUDA graph per plan
(`runtime.shared_graphs`), matching the eager pass: it pays where launch overhead matters (a
short state) and is a wash once the batch is compute bound. A plan is captured the
first time it is seen (two warmup passes, then the capture) and keyed on its exact row lengths
and groups, so graphs help when plans recur; traffic whose plans rarely repeat is better
served eagerly. Input embeddings (the VL wrapper) run the shared pass eagerly.

Per-token cosine with stock HF over every token: 0.9990 to 0.9997 mean, the same as rows run
in full. With sharing on, a batch that shares nothing costs only its planning. `validate()` checks the shared pass (token ids and embeddings) against the
model's own forward even while sharing is off, reporting `shared_cos_mean` / `shared_cos_min`
and, on a miss, `shared_prefix_rejected`.

## EmbeddingGemma 2

`EmbeddingGemma2` packs EmbeddingGemma 2 text backbones: 24 bidirectional GQA layers, sliding
(|q − k| ≤ 512, head dim 256, two KV heads) except every sixth, which is global (head dim 512, one
KV head), with attention scale 1 and RMSNorm on queries, keys and values; post-norm ("sandwich")
blocks around attention, a tanh-GELU GeGLU MLP and a per-layer-embedding (PLE) block, each layer
ending in a learned scalar; and a per-token output projection (512 → 768) after the final norm.
Norms scale by `weight` (no `1 +`).

`HFEmbeddingGemma2Adapter` binds the text backbone (`EmbeddingGemma2TextModel`) as the weight
source. Its patch target is the multimodal wrapper (`EmbeddingGemma2Model`, which `AutoModel` and
SentenceTransformers load) when it is given one, else the backbone. The wrapper always calls the
backbone with `inputs_embeds`, so the patched wrapper takes text-only calls itself with the ids it
was given, mapping image, video and audio placeholders to padding as the wrapper does. Each patched
forward accepts `input_ids` (or, at the backbone, `inputs_embeds`) plus `attention_mask`, right or
left padded, and returns `last_hidden_state`. Parity covers real tokens: packed padding slots are
zero, while HF's padding outputs can be nonzero. Pooling and normalization run unchanged when
pooling uses the attention mask; consumers must mask padding rather than compare or pool those slots.

```python
st = SentenceTransformer("google/embeddinggemma-2", device="cuda", model_kwargs={"dtype": torch.bfloat16})
pe.pack(st)                                    # patches the EmbeddingGemma2Model inside
queries = st.encode_query(texts, batch_size=32)
```

Token-id batches replay CUDA graphs when they fit a bucket, else run the eager packed engine;
input-embedding batches at the backbone run eagerly. Image, video or audio inputs, gradients,
active training dropout, explicit `position_ids`, extra outputs, masks with holes and prepared
mask dictionaries run the original forward unchanged.
Bidirectional models never share prefixes.

`EmbeddingGemma2Pieces` selects 7 pieces: projection; RMSNorm over rows (the input norm, the PLE
norm with its input scale rounded first, and the unscaled V norm); the post-norm add (`residual +
RMSNorm(delta) * w`, times the layer scalar, then the next block's input norm, in one launch); the
tanh-GELU gate; paired Q/K RMSNorm + RoPE; and packed and segmented-padded attention. q|k|v and
gate|up are merged GEMMs sharing the HF parameters' storage; fp32 norm and scalar copies refresh
after in-place updates (`sync_norms`). Attention is chosen per kind: each choice carries its window
and scale, and the pack-time probe runs rows longer than twice the window, so a kernel that ignores
the window, or is off by one, is rejected. On an RTX 4090 the sliding layers select FlashAttention 2
and the global layers (head dim 512, beyond its limit) SDPA. Biases, unmerged adapters, other
activations and causal configs are rejected before any mutation.

Pack-time validation holds eager, input-embedding and graphed output to the model's own forward by
the batch's per-token cosine mean (≥ 0.99) and each sequence's mean (≥ 0.98). The sequence gate
prevents long rows from hiding corruption of whole short rows. On the real checkpoint with an
RTX 5090, Torch 2.11 and Transformers 5.19, the lowest sequence mean was 0.98935 eager and
0.99048 graphed, so a per-sequence threshold of 0.99 would reject the correct eager engine.
The report records the lowest sequence mean for each path in `*_sequence_cos_mean_min`.
Token minima remain informational: the output projection amplifies bf16 rounding on a few
tokens, so the model's own bf16 forward is far from its fp32 forward on those tokens. Measured on
the validation batches (random ids, rows of 7/129/1,100/64 and 3/21/600 tokens), per-token cosine
mean (min):

| Forward | vs stock bf16 | vs fp32 |
|---|---|---|
| Stock bf16 | — | 0.99607 (0.861), 0.99637 (0.822) |
| Packed engine | 0.99447 (0.803), 0.99490 (0.873) | 0.99570 (0.709), 0.99424 (0.852) |
| No sliding window | 0.79164, 0.96192 | 0.79319, 0.96349 |
| No V norm | 0.65285, 0.62541 | 0.65474, 0.62624 |
| Layer scalar ignored | 0.14404, 0.14314 | 0.14411, 0.14379 |

A window of 511 instead of 512 scores like the engine here (it drops only the keys exactly 512
tokens away); the attention probe catches it. On real text the pooled, normalized embeddings match stock at cosine
≥ 0.99986.

Measured on google/embeddinggemma-2, RTX 4090, torch 2.11, transformers 5.19, FlashAttention 2,
random rows, ms per batch:

| Batch | Padding | Stock HF (sdpa, padded) | Packed, eager | Packed, graphs | Speedup |
|---|---|---|---|---|---|
| 1 × 128 tokens | 0% | 33.0 | 12.4 | 2.0 | 16.2x |
| 8 rows, 32–512 tokens | 47% | 32.1 | 20.8 | 13.4 | 2.40x |
| 32 rows, 32–512 tokens | 39% | 127.9 | 49.7 | 39.5 | 3.24x |
| 8 × 1,024 tokens | 0% | 66.8 | 39.8 | 33.5 | 2.00x |

```bash
python benchmarks/hf_forward_bench.py --model google/embeddinggemma-2 --out forward.json
```

Unlike the hybrids, graphs win every batch here: the model is small (137M parameters in the
backbone, without the embedding table), so even a 32-row batch is partly launch bound. End to end
through `SentenceTransformer.encode` at batch 32 (tokenization included, same GPU), 2,048 queries
of 6–25 words ran at 806/s stock and 3,326/s packed (4.12x), and 512 documents of 60–960 words at
103/s and 279/s (2.71x).
