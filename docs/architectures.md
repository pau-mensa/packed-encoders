# Architectures

`pe.pack(model)` finds the backbone inside whatever it is handed (a Hugging Face model, a
`SentenceTransformer`, a PyLate `ColBERT`, a `MultiVectorEncoder`) and asks each registered
architecture whether it patches that module. The first match installs its engine in place.
`unpack()`, `validate()`, `set_cuda_graph()` and `no_cuda_graph()` route the same way.

| name         | backbone                                                   | patched entry point        |
|--------------|------------------------------------------------------------|----------------------------|
| `modernbert` | ModernBERT, Ettin, mmBERT and their finetunes              | `ModernBertModel.forward`  |
| `qwen3_5`    | Qwen3.5 hybrid (GatedDeltaNet + gated softmax attention): [topk-embed-v1-xsmall / -small](https://huggingface.co/topk-io), [pplx-embed-v2-context](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-9b-preview), any stock HF Qwen3.5 | `TopkEmbedModel.forward`; `Qwen3_5Model.forward` / `Qwen3_5TextModel.forward` |

Each architecture brings its own kernel toolchain, loaded only when a model of that
architecture is packed: CuteDSL for ModernBERT, fla (flash-linear-attention) for Qwen3.5.
`import packed_encoders` itself pulls in neither.

## Qwen3.5 hybrid (topk-embed-v1)

```python
import packed_encoders as pe
from sentence_transformers.multi_vector_encoder import MultiVectorEncoder

model = MultiVectorEncoder("topk-io/topk-embed-v1-xsmall", trust_remote_code=True, device="cuda",
                           model_kwargs={"torch_dtype": torch.bfloat16})
pe.pack(model)                                      # probes, validates, installs; CUDA graphs on
docs = model.encode(texts, task="document", batch_size=8)
pe.unpack(model)                                    # the shipped forward, weights untouched
```

Requirements: topk's own stack (torch 2.11, transformers 5.9, `flash-linear-attention` 0.5.1)
on an sm_80+ GPU. On torch 2.8 topk's shipped forward itself fails on short batches (its
compiled flex attention), so there is no reference to validate against and `pack()` refuses.
Image inputs and gradient-enabled calls fall through to the original forward.

### Stock HF Qwen3.5 models (pplx-embed-v2-context)

The same engine serves any model built on transformers' own `Qwen3_5TextModel` or
`Qwen3_5Model` (subclasses included), through a second entry that keeps HF's contract:
right-padded `input_ids` / `attention_mask` in, `last_hidden_state` out (pads as zeros).
Attention is causal or bidirectional as `config.is_causal` says, exactly as HF builds its
masks. The model's own pooling then runs unchanged:

```python
model = AutoModel.from_pretrained("perplexity-ai/pplx-embed-v2-context-9b-preview",
                                  trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
pe.pack(model, cuda_graph=False)                     # a 9B is GPU-bound: graphs don't pay here (below)
chunk_embeddings = model.encode(doc_chunks)          # its own chunk pooling + int8, on the engine
```

Calls the engine can't serve exactly run the original forward: images, gradients, a KV cache
(`past_key_values`, `use_cache=True`), explicit `position_ids` / `inputs_embeds`, extra outputs
(`output_hidden_states`, ...), and left padding. pplx ships fp32 weights; the engine runs bf16,
so load it in bf16 and check retrieval against fp32 for your data.

**CUDA graphs and model size.** Graphs remove launch overhead, which is what limits a small
model (topk's speedups come from them). A 9B model keeps the GPU busy on its own, and a graph
runs every row padded to its bucket, so for pplx on an L40S graphs gained nothing on batch-8
queries and halved batch-32 query throughput: pack large models with `cuda_graph=False`.
Graphs also cost memory: their pool can't borrow from PyTorch's regular cache, and with a 9B
on a 48 GB GPU the default buckets (up to 16k tokens) don't fit beside the weights. If a call
runs out of GPU memory while graphs are held, the engine frees them, warns once, and continues
eager.

What the engine does (`arch/qwen3_5/engine.py`), all exact rewrites of the model's math:

- **Merged projections.** GatedDeltaNet q|k|v|z|b|a, attention q|gate|k|v and MLP gate|up are
  one GEMM each. The merged weight *is* the HF weight: each parameter is re-pointed at its
  rows, so packing costs no memory, and `unpack()` gives every parameter its storage back.
- **GatedDeltaNet through fla** with the decay gate, β sigmoid and q/k L2 norm inside the
  kernel. Inside CUDA graphs, where launches cost no host time, the chunked kernel always runs;
  it was fastest at every setting measured on L40S and H100. The eager forward (batches larger
  than the biggest graph bucket) is host-bound, so a batch whose rows are all up to 64 tokens
  takes the single-launch recurrent kernel instead of the chunked kernel's ~6 launches:
  128-query batches on xsmall run 1.7x faster, on small within noise.
- **Fusions:** the q|k|v causal conv and the b|a gate copies in one launch; the gated RMSNorm
  reading z in place; q and k RMSNorm + partial RoPE in one launch; the attention output gate
  in one kernel.
- **CUDA graphs over padded `(rows, S)` buckets.** Padding is exact here: the mixers are
  causal, and attention sees each row as `[real | pad]` segments, so real tokens never read a
  pad. A planner splits mixed-length batches into groups when padding would cost more than
  an extra replay; a batch larger than the biggest bucket runs the eager packed forward.
  Graphs are on by default.
- **No device-to-host syncs** in the forward except the batch lengths themselves; all
  metadata goes to the GPU in one pinned copy.

### Nothing is installed unmeasured

`pack()` measures on the GPU in hand before it installs anything:

| check | against | on failure |
|---|---|---|
| each attention kernel (FA4, FA2, torch varlen, SDPA) | fp32 SDPA | next candidate (none left: `ValidationError`) |
| chunked GatedDeltaNet kernel (packed and padded) and recurrent one (packed, the only way it runs) | fp32 token-by-token recurrence | chunked must pass (else `ValidationError`); recurrent off |
| fused q/k norm + RoPE | the model's own `q_norm`, `k_norm`, `apply_rotary_pos_emb` | `ValidationError` |
| each fusion | the op it replaces | fusions off |
| the whole forward, eager and graphed, long and short batches | the model's own forward | `ValidationError` |

`pe.validate(model)` runs the same gate and returns the report (`Qwen35Report`): which
kernels were chosen, what was rejected and why, and the cosine to the model's own output.

### LoRA, training and devices

- **LoRA:** merge adapters before packing (`model = model.merge_and_unload()`). The engine reads
  plain dense weights, so every projection it reads is checked, and one carrying an unmerged
  adapter (a peft layer: `lora_A` or `base_layer`) or a bias is refused with a message saying so.
- **Training after packing:** the projections, conv and gate parameters are the model's own
  storage, so updates reach the engine as they happen. The fp32 `1 + w` norm scales are copies;
  each call re-copies any whose parameter's version changed (an optimizer step,
  `load_state_dict`), in place, so captured graphs stay valid. Writes through `.data` bypass
  the version counter: re-pack after those.
- **Devices:** the engine runs on its weights' device, whatever the current CUDA device is.

## Adding an architecture

An architecture is a *backbone*. A new model on a backbone that is already registered needs
an entry inside that plugin, not a new registration: one function that finds the backbone in
the model's wrapper and one patched forward that keeps the wrapper's contract
(`arch/qwen3_5` has two, `topk` and `hf`). A new backbone is a new plugin:

1. Implement the `Architecture` protocol (`arch/base.py`) in `arch/<name>/`:
   `match(module)` (exact: two plugins must never both match), `validate`, `pack`, `unpack`.
2. Put the math in an engine with an eager packed forward: the same function as the module's
   `forward` in fewer kernels, gated by `validate` against the module's own forward on the GPU
   in hand. The state it stores on the module exposes `graph_enabled` and
   `set_cuda_graph(enabled, config)`. To get CUDA graphs, give it
   `prepare_static(static)` and `padded_core(static)` and hand it to
   `runtime.PaddedGraphRunner`; the runner owns bucketing, capture, the shared pool and
   gathering real tokens back out. Graphs over padded rows are exact only if real tokens never
   read pads, which is the engine's contract to keep.
3. Reuse the probes: `runtime.select_attention` for varlen attention, `runtime.PinnedStager`
   for host-to-device metadata. Per-shape constants belong in `prepare_static`, which runs once
   per bucket before capture.
4. Import the toolchain lazily (inside `pack`/`validate`), so `import packed_encoders` stays
   free of it, and `register()` the plugin in `arch/__init__.py`.
