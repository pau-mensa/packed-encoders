"""EmbeddingGemma 2 text backbones (bidirectional GQA with sliding and full layers, per-layer
embeddings, a per-token output projection).

The engine (`engine.EmbeddingGemma2Engine`) is per text backbone (`EmbeddingGemma2TextModel`). The
adapter patches the multimodal wrapper (`EmbeddingGemma2Model`, which `AutoModel` and
SentenceTransformers load) when it is given one, else the text backbone itself; each patched forward
accepts `input_ids` (or, for the backbone, `inputs_embeds`) plus `attention_mask` (right or left
padded) and returns padded `last_hidden_state`. Packed output parity covers real tokens; padding
slots are zero, unlike HF's generally nonzero padding outputs. Mask-aware pooling runs unchanged.

What `pack()` installs, per batch:
  - token-id batches, no grad: CUDA graph replay when the batch fits a `(rows, S)` bucket, else the
    eager packed engine. The wrapper's text-only calls take this path with the ids it was given (its
    own forward would turn them into input embeddings first), mapping image, video and audio
    placeholders to padding as it does.
  - input-embedding batches at the backbone, no grad: the eager packed engine.
  - image, video or audio inputs, grad-enabled calls, active training dropout, extra outputs,
    explicit `position_ids`, a mask with holes or a prepared mask dictionary: the original forward.
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from packed_encoders.arch.weights import validation_ids_below
from packed_encoders.runtime.cuda_epilogue import CUDAEpilogueSupport
from packed_encoders.engine import Capabilities, ModelBinding, ValidationResult
from packed_encoders.errors import PackedEncodersError, ValidationError
from packed_encoders.runtime.graphs import (
    PaddedGraphConfig,
    PaddedGraphRunner,
    graphs_globally_disabled,
)
from packed_encoders.runtime.staging import PinnedStager
from packed_encoders.state import ATTR

MIN_CAPABILITY = (8, 0)   # bf16 tensor cores; the Triton kernels target sm_80+
# Validation measures the engine against the model's own forward on this device: per-token cosine
# over real tokens, bf16 vs bf16, for eager and graphed alike. The output projection amplifies
# rounding on a few tokens, so token minima are reported, not gated. Keep the batch mean gate
# and also check every sequence: long rows must not hide a broken short row. On the real
# checkpoint / RTX 5090 the lowest sequence mean was 0.98935; 0.98 leaves rounding margin
# while rejecting sequence-level corruption (measurements: docs/architectures.md).
MEAN_COS = 0.99
SEQUENCE_MEAN_COS = 0.98
_BACKENDS = {None: None, "auto": None, "flash": ("flash4", "flash2", "torch_varlen"), "sdpa": ("sdpa",)}
_MODULE = "transformers.models.embedding_gemma2."
_PLACEHOLDERS = ("image_token_id", "video_token_id", "audio_token_id")


def _is(module: object, name: str) -> bool:
    """By class name along the MRO, so matching never imports transformers' model code."""
    return isinstance(module, nn.Module) and any(c.__name__ == name and c.__module__.startswith(_MODULE)
                                                 for c in type(module).__mro__)


@dataclass
class EmbeddingGemma2Report:
    capability: tuple[int, int]
    torch_version: str
    attention_backend: dict[str, str]          # attention kind -> the kernel its probe chose
    attention_max_abs_err: dict[str, float]
    attention_rejected: dict[str, dict[str, str]]
    kernel_errors: dict[str, float]
    batches: tuple[tuple[int, ...], ...]
    eager_cos_mean: float | None = None        # the worst batch's per-token cosine vs the model's forward
    eager_cos_min: float | None = None
    embeds_cos_mean: float | None = None       # the inputs_embeds entry, eager
    embeds_cos_min: float | None = None
    graph_cos_mean: float | None = None        # graph replay, also against the model's forward
    graph_cos_min: float | None = None
    eager_sequence_cos_mean_min: float | None = None  # lowest mean over any individual sequence
    embeds_sequence_cos_mean_min: float | None = None
    graph_sequence_cos_mean_min: float | None = None
    pieces: dict = field(default_factory=dict)

    def record(self, name: str, got: Tensor, ref: Tensor, lengths) -> tuple[float, float, float]:
        """Record the batch mean, token minimum and worst sequence mean, retaining the worst seen."""
        cos = F.cosine_similarity(got, ref, dim=-1)
        mean, low = cos.mean().item(), cos.min().item()
        sequence_low = torch.stack([row.mean() for row in cos.split(list(lengths))]).min().item()
        for stat, value in (("cos_mean", mean), ("cos_min", low), ("sequence_cos_mean_min", sequence_low)):
            prev = getattr(self, f"{name}_{stat}")
            setattr(self, f"{name}_{stat}", value if prev is None else min(prev, value))
        return mean, low, sequence_low


@dataclass
class EmbeddingGemma2State:
    arch: str
    engine: Any
    original_forward: Any
    runner: PaddedGraphRunner | None
    graph_enabled: bool
    stager: PinnedStager
    report: EmbeddingGemma2Report | None = None
    fallback_warned: bool = False

    def set_cuda_graph(self, enabled: bool, config: PaddedGraphConfig | None = None) -> None:
        if config is not None and not isinstance(config, PaddedGraphConfig):
            raise PackedEncodersError(f"EmbeddingGemma 2 graphs take a runtime.PaddedGraphConfig, got {type(config).__name__}")
        if enabled and (self.runner is None or config is not None):
            self.runner = PaddedGraphRunner(self.engine, config)
        self.graph_enabled = enabled


def _hidden_once(state: EmbeddingGemma2State, ids: Tensor | None, lengths, embeds: Tensor | None,
                 graphs: bool) -> Tensor:
    hidden = None
    if (embeds is None and graphs and state.runner is not None and state.graph_enabled
            and not graphs_globally_disabled() and not torch.is_autocast_enabled("cuda")):
        state.engine.sync_norms()             # norms trained since packing reach the graphs too
        hidden = state.runner(ids, lengths)
    if hidden is None:
        hidden = state.engine.forward_packed(ids, lengths, embeds=embeds)
    return hidden


def _hidden(state: EmbeddingGemma2State, ids: Tensor | None, lengths, *, embeds: Tensor | None = None,
            graphs: bool = True) -> Tensor:
    # Each attempt in its own frame, so a failed attempt's allocations are released before the retry.
    for attempt in (0, 1):
        try:
            return _hidden_once(state, ids, lengths, embeds, graphs)
        except torch.OutOfMemoryError as exc:
            if attempt or state.runner is None:
                raise
            reason = (str(exc).splitlines() or ["CUDA out of memory"])[0][:160]
        _drop_graphs(state, reason)          # outside the handler: its traceback would keep the graphs alive


def _drop_graphs(state: EmbeddingGemma2State, reason: str) -> None:
    """Out of GPU memory with graphs held: free them all (their pool can't lend to eager) and go on."""
    with torch.cuda.device(state.engine.device):
        state.runner, state.graph_enabled = None, False
        gc.collect()
        torch.cuda.empty_cache()
    warnings.warn(f"packed-encoders: out of GPU memory with CUDA graphs held ({reason}); dropped them and "
                  "continuing without. Disable graphs with cuda_graph=False or reduce the buckets with "
                  "PaddedGraphConfig(max_tokens=...).", stacklevel=3)


def _hf_rows(state: EmbeddingGemma2State, input_ids, embeds, attention_mask, args,
             kwargs) -> tuple[list[int], list[int]] | None:
    """Per-row token counts and first real column, or None when the engine can't serve the call."""
    if args or torch.is_grad_enabled() or (input_ids is None) == (embeds is None):
        return None
    if any(layer.module.training and layer.module.attention_dropout for layer in state.engine.layers):
        return None          # no_grad does not disable HF's training dropout
    if input_ids is not None and input_ids.dim() != 2:
        return None
    if embeds is not None and (embeds.dim() != 3 or embeds.shape[-1] != state.engine.model_dim):
        return None
    cfg = state.engine.cfg
    if any(kwargs.get(k, getattr(cfg, k, False)) for k in ("output_hidden_states", "output_attentions")):
        return None
    for k, v in kwargs.items():
        if k == "return_dict" or v is None or v is False:
            continue
        return None          # position_ids, image/video/audio inputs, extra outputs
    B, S = (input_ids if embeds is None else embeds).shape[:2]
    if attention_mask is None:
        return [S] * B, [0] * B
    if not isinstance(attention_mask, Tensor) or attention_mask.shape != (B, S):
        return None
    real = attention_mask != 0
    lens = real.sum(1)
    col = torch.arange(S, device=real.device)
    right = (real == (col < lens[:, None])).all(1)
    left = (real == (col >= S - lens[:, None])).all(1)
    flat = torch.cat([lens, right.long(), left.long()]).tolist()      # the only device->host read
    lengths, right, left = flat[:B], flat[B:2 * B], flat[2 * B:]
    if min(lengths) == 0 or not all(r or l for r, l in zip(right, left)):
        return None          # an empty row, or a mask with holes
    return lengths, [0 if r else S - n for n, r in zip(lengths, right)]


def _make_hf_forward(module: nn.Module, state: EmbeddingGemma2State, *, wrapper: bool):
    if wrapper:
        from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
            EmbeddingGemma2ModelOutput as Output,
        )
        pad = module.config.text_config.pad_token_id
        placeholders = [i for i in (getattr(module.config, k, None) for k in _PLACEHOLDERS) if i is not None]
    else:
        from transformers.modeling_outputs import BaseModelOutput as Output

    def serve(input_ids, attention_mask, args, kwargs, original):
        embeds = kwargs.pop("inputs_embeds", None)
        # The wrapper builds its own embeddings; given some, it reads its placeholders off them.
        rows = None if wrapper and embeds is not None else _hf_rows(state, input_ids, embeds, attention_mask, args, kwargs)
        if rows is None:
            if not state.fallback_warned:
                state.fallback_warned = True
                warnings.warn("packed-encoders: this call runs the model's original forward (gradients, training "
                              "dropout, media inputs, position_ids, extra outputs or unsupported masks)", stacklevel=3)
            if embeds is not None:
                kwargs["inputs_embeds"] = embeds
            return original()
        lengths, starts = rows
        if wrapper and placeholders:
            hits = input_ids == placeholders[0]
            for i in placeholders[1:]:
                hits |= input_ids == i
            input_ids = input_ids.masked_fill(hits, pad)
        B, S = (input_ids if embeds is None else embeds).shape[:2]
        ids = None if input_ids is None else input_ids.reshape(-1)
        flat = None if embeds is None else embeds.reshape(B * S, -1)
        if sum(lengths) == B * S:
            out = _hidden(state, ids, lengths, embeds=flat).view(B, S, -1)
        else:
            # Row i's tokens are the flat slots [i*S + start_i, i*S + start_i + len_i); built on host.
            lens = torch.as_tensor(lengths, dtype=torch.long)
            first = torch.arange(B) * S + torch.as_tensor(starts, dtype=torch.long)
            idx = torch.arange(int(lens.sum())) + torch.repeat_interleave(first - (torch.cumsum(lens, 0) - lens), lens)
            (d_idx,) = state.stager.put([idx])
            hidden = _hidden(state, None if ids is None else ids.index_select(0, d_idx), lengths,
                             embeds=None if flat is None else flat.index_select(0, d_idx))
            out = hidden.new_zeros((B * S, hidden.shape[-1])).index_copy_(0, d_idx, hidden).view(B, S, -1)
        return_dict = kwargs.get("return_dict")
        if return_dict is None:
            return_dict = getattr(module.config, "return_dict", True)
        return Output(last_hidden_state=out) if return_dict else (out,)

    if wrapper:     # its second positional parameter is pixel_values
        def forward(input_ids=None, *args, **kwargs):
            mask = kwargs.pop("attention_mask", None)
            return serve(input_ids, mask, args, kwargs,
                         lambda: state.original_forward(input_ids, *args, attention_mask=mask, **kwargs))
    else:
        def forward(input_ids=None, attention_mask=None, *args, **kwargs):
            return serve(input_ids, attention_mask, args, kwargs,
                         lambda: state.original_forward(input_ids, attention_mask, *args, **kwargs))
    return forward


class HFEmbeddingGemma2Adapter:
    name = "hf-embedding_gemma2"

    def bind(self, module: object) -> ModelBinding | None:
        if _is(module, "EmbeddingGemma2TextModel"):
            return ModelBinding(self, module, module)
        lm = getattr(module, "language_model", None)
        if _is(module, "EmbeddingGemma2Model") and _is(lm, "EmbeddingGemma2TextModel"):
            return ModelBinding(self, module, lm)
        return None

    def install(self, binding: ModelBinding, packed) -> None:
        target = binding.patch_target
        setattr(target, ATTR, packed.state)
        target.forward = _make_hf_forward(target, packed.state, wrapper=target is not binding.weight_source)


def _check_options(options):
    unknown = options.keys() - {"cuda_graph", "train_cuda_graph", "attention_backend", "validate"}
    if unknown:
        raise PackedEncodersError(f"unsupported EmbeddingGemma 2 options: {sorted(unknown)}")
    graph = options.get("cuda_graph")
    if graph is not None and not isinstance(graph, (bool, PaddedGraphConfig)):
        raise PackedEncodersError("EmbeddingGemma 2 graphs take a bool or PaddedGraphConfig")
    if options.get("train_cuda_graph") not in (None, False):
        raise PackedEncodersError("training graphs are not implemented for EmbeddingGemma 2")
    if options.get("attention_backend") not in _BACKENDS:
        raise PackedEncodersError(f"unsupported EmbeddingGemma 2 attention_backend: {options['attention_backend']!r}")


class EmbeddingGemma2:
    name = "embedding_gemma2"
    adapters = (HFEmbeddingGemma2Adapter(),)

    def __init__(self, *, pieces=None):
        self.pieces = pieces

    @staticmethod
    def _require_env(module: nn.Module) -> torch.device:
        try:
            param = next(module.parameters())
        except StopIteration as exc:
            raise ValidationError("the model has no parameters") from exc
        if param.device.type != "cuda":
            raise ValidationError(f"the EmbeddingGemma 2 engine needs CUDA weights; the model is on {param.device}")
        cap = torch.cuda.get_device_capability(param.device)
        if cap < MIN_CAPABILITY:
            raise ValidationError(f"sm_{cap[0]}{cap[1]} is below sm_80 (bf16 + Triton kernels)")
        if param.dtype != torch.bfloat16:
            raise ValidationError("the EmbeddingGemma 2 engine runs bf16 weights; load with dtype=torch.bfloat16")
        return param.device

    def _validate(self, binding: ModelBinding, *,
                  batches: tuple[tuple[int, ...], ...] = ((7, 129, 1100, 64), (3, 21, 600)), graphs: bool = True,
                  _engine=None, _state: EmbeddingGemma2State | None = None) -> EmbeddingGemma2Report:
        """Build (or reuse) the engine and compare it with the model's own forward on random text
        rows, on this device: token ids eager and graphed, and input embeddings eager. Rows longer
        than twice the sliding window exercise it on both sides. The report keeps the worst batch.
        Raises ValidationError on any miss."""
        module = binding.patch_target
        device = self._require_env(module)
        state = _state if _state is not None else getattr(module, ATTR, None)
        engine = _engine or (state.engine if state is not None else None)
        built_here = engine is None
        if built_here:
            engine = self._engine(binding.weight_source, None)
        try:
            report = EmbeddingGemma2Report(
                capability=torch.cuda.get_device_capability(device), torch_version=torch.__version__,
                attention_backend={k: c.name for k, c in engine.attention.items()},
                attention_max_abs_err={k: c.max_abs_err for k, c in engine.attention.items()},
                attention_rejected={k: dict(c.rejected) for k, c in engine.attention.items()},
                kernel_errors=dict(engine.kernel_errors), batches=tuple(tuple(b) for b in batches))
            with torch.cuda.device(device):
                report.pieces = engine.composition.validate(engine)
            oracle = state.original_forward if state is not None else module.forward
            probe_state = state or EmbeddingGemma2State(arch=self.name, engine=engine, original_forward=oracle,
                                                         runner=None, graph_enabled=False,
                                                         stager=PinnedStager(engine.device))
            # The wrapper names its image/video/audio placeholders (no vocab_size): draw below those too.
            named = SimpleNamespace(**{**vars(module.config), "vocab_size": engine.cfg.vocab_size})
            below = min(validation_ids_below(engine.cfg), validation_ids_below(named))
            for lengths in batches:
                self._check_numerics(oracle, probe_state, lengths, graphs, below, report)
        finally:
            if built_here:
                engine.release(rollback=True)
        return report

    @staticmethod
    def _reference(oracle, engine, seqs: list[Tensor]) -> Tensor:
        """The model's own forward on right-padded `seqs`: real tokens in packed order (fp32)."""
        lengths = [len(x) for x in seqs]
        ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(engine.device)
        mask = (torch.arange(ids.shape[1])[None] < torch.tensor(lengths)[:, None]).to(engine.device)
        try:
            return oracle(input_ids=ids, attention_mask=mask.long())[0][mask].float()
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"the model's own forward failed on the validation batch ({type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:200]}); cannot certify the engine here. Pass validate=False to "
                "skip at your own risk.") from exc

    def _check_numerics(self, oracle, state: EmbeddingGemma2State, lengths, graphs, below,
                        report: EmbeddingGemma2Report) -> None:
        engine = state.engine
        g = torch.Generator().manual_seed(0)
        seqs = [torch.randint(0, below, (n,), generator=g) for n in lengths]
        B = len(seqs)
        packed = torch.cat(seqs).to(engine.device)
        with torch.no_grad():
            ref = self._reference(oracle, engine, seqs)
            eager = _hidden(state, packed, list(lengths), graphs=False).float()
            via_embeds = _hidden(state, None, list(lengths), embeds=engine._embed(packed), graphs=False).float()
            for name, got in (("eager", eager), ("embeds", via_embeds)):
                mean, low, sequence_low = report.record(name, got, ref, lengths)
                if not (mean >= MEAN_COS and sequence_low >= SEQUENCE_MEAN_COS):
                    raise ValidationError(f"engine ({name}) diverged from the model's forward on rows {tuple(lengths)}: "
                                          f"per-token cosine mean {mean:.6f} min {low:.5f}; "
                                          f"worst sequence mean {sequence_low:.6f}")
            if graphs:
                runner = PaddedGraphRunner(engine, PaddedGraphConfig(row_buckets=(B,), max_graphs=1))
                graphed = runner(packed, list(lengths)).float()
                del runner
                mean, low, sequence_low = report.record("graph", graphed, ref, lengths)
                if not (mean >= MEAN_COS and sequence_low >= SEQUENCE_MEAN_COS):
                    raise ValidationError(f"CUDA-graph replay diverged from the model's forward on rows {tuple(lengths)}: "
                                          f"per-token cosine mean {mean:.6f} min {low:.5f}; "
                                          f"worst sequence mean {sequence_low:.6f}")

    def _engine(self, text_model: nn.Module, attention_backend: str | None):
        from packed_encoders.arch.embedding_gemma2.engine import EmbeddingGemma2Engine

        if attention_backend not in _BACKENDS:
            raise PackedEncodersError(f"attention_backend for EmbeddingGemma 2 must be one of {sorted(map(str, _BACKENDS))}")
        return EmbeddingGemma2Engine(text_model, attention_order=_BACKENDS[attention_backend], pieces=self.pieces)

    def validate(self, binding: ModelBinding, **kwargs) -> ValidationResult:
        return ValidationResult(self.name, self._validate(binding, **kwargs))

    def prepare(self, binding: ModelBinding, options):
        _check_options(options)
        self._require_env(binding.patch_target)
        engine = self._engine(binding.weight_source, options.get("attention_backend"))
        try:
            graph = options.get("cuda_graph")
            use_graphs = graph is None or bool(graph)
            runner = PaddedGraphRunner(engine, graph if isinstance(graph, PaddedGraphConfig) else None) if use_graphs else None
            state = EmbeddingGemma2State(arch=self.name, engine=engine, original_forward=binding.patch_target.forward,
                                         runner=runner, graph_enabled=use_graphs, stager=PinnedStager(engine.device))
            packed = PackedEmbeddingGemma2(self, binding, state, options.get("attention_backend"))
            if options.get("validate", True):
                state.report = packed.validate(graphs=use_graphs).details
            return packed
        except BaseException:
            engine.release(rollback=True)
            raise


class PackedEmbeddingGemma2(CUDAEpilogueSupport):
    capabilities = Capabilities(inference_capture=True, original_forward_fallback=True)

    def __init__(self, owner, binding, state, backend):
        self.owner, self.binding, self.state = owner, binding, state
        self.attention_backend = backend
        self.composition = state.engine.composition
        self.pieces = self.composition.all()
        self._closed = False
        self._init_epilogues()

    def _require_open(self):
        if self._closed:
            raise PackedEncodersError("this packed engine has been closed; pack the model again")

    @property
    def graph_enabled(self):
        return self.state.graph_enabled

    @graph_enabled.setter
    def graph_enabled(self, enabled):
        self._require_open()
        self.state.graph_enabled = enabled

    def forward_packed(self, batch):
        """Projected per-token hidden states for flat device int64 ids and positive host_lengths;
        positions restart at zero for each sequence. Inference only."""
        self._require_open()
        ids, lengths = batch.input_ids, batch.host_lengths
        if torch.is_grad_enabled():
            raise PackedEncodersError("EmbeddingGemma 2 forward_packed requires no_grad or inference_mode")
        if ids.ndim != 1 or ids.dtype != torch.int64 or ids.device != self.state.engine.device:
            raise PackedEncodersError("EmbeddingGemma 2 requires flat int64 token IDs on the weights' device")
        if not lengths or any(type(n) is not int or n <= 0 for n in lengths) or sum(lengths) != ids.numel():
            raise PackedEncodersError("EmbeddingGemma 2 requires positive host_lengths summing to the token count")
        if batch.cu_seqlens is not None or batch.position_ids is not None or batch.max_seqlen is not None:
            raise PackedEncodersError("EmbeddingGemma 2 takes host_lengths only; positions restart per sequence")
        return _hidden(self.state, ids, lengths)

    def validate(self, **kwargs):
        self._require_open()
        return ValidationResult(self.owner.name, self.owner._validate(
            self.binding, _engine=self.state.engine, _state=self.state, **kwargs))

    def configure(self, options):
        self._require_open()
        _check_options(options)
        if "attention_backend" in options and options["attention_backend"] != self.attention_backend:
            raise PackedEncodersError("unpack before changing the prepared attention backend")
        if "cuda_graph" in options and options["cuda_graph"] is not None:
            graph = options["cuda_graph"]
            self.set_cuda_graph(bool(graph), graph if isinstance(graph, PaddedGraphConfig) else None)

    def set_cuda_graph(self, enabled, config=None):
        self._require_open()
        self.state.set_cuda_graph(enabled, config)

    def set_train_cuda_graph(self, enabled, config=None):
        self._require_open()
        if enabled or config is not None:
            raise PackedEncodersError("training graphs are not implemented for EmbeddingGemma 2")

    def close(self, *, rollback=False):
        if self._closed:
            return
        self._close_epilogues()
        self.state.runner = None
        self.state.graph_enabled = False
        self.state.engine.release(rollback=rollback)
        self.state.engine._stager = None
        self.state.stager = None
        self._closed = True


__all__ = ["EmbeddingGemma2", "EmbeddingGemma2Report", "HFEmbeddingGemma2Adapter", "PackedEmbeddingGemma2"]
