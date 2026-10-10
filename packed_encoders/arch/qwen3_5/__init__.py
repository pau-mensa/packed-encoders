"""Qwen3.5 hybrid backbones (GatedDeltaNet + gated softmax attention).

The engine (`engine.Qwen35Engine`) is per *backbone*; entry adapters are per *wrapper
signature*, and each patched forward keeps its wrapper's contract exactly:
  - topk-embed-v1 (`TopkEmbedModel`, xsmall and small): its HF forward receives an
    already-packed batch and returns padded per-token vectors, so
    `SentenceTransformer.encode()` inherits the speedup with no adapter.
  - stock HF (`Qwen3_5TextModel`, or `Qwen3_5Model` and its subclasses, such as embedding
    models and the backbones of decision models): `input_ids` + `attention_mask` (right or
    left padded) in, padded `last_hidden_state` out, causal or bidirectional as
    `config.is_causal` says. The model's own pooling or readout then runs unchanged.

What `pack()` installs, per batch:
  - text batches, no grad, rows sharing at least `min_shared_prefix` leading tokens (causal
    engines, opt in; a decision model asking several questions about one context): each
    shared prefix runs once (`engine.forward_shared`).
  - other text batches, no grad: CUDA graph replay when the batch fits a `(rows, S)` bucket, else
    the eager packed engine; then (topk) the model's own head, slice, and L2 norm.
  - image batches, grad-enabled calls, and anything else the engine can't serve: the
    original forward, untouched.

fla (flash-linear-attention) is imported only when a Qwen3.5 model is actually packed.
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from packed_encoders.arch.weights import validation_ids_below as _validation_ids_below
from packed_encoders.errors import PackedEncodersError, ValidationError
from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner, graphs_globally_disabled
from packed_encoders.runtime.staging import PinnedStager
from packed_encoders.state import ATTR
from packed_encoders.engine import Capabilities, ModelBinding, ValidationResult

MIN_CAPABILITY = (8, 0)   # bf16 tensor cores; fla's Triton kernels target sm_80+
# Validation is a measurement on the actual device (engine vs the model's own forward), not a
# table of blessed GPUs. Per-token cosine over real tokens, bf16 vs bf16:
EAGER_MEAN_COS, EAGER_MIN_COS = 0.999, 0.98
GRAPH_MEAN_COS, GRAPH_MIN_COS = 0.9995, 0.99
_BACKENDS = {None: None, "auto": None, "flash": ("flash4", "flash2", "torch_varlen"), "sdpa": ("sdpa",)}


def _topk_text_model(module: nn.Module) -> nn.Module | None:
    cfg = getattr(module, "config", None)
    if getattr(cfg, "model_type", None) != "topk_embed" or not isinstance(getattr(module, "head", None), nn.Linear):
        return None
    lm = getattr(getattr(module, "model", None), "language_model", None)
    if getattr(getattr(lm, "config", None), "model_type", None) != "qwen3_5_text":
        return None
    return lm


def _hf_text_model(module: nn.Module) -> nn.Module | None:
    """`Qwen3_5TextModel`, or the one a `Qwen3_5Model` (or subclass) holds as `.language_model`.
    By class name along the MRO, so matching never imports transformers' Qwen3.5 code."""
    names = {c.__name__ for c in type(module).__mro__ if c.__module__.startswith("transformers.models.qwen3_5.")}
    if "Qwen3_5TextModel" in names:
        return module
    if "Qwen3_5Model" in names:
        lm = getattr(module, "language_model", None)
        return lm if lm is not None and _hf_text_model(lm) is lm else None
    return None


@dataclass
class Qwen35Report:
    capability: tuple[int, int]
    torch_version: str
    fla_version: str
    attention_backend: str
    attention_max_abs_err: float
    attention_rejected: dict[str, str]
    qk_kernel_max_abs_err: float | None
    gdn_kernel_rel_err: dict[str, float]
    gdn_rejected: dict[str, str]
    gdn_recurrent_max_len: int
    batches: tuple[tuple[int, ...], ...]
    fused: bool = False
    fusion_errors: dict[str, float] = field(default_factory=dict)
    fusion_rejected: dict[str, str] = field(default_factory=dict)
    eager_cos_mean: float = 0.0
    eager_cos_min: float = 0.0
    graph_cos_mean: float | None = None
    graph_cos_min: float | None = None
    shared_prefix_rejected: str | None = None  # why rows can't share prefixes here (None: they can, opt in)
    shared_cos_mean: float | None = None       # rows continuing shared prefixes vs the model's forward
    shared_cos_min: float | None = None
    notes: list[str] = field(default_factory=list)
    pieces: dict = field(default_factory=dict)


@dataclass
class Qwen35State:
    """Attached to the packed model under `ATTR`; a stock HF model has no head (`head=None`)."""

    arch: str
    engine: Any
    original_forward: Any
    runner: PaddedGraphRunner | None
    graph_enabled: bool
    head: Tensor | None
    dim: int
    normalize: bool
    stager: PinnedStager
    report: Qwen35Report | None = None
    fallback_warned: bool = False
    shared_runner: Any = None

    def set_cuda_graph(self, enabled: bool, config: PaddedGraphConfig | None = None) -> None:
        if config is not None and not isinstance(config, PaddedGraphConfig):
            raise PackedEncodersError("Qwen3.5 graphs take a runtime.PaddedGraphConfig, "
                                      f"got {type(config).__name__}")
        if enabled and (self.runner is None or config is not None):
            self.runner = PaddedGraphRunner(self.engine, config)
            self.shared_runner = None
        self.graph_enabled = enabled


def _vectors(state: Qwen35State, hidden: Tensor) -> Tensor:
    if state.head is None:
        return hidden
    v = F.linear(hidden, state.head).float()[..., : state.dim]
    return F.normalize(v, p=2, dim=-1) if state.normalize else v


def _hidden_once(state: Qwen35State, ids: Tensor, lengths, *, graphs: bool) -> Tensor:
    state.engine.sync_norms()                 # norms trained since packing reach the graphs too
    plan = state.engine.plan_sharing(ids, lengths)
    use_graph = (graphs and state.runner is not None and state.graph_enabled
                 and not graphs_globally_disabled() and not torch.is_autocast_enabled("cuda"))
    if plan is not None:
        if use_graph:
            from packed_encoders.arch.qwen3_5.prefix_graphs import SharedGraphRunner
            if state.shared_runner is None:
                state.shared_runner = SharedGraphRunner(state.engine, state.runner.config)
            hidden = state.shared_runner(ids, plan)
            if hidden is not None:
                return hidden
        return state.engine.forward_shared(ids, plan)
    hidden = None
    if use_graph:
        hidden = state.runner(ids, lengths)
    if hidden is None:
        hidden = state.engine.forward_packed(ids, lengths)
    return hidden


def _with_graph_recovery(state: Qwen35State, forward) -> Tensor:
    # Keep each attempt in its own frame so failed outputs and the exception's
    # traceback release their allocations before graph cleanup and eager retry.
    for attempt in (0, 1):
        try:
            return forward()
        except torch.OutOfMemoryError as exc:
            if attempt or state.runner is None:
                raise
            reason = (str(exc).splitlines() or ["CUDA out of memory"])[0][:160]
        _drop_graphs(state, reason)          # outside the handler: its traceback would keep the graphs alive


def _drop_graphs(state: Qwen35State, reason: str) -> None:
    """Graphs are an optimisation, and their private memory pool can't lend to the eager path: out of
    GPU memory with graphs held, free them all and continue eager."""
    with torch.cuda.device(state.engine.device):
        state.runner, state.graph_enabled = None, False
        state.shared_runner = None
        gc.collect()
        torch.cuda.empty_cache()
    warnings.warn(f"packed-encoders: out of GPU memory with CUDA graphs held ({reason}); dropped them and "
                  "continuing without. Disable graphs with cuda_graph=False or reduce the buckets with "
                  "PaddedGraphConfig(max_tokens=...).", stacklevel=3)


def _hidden(state: Qwen35State, ids: Tensor, lengths, *, graphs: bool) -> Tensor:
    return _with_graph_recovery(state, lambda: _hidden_once(state, ids, lengths, graphs=graphs))


def _encode_text(state: Qwen35State, ids: Tensor, lengths, *, graphs: bool) -> Tensor:
    return _with_graph_recovery(state, lambda: _vectors(state, _hidden_once(state, ids, lengths, graphs=graphs)))


def _make_topk_forward(module: nn.Module, state: Qwen35State):
    def forward(input_ids, attention_mask, packed_ids, position_ids, cu_seqlens, seq_idx, pixel_values=None,
                image_scatter_index=None, vision_inputs=None, return_dict=True):
        if any(x is not None for x in (pixel_values, image_scatter_index, vision_inputs)) or torch.is_grad_enabled():
            if not state.fallback_warned:
                state.fallback_warned = True
                warnings.warn("packed-encoders: image inputs and grad-enabled calls run topk's original forward",
                              stacklevel=2)
            return state.original_forward(input_ids, attention_mask, packed_ids, position_ids, cu_seqlens, seq_idx,
                                          pixel_values=pixel_values, image_scatter_index=image_scatter_index,
                                          vision_inputs=vision_inputs, return_dict=return_dict)
        B, S = input_ids.shape
        lengths = cu_seqlens.diff().tolist()          # B small ints: the only device->host read
        vectors = _encode_text(state, packed_ids.reshape(-1), lengths, graphs=True)
        # Same layout as the original: (B, S, dim), real tokens row-major, zeros elsewhere.
        # Right padding makes row i's tokens the flat slots [i*S, i*S + len_i); built on host.
        lens = torch.as_tensor(lengths, dtype=torch.long)
        cu = torch.zeros(B + 1, dtype=torch.long)
        torch.cumsum(lens, 0, out=cu[1:])
        dst = torch.arange(int(cu[-1])) - torch.repeat_interleave(cu[:-1] - torch.arange(B) * S, lens)
        (d_dst,) = state.stager.put([dst])
        out = vectors.new_zeros((B * S, state.dim))
        out.index_copy_(0, d_dst, vectors)
        return out.view(B, S, state.dim)

    return forward


def _hf_rows(state: Qwen35State, input_ids, attention_mask, args, kwargs) -> tuple[list[int], list[int]] | None:
    """Per-row token counts and first real column, or None when the engine can't serve the call."""
    if args or input_ids is None or input_ids.dim() != 2 or torch.is_grad_enabled():
        return None
    # HF resolves cache defaults on the text backbone (also for Qwen3_5Model).
    # Its output-capture decorator uses the config only for omitted flags: an
    # explicit None disables capture, whereas use_cache=None inherits the config.
    cfg = state.engine.cfg
    use_cache = kwargs.get("use_cache")
    if use_cache is None:
        use_cache = getattr(cfg, "use_cache", False)
    if use_cache or any(kwargs.get(k, getattr(cfg, k, False))
                        for k in ("output_hidden_states", "output_attentions")):
        return None
    for k, v in kwargs.items():
        if k == "is_causal":
            if v is None or bool(v) == state.engine.causal:
                continue
            return None
        # mm_token_type_ids (Qwen3-VL processors always return it) only matters next to image or video inputs.
        if k in ("return_dict", "mm_token_type_ids") or v is None or v is False:
            continue
        return None          # images, inputs_embeds, position_ids, a cache or use_cache=True, extra outputs
    B, S = input_ids.shape
    if attention_mask is None:
        return [S] * B, [0] * B
    if attention_mask.shape != input_ids.shape:
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


def _make_hf_forward(module: nn.Module, state: Qwen35State):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast

    def forward(input_ids=None, attention_mask=None, *args, **kwargs):
        rows = _hf_rows(state, input_ids, attention_mask, args, kwargs)
        if rows is None:
            if not state.fallback_warned:
                state.fallback_warned = True
                warnings.warn("packed-encoders: this call runs the model's original forward (images, gradients, "
                              "a KV cache, position_ids / inputs_embeds, extra outputs or a mask with holes)",
                              stacklevel=2)
            return state.original_forward(input_ids, attention_mask, *args, **kwargs)
        lengths, starts = rows
        B, S = input_ids.shape
        ids = input_ids.reshape(-1)
        if sum(lengths) == B * S:
            out = _encode_text(state, ids, lengths, graphs=True).view(B, S, -1)
        else:
            # Row i's tokens are the flat slots [i*S + start_i, i*S + start_i + len_i); built on host.
            lens = torch.as_tensor(lengths, dtype=torch.long)
            first = torch.arange(B) * S + torch.as_tensor(starts, dtype=torch.long)
            idx = torch.arange(int(lens.sum())) + torch.repeat_interleave(first - (torch.cumsum(lens, 0) - lens), lens)
            (d_idx,) = state.stager.put([idx])
            hidden = _encode_text(state, ids.index_select(0, d_idx), lengths, graphs=True)
            out = hidden.new_zeros((B * S, hidden.shape[-1])).index_copy_(0, d_idx, hidden).view(B, S, -1)
        return_dict = kwargs.get("return_dict")
        if return_dict is None:
            return_dict = getattr(module.config, "return_dict", True)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=out) if return_dict else (out,)

    return forward


class TopkEmbedAdapter:
    name = "topk-embed-v1"

    def bind(self, module: object) -> ModelBinding | None:
        lm = _topk_text_model(module)
        return ModelBinding(self, module, lm) if lm is not None else None

    def install(self, binding: ModelBinding, packed) -> None:
        setattr(binding.patch_target, ATTR, packed.state)
        binding.patch_target.forward = _make_topk_forward(binding.patch_target, packed.state)


class HFQwen35Adapter:
    name = "hf-qwen3_5"

    def bind(self, module: object) -> ModelBinding | None:
        lm = _hf_text_model(module)
        return ModelBinding(self, module, lm) if lm is not None else None

    def install(self, binding: ModelBinding, packed) -> None:
        setattr(binding.patch_target, ATTR, packed.state)
        binding.patch_target.forward = _make_hf_forward(binding.patch_target, packed.state)


@dataclass(frozen=True)
class DecisionBinding(ModelBinding):
    causal: bool


class DecisionModelAdapter(HFQwen35Adapter):
    """Read autojev's attention contract without modifying its config or mask hook."""
    name = "pplx-decider"

    def bind(self, module: object) -> ModelBinding | None:
        if not any(c.__name__ == "DecisionModel" and c.__module__ == "autojev.model"
                   for c in type(module).__mro__):
            return None
        backbone = getattr(module, "backbone", None)
        lm = _hf_text_model(backbone)
        if lm is None:
            return None
        mode = getattr(module, "attention_mode", "causal")
        if mode not in ("causal", "noncausal_full_attention"):
            raise PackedEncodersError(f"unsupported decision attention mode: {mode!r}")
        return DecisionBinding(self, backbone, lm, causal=mode == "causal")


def _check_options(options):
    unknown = options.keys() - {"cuda_graph", "train_cuda_graph", "attention_backend", "validate"}
    if unknown:
        raise PackedEncodersError(f"unsupported Qwen3.5 options: {sorted(unknown)}")
    graph = options.get("cuda_graph")
    if graph is not None and not isinstance(graph, (bool, PaddedGraphConfig)):
        raise PackedEncodersError("Qwen3.5 graphs take a bool or PaddedGraphConfig")
    if options.get("train_cuda_graph") not in (None, False):
        raise PackedEncodersError("training graphs are not implemented for Qwen3.5")
    if options.get("attention_backend") not in _BACKENDS:
        raise PackedEncodersError(f"unsupported Qwen3.5 attention_backend: {options['attention_backend']!r}")


class Qwen35Hybrid:
    name = "qwen3_5"
    adapters = (DecisionModelAdapter(), TopkEmbedAdapter(), HFQwen35Adapter())

    def __init__(self, *, pieces=None):
        self.pieces = pieces

    # ------------------------------------------------------------------ gates
    @staticmethod
    def _require_env(module: nn.Module) -> tuple[torch.device, str]:
        try:
            device = next(module.parameters()).device
        except StopIteration as exc:
            raise ValidationError("the model has no parameters") from exc
        if device.type != "cuda":
            raise ValidationError(f"the Qwen3.5 engine needs CUDA weights; the model is on {device}")
        cap = torch.cuda.get_device_capability(device)
        if cap < MIN_CAPABILITY:
            raise ValidationError(f"sm_{cap[0]}{cap[1]} is below sm_80 (bf16 + fla kernels)")
        if next(module.parameters()).dtype != torch.bfloat16:
            raise ValidationError("the Qwen3.5 engine runs bf16 weights; load with torch_dtype=torch.bfloat16")
        try:
            import fla
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        except ImportError as exc:
            raise ValidationError("Qwen3.5 packing needs flash-linear-attention; see docs/torch-2.11-review.md "
                                  "for the topk checkpoint environment") from exc
        # The kernel flags pass through fla's **kwargs, so no signature check: the engine probes the
        # kernels numerically when it is built (engine.select_gdn).
        return device, fla.__version__

    def _validate(self, module: nn.Module, *, batches: tuple[tuple[int, ...], ...] = ((7, 129, 300, 64), (3, 21, 30)),
                 graphs: bool = True, _engine=None, _state: Qwen35State | None = None, causal: bool | None = None) -> Qwen35Report:
        """Build (or reuse) the engine and compare it with the model's own forward on random text
        rows, on this device, eager and graphed. The default batches cover both GatedDeltaNet
        kernels (eager, a batch whose rows are all <= RECURRENT_MAX_LEN tokens runs the recurrent
        one, any other the chunked one; graphs always run the chunked one). The report keeps the
        worst batch. Raises ValidationError on any miss."""
        device, fla_version = self._require_env(module)
        state = _state if _state is not None else getattr(module, ATTR, None)
        engine = _engine or (state.engine if state is not None else None)
        built_here = engine is None
        if built_here:
            engine = self._engine(module, None, causal=causal)
        try:
            report = Qwen35Report(
                capability=torch.cuda.get_device_capability(device), torch_version=torch.__version__,
                fla_version=fla_version, attention_backend=engine.attention.name if engine.attention else "none",
                attention_max_abs_err=engine.attention.max_abs_err if engine.attention else 0.0,
                attention_rejected=dict(engine.attention.rejected) if engine.attention else {},
                qk_kernel_max_abs_err=engine.qk_check_err,
                gdn_kernel_rel_err=dict(engine.gdn.errors) if engine.gdn else {},
                gdn_rejected=dict(engine.gdn.rejected) if engine.gdn else {},
                gdn_recurrent_max_len=engine.gdn.recurrent_max_len if engine.gdn else 0,
                batches=tuple(tuple(b) for b in batches), eager_cos_min=1.0, fused=engine.fused,
                fusion_errors=dict(engine.fusion_errors), fusion_rejected=dict(engine.fusion_rejected))
            with torch.cuda.device(device):
                report.pieces = engine.composition.validate(engine)
            oracle = state.original_forward if state is not None else module.forward
            probe_state = state or self._state(module, engine, oracle, runner=None, graph_enabled=False)
            for lengths in batches:
                self._check_numerics(module, oracle, probe_state, lengths, graphs, report)
            if engine.share_rejected is None:
                self._check_sharing(module, oracle, probe_state, report)
            report.shared_prefix_rejected = engine.share_rejected
            if engine.share_rejected and engine.causal:
                report.notes.append(f"shared prefixes unavailable: {engine.share_rejected}")
                warnings.warn(f"packed-encoders: {engine.share_rejected}; every row runs in full", stacklevel=4)
        finally:
            if built_here:
                engine.release(rollback=True)
        return report

    def _check_sharing(self, module, oracle, state: Qwen35State, report: Qwen35Report) -> None:
        """Rows continuing two shared prefixes (children of 1 to 20 tokens) beside rows sharing nothing,
        against the model's own forward, whether or not sharing is on. A miss makes sharing unavailable
        (recorded) rather than failing the pack."""
        from packed_encoders.arch.qwen3_5.engine import SHARED_PREFIX_MIN

        engine, threshold = state.engine, state.engine.min_shared_prefix
        m = threshold or SHARED_PREFIX_MIN
        g = torch.Generator().manual_seed(1)

        def rand(n):
            return torch.randint(0, _validation_ids_below(engine.cfg), (n,), generator=g)

        a, b = rand(m + 30), rand(m + 6)
        seqs = [torch.cat([a, rand(7)]), rand(40), torch.cat([b, rand(12)]), torch.cat([a, rand(1)]),
                rand(m + 50), torch.cat([b, rand(3)]), torch.cat([a, rand(20)])]
        reason = None
        try:
            engine.min_shared_prefix = m
            if engine.plan_sharing(torch.cat(seqs).to(engine.device), [len(s) for s in seqs]) is None:
                raise ValidationError("the shared-prefix validation batch formed no group")
            shared, ref, _ = self._compare(module, oracle, state, seqs)
            cos = F.cosine_similarity(shared, ref, dim=-1)
            report.shared_cos_mean, report.shared_cos_min = cos.mean().item(), cos.min().item()
            if not (report.shared_cos_mean >= EAGER_MEAN_COS and report.shared_cos_min >= EAGER_MIN_COS):
                reason = (f"rows continuing shared prefixes diverged from the model's forward: per-token cosine "
                          f"mean {report.shared_cos_mean:.6f} min {report.shared_cos_min:.5f}")
        except Exception as exc:  # optional optimization; ordinary numerics have already passed
            reason = f"shared-prefix validation failed: {type(exc).__name__}: {str(exc)[:200]}"
        finally:
            engine.min_shared_prefix = threshold if reason is None else 0
        if reason is not None:
            engine.share_rejected = reason

    def _compare(self, module, oracle, state: Qwen35State, seqs: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """The eager engine and the model's own forward on `seqs`, real tokens in packed order: (eager, ref, ids)."""
        device = state.engine.device
        topk = _topk_text_model(module) is not None
        lengths = [len(x) for x in seqs]
        B, S = len(seqs), max(lengths)
        lens = torch.tensor(lengths)
        packed = torch.cat(seqs)
        cu = torch.zeros(B + 1, dtype=torch.long)
        torch.cumsum(lens, 0, out=cu[1:])
        pos = torch.arange(len(packed)) - torch.repeat_interleave(cu[:-1], lens)
        mask = torch.arange(S)[None] < lens[:, None]
        feats = {"input_ids": torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True), "attention_mask": mask}
        if topk:
            feats.update(packed_ids=packed[None], position_ids=pos[None], cu_seqlens=cu,
                         seq_idx=torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens))
        else:
            feats.update(attention_mask=mask.long(), use_cache=False)
        feats = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in feats.items()}
        mask = mask.to(device)
        with torch.no_grad():
            try:
                out = oracle(**feats)
                ref = (out if topk else out[0])[mask].float()
            except Exception as exc:  # noqa: BLE001
                raise ValidationError(
                    f"the model's own forward failed on the validation batch ({type(exc).__name__}: "
                    f"{str(exc).splitlines()[0][:200]}); cannot certify the engine here. Pass validate=False to "
                    "skip at your own risk.") from exc
            ids = packed.to(device)
            return _encode_text(state, ids, lengths, graphs=False).float(), ref, ids

    def _check_numerics(self, module, oracle, state: Qwen35State, lengths, graphs, report: Qwen35Report) -> None:
        g = torch.Generator().manual_seed(0)
        seqs = [torch.randint(0, _validation_ids_below(state.engine.cfg), (n,), generator=g) for n in lengths]
        B = len(seqs)
        eager, ref, ids = self._compare(module, oracle, state, seqs)
        with torch.no_grad():
            cos = F.cosine_similarity(eager, ref, dim=-1)
            mean, low = cos.mean().item(), cos.min().item()
            report.eager_cos_mean = mean if report.eager_cos_mean == 0.0 else min(report.eager_cos_mean, mean)
            report.eager_cos_min = min(report.eager_cos_min, low)
            if not (mean >= EAGER_MEAN_COS and low >= EAGER_MIN_COS):
                raise ValidationError(f"engine diverged from the model's forward on rows {tuple(lengths)}: per-token "
                                      f"cosine mean {mean:.6f} min {low:.5f}")
            if graphs:
                runner = PaddedGraphRunner(state.engine, PaddedGraphConfig(row_buckets=(B,), max_graphs=1))
                hidden = runner(ids, list(lengths))
                graphed = _vectors(state, hidden).float()
                cos = F.cosine_similarity(graphed, eager, dim=-1)
                mean, low = cos.mean().item(), cos.min().item()
                del runner
                report.graph_cos_mean = mean if report.graph_cos_mean is None else min(report.graph_cos_mean, mean)
                report.graph_cos_min = low if report.graph_cos_min is None else min(report.graph_cos_min, low)
                if not (mean >= GRAPH_MEAN_COS and low >= GRAPH_MIN_COS):
                    raise ValidationError(f"CUDA-graph replay diverged from the eager engine on rows {tuple(lengths)}: "
                                          f"per-token cosine mean {mean:.6f} min {low:.5f}")

    # ------------------------------------------------------------------ install
    def _engine(self, module: nn.Module, attention_backend: str | None, *, causal: bool | None = None):
        from packed_encoders.arch.qwen3_5.engine import Qwen35Engine

        if attention_backend not in _BACKENDS:
            raise PackedEncodersError(f"attention_backend for Qwen3.5 must be one of {sorted(map(str, _BACKENDS))}")
        lm = _topk_text_model(module)
        if lm is not None:   # topk flags causal documents on the module
            causal = getattr(lm, "document_causal", False) if causal is None else causal
        else:                # stock HF follows config.is_causal, as its own masks do (default causal)
            lm = _hf_text_model(module)
            causal = getattr(lm.config, "is_causal", True) if causal is None else causal
        return Qwen35Engine(lm, causal=bool(causal), attention_order=_BACKENDS[attention_backend], pieces=self.pieces)

    def _state(self, module: nn.Module, engine, original_forward, *, runner, graph_enabled) -> Qwen35State:
        if _topk_text_model(module) is not None:
            head, dim, normalize = (module.head.weight, module.config.output_dim or module.config.dim,
                                    bool(module.config.normalize))
        else:                # stock HF: the final-normed hidden states are the output
            head, dim, normalize = None, engine.cfg.hidden_size, False
        return Qwen35State(arch=self.name, engine=engine, original_forward=original_forward, runner=runner,
                         graph_enabled=graph_enabled, head=head, dim=dim, normalize=normalize,
                         stager=PinnedStager(engine.device))

    def validate(self, binding: ModelBinding, **kwargs) -> ValidationResult:
        return ValidationResult(self.name, self._validate(binding.patch_target, causal=getattr(binding, "causal", None), **kwargs))

    def prepare(self, binding: ModelBinding, options):
        _check_options(options)
        module = binding.patch_target
        self._require_env(module)
        # The adapter reads the head directly as a bias-free projection too.
        from packed_encoders.arch.qwen3_5.engine import _require_plain

        if _topk_text_model(module) is not None:
            _require_plain([module.head])
        engine = self._engine(module, options.get("attention_backend"), causal=getattr(binding, "causal", None))
        try:
            graph = options.get("cuda_graph")
            use_graphs = graph is None or bool(graph)
            state = self._state(
                module, engine, module.forward,
                runner=PaddedGraphRunner(engine, graph if isinstance(graph, PaddedGraphConfig) else None)
                if use_graphs else None, graph_enabled=use_graphs)
            packed = PackedQwen35(self, binding, state, options.get("attention_backend"))
            if options.get("validate", True):
                state.report = packed.validate(graphs=use_graphs).details
            return packed
        except BaseException:
            engine.release(rollback=True)
            raise


class PackedQwen35:
    capabilities = Capabilities(inference_capture=True, original_forward_fallback=True)

    def __init__(self, owner, binding, state, backend):
        self.owner, self.binding, self.state = owner, binding, state
        self.attention_backend = backend
        self.composition = state.engine.composition
        self.pieces = self.composition.all()
        self._closed = False

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

    @property
    def min_shared_prefix(self) -> int:
        """Rows that share at least this many leading tokens run that prefix once; 0, the default,
        runs every row in full. It pays where a forward is compute bound (engine.SHARED_PREFIX_MIN
        has measurements); the report's `shared_prefix_rejected` says why a model can't share."""
        return self.state.engine.min_shared_prefix

    @min_shared_prefix.setter
    def min_shared_prefix(self, tokens: int) -> None:
        self._require_open()
        engine = self.state.engine
        if tokens < 0:
            raise PackedEncodersError("min_shared_prefix must be >= 0")
        if tokens and engine.share_rejected:
            raise PackedEncodersError(f"shared prefixes are unavailable here: {engine.share_rejected}")
        engine.min_shared_prefix = max(int(tokens), engine.conv_width) if tokens else 0

    def forward_packed(self, batch):
        """Return final-normed hidden states (for topk, before its head and normalization).

        Requires flat device int64 IDs and positive host_lengths. Positions restart
        at zero for each sequence; omit other metadata (it is never silently ignored).
        This entry is inference-only; the adapters own the original-forward fallback.
        """
        self._require_open()
        ids, lengths = batch.input_ids, batch.host_lengths
        if torch.is_grad_enabled():
            raise PackedEncodersError("Qwen3.5 forward_packed requires no_grad or inference_mode")
        if ids.ndim != 1 or ids.dtype != torch.int64 or ids.device != self.state.engine.device:
            raise PackedEncodersError("Qwen3.5 requires flat int64 token IDs on the weights' device")
        if not lengths or any(type(n) is not int or n <= 0 for n in lengths) or sum(lengths) != ids.numel():
            raise PackedEncodersError("Qwen3.5 requires positive host_lengths summing to the token count")
        if batch.cu_seqlens is not None or batch.position_ids is not None or batch.max_seqlen is not None:
            raise PackedEncodersError("Qwen3.5 takes host_lengths only; positions restart per sequence")
        return _hidden(self.state, ids, lengths, graphs=True)

    def prepare_prefix(self, input_ids, *, max_bytes=256 << 20):
        """Cache one causal prefix on the GPU; see PreparedPrefix.forward_suffixes.

        The budget bounds retained cache tensors, excluding weights and temporary
        forward workspace. Handles must be closed before out-of-band weight writes.
        """
        from packed_encoders.arch.qwen3_5.prefix_cache import PreparedPrefix
        return PreparedPrefix(self, input_ids, max_bytes)

    def validate(self, **kwargs):
        self._require_open()
        return ValidationResult(self.owner.name, self.owner._validate(
            self.binding.patch_target, _engine=self.state.engine, _state=self.state, **kwargs))

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
            raise PackedEncodersError("training graphs are not implemented for Qwen3.5")

    def close(self, *, rollback=False):
        if self._closed:
            return
        self.state.runner = None
        self.state.shared_runner = None
        self.state.graph_enabled = False
        engine = self.state.engine
        engine.release(rollback=rollback)
        engine._stager = None
        self.state.stager = None
        self._closed = True


__all__ = ["HFQwen35Adapter", "Qwen35Hybrid", "Qwen35Report", "PackedQwen35", "TopkEmbedAdapter"]
