"""Qwen3.5 hybrid backbones (GatedDeltaNet + gated softmax attention).

The engine (`engine.Qwen35Engine`) is per *backbone*; entries are per *wrapper signature*, and
each patched forward keeps its wrapper's contract exactly, so the model's own `encode()` inherits
the speedup with no adapter:

  - `topk`: topk-embed-v1 (`TopkEmbedModel`, xsmall and small). Its forward receives an
    already-packed batch and returns padded per-token vectors (head, slice, L2 norm).
  - `hf`: stock HF `Qwen3_5TextModel`, or `Qwen3_5Model` and its subclasses (e.g.
    pplx-embed-v2-context, which pools chunks itself). Right-padded `input_ids` /
    `attention_mask` in, padded `last_hidden_state` out; causal or bidirectional as
    `config.is_causal` says, exactly as HF's own masks do.

What `pack()` installs, per batch:
  - text batches, no grad: CUDA graph replay when the batch fits a `(rows, S)` bucket, else the
    eager packed engine.
  - anything else (images, grad-enabled calls; for `hf` also a KV cache, explicit
    `position_ids` / `inputs_embeds`, extra outputs, left padding): the original forward.

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

from packed_encoders.errors import PackedEncodersError, UnsupportedTargetError, ValidationError
from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner, graphs_globally_disabled
from packed_encoders.runtime.staging import PinnedStager
from packed_encoders.state import ATTR

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


def _entry(module: nn.Module) -> tuple[str, nn.Module] | None:
    """`(entry, text model)`; topk first, since its wrapper holds a stock `Qwen3_5Model`."""
    for name, find in (("topk", _topk_text_model), ("hf", _hf_text_model)):
        lm = find(module)
        if lm is not None:
            return name, lm
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
    notes: list[str] = field(default_factory=list)


@dataclass
class Qwen35State:
    """Attached to the packed module under `ATTR`. `head` is None for the `hf` entry (hidden states out)."""

    arch: str
    entry: str
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

    def set_cuda_graph(self, enabled: bool, config: PaddedGraphConfig | None = None) -> None:
        if config is not None and not isinstance(config, PaddedGraphConfig):
            raise PackedEncodersError("Qwen3.5 graphs take a runtime.PaddedGraphConfig, "
                                      f"got {type(config).__name__}")
        if enabled and (self.runner is None or config is not None):
            self.runner = PaddedGraphRunner(self.engine, config)
        self.graph_enabled = enabled


def _new_state(module: nn.Module, entry: str, engine, original_forward, runner, graph_enabled: bool) -> Qwen35State:
    if entry == "topk":
        cfg = module.config
        head, dim, normalize = module.head.weight, cfg.output_dim or cfg.dim, bool(cfg.normalize)
    else:
        head, dim, normalize = None, engine.hidden_size, False
    return Qwen35State(arch=Qwen35Hybrid.name, entry=entry, engine=engine, original_forward=original_forward,
                       runner=runner, graph_enabled=graph_enabled, head=head, dim=dim, normalize=normalize,
                       stager=PinnedStager(engine.device))


def _vectors(state: Qwen35State, hidden: Tensor) -> Tensor:
    if state.head is None:
        return hidden
    v = F.linear(hidden, state.head).float()[..., : state.dim]
    return F.normalize(v, p=2, dim=-1) if state.normalize else v


def _encode_text(state: Qwen35State, ids: Tensor, lengths: list[int], *, graphs: bool) -> Tensor:
    state.engine.sync_norms()                 # norms trained since packing reach the graphs too
    for attempt in (0, 1):
        try:
            hidden = None
            if graphs and state.runner is not None and state.graph_enabled and not graphs_globally_disabled() \
                    and not torch.is_autocast_enabled("cuda"):
                hidden = state.runner(ids, lengths)
            if hidden is None:
                hidden = state.engine.forward_packed(ids, lengths)
            return _vectors(state, hidden)
        except torch.OutOfMemoryError as exc:
            if attempt or state.runner is None:
                raise
            reason = str(exc).splitlines()[0][:160]
        _drop_graphs(state, reason)          # outside the handler: its traceback would keep the graphs alive


def _drop_graphs(state: Qwen35State, reason: str) -> None:
    """Graphs are an optimisation, and their private memory pool can't lend to the eager path: out of
    GPU memory with graphs held, free them all and continue eager."""
    state.runner, state.graph_enabled = None, False
    gc.collect()
    torch.cuda.empty_cache()
    warnings.warn(f"packed-encoders: out of GPU memory with CUDA graphs held ({reason}); dropped them and "
                  "continuing without. Large models gain little from graphs: pack with cuda_graph=False, or "
                  "cap them with a smaller PaddedGraphConfig(max_tokens=...)", stacklevel=3)


def _make_topk_forward(module: nn.Module, state: Qwen35State):
    def forward(input_ids, attention_mask, packed_ids, position_ids, cu_seqlens, seq_idx, pixel_values=None,
                image_scatter_index=None, vision_inputs=None, return_dict=True):
        if pixel_values is not None or torch.is_grad_enabled():
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


def _hf_lengths(state: Qwen35State, input_ids, attention_mask, args, kwargs) -> list[int] | None:
    """Row lengths if this call can take the engine, else None (the original forward runs)."""
    if args or input_ids is None or input_ids.dim() != 2 or torch.is_grad_enabled():
        return None
    for k, v in kwargs.items():
        if k == "return_dict" or v is None or v is False or (k == "is_causal" and bool(v) == state.engine.causal):
            continue
        return None          # images, inputs_embeds, position_ids, a cache or use_cache=True, extra outputs
    B, S = input_ids.shape
    if attention_mask is None:
        return [S] * B
    if attention_mask.shape != input_ids.shape:
        return None
    lens = (attention_mask != 0).sum(1)
    right = (attention_mask == (torch.arange(S, device=lens.device) < lens[:, None])).all()
    *lengths, ok = torch.cat([lens, right[None].long()]).tolist()      # the only device->host read
    return lengths if ok and min(lengths) > 0 else None


def _make_hf_forward(module: nn.Module, state: Qwen35State):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast

    def forward(input_ids=None, attention_mask=None, *args, **kwargs):
        lengths = _hf_lengths(state, input_ids, attention_mask, args, kwargs)
        if lengths is None:
            if not state.fallback_warned:
                state.fallback_warned = True
                warnings.warn("packed-encoders: this call runs the model's original forward (images, gradients, "
                              "a KV cache, position_ids / inputs_embeds, extra outputs or left padding)", stacklevel=2)
            return state.original_forward(input_ids, attention_mask, *args, **kwargs)
        B, S = input_ids.shape
        ids = input_ids.reshape(-1)
        if sum(lengths) == B * S:
            out = _encode_text(state, ids, lengths, graphs=True).view(B, S, -1)
        else:
            # Right padding: row i's tokens are the flat slots [i*S, i*S + len_i); pads come back as zeros.
            lens = torch.as_tensor(lengths, dtype=torch.long)
            starts = torch.repeat_interleave(torch.arange(B) * S, lens)
            idx = starts + torch.arange(int(lens.sum())) - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
            (d_idx,) = state.stager.put([idx])
            hidden = _encode_text(state, ids.index_select(0, d_idx), lengths, graphs=True)
            out = hidden.new_zeros((B * S, hidden.shape[-1])).index_copy_(0, d_idx, hidden).view(B, S, -1)
        return_dict = kwargs.get("return_dict")
        if return_dict is None:
            return_dict = getattr(module.config, "return_dict", True)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=out) if return_dict else (out,)

    return forward


class Qwen35Hybrid:
    name = "qwen3_5"

    def match(self, module: nn.Module) -> bool:
        return _entry(module) is not None

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
            raise ValidationError("the Qwen3.5 engine runs bf16 weights; load with dtype=torch.bfloat16 "
                                  "(torch_dtype= on older transformers)")
        try:
            import fla
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        except ImportError as exc:
            raise ValidationError("Qwen3.5 packing needs flash-linear-attention (pip install "
                                  "'packed-encoders[qwen3_5]')") from exc
        # The kernel flags pass through fla's **kwargs, so no signature check: the engine probes the
        # kernels numerically when it is built (engine.select_gdn).
        return device, fla.__version__

    def validate(self, module: nn.Module, *, batches: tuple[tuple[int, ...], ...] = ((7, 129, 300, 64), (3, 21, 30)),
                 graphs: bool = True, _engine=None, _state: Qwen35State | None = None) -> Qwen35Report:
        """Build (or reuse) the engine and compare it with the model's own forward on random text
        rows, on this device, eager and graphed. The default batches cover both GatedDeltaNet
        kernels (eager, a batch whose rows are all <= RECURRENT_MAX_LEN tokens runs the recurrent
        one, any other the chunked one; graphs always run the chunked one). The report keeps the
        worst batch. Raises ValidationError on any miss."""
        entry, lm = _entry(module)
        device, fla_version = self._require_env(lm)
        state = _state if _state is not None else getattr(module, ATTR, None)
        engine = _engine or (state.engine if state is not None else None)
        built_here = engine is None
        if built_here:
            engine = self._engine(module, None)
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
            oracle = state.original_forward if state is not None else module.forward
            probe_state = state or _new_state(module, entry, engine, oracle, runner=None, graph_enabled=False)
            for lengths in batches:
                self._check_numerics(module, oracle, probe_state, lengths, graphs, report)
        finally:
            if built_here:
                from packed_encoders.arch.qwen3_5.engine import unshare_rows

                unshare_rows(engine.shared)
        return report

    def _check_numerics(self, module, oracle, state: Qwen35State, lengths, graphs, report: Qwen35Report) -> None:
        device = state.engine.device
        vocab = state.engine.embed.shape[0]
        g = torch.Generator().manual_seed(0)
        seqs = [torch.randint(min(1000, vocab // 2), min(150000, vocab), (n,), generator=g) for n in lengths]
        B, S = len(seqs), max(lengths)
        lens = torch.tensor(lengths)
        packed = torch.cat(seqs)
        cu = torch.zeros(B + 1, dtype=torch.long)
        torch.cumsum(lens, 0, out=cu[1:])
        pos = torch.arange(len(packed)) - torch.repeat_interleave(cu[:-1], lens)
        mask = torch.arange(S)[None] < lens[:, None]
        if state.entry == "topk":
            feats = {
                "input_ids": torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True), "attention_mask": mask,
                "packed_ids": packed[None], "position_ids": pos[None], "cu_seqlens": cu,
                "seq_idx": torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens),
            }
        else:
            feats = {"input_ids": torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True), "attention_mask": mask.long()}
        feats = {k: v.to(device) for k, v in feats.items()}
        mask = mask.to(device)
        with torch.no_grad():
            try:
                out = oracle(**feats) if state.entry == "topk" else oracle(**feats, use_cache=False)[0]
                ref = out[mask].float()
            except Exception as exc:  # noqa: BLE001
                raise ValidationError(
                    f"the model's own forward failed on the validation batch ({type(exc).__name__}: "
                    f"{str(exc).splitlines()[0][:200]}); cannot certify the engine here. Pass validate=False to "
                    "skip at your own risk.") from exc
            ids = packed.to(device)
            eager = _encode_text(state, ids, list(lengths), graphs=False).float()
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
    @staticmethod
    def _engine(module: nn.Module, attention_backend: str | None):
        from packed_encoders.arch.qwen3_5.engine import Qwen35Engine

        if attention_backend not in _BACKENDS:
            raise PackedEncodersError(f"attention_backend for Qwen3.5 must be one of {sorted(map(str, _BACKENDS))}")
        entry, lm = _entry(module)
        # topk flags causal documents on the module; stock HF follows config.is_causal (default causal).
        causal = getattr(lm, "document_causal", False) if entry == "topk" else getattr(lm.config, "is_causal", True)
        return Qwen35Engine(lm, causal=bool(causal), attention_order=_BACKENDS[attention_backend])

    def pack(self, target: object, module: nn.Module, *, cuda_graph: bool | PaddedGraphConfig | None = None,
             attention_backend: str | None = None, validate: bool = True, train_cuda_graph: Any = False,
             **ignored: Any) -> Qwen35State:
        existing = getattr(module, ATTR, None)
        if existing is not None:                     # idempotent; an explicit cuda_graph still applies
            if cuda_graph is not None:
                existing.set_cuda_graph(bool(cuda_graph),
                                        cuda_graph if isinstance(cuda_graph, PaddedGraphConfig) else None)
            return existing
        if train_cuda_graph:
            raise PackedEncodersError("training graphs are not implemented for Qwen3.5 (inference only)")
        if cuda_graph is not None and not isinstance(cuda_graph, (bool, PaddedGraphConfig)):
            raise PackedEncodersError("Qwen3.5 graphs take cuda_graph=True/False or a runtime.PaddedGraphConfig")
        entry, lm = _entry(module)
        self._require_env(lm)
        engine = self._engine(module, attention_backend)
        use_graphs = cuda_graph is None or bool(cuda_graph)    # graphs are the point here: on by default
        runner = PaddedGraphRunner(engine, cuda_graph if isinstance(cuda_graph, PaddedGraphConfig) else None) \
            if use_graphs else None
        state = _new_state(module, entry, engine, module.forward, runner, use_graphs)
        try:
            if validate:
                state.report = self.validate(module, graphs=use_graphs, _engine=engine, _state=state)
        except Exception:
            from packed_encoders.arch.qwen3_5.engine import unshare_rows

            unshare_rows(engine.shared)
            raise
        setattr(module, ATTR, state)
        module.forward = (_make_topk_forward if entry == "topk" else _make_hf_forward)(module, state)
        return state

    def unpack(self, module: nn.Module) -> None:
        state = getattr(module, ATTR, None)
        if state is None:
            return
        from packed_encoders.arch.qwen3_5.engine import unshare_rows

        module.forward = state.original_forward
        unshare_rows(state.engine.shared)
        delattr(module, ATTR)


__all__ = ["Qwen35Hybrid", "Qwen35Report", "Qwen35State", "UnsupportedTargetError"]
