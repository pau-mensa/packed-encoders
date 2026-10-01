"""Qwen3.5 hybrid backbones (GatedDeltaNet + gated softmax attention).

The engine (`engine.Qwen35Engine`) is per *backbone*; entry adapters are per *wrapper
signature*. Today there is one entry, topk-embed-v1 (`TopkEmbedModel`, xsmall and small):
its HF forward receives an already-packed batch and returns padded per-token vectors, and
the patched forward keeps that contract exactly, so `SentenceTransformer.encode()` inherits
the speedup with no adapter. A plain HF `Qwen3_5TextModel` entry (causal, padded) would be a
second adapter over the same engine.

What `pack()` installs, per batch:
  - text batches, no grad: CUDA graph replay when the batch fits a `(rows, S)` bucket, else the
    eager packed engine; then the model's own head, slice, and L2 norm.
  - image batches or grad-enabled calls: the original forward, untouched.

fla (flash-linear-attention) is imported only when a Qwen3.5 model is actually packed.
"""

from __future__ import annotations

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
class TopkState:
    """Attached to the packed `TopkEmbedModel` under `ATTR`."""

    arch: str
    engine: Any
    original_forward: Any
    runner: PaddedGraphRunner | None
    graph_enabled: bool
    head: Tensor
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


def _vectors(state: TopkState, hidden: Tensor) -> Tensor:
    v = F.linear(hidden, state.head).float()[..., : state.dim]
    return F.normalize(v, p=2, dim=-1) if state.normalize else v


def _encode_text(state: TopkState, ids: Tensor, lengths: list[int], *, graphs: bool) -> Tensor:
    state.engine.sync_norms()                 # norms trained since packing reach the graphs too
    hidden = None
    if graphs and state.runner is not None and state.graph_enabled and not graphs_globally_disabled() \
            and not torch.is_autocast_enabled("cuda"):
        hidden = state.runner(ids, lengths)
    if hidden is None:
        hidden = state.engine.forward_packed(ids, lengths)
    return _vectors(state, hidden)


def _make_topk_forward(module: nn.Module, state: TopkState):
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


class Qwen35Hybrid:
    name = "qwen3_5"

    def match(self, module: nn.Module) -> bool:
        return _topk_text_model(module) is not None

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
            raise ValidationError("Qwen3.5 packing needs flash-linear-attention (pip install "
                                  "'packed-encoders[qwen3_5]')") from exc
        # The kernel flags pass through fla's **kwargs, so no signature check: the engine probes the
        # kernels numerically when it is built (engine.select_gdn).
        return device, fla.__version__

    def validate(self, module: nn.Module, *, batches: tuple[tuple[int, ...], ...] = ((7, 129, 300, 64), (3, 21, 30)),
                 graphs: bool = True, _engine=None, _state: TopkState | None = None) -> Qwen35Report:
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
            probe_state = state or TopkState(
                arch=self.name, engine=engine, original_forward=oracle, runner=None, graph_enabled=False,
                head=module.head.weight, dim=module.config.output_dim or module.config.dim,
                normalize=bool(module.config.normalize), stager=PinnedStager(device))
            for lengths in batches:
                self._check_numerics(module, oracle, probe_state, lengths, graphs, report)
        finally:
            if built_here:
                from packed_encoders.arch.qwen3_5.engine import unshare_rows

                unshare_rows(engine.shared)
        return report

    def _check_numerics(self, module, oracle, state: TopkState, lengths, graphs, report: Qwen35Report) -> None:
        device = state.head.device
        g = torch.Generator().manual_seed(0)
        seqs = [torch.randint(1000, 150000, (n,), generator=g) for n in lengths]
        B, S = len(seqs), max(lengths)
        lens = torch.tensor(lengths)
        packed = torch.cat(seqs)
        cu = torch.zeros(B + 1, dtype=torch.long)
        torch.cumsum(lens, 0, out=cu[1:])
        pos = torch.arange(len(packed)) - torch.repeat_interleave(cu[:-1], lens)
        feats = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True),
            "attention_mask": torch.arange(S)[None] < lens[:, None],
            "packed_ids": packed[None], "position_ids": pos[None], "cu_seqlens": cu,
            "seq_idx": torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens),
        }
        feats = {k: v.to(device) for k, v in feats.items()}
        with torch.no_grad():
            try:
                ref = oracle(**feats)[feats["attention_mask"]].float()
            except Exception as exc:  # noqa: BLE001
                raise ValidationError(
                    f"the model's own forward failed on the validation batch ({type(exc).__name__}: "
                    f"{str(exc).splitlines()[0][:200]}); cannot certify the engine here. Pass validate=False to "
                    "skip at your own risk.") from exc
            ids = feats["packed_ids"].reshape(-1)
            eager = _encode_text(state, ids, list(lengths), graphs=False)
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
                graphed = _vectors(state, hidden)
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
        lm = _topk_text_model(module)
        return Qwen35Engine(lm, causal=bool(getattr(lm, "document_causal", False)),
                            attention_order=_BACKENDS[attention_backend])

    def pack(self, target: object, module: nn.Module, *, cuda_graph: bool | PaddedGraphConfig | None = None,
             attention_backend: str | None = None, validate: bool = True, train_cuda_graph: Any = False,
             **ignored: Any) -> TopkState:
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
        self._require_env(module)
        engine = self._engine(module, attention_backend)
        use_graphs = cuda_graph is None or bool(cuda_graph)    # graphs are the point here: on by default
        state = TopkState(
            arch=self.name, engine=engine, original_forward=module.forward,
            runner=PaddedGraphRunner(engine, cuda_graph if isinstance(cuda_graph, PaddedGraphConfig) else None)
            if use_graphs else None,
            graph_enabled=use_graphs, head=module.head.weight,
            dim=module.config.output_dim or module.config.dim, normalize=bool(module.config.normalize),
            stager=PinnedStager(engine.device))
        try:
            if validate:
                state.report = self.validate(module, graphs=use_graphs, _engine=engine, _state=state)
        except Exception:
            from packed_encoders.arch.qwen3_5.engine import unshare_rows

            unshare_rows(engine.shared)
            raise
        setattr(module, ATTR, state)
        module.forward = _make_topk_forward(module, state)
        return state

    def unpack(self, module: nn.Module) -> None:
        state = getattr(module, ATTR, None)
        if state is None:
            return
        from packed_encoders.arch.qwen3_5.engine import unshare_rows

        module.forward = state.original_forward
        unshare_rows(state.engine.shared)
        delattr(module, ATTR)


__all__ = ["Qwen35Hybrid", "Qwen35Report", "TopkState", "UnsupportedTargetError"]
