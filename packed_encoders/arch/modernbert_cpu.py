"""Inference-only, padding-free FP32 ModernBERT CPU engine."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from packed_encoders.batch import PackedBatch
from packed_encoders.config import _rope_thetas
from packed_encoders.cpu.attention import CPUAttention
from packed_encoders.engine import Capabilities, ModelBinding, ValidationResult
from packed_encoders.errors import PackedEncodersError, UnsupportedTargetError, ValidationError
from packed_encoders.locate import is_modernbert_encoder
from packed_encoders.state import ATTR


def _forward_hooks(module):
    # Transformers 5 installs these permanently on the first request for extra
    # outputs. Such requests already delegate to HF; ordinary calls leave them
    # inactive. Other hooks can change values and cannot be bypassed.
    return tuple(key for key, hook in module._forward_hooks.items()
                 if not (getattr(hook, "__module__", "") == "transformers.utils.output_capturing"
                         and getattr(hook, "__qualname__", "") ==
                         "install_output_capuring_hook.<locals>.output_capturing_hook"))


def _options(options):
    unknown = options.keys() - {"validate", "attention_backend", "cuda_graph", "train_cuda_graph", "cuda_graph_seq_cutoff"}
    if unknown:
        raise PackedEncodersError(f"unsupported CPU options: {sorted(unknown)}")
    for key in ("cuda_graph", "train_cuda_graph"):
        if options.get(key) is not None and options[key] is not False:
            raise PackedEncodersError("the CPU engine does not support CUDA graphs")
    if options.get("cuda_graph_seq_cutoff") is not None:
        raise PackedEncodersError("cuda_graph_seq_cutoff is unsupported on CPU")
    if options.get("attention_backend") not in (None, "auto", "native", "sdpa"):
        raise PackedEncodersError("CPU attention_backend must be 'auto', 'native' or 'sdpa'")


@dataclass(frozen=True)
class CPUValidationReport:
    attention_backend: str
    min_cosine: float
    relative_l2: float
    max_abs: float
    sequence_lengths: tuple[int, ...]


class CPUAdapter:
    name = "hf-modernbert-cpu"

    def bind(self, module):
        if is_modernbert_encoder(module) and next(module.parameters()).device.type == "cpu":
            return ModelBinding(self, module, module)
        return None

    def install(self, binding, packed):
        packed.install_forward()


class ModernBertCPU:
    """CPU configuration. Thread count controls native attention/RoPE workers.

    Matrix operations use PyTorch/Accelerate's existing thread settings. This
    engine never changes process-global threading configuration.
    """
    name = "modernbert-cpu"
    adapters = (CPUAdapter(),)

    def __init__(self, *, threads=None, max_tokens=32768):
        if threads is not None and (type(threads) is not int or threads <= 0):
            raise ValueError("threads must be a positive integer")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self.threads, self.max_tokens = threads, max_tokens

    def prepare(self, binding, options):
        _options(options)
        packed = PackedModernBertCPU(binding, self, options.get("attention_backend") or "auto")
        try:
            if options.get("validate", True):
                packed.validate()
            return packed
        except BaseException:
            packed.close(rollback=True)
            raise

    def validate(self, binding, **kwargs):
        packed = self.prepare(binding, {"validate": False})
        try:
            return packed.validate(**kwargs)
        finally:
            packed.close()


class PackedModernBertCPU:
    capabilities = Capabilities(original_forward_fallback=True)
    pieces = ()

    def __init__(self, binding, engine, backend):
        self.binding, self.model = binding, binding.weight_source
        self.original_forward = binding.patch_target.forward
        self._lock = threading.RLock()
        self._closed = False
        self.max_tokens, self.requested_backend = engine.max_tokens, backend
        cfg = self.model.config
        self.hidden, self.heads = cfg.hidden_size, cfg.num_attention_heads
        if self.hidden % self.heads or (self.hidden // self.heads) % 2:
            raise UnsupportedTargetError("CPU ModernBERT requires equal heads with even head_dim")
        self.dim, self.half = self.hidden // self.heads, cfg.local_attention // 2
        self.max_position = cfg.max_position_embeddings
        if self.max_position <= 0 or self.half < 0 or cfg.local_attention % 2:
            raise UnsupportedTargetError("unsupported CPU position/window configuration")
        rope = getattr(cfg, "rope_parameters", None) or {}
        if any(p.get("rope_type", "default") != "default" for p in rope.values() if isinstance(p, dict)):
            raise UnsupportedTargetError("CPU ModernBERT currently supports default split-half RoPE only")
        if cfg.hidden_activation != "gelu":
            raise UnsupportedTargetError("CPU ModernBERT requires exact-erf GeGLU")
        self.kinds = tuple(getattr(cfg, "layer_types", None) or (
            "full_attention" if i % cfg.global_attn_every_n_layers == 0 else "sliding_attention"
            for i in range(len(self.model.layers))))
        if len(self.kinds) != len(self.model.layers) or set(self.kinds) - {"full_attention", "sliding_attention"}:
            raise UnsupportedTargetError("unsupported CPU ModernBERT layer schedule")
        for module in self.model.modules():
            if module is not self.model and (_forward_hooks(module) or module._forward_pre_hooks):
                raise UnsupportedTargetError("remove internal forward hooks before packing the CPU encoder")
        self.layers = tuple(self.model.layers)
        for layer in self.layers:
            for projection in (layer.attn.Wqkv, layer.attn.Wo, layer.mlp.Wi, layer.mlp.Wo):
                if type(projection) is not nn.Linear or "forward" in projection.__dict__:
                    raise UnsupportedTargetError("CPU projections must be plain nn.Linear modules; merge adapters first")
        self._weights = self._weight_signature()
        for param in self.model.parameters():
            if param.device.type != "cpu" or param.dtype != torch.float32 or not param.is_contiguous():
                raise UnsupportedTargetError("CPU ModernBERT requires contiguous FP32 CPU parameters")
        self._configuration = self._config_signature()
        self.attention = CPUAttention(backend, self.dim, self.half, engine.threads or torch.get_num_threads())
        self.state = SimpleNamespace(attention_backend=self.attention.backend, validation_report=None)
        self.tables = {}
        # Tables are engine-owned; HF parameters/storage and state_dict stay intact.
        for kind, theta in zip(("full_attention", "sliding_attention"), _rope_thetas(cfg)):
            inv = 1.0 / (theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
            angles = torch.outer(torch.arange(self.max_position).float(), inv).repeat(1, 2)
            self.tables[kind] = (angles.cos(), angles.sin())

    def _weight_signature(self):
        return tuple((name, id(p), p.data_ptr(), p.device, p.dtype, tuple(p.shape), p.stride())
                     for name, p in self.model.named_parameters())

    def _config_signature(self):
        cfg = self.model.config
        return (cfg.hidden_size, cfg.num_attention_heads, cfg.local_attention,
                cfg.max_position_embeddings, repr(getattr(cfg, "layer_types", None)),
                getattr(cfg, "global_attn_every_n_layers", None), repr(getattr(cfg, "rope_parameters", None)),
                _rope_thetas(cfg), cfg.hidden_activation,
                tuple((name, id(m), _forward_hooks(m), tuple(m._forward_pre_hooks))
                      for name, m in self.model.named_modules() if m is not self.model))

    def _check(self):
        if self._closed:
            raise PackedEncodersError("this CPU engine has been closed; pack the model again")
        if self._weights != self._weight_signature() or self._configuration != self._config_signature():
            raise PackedEncodersError("CPU model structure/storage/device/dtype changed; unpack and repack")

    @property
    def graph_enabled(self):
        return False

    @graph_enabled.setter
    def graph_enabled(self, enabled):
        self.set_cuda_graph(enabled)

    def configure(self, options):
        self._check()
        _options(options)
        backend = options.get("attention_backend")
        if backend is not None and backend not in (self.requested_backend, self.attention.backend):
            raise PackedEncodersError("unpack before changing the prepared CPU attention backend")

    def set_cuda_graph(self, enabled, config=None):
        self._check()
        if enabled or config is not None:
            raise PackedEncodersError("the CPU engine does not support CUDA graphs")

    set_train_cuda_graph = set_cuda_graph

    def install_forward(self):
        setattr(self.binding.patch_target, ATTR, self.state)
        self.binding.patch_target.forward = self.forward

    def forward(self, input_ids=None, attention_mask=None, *args, **kwargs):
        with self._lock:
            self._check()
            cfg = self.model.config
            unsupported = set(kwargs) - {"return_dict", "output_hidden_states", "output_attentions", "token_type_ids"}
            extra_outputs = any(kwargs.get(key) if kwargs.get(key) is not None else getattr(cfg, key, False)
                                for key in ("output_hidden_states", "output_attentions"))
            token_types = kwargs.get("token_type_ids")
            fallback = (args or unsupported or extra_outputs or any(m.training for m in self.model.modules()) or torch.is_grad_enabled()
                        or torch.is_autocast_enabled("cpu") or input_ids is None
                        or (token_types is not None and bool(token_types.any())))
            if fallback:
                return self.original_forward(input_ids, attention_mask, *args, **kwargs)
            self._ids(input_ids, ndim=2)
            b, s = input_ids.shape
            if attention_mask is None:
                mask = torch.ones_like(input_ids, dtype=torch.bool)
            elif (attention_mask.device.type != "cpu" or attention_mask.shape != input_ids.shape
                  or not bool(((attention_mask == 0) | (attention_mask == 1)).all())):
                return self.original_forward(input_ids, attention_mask, **kwargs)
            else:
                mask = attention_mask.bool()
            lengths = tuple(mask.sum(1).tolist())
            # Compaction must preserve local-window distances. Holey or empty
            # rows delegate to HF; left/right padding preserves column RoPE.
            columns = torch.arange(s).expand(b, s)
            starts = columns.masked_fill(~mask, s).amin(1)
            ends = columns.masked_fill(~mask, -1).amax(1)
            if not all(lengths) or not torch.equal(ends - starts + 1, mask.sum(1)):
                return self.original_forward(input_ids, attention_mask, **kwargs)
            indices = mask.flatten().nonzero().flatten()
            ids = input_ids.flatten()[indices]
            hidden = self._packed(ids, lengths, indices % s)
            out = hidden.new_zeros(b * s, self.hidden)
            out.index_copy_(0, indices, hidden)
            out = out.view(b, s, self.hidden)
            return_dict = kwargs.get("return_dict")
            if return_dict is None:
                return_dict = getattr(cfg, "return_dict", True)
            if not return_dict:
                return (out,)
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(last_hidden_state=out)

    def _ids(self, ids, *, ndim):
        if ids.device.type != "cpu" or ids.dtype != torch.int64 or ids.ndim != ndim or not ids.numel():
            raise PackedEncodersError(f"CPU token IDs must be a nonempty int64 CPU tensor of rank {ndim}")
        if int(ids.min()) < 0 or int(ids.max()) >= self.model.embeddings.tok_embeddings.num_embeddings:
            raise PackedEncodersError("token IDs are outside the embedding vocabulary")

    def forward_packed(self, batch: PackedBatch):
        with self._lock:
            self._check()
            if any(m.training for m in self.model.modules()) or torch.is_grad_enabled() or torch.is_autocast_enabled("cpu"):
                raise PackedEncodersError("CPU forward_packed requires eval(), no_grad/inference_mode, and no CPU autocast")
            self._ids(batch.input_ids, ndim=1)
            lengths = batch.host_lengths
            if batch.cu_seqlens is not None:
                cu = batch.cu_seqlens
                if cu.device.type != "cpu" or cu.dtype != torch.int32 or cu.ndim != 1 or len(cu) < 2 or int(cu[0]) != 0:
                    raise PackedEncodersError("CPU cu_seqlens must be int32 [B+1] starting at zero")
                derived = tuple(cu.diff().tolist())
                if lengths is not None and tuple(lengths) != derived:
                    raise PackedEncodersError("host_lengths and cu_seqlens disagree")
                lengths = derived
            if not lengths or any(type(n) is not int or n <= 0 for n in lengths) or sum(lengths) != batch.input_ids.numel():
                raise PackedEncodersError("positive host_lengths or cu_seqlens must cover every packed token")
            if batch.max_seqlen is not None and (type(batch.max_seqlen) is not int or batch.max_seqlen < max(lengths)):
                raise PackedEncodersError("max_seqlen is smaller than a sequence")
            positions = batch.position_ids
            if positions is None:
                positions = torch.cat([torch.arange(n) for n in lengths])
            return self._packed(batch.input_ids, tuple(lengths), positions)

    def _packed(self, ids, lengths, positions):
        if (positions.device.type != "cpu" or positions.dtype != torch.int64 or positions.shape != ids.shape
                or int(positions.min()) < 0 or int(positions.max()) >= self.max_position):
            raise PackedEncodersError("CPU positions must be int64, match IDs, and lie within max_position_embeddings")
        if max(lengths) > self.max_tokens or max(lengths) > self.max_position:
            raise PackedEncodersError("sequence exceeds the engine's max_tokens or max_position_embeddings")
        outputs, offset, group = [], 0, []
        for length in lengths:
            if group and sum(group) + length > self.max_tokens:
                count = sum(group)
                outputs.append(self._core(ids[offset:offset+count], tuple(group), positions[offset:offset+count]))
                offset += count
                group = []
            group.append(length)
        outputs.append(self._core(ids[offset:], tuple(group), positions[offset:]))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    def _core(self, ids, lengths, positions):
        tables = {kind: (cos[positions][None, None], sin[positions][None, None])
                  for kind, (cos, sin) in self.tables.items()}
        boundaries = torch.tensor(lengths, dtype=torch.int32)
        x = self.model.embeddings(ids[None])
        for layer, kind in zip(self.layers, self.kinds):
            qkv = F.linear(layer.attn_norm(x), layer.attn.Wqkv.weight, layer.attn.Wqkv.bias)
            cos, sin = tables[kind]
            q, k, v = self.attention.rope_layout(qkv, cos, sin, self.heads, self.dim)
            context = (self.attention.local(q, k, v, lengths, boundaries) if kind == "sliding_attention"
                       else self.attention.global_(q, k, v, lengths))
            context = context.transpose(1, 2).reshape_as(x)
            x = x + F.linear(context, layer.attn.Wo.weight, layer.attn.Wo.bias)
            hidden = F.linear(layer.mlp_norm(x), layer.mlp.Wi.weight, layer.mlp.Wi.bias)
            x = x + F.linear(self.attention.geglu(hidden), layer.mlp.Wo.weight, layer.mlp.Wo.bias)
        return self.model.final_norm(x)[0]

    def validate(self):
        with self._lock:
            self._check()
            states = [(module, module.training) for module in self.model.modules()]
            lengths = (min(7, self.max_position, self.max_tokens), min(129, self.max_position, self.max_tokens))
            ids = torch.randint(self.model.config.vocab_size, (2, max(lengths)), generator=torch.Generator().manual_seed(17))
            mask = torch.arange(ids.shape[1])[None] < torch.tensor(lengths)[:, None]
            try:
                self.model.eval()
                with torch.no_grad(), torch.autocast("cpu", enabled=False):
                    reference = self.original_forward(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state[mask]
                    actual = self._packed(ids[mask], lengths, mask.nonzero()[:, 1])
                    diff = actual - reference
                    cosine = F.cosine_similarity(actual, reference, dim=-1)
                    cosine = torch.where((actual.norm(dim=-1) == 0) & (reference.norm(dim=-1) == 0), 1., cosine)
                    report = CPUValidationReport(self.attention.backend, float(cosine.min()),
                                                 float(diff.norm() / reference.norm().clamp_min(1e-12)),
                                                 float(diff.abs().max()), lengths)
                    if not torch.isfinite(actual).all() or report.min_cosine < 0.99999 or report.relative_l2 > 2e-4:
                        raise ValidationError(f"CPU ModernBERT output validation failed: {report}")
            finally:
                for module, training in states:
                    module.training = training
            self.state.validation_report = report
            return ValidationResult("modernbert-cpu", report)

    def close(self, *, rollback=False):
        with self._lock:
            self.attention.close()
            self.tables.clear()
            self.layers = ()
            self._closed = True
