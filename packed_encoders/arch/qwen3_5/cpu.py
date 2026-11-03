"""Inference-only Qwen3.5 CPU engine, with packed FP32 and optional native INT8 projections."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from types import SimpleNamespace
import threading
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from packed_encoders.arch.qwen3_5 import _hf_text_model, _topk_text_model
from packed_encoders.arch.modernbert_cpu import _forward_hooks
from packed_encoders.arch.weights import validation_ids_below
from packed_encoders.batch import PackedBatch
from packed_encoders.cpu.qwen35 import GatedDeltaNet, reference_gdn
from packed_encoders.cpu.topology import default_threads
from packed_encoders.engine import Capabilities, ModelBinding, ValidationResult
from packed_encoders.errors import PackedEncodersError, UnsupportedTargetError, ValidationError
from packed_encoders.runtime.epilogue import EpilogueSupport
from packed_encoders.state import ATTR


def _options(options):
    unknown = options.keys() - {"validate", "precision", "attention_backend", "cuda_graph", "train_cuda_graph",
                                "cuda_graph_seq_cutoff"}
    if unknown:
        raise PackedEncodersError(f"unsupported Qwen3.5 CPU options: {sorted(unknown)}")
    if any(options.get(k) is not None and options[k] is not False for k in ("cuda_graph", "train_cuda_graph")):
        raise PackedEncodersError("the CPU engine does not support CUDA graphs")
    if options.get("cuda_graph_seq_cutoff") is not None:
        raise PackedEncodersError("cuda_graph_seq_cutoff is unsupported on CPU")
    if options.get("attention_backend") not in (None, "auto", "native", "sdpa"):
        raise PackedEncodersError("CPU attention_backend must be 'auto', 'native' or 'sdpa'")
    if options.get("precision") not in (None, "fp32", "int8"):
        raise PackedEncodersError("CPU precision must be 'fp32' or 'int8'")


class CPUAdapter:
    name = "qwen3_5-cpu"

    def bind(self, module):
        model = _topk_text_model(module) or _hf_text_model(module)
        if model is not None and next(model.parameters()).device.type == "cpu":
            return ModelBinding(self, module, model)
        return None

    def install(self, binding, packed):
        packed.install_forward()


@dataclass(frozen=True)
class Qwen35CPUReport:
    precision: str
    gdn_backend: str
    sequence_lengths: tuple[int, ...]
    min_cosine: float
    median_cosine: float
    relative_l2: float
    max_abs: float


class Qwen35CPU:
    """CPU Qwen3.5 text inference. ``threads`` controls engine-owned native workers;
    PyTorch/BLAS keeps the caller's process thread settings. ``max_tokens`` bounds
    packed microbatches without splitting a sequence. Native GatedDeltaNet runs on
    macOS and Linux; attention uses per-sequence SDPA with grouped query heads.

    FP32 projection copies refresh on ordinary in-place updates. INT8 weights are
    prepared once; updates require unpack/repack. Parameter replacement, storage
    swaps, config edits and device/dtype moves require repacking in both modes.
    """
    name = "qwen3_5-cpu"
    adapters = (CPUAdapter(),)

    def __init__(self, *, threads=None, max_tokens=32768, prepack_weights=True):
        if threads is not None and (type(threads) is not int or threads <= 0):
            raise ValueError("threads must be a positive integer")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if type(prepack_weights) is not bool:
            raise ValueError("prepack_weights must be a bool")
        self.threads, self.max_tokens, self.prepack_weights = threads, max_tokens, prepack_weights

    def prepare(self, binding, options):
        _options(options)
        packed = PackedQwen35CPU(binding, self, options)
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


class Projection:
    """Merged projection owned by the engine; source Parameters and storage stay intact."""
    def __init__(self, modules, align=64):
        self.parameters = tuple(p.weight for p in modules)
        self.sizes = tuple(p.shape[0] for p in self.parameters)
        self.padding = -sum(self.sizes) % align
        self.shape = (sum(self.sizes) + self.padding, self.parameters[0].shape[1])
        self.alias = len(self.parameters) == 1 and self.padding == 0
        with torch.inference_mode(False):
            self.weight = self.parameters[0].detach() if self.alias else torch.empty(self.shape)
        self.versions = None
        self.refresh()

    def refresh(self):
        if self.alias:
            return
        # Inference tensors have no version counter; keep refreshing those copies.
        try:
            versions = tuple(p._version for p in self.parameters)
        except RuntimeError:
            versions = None
        if versions is None or versions != self.versions:
            with torch.no_grad():
                offset = 0
                for p, size in zip(self.parameters, self.sizes):
                    self.weight[offset:offset + size].copy_(p)
                    offset += size
                self.weight[offset:].zero_()
            self.versions = versions


class PackedQwen35CPU(EpilogueSupport):
    capabilities = Capabilities(original_forward_fallback=True)
    pieces = ()

    def __init__(self, binding, engine, options):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        self._rms_norm_type = Qwen3_5RMSNorm
        self.binding, self.model = binding, binding.weight_source
        self.original_forward = binding.patch_target.forward
        self.topk = _topk_text_model(binding.patch_target) is not None
        self._lock, self._closed = threading.RLock(), False
        self._init_epilogues(lock=self._lock)
        self.threads, self.max_tokens = engine.threads or default_threads(), engine.max_tokens
        self.prepack_weights, self.tune_gemm = engine.prepack_weights, False
        self.requested_backend = options.get("attention_backend") or "auto"
        self.requested_precision = options.get("precision") or "fp32"
        self.cfg = self.model.config
        self.causal = (getattr(self.model, "document_causal", False) if self.topk else
                       getattr(binding, "causal", getattr(self.cfg, "is_causal", True)))
        self.hidden, self.max_position = self.cfg.hidden_size, self.cfg.max_position_embeddings
        self.native = self.sme = self.int8 = None
        self.projections, self.layers = [], []
        if self.topk:
            head = binding.patch_target.head
            if (type(head) is not nn.Linear or head.bias is not None or "forward" in head.__dict__
                    or head._forward_hooks or head._forward_pre_hooks or head.weight.device.type != "cpu"
                    or head.weight.dtype != torch.float32):
                raise UnsupportedTargetError("the topk CPU head must be a plain bias-free FP32 CPU Linear")
        if self.cfg.hidden_act != "silu":
            raise UnsupportedTargetError("CPU Qwen3.5 requires SiLU activation")
        for p in self.model.parameters():
            if p.device.type != "cpu" or p.dtype != torch.float32 or not p.is_contiguous():
                raise UnsupportedTargetError("CPU Qwen3.5 requires contiguous FP32 CPU parameters")
        for m in self.model.modules():
            if m is not self.model and (_forward_hooks(m) or m._forward_pre_hooks):
                raise UnsupportedTargetError("remove internal forward hooks before packing CPU Qwen3.5")
            if isinstance(m, nn.Linear) and (type(m) is not nn.Linear or "forward" in m.__dict__ or m.bias is not None):
                raise UnsupportedTargetError("CPU projections must be plain bias-free nn.Linear modules; merge adapters first")
        if len(self.model.layers) != len(self.cfg.layer_types) or set(self.cfg.layer_types) - {"linear_attention", "full_attention"}:
            raise UnsupportedTargetError("unsupported Qwen3.5 layer schedule")
        # Preparation never mutates the HF model. All merged and quantized storage is private.
        for layer, kind in zip(self.model.layers, self.cfg.layer_types):
            linear = kind == "linear_attention"
            mixer = layer.linear_attn if linear else layer.self_attn
            if linear:
                if mixer.num_v_heads % mixer.num_k_heads:
                    raise UnsupportedTargetError("GatedDeltaNet requires divisible grouped value heads")
                modules = (mixer.in_proj_qkv, mixer.in_proj_z, mixer.in_proj_b, mixer.in_proj_a)
                out = mixer.out_proj
            else:
                modules = (mixer.q_proj, mixer.k_proj, mixer.v_proj)
                out = mixer.o_proj
                if mixer.q_proj.out_features != 2 * self.cfg.num_attention_heads * mixer.head_dim:
                    raise UnsupportedTargetError("CPU Qwen3.5 requires gated softmax attention")
            entry = SimpleNamespace(linear=linear, mixer=mixer, input=layer.input_layernorm,
                                    post=layer.post_attention_layernorm,
                                    mixed=Projection(modules), out=Projection((out,)),
                                    up=Projection((layer.mlp.gate_proj, layer.mlp.up_proj)),
                                    down=Projection((layer.mlp.down_proj,)))
            self.layers.append(entry)
            self.projections.extend((entry.mixed, entry.out, entry.up, entry.down))
        try:
            if self.requested_backend != "sdpa":
                try:
                    self.native = GatedDeltaNet(self.threads)
                except PackedEncodersError as exc:
                    if self.requested_backend == "native":
                        raise
                    warnings.warn(f"{exc}; using torch GatedDeltaNet", stacklevel=3)
            if self.requested_precision == "int8":
                from packed_encoders.cpu.qwen35_int8 import Int8Projections
                self.int8 = Int8Projections(self.projections, self.threads,
                                            None if self.native is None else self.native.pool)
                # INT8 owns the prepared copies; merged FP32 buffers are no longer used.
                for projection in self.projections:
                    projection.weight = None
            elif self.prepack_weights:
                from packed_encoders.cpu.sme import FP32Projections, sme2_available
                if sme2_available():
                    try:
                        self.sme = FP32Projections([(p.weight, None) for p in self.projections], self.threads,
                                                   None if self.native is None else self.native.pool)
                        # Qwen's wider projections reach Accelerate's crossover earlier
                        # than ModernBERT's. M4 Pro full-model probes favor SME at 32
                        # tokens, Accelerate at 128 and 256; keep SME on short queries.
                        self.sme.max_rows = min(self.sme.max_rows, 64)
                    except PackedEncodersError as exc:
                        warnings.warn(str(exc), stacklevel=3)
            self.precision = "int8" if self.int8 else "fp32"
            self.state = SimpleNamespace(precision=self.precision, attention_backend="sdpa",
                                         gdn_backend="native" if self.native else "torch", validation_report=None)
        except BaseException:
            self.close(rollback=True)
            raise

    def _require_open(self):
        if self._closed:
            raise PackedEncodersError("this CPU engine has been closed; pack the model again")

    def install_forward(self):
        setattr(self.binding.patch_target, ATTR, self.state)
        self.binding.patch_target.forward = self.topk_forward if self.topk else self.forward

    @property
    def graph_enabled(self):
        return False

    @graph_enabled.setter
    def graph_enabled(self, enabled):
        self.set_cuda_graph(enabled)

    def set_cuda_graph(self, enabled, config=None):
        self._require_open()
        if enabled or config is not None:
            raise PackedEncodersError("the CPU engine does not support CUDA graphs")

    set_train_cuda_graph = set_cuda_graph

    def configure(self, options):
        self._require_open()
        _options(options)
        if options.get("precision") not in (None, self.precision):
            raise PackedEncodersError("unpack before changing the CPU precision")
        if options.get("attention_backend") not in (None, self.requested_backend):
            raise PackedEncodersError("unpack before changing the CPU backend")

    def _linear(self, x, p):
        if self.int8 is not None:
            return self.int8.linear(x.contiguous(), p)
        p.refresh()
        out = None if self.sme is None else self.sme.linear(x.contiguous(), p.weight, None, self.threads)
        return F.linear(x, p.weight) if out is None else out

    def _qk_rope(self, p, attention, cos, sin):
        nq, nk, hd = self.cfg.num_attention_heads, self.cfg.num_key_value_heads, attention.head_dim
        # M4 full-model probes favor fusion at 32/128 packed tokens, but regress
        # at 256 despite a faster isolated kernel. Keep larger batches on Torch.
        if (p.shape[0] <= 128 and self.native is not None
                and p.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
                and all(type(norm) is self._rms_norm_type and "forward" not in norm.__dict__
                        for norm in (attention.q_norm, attention.k_norm))):
            return self.native.qk_rope(p, attention, cos, sin, nq, nk)
        n, qw, kw = p.shape[0], 2 * nq * hd, nk * hd
        q = attention.q_norm(p[:, :qw].reshape(n, nq, 2 * hd)[..., :hd])
        k = attention.k_norm(p[:, qw:qw + kw].reshape(n, nk, hd))
        r = cos.shape[-1]
        def rope(t):
            rot = t[..., :r]
            rotated = torch.cat((-rot[..., r // 2:], rot[..., :r // 2]), -1)
            return torch.cat((rot * cos[:, None] + rotated * sin[:, None], t[..., r:]), -1)
        return rope(q), rope(k)

    def _attention(self, x, layer, lengths, cos, sin):
        a = layer.mixer
        n, nq, nk, hd = x.shape[0], self.cfg.num_attention_heads, self.cfg.num_key_value_heads, a.head_dim
        p = self._linear(x, layer.mixed)
        qw, kw = nq * 2 * hd, nk * hd
        gate = p[:, :qw].reshape(n, nq, 2 * hd)[..., hd:]
        v = p[:, qw + kw:qw + 2 * kw].reshape(n, nk, hd)
        q, k = self._qk_rope(p, a, cos, sin)
        if all(length == lengths[0] for length in lengths):
            # Keep sequence boundaries in the batch dimension and launch SDPA once.
            batch, length = len(lengths), lengths[0]
            qq, kk, vv = (t.reshape(batch, length, heads, hd).transpose(1, 2)
                          for t, heads in ((q, nq), (k, nk), (v, nk)))
            # Explicit KV expansion reaches PyTorch's optimized CPU SDPA path.
            kk, vv = (t.repeat_interleave(nq // nk, 1) for t in (kk, vv))
            y = F.scaled_dot_product_attention(qq, kk, vv, is_causal=self.causal, scale=a.scaling)
            context = y.transpose(1, 2).reshape(n, nq * hd)
        else:
            out, offset = [], 0
            for length in lengths:
                sl = slice(offset, offset + length)
                qq, kk, vv = (t[sl].transpose(0, 1)[None] for t in (q, k, v))
                kk, vv = (t.repeat_interleave(nq // nk, 1) for t in (kk, vv))
                y = F.scaled_dot_product_attention(qq, kk, vv, is_causal=self.causal, scale=a.scaling)
                out.append(y[0].transpose(0, 1).reshape(length, nq * hd))
                offset += length
            context = torch.cat(out)
        context = context * gate.reshape(n, nq * hd).sigmoid()
        return self._linear(context, layer.out)[:, :self.hidden]

    def _add_norm(self, x, branch, norm, *, owned=False):
        if (self.native is not None and type(norm) is self._rms_norm_type
                and "forward" not in norm.__dict__ and x.is_contiguous() and branch.is_contiguous()):
            return self.native.add_norm(x, branch, norm, owned=owned)
        x = x + branch
        return x, norm(x)

    def _swiglu(self, x, width):
        if self.native is not None and x.is_contiguous():
            return self.native.swiglu(x, width)
        return F.silu(x[:, :width]) * x[:, width:2 * width]

    def _core(self, ids, lengths, positions):
        if self.int8 is not None:
            self.int8.check_versions()
        x = F.embedding(ids, self.model.embed_tokens.weight)
        cos, sin = self.model.rotary_emb(x[None], positions[None, None].expand(3, 1, -1))
        cos, sin = cos[0], sin[0]
        cu = torch.tensor((0, *lengths), dtype=torch.int32).cumsum(0, dtype=torch.int32)
        h = self.layers[0].input(x)
        for i, layer in enumerate(self.layers):
            if layer.linear:
                projection = self._linear(h, layer.mixed)
                h = (self.native(projection, layer, cu) if self.native else reference_gdn(projection, layer, lengths))
                mix = self._linear(h, layer.out)[:, :self.hidden]
            else:
                mix = self._attention(h, layer, lengths, cos, sin)
            x, h = self._add_norm(x, mix, layer.post)
            h = self._linear(h, layer.up)
            width = layer.up.sizes[0]
            h = self._swiglu(h, width)
            final = i + 1 == len(self.layers)
            norm = self.model.norm if final else self.layers[i + 1].input
            x, h = self._add_norm(x, self._linear(h.contiguous(), layer.down)[:, :self.hidden], norm, owned=final)
        return h

    def _ids(self, ids):
        if ids.device.type != "cpu" or ids.dtype not in (torch.int64, torch.int32) or ids.numel() == 0:
            raise PackedEncodersError("CPU token IDs must be nonempty int32/int64 CPU tensors")
        if int(ids.min()) < 0 or int(ids.max()) >= self.model.embed_tokens.num_embeddings:
            raise PackedEncodersError("token IDs are outside the embedding vocabulary")

    def _packed(self, ids, lengths, positions, epilogue=None):
        if max(lengths) > min(self.max_tokens, self.max_position):
            raise PackedEncodersError("sequence exceeds max_tokens or max_position_embeddings")
        if (positions.device.type != "cpu" or positions.dtype != torch.int64 or positions.shape != ids.shape
                or int(positions.min()) < 0 or int(positions.max()) >= self.max_position):
            raise PackedEncodersError("CPU positions must be int64, match IDs and lie within max_position_embeddings")
        out, offset, group = [], 0, []
        def run(count):
            batch = PackedBatch(ids[offset:offset + count], host_lengths=tuple(group),
                                position_ids=positions[offset:offset + count], max_seqlen=max(group),
                                cu_seqlens=(torch.tensor((0, *group), dtype=torch.int32).cumsum(0, dtype=torch.int32)
                                            if epilogue is not None else None))
            hidden = self._core(batch.input_ids, batch.host_lengths, batch.position_ids)
            return hidden if epilogue is None else epilogue(hidden, batch)
        for length in lengths:
            if group and sum(group) + length > self.max_tokens:
                count = sum(group)
                out.append(run(count))
                offset += count
                group = []
            group.append(length)
        out.append(run(sum(group)))
        return out[0] if len(out) == 1 else torch.cat(out)

    def forward_packed(self, batch):
        return self._forward_packed(batch)

    def _forward_packed(self, batch, epilogue=None):
        with self._lock:
            self._require_open()
            if self.model.training or torch.is_grad_enabled() or torch.is_autocast_enabled("cpu"):
                raise PackedEncodersError("CPU forward_packed requires eval(), no_grad/inference_mode and no CPU autocast")
            self._ids(batch.input_ids)
            if batch.input_ids.ndim != 1:
                raise PackedEncodersError("packed IDs must have shape [tokens]")
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
            return self._packed(batch.input_ids, tuple(lengths), positions, epilogue)

    def forward(self, input_ids=None, attention_mask=None, *args, **kwargs):
        with self._lock:
            self._require_open()
            use_cache = kwargs.get("use_cache")
            if use_cache is None:
                use_cache = getattr(self.cfg, "use_cache", False)
            allowed = {"return_dict", "use_cache", "output_attentions", "output_hidden_states"}
            fallback = (args or self.model.training or torch.is_grad_enabled() or torch.is_autocast_enabled("cpu")
                        or input_ids is None or use_cache
                        or any(v is not None for k, v in kwargs.items() if k not in allowed)
                        or any(kwargs.get(k, getattr(self.cfg, k, False)) for k in ("output_attentions", "output_hidden_states")))
            if fallback:
                return self.original_forward(input_ids, attention_mask, *args, **kwargs)
            self._ids(input_ids)
            if input_ids.ndim != 2:
                raise PackedEncodersError("padded input_ids must have shape [batch, sequence]")
            b, s = input_ids.shape
            if attention_mask is None:
                mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                if (attention_mask.device.type != "cpu" or attention_mask.shape != input_ids.shape
                        or not bool(((attention_mask == 0) | (attention_mask == 1)).all())):
                    return self.original_forward(input_ids, attention_mask, **kwargs)
                mask = attention_mask.bool()
            counts = mask.sum(1)
            col = torch.arange(s)
            right = (mask == (col < counts[:, None])).all(1)
            # Left padding reaches GDN's convolution state even when projected inputs are zero
            # (a biased convolution can produce nonzero states), so keep HF for left padding.
            if not bool((counts > 0).all()) or not bool(right.all()):
                return self.original_forward(input_ids, attention_mask, **kwargs)
            lengths = tuple(counts.tolist())
            hidden = self._packed(input_ids[mask], lengths, col.expand(b, s)[mask])
            out = hidden.new_zeros(b, s, self.hidden)
            out[mask] = hidden
            return_dict = kwargs.get("return_dict")
            if return_dict is None:
                return_dict = getattr(self.binding.patch_target.config, "return_dict", True)
            if not return_dict:
                return (out,)
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast
            return Qwen3_5ModelOutputWithPast(last_hidden_state=out)

    def topk_forward(self, input_ids, attention_mask, packed_ids, position_ids, cu_seqlens, seq_idx,
                     pixel_values=None, image_scatter_index=None, vision_inputs=None, return_dict=True):
        with self._lock:
            self._require_open()
            if (self.binding.patch_target.training or torch.is_grad_enabled() or torch.is_autocast_enabled("cpu")
                    or any(x is not None for x in (pixel_values, image_scatter_index, vision_inputs))):
                return self.original_forward(input_ids, attention_mask, packed_ids, position_ids, cu_seqlens, seq_idx,
                                             pixel_values=pixel_values, image_scatter_index=image_scatter_index,
                                             vision_inputs=vision_inputs, return_dict=return_dict)
            hidden = self.forward_packed(PackedBatch(packed_ids.reshape(-1), cu_seqlens=cu_seqlens))
            module = self.binding.patch_target
            v = module.head(hidden).float()[..., :(module.config.output_dim or module.config.dim)]
            if getattr(module.config, "normalize", True):
                v = F.normalize(v, p=2, dim=-1)
            out = v.new_zeros(*input_ids.shape, v.shape[-1])
            out[attention_mask.bool()] = v
            return out

    def validate(self, *, seq_lens=None, cos_threshold=None):
        with self._lock:
            self._require_open()
            limit = min(self.max_tokens, self.max_position)
            lengths = tuple(seq_lens) if seq_lens is not None else (min(7, limit), min(65, limit))
            if not lengths or any(type(n) is not int or not 0 < n <= limit for n in lengths):
                raise PackedEncodersError("seq_lens must be positive integers within the engine limits")
            states = [(m, m.training) for m in self.model.modules()]
            try:
                self.model.eval()
                ids = torch.randint(validation_ids_below(self.cfg), (len(lengths), max(lengths)),
                                    generator=torch.Generator().manual_seed(17))
                mask = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
                # topk replaces the backbone's classes with CUDA-only packed forwards.
                # Bind the same weights to stock HF classes on meta (no weight allocation)
                # for an independent CPU reference, with topk's attention direction.
                reference = None
                if self.topk:
                    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
                    cfg = copy.deepcopy(self.cfg)
                    cfg.is_causal, cfg.use_cache, cfg._attn_implementation = self.causal, False, "sdpa"
                    with torch.device("meta"):
                        reference = Qwen3_5TextModel(cfg)
                    reference.load_state_dict(self.model.state_dict(), assign=True)
                    reference.rotary_emb = self.model.rotary_emb
                    reference.eval()
                    original = reference.forward
                else:
                    original = self.original_forward if self.binding.patch_target is self.model else self.model.forward
                with torch.no_grad(), torch.autocast("cpu", enabled=False):
                    want = original(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state[mask]
                    got = self._packed(ids[mask], lengths, mask.nonzero()[:, 1])
                    cos = F.cosine_similarity(got, want, dim=-1)
                    cos = torch.where((got.norm(dim=-1) == 0) & (want.norm(dim=-1) == 0), 1., cos)
                    diff = got - want
                    report = Qwen35CPUReport(self.precision, self.state.gdn_backend, lengths, float(cos.min()),
                                             float(cos.median()), float(diff.norm() / want.norm().clamp_min(1e-12)),
                                             float(diff.abs().max()))
                    threshold = cos_threshold if cos_threshold is not None else (.99999 if self.precision == "fp32" else .99)
                    score = report.min_cosine if self.precision == "fp32" else report.median_cosine
                    if not torch.isfinite(got).all() or score < threshold or report.relative_l2 > (2e-4 if self.precision == "fp32" else .25):
                        raise ValidationError(f"CPU Qwen3.5 output validation failed: {report}")
                    self.state.validation_report = report
                    return ValidationResult(Qwen35CPU.name, report)
            finally:
                for m, training in states:
                    m.training = training

    def _epilogue_context(self):
        # A plain FP32 head; backbone projection precision is independent.
        from packed_encoders.runtime.epilogue import PreparationContext, PreparedLinear
        class Linear(PreparedLinear):
            def _execute_owned(self, x):
                return F.linear(x, self._weight, self._bias)
        class Context(PreparationContext):
            input_dtypes = (torch.float32,)
            def _linear_dtype(self, projection, precision):
                if precision not in (None, "fp32"):
                    raise PackedEncodersError("CPU epilogues require FP32")
                return torch.float32
            def _make_linear(self, weight, bias):
                return Linear(self, weight, bias)
        return Context(self)

    def _forward_epilogue(self, batch, apply):
        return self._forward_packed(batch, apply)

    def close(self, *, rollback=False):
        with self._lock:
            if self._closed:
                return
            self._close_epilogues()
            for resource in (self.int8, self.sme, self.native):
                if resource is not None:
                    resource.close()
            self.int8 = self.sme = self.native = None
            self.layers.clear()
            self.projections.clear()
            self._closed = True
