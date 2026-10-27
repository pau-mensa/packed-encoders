"""Inference-only, padding-free ModernBERT CPU engine: exact FP32, or opt-in int8 projections."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
import threading
from types import SimpleNamespace
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from packed_encoders.batch import PackedBatch
from packed_encoders.config import _rope_thetas
from packed_encoders.cpu.attention import CPUAttention, ResolvedNorm
from packed_encoders.cpu.memory import rehome
from packed_encoders.engine import Capabilities, ModelBinding, ValidationResult
from packed_encoders.errors import PackedEncodersError, UnsupportedTargetError, ValidationError
from packed_encoders.locate import is_modernbert_encoder
from packed_encoders.runtime.epilogue import EpilogueSupport
from packed_encoders.state import ATTR

# Pack-time gate for precision="int8" against the FP32 HF forward (see validate()).
INT8_MEDIAN_COSINE, INT8_RELATIVE_L2 = 0.99, 0.25


def _alias(tensor):
    return None if tensor is None else tensor.detach()


def _forward_hooks(module):
    # Transformers 5 installs these permanently on the first request for extra
    # outputs. Such requests already delegate to HF; ordinary calls leave them
    # inactive. Other hooks can change values and cannot be bypassed.
    return tuple(key for key, hook in module._forward_hooks.items()
                 if not (getattr(hook, "__module__", "") == "transformers.utils.output_capturing"
                         and getattr(hook, "__qualname__", "") ==
                         "install_output_capuring_hook.<locals>.output_capturing_hook"))


def _options(options):
    unknown = options.keys() - {"validate", "attention_backend", "cuda_graph", "train_cuda_graph", "cuda_graph_seq_cutoff",
                                "precision"}
    if unknown:
        raise PackedEncodersError(f"unsupported CPU options: {sorted(unknown)}")
    for key in ("cuda_graph", "train_cuda_graph"):
        if options.get(key) is not None and options[key] is not False:
            raise PackedEncodersError("the CPU engine does not support CUDA graphs")
    if options.get("cuda_graph_seq_cutoff") is not None:
        raise PackedEncodersError("cuda_graph_seq_cutoff is unsupported on CPU")
    if options.get("attention_backend") not in (None, "auto", "native", "sdpa"):
        raise PackedEncodersError("CPU attention_backend must be 'auto', 'native' or 'sdpa'")
    if options.get("precision") not in (None, "fp32", "int8"):
        raise PackedEncodersError("CPU precision must be 'fp32' or 'int8'")


@dataclass(frozen=True)
class CPUValidationReport:
    attention_backend: str
    min_cosine: float
    relative_l2: float
    max_abs: float
    sequence_lengths: tuple[int, ...]
    precision: str = "fp32"
    median_cosine: float = 1.0


class CPUAdapter:
    name = "hf-modernbert-cpu"

    def bind(self, module):
        if is_modernbert_encoder(module) and next(module.parameters()).device.type == "cpu":
            return ModelBinding(self, module, module)
        return None

    def install(self, binding, packed):
        packed.install_forward()


class ModernBertCPU:
    """CPU configuration. Thread count controls the native kernels' workers
    (attention, row kernels, native FP32 projections and int8 projections); it
    defaults to PyTorch's thread count at packing, capped to the current Linux
    CPU affinity mask and to its physical cores when topology is available.

    Other matrix operations (unprepared Apple FP32 projections, SDPA) use PyTorch/Accelerate's
    existing thread settings. This engine never changes process-global threading
    configuration.

    ``rehome_weights=True`` moves the projection weights into fresh process
    memory when packing (same Parameter objects and values, new storage). It
    can speed up FP32 when they sit in from_pretrained's safetensors file mapping.
    On Linux, best-effort page eviction limits the extra resident memory; on other
    platforms this only copies. In-place updates and state_dict keep
    working; references to the old storage taken before packing stop aliasing
    the model. Unpacking does not move them back.

    Native x86 FP32 projections select register tiles using the detected L1 data
    cache size. ``tune_gemm=True`` additionally measures candidate tiles during
    packing, using the model's live weights at 16/32/256 tokens and the configured
    thread count. Other lengths use cache-aware selection. Repack after changing
    CPU affinity; tuning adds setup time and never runs during inference.

    On SME2-capable Apple Silicon, ``prepack_weights=True`` prepares FP32 SME weight copies
    (about 421 MiB extra for ModernBERT-base), used up to 128 packed tokens on one thread and
    up to 1,024 or 4,096 with more threads; with ``threads`` above one they, native attention
    and the row kernels run on an engine-owned worker pool (otherwise GCD). Ordinary in-place
    updates refresh the copies. Disable it
    to retain Accelerate and save memory. Other CPUs and int8 do not prepare these FP32 copies.
    """
    name = "modernbert-cpu"
    adapters = (CPUAdapter(),)

    def __init__(self, *, threads=None, max_tokens=32768, rehome_weights=False, tune_gemm=False,
                 prepack_weights=True):
        if threads is not None and (type(threads) is not int or threads <= 0):
            raise ValueError("threads must be a positive integer")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if type(rehome_weights) is not bool:
            raise ValueError("rehome_weights must be a bool")
        if type(tune_gemm) is not bool:
            raise ValueError("tune_gemm must be a bool")
        if type(prepack_weights) is not bool:
            raise ValueError("prepack_weights must be a bool")
        self.threads, self.max_tokens, self.rehome_weights = threads, max_tokens, rehome_weights
        self.tune_gemm = tune_gemm
        self.prepack_weights = prepack_weights

    def prepare(self, binding, options):
        _options(options)
        packed = PackedModernBertCPU(binding, self, options.get("attention_backend") or "auto",
                                     options.get("precision") or "fp32")
        try:
            if options.get("validate", True):
                packed.validate()
            return packed
        except BaseException:
            packed.close(rollback=True)
            raise

    def validate(self, binding, **kwargs):
        # Validating without packing leaves the model's weight storage where it is.
        engine = copy.copy(self)
        engine.rehome_weights = False
        packed = engine.prepare(binding, {"validate": False})
        try:
            return packed.validate(**kwargs)
        finally:
            packed.close()


class PackedModernBertCPU(EpilogueSupport):
    capabilities = Capabilities(original_forward_fallback=True)
    pieces = ()

    def __init__(self, binding, engine, backend, precision="fp32"):
        self.binding, self.model = binding, binding.weight_source
        self.original_forward = binding.patch_target.forward
        self._lock = threading.RLock()
        self._closed = False
        self._init_epilogues(lock=self._lock)
        self.prepack_weights, self.tune_gemm = engine.prepack_weights, engine.tune_gemm
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
        for param in self.model.parameters():
            if param.device.type != "cpu" or param.dtype != torch.float32 or not param.is_contiguous():
                raise UnsupportedTargetError("CPU ModernBERT requires contiguous FP32 CPU parameters")
        # The projection library (SME FP32 or int8) compiles while CPUAttention compiles its own,
        # 0.3-0.5 s each; the calls below then find it cached. Failures are left for them to report.
        projection_library = None
        if precision == "int8":
            from packed_encoders.cpu import int8
            if int8.sme2_available() or int8.vnni_available():
                projection_library = int8.library
        elif (engine.prepack_weights and backend != "sdpa" and self.dim == self.half == 64
              and min(self.max_tokens, self.max_position) > 1):
            from packed_encoders.cpu import sme
            if sme.sme2_available():
                projection_library = sme.library
        with ThreadPoolExecutor(1) as pool:
            if projection_library is not None:
                pool.submit(projection_library)
            from packed_encoders.cpu.topology import default_threads
            self.attention = CPUAttention(backend, self.dim, self.half, engine.threads or default_threads())
        self.attention.tune_attention(self.heads, self.max_position, self.max_tokens)
        if engine.rehome_weights:  # before the plan takes its aliases
            rehome([param for layer in self.layers
                    for projection in (layer.attn.Wqkv, layer.attn.Wo, layer.mlp.Wi, layer.mlp.Wo)
                    for param in (projection.weight, projection.bias) if param is not None])
        # What _core reads per layer, resolved once: the attention kind, (weight, bias) aliases
        # of the four projections, the two norms it applies, and whether the second is the final
        # norm. Aliases share the parameters' storage, so in-place updates are read. Requests
        # never re-walk the module tree (a cold walk cost ~0.3 ms per call): other changes after
        # packing (replaced parameters or modules, .data swaps, device/dtype moves, internal
        # hooks, config edits) need a repack and go undetected, but never reach native code as
        # stale pointers.
        following = (*(layer.attn_norm for layer in self.layers[1:]), self.model.final_norm)
        self.plan = tuple(
            (kind, *((_alias(p.weight), _alias(p.bias)) for p in (layer.attn.Wqkv, layer.attn.Wo, layer.mlp.Wi,
                                                                   layer.mlp.Wo)),
             ResolvedNorm(layer.mlp_norm), ResolvedNorm(after), after is self.model.final_norm)
            for layer, kind, after in zip(self.layers, self.kinds, following))
        # Private ablation switch: the fast path for unpadded/right-padded masks.
        self.fast_masks = True
        self.state = SimpleNamespace(attention_backend=self.attention.backend, validation_report=None,
                                     precision="fp32")
        self.tables = {}
        # Tables are engine-owned; HF parameters/storage and state_dict stay intact.
        for kind, theta in zip(("full_attention", "sliding_attention"), _rope_thetas(cfg)):
            inv = 1.0 / (theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
            angles = torch.outer(torch.arange(self.max_position).float(), inv).repeat(1, 2)
            self.tables[kind] = (angles.cos(), angles.sin())
        # Prepared int8 weights are engine-owned too; calibration runs the FP32 path above.
        # Machines that cannot run them (M1-M3, other CPUs, no M4-capable compiler) warn
        # and stay exact FP32; unsupported model shapes still raise.
        self.requested_precision, self.int8 = precision, None
        if precision == "int8":
            from packed_encoders.cpu.int8 import Int8Projections, Int8Unavailable
            try:
                self.int8 = Int8Projections(self)
            except Int8Unavailable as exc:
                warnings.warn(f"{exc}; using exact FP32", stacklevel=4)
        self.precision = self.state.precision = "fp32" if self.int8 is None else "int8"
        if engine.prepack_weights and self.precision == "fp32" and min(self.max_tokens, self.max_position) > 1:
            self.attention.prepare_sme([projection for layer in self.plan for projection in layer[1:5]])
        if engine.tune_gemm and self.precision == "fp32":
            self.attention.tune_linear([projection for layer in self.plan for projection in layer[1:5]],
                                       self.max_tokens)

    def _check(self):
        if self._closed:
            raise PackedEncodersError("this CPU engine has been closed; pack the model again")

    _require_open = _check

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
        if options.get("precision") not in (None, self.requested_precision, self.precision):
            raise PackedEncodersError("unpack before changing the CPU precision")

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
            # HF task heads pass position_ids=None and inputs_embeds=None; only set values change the forward.
            unsupported = {key for key, value in kwargs.items() if value is not None} - {
                "return_dict", "output_hidden_states", "output_attentions", "token_type_ids"}
            extra_outputs = any(kwargs.get(key) if kwargs.get(key) is not None else getattr(cfg, key, False)
                                for key in ("output_hidden_states", "output_attentions"))
            token_types = kwargs.get("token_type_ids")
            fallback = (args or unsupported or extra_outputs or self.model.training or torch.is_grad_enabled()
                        or torch.is_autocast_enabled("cpu") or input_ids is None
                        or (token_types is not None and bool(token_types.any())))
            if fallback:
                return self.original_forward(input_ids, attention_mask, *args, **kwargs)
            if input_ids.dtype == torch.int32:  # HF's embedding accepts int32 IDs too
                input_ids = input_ids.long()
            self._ids(input_ids, ndim=2)
            out = self._prefixes(input_ids, attention_mask) if self.fast_masks else None
            if out is None:
                out = self._compacted(input_ids, attention_mask)
                if out is None:
                    return self.original_forward(input_ids, attention_mask, **kwargs)
            return_dict = kwargs.get("return_dict")
            if return_dict is None:
                return_dict = getattr(cfg, "return_dict", True)
            if not return_dict:
                return (out,)
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(last_hidden_state=out)

    def _prefixes(self, input_ids, attention_mask):
        """The padded output when every row is a nonempty prefix (unpadded or right-padded
        rows), else None for the general path. Equality with a boolean prefix also rejects
        values other than 0/1."""
        b, s = input_ids.shape
        columns = torch.arange(s)
        if attention_mask is not None:
            if attention_mask.device.type != "cpu" or attention_mask.shape != input_ids.shape:
                return None
            counts = (attention_mask != 0).sum(1)  # integer lengths for float masks too
            prefix = columns < counts[:, None]
            if not bool((attention_mask == prefix).all()):
                return None
            lengths = tuple(counts.tolist())
            if not all(lengths):
                return None
        if attention_mask is None or all(n == s for n in lengths):
            return self._packed(input_ids.reshape(-1), (s,) * b, columns.repeat(b) if b > 1 else columns).view(b, s, -1)
        hidden = self._packed(input_ids[prefix], lengths, columns.expand(b, s)[prefix])
        out = hidden.new_zeros(b, s, self.hidden)
        out[prefix] = hidden
        return out

    def _compacted(self, input_ids, attention_mask):
        """The padded output for any 0/1 mask, or None to delegate to HF."""
        b, s = input_ids.shape
        if attention_mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        elif (attention_mask.device.type != "cpu" or attention_mask.shape != input_ids.shape
              or not bool(((attention_mask == 0) | (attention_mask == 1)).all())):
            return None
        else:
            mask = attention_mask.bool()
        lengths = tuple(mask.sum(1).tolist())
        # Compaction must preserve local-window distances. Holey or empty
        # rows delegate to HF; left/right padding preserves column RoPE.
        columns = torch.arange(s).expand(b, s)
        starts = columns.masked_fill(~mask, s).amin(1)
        ends = columns.masked_fill(~mask, -1).amax(1)
        if not all(lengths) or not torch.equal(ends - starts + 1, mask.sum(1)):
            return None
        indices = mask.flatten().nonzero().flatten()
        ids = input_ids.flatten()[indices]
        hidden = self._packed(ids, lengths, indices % s)
        out = hidden.new_zeros(b * s, self.hidden)
        out.index_copy_(0, indices, hidden)
        return out.view(b, s, self.hidden)

    def _ids(self, ids, *, ndim):
        if ids.device.type != "cpu" or ids.dtype != torch.int64 or ids.ndim != ndim or not ids.numel():
            raise PackedEncodersError(f"CPU token IDs must be a nonempty int64 CPU tensor of rank {ndim}")
        if int(ids.min()) < 0 or int(ids.max()) >= self.model.embeddings.tok_embeddings.num_embeddings:
            raise PackedEncodersError("token IDs are outside the embedding vocabulary")

    def _epilogue_context(self):
        from packed_encoders.cpu.epilogue import CPUPreparation
        return CPUPreparation(self)

    def _forward_epilogue(self, batch, apply):
        return self._forward_packed(batch, epilogue=apply)

    def forward_packed(self, batch: PackedBatch):
        return self._forward_packed(batch)

    def _forward_packed(self, batch: PackedBatch, epilogue=None):
        with self._lock:
            self._check()
            if self.model.training or torch.is_grad_enabled() or torch.is_autocast_enabled("cpu"):
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
            return self._packed(batch.input_ids, tuple(lengths), positions, epilogue=epilogue)

    def _packed(self, ids, lengths, positions, epilogue=None):
        if (positions.device.type != "cpu" or positions.dtype != torch.int64 or positions.shape != ids.shape
                or int(positions.min()) < 0 or int(positions.max()) >= self.max_position):
            raise PackedEncodersError("CPU positions must be int64, match IDs, and lie within max_position_embeddings")
        if max(lengths) > self.max_tokens or max(lengths) > self.max_position:
            raise PackedEncodersError("sequence exceeds the engine's max_tokens or max_position_embeddings")
        core = self._core if self.int8 is None else self.int8.core

        def run(chunk_ids, chunk_lengths, chunk_positions):
            hidden = core(chunk_ids, chunk_lengths, chunk_positions)
            if epilogue is None:
                return hidden
            cu = torch.tensor((0, *chunk_lengths), dtype=torch.int32).cumsum(0, dtype=torch.int32)
            batch = PackedBatch(chunk_ids, cu_seqlens=cu, max_seqlen=max(chunk_lengths),
                                position_ids=chunk_positions, host_lengths=chunk_lengths)
            return epilogue(hidden, batch)

        outputs, offset, group = [], 0, []
        for length in lengths:
            if group and sum(group) + length > self.max_tokens:
                count = sum(group)
                outputs.append(run(ids[offset:offset+count], tuple(group), positions[offset:offset+count]))
                offset += count
                group = []
            group.append(length)
        outputs.append(run(ids[offset:], tuple(group), positions[offset:]))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    def _core(self, ids, lengths, positions, linear=None):
        tables = {kind: (cos[positions][None, None], sin[positions][None, None])
                  for kind, (cos, sin) in self.tables.items()}
        boundaries = torch.tensor(lengths, dtype=torch.int32)
        x = self.model.embeddings(ids[None])
        normed = self.layers[0].attn_norm(x)
        cpu = self.attention
        linear = linear or cpu.linear
        # Each residual add feeds the next norm: this layer's MLP norm, then the
        # next layer's attention norm or the final norm.
        for kind, (wqkv, bqkv), (wo, bo), (wi, bi), (wmo, bmo), mlp_norm, after, final in self.plan:
            qkv = linear(normed, wqkv, bqkv)
            cos, sin = tables[kind]
            q, k, v = cpu.rope_layout(qkv, cos, sin, self.heads, self.dim)
            attention = cpu.local if kind == "sliding_attention" else cpu.global_
            context = attention(q, k, v, lengths, boundaries, token_major=True)
            x, normed = cpu.add_norm(x, linear(context, wo, bo), mlp_norm)
            x, normed = cpu.add_norm(x, linear(cpu.geglu(linear(normed, wi, bi)), wmo, bmo), after, owned=final)
        return normed[0]

    def validate(self, *, seq_lens=None, cos_threshold=None):
        """Compare with the HF forward. ``seq_lens`` replaces the default ragged batch (7 and
        129 tokens) with one batch of those lengths; ``cos_threshold`` replaces the cosine
        bound, which applies to every token in FP32 (0.99999) and to the median in int8."""
        with self._lock:
            self._check()
            states = [(module, module.training) for module in self.model.modules()]
            limit = min(self.max_position, self.max_tokens)
            if seq_lens is None:
                lengths = (min(7, limit), min(129, limit))
            else:
                lengths = tuple(seq_lens)
                if not lengths or any(type(n) is not int or not 0 < n <= limit for n in lengths):
                    raise PackedEncodersError(f"seq_lens must be positive integers of at most {limit} tokens")
            if cos_threshold is None:
                cos_threshold = 0.99999 if self.precision == "fp32" else INT8_MEDIAN_COSINE
            cases = [lengths]
            if self.attention.sme is not None and self.attention.sme.weights:
                # One ragged call may exceed SME's dispatch limit. Exercise its register tiles,
                # row-block loop, and (within that limit) the 512-row tile groups.
                cases.extend((min(n, self.max_position, self.max_tokens),)
                             for n in (16, 32, min(self.attention.sme.max_rows, 545)))
            ids = torch.randint(self.model.config.vocab_size, (max(map(len, cases)), max(map(max, cases))),
                                generator=torch.Generator().manual_seed(17))
            reports = []
            try:
                self.model.eval()
                with torch.no_grad(), torch.autocast("cpu", enabled=False):
                    for case in dict.fromkeys(cases):
                        inputs = ids[:len(case), :max(case)]
                        mask = torch.arange(inputs.shape[1])[None] < torch.tensor(case)[:, None]
                        reference = self.original_forward(input_ids=inputs, attention_mask=mask, return_dict=True).last_hidden_state[mask]
                        actual = self._packed(inputs[mask], case, mask.nonzero()[:, 1])
                        diff = actual - reference
                        cosine = F.cosine_similarity(actual, reference, dim=-1)
                        cosine = torch.where((actual.norm(dim=-1) == 0) & (reference.norm(dim=-1) == 0), 1., cosine)
                        report = CPUValidationReport(self.attention.backend, float(cosine.min()),
                                                     float(diff.norm() / reference.norm().clamp_min(1e-12)),
                                                     float(diff.abs().max()), case, self.precision,
                                                     float(cosine.median()))
                        # int8 bounds typical-token cosine and overall error; FP32 bounds every token.
                        if self.precision == "fp32":
                            failed = report.min_cosine < cos_threshold or report.relative_l2 > 2e-4
                        else:
                            failed = report.median_cosine < cos_threshold or report.relative_l2 > INT8_RELATIVE_L2
                        if not torch.isfinite(actual).all() or failed:
                            raise ValidationError(f"CPU ModernBERT output validation failed: {report}")
                        reports.append(report)
            finally:
                for module, training in states:
                    module.training = training
            report = CPUValidationReport(self.attention.backend, min(r.min_cosine for r in reports),
                                         max(r.relative_l2 for r in reports), max(r.max_abs for r in reports),
                                         tuple(n for r in reports for n in r.sequence_lengths), self.precision,
                                         min(r.median_cosine for r in reports))
            self.state.validation_report = report
            return ValidationResult("modernbert-cpu", report)

    def close(self, *, rollback=False):
        with self._lock:
            self._close_epilogues()
            if self.int8 is not None:
                self.int8.close()
                self.int8 = None
            self.attention.close()
            self.tables.clear()
            self.layers = self.plan = ()
            self._closed = True
