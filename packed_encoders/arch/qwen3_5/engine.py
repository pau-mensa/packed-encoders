"""Exact fast forward for a Qwen3.5 hybrid text backbone (GatedDeltaNet + gated softmax attention).

Same math as the model, fewer and fatter kernels:

- merged projections: GDN q|k|v|z|b|a, attention q|gate|k|v, MLP gate|up — one GEMM each.
  The merged weight *is* the HF weight: each HF Linear's parameter is re-pointed at its row
  range of the merged buffer, so packing costs no extra persistent projection storage (`unpack()` separates them).
- GatedDeltaNet through fla with the decay gate (A_log, dt_bias), beta sigmoid, and q/k L2
  norm inside the kernel: the chunked kernel in graphs and for long eager batches, the fused
  recurrent one (a single launch) for short eager batches. Both are probed at build time against
  an fp32 recurrence (`select_gdn`). The causal conv reads strided slices of the merged projection, so fla's
  contiguity guard makes no copies.
- q/k RMSNorm + partial RoPE in one Triton kernel reading the interleaved q|gate in place.
- fla residual-add + RMSNorm and SwiGLU; varlen attention chosen by probe (runtime.attention).
- Fusions (`fused=True`, each probed against the unfused op at build time, else off): the
  q|k|v conv with the b|a gate copies in one launch (fla's kernels would copy the strided
  gates), the gated RMSNorm reading z in place, q and k norm + RoPE in one launch, and the
  attention output gate in one kernel.

Three entry points share the layer code: `forward_packed` (eager, padding-free, any batch),
`padded_core` (the CUDA-graph region, right-padded `(rows, S)`; exact because the
mixers are causal and attention sees `[real | pad]` segments — see runtime.graphs), and
`forward_shared` (eager, causal, opt in: rows that start with the same tokens run that prefix
once — see sharing).
"""

from __future__ import annotations

import contextlib
import weakref
from types import SimpleNamespace
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import fla.utils
from fla.modules import FusedRMSNormGated
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from packed_encoders.arch.qwen3_5.kernels import conv_split, gated_rms_norm, qk_norm_rope, qk_norm_rope_pair, sigmoid_gate
from packed_encoders.arch.qwen3_5.execution import ORDINARY, PREFIX
from packed_encoders.arch.weights import (
    require_independent_parameters, require_plain as _require_plain, share_rows as _share_rows, unshare_rows,
)
from packed_encoders.runtime.sharing import SharedPlan, plan_shared_prefixes
from packed_encoders.errors import UnsupportedTargetError, ValidationError
from packed_encoders.pieces.hybrid import reference_delta as _gdn_reference
from packed_encoders.runtime.attention import AttentionChoice, select_attention
from packed_encoders.runtime.graphs import PaddedStatic
from packed_encoders.runtime.staging import PinnedStager, packed_layout_host


@contextlib.contextmanager
def fla_tensor_cache():
    """Enable fla's identity cache (chunk indices computed once per batch, not once per GDN
    layer) for the duration of our forward only. fla reads the flag at call time, and the
    stock topk model turns it off globally, so we restore whatever was set. The flag is
    process-global: don't run this forward concurrently with another fla model on other threads."""
    prev = fla.utils.FLA_DISABLE_TENSOR_CACHE
    fla.utils.FLA_DISABLE_TENSOR_CACHE = False
    try:
        yield
    finally:
        fla.utils.FLA_DISABLE_TENSOR_CACHE = prev


def _require_independent_parameters(model: nn.Module) -> None:
    require_independent_parameters(model, "Qwen3.5")


GDN_PROBE_TOLERANCE = 3e-2     # max |kernel - fp32 recurrence|, relative to the recurrence's max |output|
FUSION_TOLERANCE = 2e-2        # max |fused - unfused op|, relative to the unfused op's max |output| (bf16: a few ulp)
# Which GatedDeltaNet kernel: inside a CUDA graph, launches cost no host time, so the chunked kernel (parallel
# over chunks, ~6 launches) always runs. Eager, launches are the cost, so a batch whose rows are all up to this
# many tokens takes the fused recurrent kernel (one launch, a sequential token loop). Measured, queries/s (same process per row):
#   graphed, xsmall, L40S batch 8: chunked 2200, <=32 2116, <=64 1895 | H100: chunked 3092, <=32 2927, <=64 2525
#   eager (128 rows > the largest bucket), L40S: xsmall <=64 4418, <=32 2534; small <=64 2263, <=32 2408 (small
#   has more GPU work per launch, so the recurrent kernel's slower GPU time shows; end to end it is a wash).
# (A pre-fusion L40S sweep had <=32 winning graphed batch 8; it no longer does.) 0: chunked everywhere.
RECURRENT_MAX_LEN = 64
# Shared prefixes are opt in: `min_shared_prefix` starts at 0 (off); a caller sets how many leading tokens rows
# must agree on to run their common prefix once. The shared pass is eager and runs GatedDeltaNet twice per layer
# (prefixes, then continuations), so it pays only where a forward is compute bound. Measured, batch 8, shared
# at 64 tokens vs the faster of graphs and eager, rows per prefix 2 / 4 / 8:
#   27B backbone (pplx-decider-v1-27b), H100, 71- to 3,096-token prefixes, short questions: 1.8x / 3.0x / 4.2x
#   Qwen3.5-0.8B, L40S: 1,024-token prefixes + up to 1,024 own tokens 1.31x / 1.35x / 1.37x; shorter prefixes
#   or rows 0.12x-0.99x (the shared pass has a floor of ~85 ms per batch, however few tokens it computes).
# benchmarks/qwen35_shared_prefix_bench.py measures a model. validate() checks the shared pass at this threshold.
SHARED_PREFIX_MIN = 64


def _chunk(q, k, v, a, b, L, cu=None, cu_cpu=None):
    return chunk_gated_delta_rule(q, k, v, g=a, beta=b, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                                  A_log=L.A_log, dt_bias=L.dt_bias, use_beta_sigmoid_in_kernel=True,
                                  cu_seqlens=cu, cu_seqlens_cpu=cu_cpu)[0]


def _recurrent(q, k, v, a, b, L, cu=None, cu_cpu=None):
    return fused_recurrent_gated_delta_rule(q, k, v, g=a, beta=b, use_qk_l2norm_in_kernel=True,
                                            use_gate_in_kernel=True, A_log=L.A_log, dt_bias=L.dt_bias,
                                            use_beta_sigmoid_in_kernel=True, cu_seqlens=cu)[0]


_GDN_KERNELS = {"chunk": _chunk, "recurrent": _recurrent}


@dataclass
class GdnChoice:
    expand_gva: bool               # repeat q/k heads up to the value heads before the kernel call
    errors: dict[str, float]       # kernel -> worst relative error over the packed and padded probes
    rejected: dict[str, str]
    recurrent_max_len: int         # 0: the chunked kernel for every length


def select_gdn(L: "_Layer", device: torch.device, dtype: torch.dtype, recurrent_max_len: int) -> GdnChoice:
    """Call each fla kernel exactly as `_gdn` will — raw gates fused in — and compare real tokens
    with `_gdn_reference`. The chunked kernel runs packed (cu_seqlens) and padded (rows, S) and must
    pass both; the recurrent one only ever runs eager, packed, and is used for short batches only if
    it passes there. Checking the numbers rather than a signature also catches a flag an fla version
    would swallow in **kwargs."""
    gen = torch.Generator(device=device).manual_seed(0)
    lengths = [5, 40, 1, 23]
    T, rep = sum(lengths), L.nv // L.nk

    def rand(*shape):
        return torch.randn(*shape, device=device, generator=gen).to(dtype)

    q, k, v = rand(T, L.nk, L.hk), rand(T, L.nk, L.hk), rand(T, L.nv, L.hv)
    a, b = rand(T, L.nv), rand(T, L.nv)
    with torch.no_grad():
        decay = -L.A_log.float().exp() * F.softplus(a.float() + L.dt_bias.float())
        ref = _gdn_reference(q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1), v, decay,
                             torch.sigmoid(b.float()), lengths)
    scale = ref.abs().max().item()
    cu_cpu = torch.tensor([0] + lengths).cumsum(0)
    cu = cu_cpu.to(device)
    rows, S = len(lengths), max(lengths)
    dst = torch.cat([torch.arange(n) + i * S for i, n in enumerate(lengths)]).to(device)

    def padded(x):
        out = x.new_zeros((rows * S, *x.shape[1:]))
        return out.index_copy_(0, dst, x).view(rows, S, *x.shape[1:])

    def worst_err(fn, expand, with_padded):
        qq, kk = (q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)) if expand else (q, k)
        with torch.no_grad():
            got = [fn(qq[None], kk[None], v[None], a[None], b[None], L, cu, cu_cpu)[0]]
            if with_padded:
                got_padded = fn(padded(qq), padded(kk), padded(v), padded(a), padded(b), L)
                got.append(got_padded.reshape(rows * S, L.nv, L.hv).index_select(0, dst))
        return max((g.float() - ref).abs().max().item() / scale for g in got)

    errors, rejected = {}, {}

    def attempt(name, expand):
        try:
            err = worst_err(_GDN_KERNELS[name], expand, with_padded=name == "chunk")
        except Exception as exc:  # noqa: BLE001
            rejected[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
            return False
        errors[name] = err
        if not err < GDN_PROBE_TOLERANCE:
            rejected[name] = f"relative error {err:.2e} >= {GDN_PROBE_TOLERANCE}"
            return False
        rejected.pop(name, None)
        return True

    # Native grouped-value heads first (no copy); expand as the model does if this fla lacks them.
    expand = False
    if not attempt("chunk", expand) and rep > 1:
        expand = attempt("chunk", True)
    if "chunk" in rejected:
        import fla

        raise ValidationError(f"fla {fla.__version__} chunk_gated_delta_rule with in-kernel gates disagrees with "
                              f"the gated delta rule: {rejected['chunk']}")
    use_recurrent = recurrent_max_len > 0 and attempt("recurrent", expand)
    return GdnChoice(expand_gva=expand and rep > 1, errors=errors, rejected=rejected,
                     recurrent_max_len=recurrent_max_len if use_recurrent else 0)


class _Layer:
    pass


@dataclass
class _Layout:
    execution = ORDINARY
    shape: tuple[int, int]            # (B, S) the causal ops see: (1, T) packed, (rows, S) padded
    cos: Tensor
    sin: Tensor
    cu: Tensor | None = None          # packed: int64 cu_seqlens (fla)
    cu_cpu: Tensor | None = None
    cu32: Tensor | None = None        # packed: int32 cu_seqlens (attention)
    max_len: int = 0
    lengths: list[int] | None = None
    static: PaddedStatic | None = None
    pos: Tensor | None = None         # (T,) position of each token in its sequence / row (the fused conv)
    layer_index: int = 0


@dataclass
class _PrefixLayout(_Layout):
    execution = PREFIX
    share: _Shared | None = None      # forward_shared: segments are roots then children
    prefix: object | None = None      # explicit persistent prefix handle
    prefix_write: bool = False
    prefix_meta: object | None = None


@dataclass
class _Shared:
    """Device side of a SharedPlan. The roots region is tokens [0, root_tokens)."""

    root_tokens: int
    cu_roots: Tensor
    cu_roots_cpu: Tensor
    cu_kids: Tensor                   # children's boundaries, relative to root_tokens
    cu_kids_cpu: Tensor
    parent: Tensor
    fix_rows: Tensor
    fix_taps: Tensor
    kv_idx: Tensor
    cu_k32: Tensor
    max_k: int
    kv_lengths: list[int]


class Qwen35Engine:
    def __init__(self, text_model: nn.Module, *, causal: bool, attention_order: Sequence[str] | None = None,
                 recurrent_max_len: int = RECURRENT_MAX_LEN, fused: bool = True, pieces=None):
        from packed_encoders.arch.qwen3_5.pieces import default_pieces

        self.composition = pieces or default_pieces()
        self.ops = self.composition.bind()
        _require_independent_parameters(text_model)
        cfg = text_model.config
        self.tm, self.cfg, self.causal = text_model, cfg, causal
        self.embed = text_model.embed_tokens.weight
        self.device, self.dtype = self.embed.device, self.embed.dtype
        if self.device.type != "cuda":
            raise UnsupportedTargetError(f"the Qwen3.5 engine needs CUDA weights; model is on {self.device}")
        self.hidden_size = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self._norms: list[list] = []            # [parameter (detached), fp32 copy, version seen]; see sync_norms
        self.w_final = self._plus_one(text_model.norm.weight)
        n = cfg.num_hidden_layers
        self.shared: list[nn.Linear] = []
        kinds = list(getattr(cfg, "layer_types", None) or [])
        if len(kinds) < n:
            raise UnsupportedTargetError("config.layer_types is missing or shorter than num_hidden_layers")
        self.layers: list[_Layer] = []
        self._prefix_caches = weakref.WeakSet()
        try:
            # Triton launches and CUDA graphs use the *current* device, not the tensors': pin it to the
            # weights' for every launch (here, `forward_packed`, the graph runner, the stager).
            with torch.cuda.device(self.device):
                for layer, kind in zip(text_model.layers[:n], kinds[:n]):
                    self.layers.append(self._prepare(layer, kind))
                gdn = next((L for L in self.layers if L.linear), None)
                self.gdn: GdnChoice | None = None
                if gdn is not None:
                    self.gdn = select_gdn(gdn, self.device, self.dtype, recurrent_max_len)
                attn = next((L for L in self.layers if not L.linear), None)
                self.attention: AttentionChoice | None = None
                self.qk_check_err = None
                if attn is not None:
                    self.attention = select_attention(n_heads=attn.nq, n_kv_heads=attn.nkv, head_dim=attn.hd,
                                                      causal=causal, device=self.device, dtype=self.dtype,
                                                      order=attention_order)
                    self.qk_check_err = self._check_qk(attn)
                self.fusion_errors: dict[str, float] = {}
                self.fusion_rejected: dict[str, str] = {}
                self.fused = fused and self._check_fusions(gdn, attn)
                self.conv_width = gdn.conv_w.shape[1] if gdn is not None else 1
                self.share_err: float | None = None
                try:
                    self.share_rejected = self._check_sharing(gdn)     # None: forward_shared can run here
                except Exception as exc:
                    self.share_rejected = f"shared-prefix probe failed: {type(exc).__name__}: {str(exc)[:200]}"
                self.min_shared_prefix = 0                          # off until the caller opts in
                # Optional CuTe acceleration; the Qwen-only environment may omit Cutlass.
                try:
                    from packed_encoders._kernels.prefix_conv import continue_conv
                    self._prefix_conv = continue_conv
                except ImportError:
                    self._prefix_conv = None
                self._stager = PinnedStager(self.device)
        except BaseException:
            self.release(rollback=True)                          # leave the model exactly as we found it
            raise

    def release(self, *, rollback: bool = False) -> None:
        """Drop the engine's merged weights, then give every projection its own storage back.

        The merged buffers must go first: each is then freed as soon as its last slice is cloned,
        so the transient cost is one projection group, not a second copy of every projection
        (which does not fit beside a 27B model's weights on an 80 GB GPU)."""
        for prefix in self._prefix_caches:
            prefix.close()
        self._prefix_caches.clear()
        self.layers.clear()
        self._norms.clear()
        self.w_final = None
        unshare_rows(self.shared, rollback=rollback)

    # ------------------------------------------------------------------ weights
    def _prepare(self, layer: nn.Module, kind: str) -> _Layer:
        L = _Layer()
        L.linear = kind == "linear_attention"
        if kind not in ("linear_attention", "full_attention"):
            raise UnsupportedTargetError(f"unsupported Qwen3.5 layer type {kind!r}")
        L.w_in = self._plus_one(layer.input_layernorm.weight)
        L.w_post = self._plus_one(layer.post_attention_layernorm.weight)
        mlp = layer.mlp
        if not all(hasattr(mlp, a) for a in ("gate_proj", "up_proj", "down_proj")):
            raise UnsupportedTargetError("only the dense SwiGLU MLP is supported (no MoE layers)")
        L.gate_up = _share_rows([mlp.gate_proj, mlp.up_proj], self.shared)
        L.inter = mlp.gate_proj.weight.shape[0]
        _require_plain([mlp.down_proj, layer.linear_attn.out_proj if L.linear else layer.self_attn.o_proj])
        L.down = mlp.down_proj.weight
        if L.linear:
            la = layer.linear_attn
            L.in_proj = _share_rows([la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a], self.shared)
            L.split_in = [la.in_proj_qkv.weight.shape[0], la.in_proj_z.weight.shape[0],
                          la.in_proj_b.weight.shape[0], la.in_proj_a.weight.shape[0]]
            conv_w = la.conv1d.weight.detach().squeeze(1)
            conv_b = None if la.conv1d.bias is None else la.conv1d.bias.detach()
            L.bounds = [0, la.key_dim, 2 * la.key_dim, 2 * la.key_dim + la.value_dim]
            L.conv = [(conv_w[i:j], None if conv_b is None else conv_b[i:j]) for i, j in zip(L.bounds, L.bounds[1:])]
            L.conv_w, L.conv_b = conv_w.contiguous(), conv_b
            L.kd, L.vd = la.key_dim, la.value_dim
            L.gate_off = L.split_in[0] + L.split_in[1]                   # b|a follow q|k|v and z
            L.act = la.activation
            L.hk, L.hv, L.nk, L.nv = la.head_k_dim, la.head_v_dim, la.num_k_heads, la.num_v_heads
            if L.nv % L.nk:
                raise UnsupportedTargetError(f"GatedDeltaNet value heads {L.nv} not a multiple of key heads {L.nk}")
            L.A_log, L.dt_bias = la.A_log.detach(), la.dt_bias.detach()
            eps = getattr(la.norm, "variance_epsilon", None) or getattr(la.norm, "eps")
            gn = FusedRMSNormGated(la.head_v_dim, eps=eps, activation=getattr(la.norm, "activation", "silu"),
                                   device=self.device, dtype=la.norm.weight.dtype)
            gn.weight.data = la.norm.weight.detach()
            L.gnorm = gn
            L.norm_module, L.gn_w, L.gn_eps = la.norm, la.norm.weight.detach(), eps
            L.gn_silu = getattr(la.norm, "activation", "silu") in ("silu", "swish")
            L.out = la.out_proj.weight
        else:
            sa = layer.self_attn
            L.hd = sa.head_dim
            L.nkv = sa.k_proj.weight.shape[0] // L.hd
            q_rows = sa.q_proj.weight.shape[0]
            L.nq = self.cfg.num_attention_heads
            # q_proj carries q|gate interleaved per head when the output gate is on; read it off the weight
            L.gated = q_rows == 2 * L.nq * L.hd
            if not L.gated and q_rows != L.nq * L.hd:
                raise UnsupportedTargetError(f"q_proj has {q_rows} rows; expected {L.nq}x{L.hd} (x2 if gated)")
            L.qkv = _share_rows([sa.q_proj, sa.k_proj, sa.v_proj], self.shared)
            L.q_norm = sa.q_norm
            L.wqk = torch.empty((2, L.hd), device=self.device, dtype=torch.float32)   # q row, k row
            L.wq = self._plus_one(sa.q_norm.weight, out=L.wqk[0])
            L.wk = self._plus_one(sa.k_norm.weight, out=L.wqk[1])
            L.k_norm = sa.k_norm
            L.qk_eps = sa.q_norm.eps
            L.o = sa.o_proj.weight
        return L

    def _plus_one(self, weight: Tensor, out: Tensor | None = None) -> Tensor:
        """fp32 `1 + weight`, the scale Qwen3.5's RMSNorm applies, for the fused norm kernels. Unlike
        the projections it is a copy, not the parameter's own storage, so it is tracked (sync_norms)."""
        src = weight.detach()                    # shares the parameter's storage and version counter
        if out is None:
            out = torch.empty(src.shape, device=src.device, dtype=torch.float32)
        out.copy_(src).add_(1.0)
        self._norms.append([src, out, src._version])
        return out

    def sync_norms(self) -> None:
        """Re-copy every norm scale whose parameter changed since it was copied, so a model trained
        after packing (an optimizer step, `load_state_dict`) encodes with its current norms. A host-side
        version check per call; the copy is in place, so captured graphs stay valid. Writes through
        `.data` bypass the version counter: re-pack after those."""
        for entry in self._norms:
            src, out, seen = entry
            if src._version != seen:
                out.copy_(src).add_(1.0)
                entry[2] = src._version

    def _check_qk(self, L: _Layer) -> float:
        """The fused q/k norm + RoPE kernel against the model's own RMSNorm + apply_rotary_pos_emb."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        g = torch.Generator(device=self.device).manual_seed(0)
        T = 37
        x = torch.randn(T, L.nq, 2 * L.hd, device=self.device, dtype=self.dtype, generator=g)[..., : L.hd]
        cos, sin = self._rope(torch.arange(T, device=self.device) * 7)
        with torch.no_grad():
            ref = apply_rotary_pos_emb(L.q_norm(x), L.q_norm(x), cos, sin, unsqueeze_dim=1)[0].float()
            err = (qk_norm_rope(x, L.wq, cos, sin, L.qk_eps).float() - ref).abs().max().item()
            # the one-launch pair, on a projection laid out as the model's: q|gate per head, then k, then v
            qw = L.nq * (2 * L.hd if L.gated else L.hd)
            proj = torch.randn(T, qw + 2 * L.nkv * L.hd, device=self.device, dtype=self.dtype, generator=g)
            qx = proj[:, :qw].view(T, L.nq, -1)[..., : L.hd]
            kx = proj[:, qw: qw + L.nkv * L.hd].view(T, L.nkv, L.hd)
            rq, rk = apply_rotary_pos_emb(L.q_norm(qx), L.k_norm(kx), cos, sin, unsqueeze_dim=1)
            pq, pk = qk_norm_rope_pair(proj, 0, qx.stride(1), L.nq, qw, L.hd, L.nkv, L.wqk, cos, sin, L.qk_eps)
            err = max(err, (pq.float() - rq.float()).abs().max().item(), (pk.float() - rk.float()).abs().max().item())
        if not err < 0.05:
            raise ValidationError(f"fused q/k norm + RoPE mismatch vs the model's modules: max abs err {err}")
        return err

    def _check_fusions(self, G: _Layer | None, A: _Layer | None) -> bool:
        """Each fused kernel against the op it replaces, on random data laid out as the model's:
        the conv against fla's causal_conv1d per sequence (+ exact gate copies), the gated norm
        against the model's own module, the output gate against torch. Any miss turns fusion off
        for the engine (recorded in `fusion_rejected`) rather than failing the pack."""
        g = torch.Generator(device=self.device).manual_seed(0)
        lengths = [5, 20, 1, 12]
        N = sum(lengths)

        def rand(*shape):
            return torch.randn(*shape, device=self.device, dtype=self.dtype, generator=g)

        def rel(got, ref):
            return (got.float() - ref.float()).abs().max().item() / max(ref.float().abs().max().item(), 1e-6)

        cu_cpu, pos = packed_layout_host(lengths)
        cu, pos = cu_cpu.to(self.device), pos.to(self.device)
        with torch.no_grad():
            if G is not None:
                if G.act not in ("silu", "swish") or not G.gn_silu:
                    self.fusion_rejected["gdn"] = f"conv activation {G.act!r} / norm gate not SiLU"
                else:
                    proj = rand(N, sum(G.split_in))
                    q, k, v, b, a = conv_split(proj, G.conv_w, G.conv_b, pos, G.kd, G.vd, G.gate_off, G.nv, G.nv, True)
                    ref = causal_conv1d(x=proj[None, :, : G.bounds[-1]], weight=G.conv_w, bias=G.conv_b,
                                        activation=G.act, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu)[0][0]
                    self.fusion_errors["conv"] = rel(torch.cat([q, k, v], 1), ref)
                    gates = proj[:, G.gate_off: G.gate_off + 2 * G.nv]
                    if not torch.equal(torch.cat([b, a], 1), gates):
                        self.fusion_rejected["conv"] = "gate columns not copied exactly"
                    o, z = rand(N, G.nv, G.hv), proj[:, G.bounds[-1]: G.bounds[-1] + G.nv * G.hv].view(N, G.nv, G.hv)
                    ref = G.gnorm(o.reshape(-1, G.hv), z.reshape(-1, G.hv)).view(N, -1)       # the unfused op
                    self.fusion_errors["gated_norm"] = rel(gated_rms_norm(o, z, G.gn_w, G.gn_eps), ref)
                    model = G.norm_module(o.reshape(-1, G.hv), z.reshape(-1, G.hv)).view(N, -1)
                    self.fusion_errors["gated_norm_vs_model"] = rel(gated_rms_norm(o, z, G.gn_w, G.gn_eps), model)
            if A is not None and A.gated:
                o, gate = rand(N, A.nq, A.hd), rand(N, A.nq, 2 * A.hd)[..., A.hd:]
                self.fusion_errors["output_gate"] = rel(sigmoid_gate(o, gate), (o * torch.sigmoid(gate)).view(N, -1))
        for name, err in self.fusion_errors.items():
            if name.endswith("_vs_model"):          # informational: the unfused path differs from it equally
                continue
            if not err < FUSION_TOLERANCE:
                self.fusion_rejected[name] = f"relative error {err:.2e} >= {FUSION_TOLERANCE}"
        return not self.fusion_rejected

    def _check_sharing(self, G: _Layer | None) -> str | None:
        """Why `forward_shared` can't run here, or None. Besides causality and an end-aligned
        attention, the GatedDeltaNet must resume exactly: roots run from a zero state, children
        from their root's final state, against the fp32 recurrence over the whole rows."""
        if not self.causal:
            return "bidirectional: a prefix's states depend on what follows"
        if self.attention is not None and self.attention.prefixed is None:
            return "no prefixed attention kernel"
        if G is None:
            return None
        if G.act not in ("silu", "swish"):
            return f"conv activation {G.act!r}"
        gen = torch.Generator(device=self.device).manual_seed(0)
        roots, kids, parent = [70, 33], [9, 1, 20], [0, 0, 1]
        rows = [roots[p] + n for p, n in zip(parent, kids)]
        rep = G.nv // G.nk

        def rand(*shape):
            return torch.randn(*shape, device=self.device, generator=gen).to(self.dtype)

        full = [(rand(n, G.nk, G.hk), rand(n, G.nk, G.hk), rand(n, G.nv, G.hv), rand(n, G.nv), rand(n, G.nv))
                for n in roots]                                # root r's tokens, then each child's own
        tail = [(rand(n, G.nk, G.hk), rand(n, G.nk, G.hk), rand(n, G.nv, G.hv), rand(n, G.nv), rand(n, G.nv))
                for n in kids]
        row = [[torch.cat([x, y]) for x, y in zip(full[p], t)] for p, t in zip(parent, tail)]
        with torch.no_grad():
            q, k, v, a, b = (torch.cat(xs) for xs in zip(*row))
            decay = -G.A_log.float().exp() * F.softplus(a.float() + G.dt_bias.float())
            ref = _gdn_reference(q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1), v, decay,
                                 torch.sigmoid(b.float()), rows).split(rows)
            want = torch.cat([ref[parent.index(r)][:n] for r, n in enumerate(roots)] +
                             [ref[i][roots[p]:] for i, p in enumerate(parent)])

            def run(parts, lengths, state):
                q, k, v, a, b = (torch.cat(xs)[None] for xs in zip(*parts))
                if self.gdn.expand_gva:
                    q, k = q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2)
                cu_cpu, _ = packed_layout_host(lengths)
                return self.ops.gdn_resume(q, k, v, a, b, G.A_log, G.dt_bias, state, cu_cpu.to(self.device), cu_cpu)

            try:
                o_roots, final = run(full, roots, None)
                o_kids, _ = run(tail, kids, final[parent])
            except Exception as exc:  # noqa: BLE001
                return f"resuming the GatedDeltaNet failed: {type(exc).__name__}: {str(exc)[:160]}"
            err = (torch.cat([o_roots[0], o_kids[0]]).float() - want).abs().max().item() / want.abs().max().item()
        self.share_err = err
        return None if err < GDN_PROBE_TOLERANCE else f"resumed GatedDeltaNet relative error {err:.2e}"

    def _rope(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        probe = torch.empty(1, device=self.device, dtype=self.dtype)
        # Text uses the same positions on all three multimodal RoPE axes.
        # Newer Transformers requires these axes explicitly; older versions
        # accept them too (and expanded a 2D input internally).
        cos, sin = self.tm.rotary_emb(probe, pos[None, None].expand(3, 1, -1))
        return cos[0], sin[0]

    # ------------------------------------------------------------------ layers
    def _gdn(self, L: _Layer, h: Tensor, lay: _Layout) -> Tensor:
        if not self.fused:
            return self._gdn_unfused(L, h, lay)
        B, S = lay.shape
        proj = self.ops.linear(h, L.in_proj)
        # one launch: q|k|v conv + SiLU, and b|a copied out contiguous (fla would copy each)
        q, k, v, b, a = self.ops.conv_split(proj, L.conv_w, L.conv_b, lay.pos, L.kd, L.vd, L.gate_off, L.nv, L.nv, True)
        lay.execution.continue_conv(self, L, proj, (q, k, v), lay)
        q, k, v = q.view(B, S, L.nk, L.hk), k.view(B, S, L.nk, L.hk), v.view(B, S, L.nv, L.hv)
        if self.gdn.expand_gva:
            q, k = q.repeat_interleave(L.nv // L.nk, dim=2), k.repeat_interleave(L.nv // L.nk, dim=2)
        o = self._delta_rule(L, q, k, v, a.view(B, S, L.nv), b.view(B, S, L.nv), lay)
        z = proj[:, L.bounds[-1]: L.bounds[-1] + L.nv * L.hv].view(-1, L.nv, L.hv)     # read in place
        return self.ops.linear(self.ops.gated_rms_norm(o.view(-1, L.nv, L.hv), z, L.gn_w, L.gn_eps), L.out)

    def _gdn_unfused(self, L: _Layer, h: Tensor, lay: _Layout) -> Tensor:
        B, S = lay.shape
        qkv, z, b, a = self.ops.linear(h, L.in_proj).split(L.split_in, dim=-1)
        qkv = qkv.view(B, S, -1)
        q, k, v = (causal_conv1d(x=qkv[..., i:j], weight=w, bias=bias, activation=L.act, cu_seqlens=lay.cu,
                                 cu_seqlens_cpu=lay.cu_cpu)[0]
                   for (w, bias), i, j in zip(L.conv, L.bounds, L.bounds[1:]))
        lay.execution.continue_conv(self, L, qkv.view(-1, qkv.shape[-1]), (q[0], k[0], v[0]), lay)
        q, k, v = q.reshape(B, S, L.nk, L.hk), k.reshape(B, S, L.nk, L.hk), v.reshape(B, S, L.nv, L.hv)
        if self.gdn.expand_gva:                       # this fla lacks grouped value heads: expand as the model does
            q, k = q.repeat_interleave(L.nv // L.nk, dim=2), k.repeat_interleave(L.nv // L.nk, dim=2)
        o = self._delta_rule(L, q, k, v, a.reshape(B, S, L.nv), b.reshape(B, S, L.nv), lay)
        o = L.gnorm(o.reshape(-1, L.hv), z.reshape(-1, L.hv))
        return self.ops.linear(o.reshape(h.shape[0], -1), L.out)

    def _delta_rule(self, L: _Layer, q, k, v, a, b, lay: _Layout) -> Tensor:
        return lay.execution.delta_rule(self, L, q, k, v, a, b, lay)

    def _attn(self, L: _Layer, h: Tensor, lay: _Layout) -> Tensor:
        N, nq, nkv, hd = h.shape[0], L.nq, L.nkv, L.hd
        out = self.ops.linear(h, L.qkv)
        qw = nq * (2 * hd if L.gated else hd)
        if L.gated:
            qg = out[:, :qw].view(N, nq, 2 * hd)
            q, gate = qg[..., :hd], qg[..., hd:]
        else:
            q, gate = out[:, :qw].view(N, nq, hd), None
        k = out[:, qw: qw + nkv * hd].view(N, nkv, hd)
        v = out[:, qw + nkv * hd:].view(N, nkv, hd)
        if self.fused:
            q, k = self.ops.qk_norm_rope_pair(out, 0, q.stride(1), nq, qw, hd, nkv, L.wqk, lay.cos, lay.sin, L.qk_eps)
        else:
            q, k = self.ops.qk_norm_rope(q, L.wq, lay.cos, lay.sin, L.qk_eps), self.ops.qk_norm_rope(k, L.wk, lay.cos, lay.sin, L.qk_eps)
        o = lay.execution.attention(self, q, k, v, lay)
        if gate is None:
            return self.ops.linear(o.reshape(N, nq * hd), L.o)
        if self.fused:
            return self.ops.linear(self.ops.sigmoid_gate(o.view(N, nq, hd), gate), L.o)
        return self.ops.linear((o * torch.sigmoid(gate)).reshape(N, nq * hd), L.o)

    def _trunk(self, x: Tensor, lay: _Layout) -> Tensor:
        residual, delta, eps = x, None, self.eps
        for index, L in enumerate(self.layers):
            lay.layer_index = index
            if delta is None:
                h = self.ops.rms_norm(residual, L.w_in, eps)
            else:
                h, residual = self.ops.add_rms_norm(delta, residual, L.w_in, eps)
            mix = self._gdn(L, h, lay) if L.linear else self._attn(L, h, lay)
            h, residual = self.ops.add_rms_norm(mix, residual, L.w_post, eps)
            g, u = self.ops.linear(h, L.gate_up).split(L.inter, dim=-1)
            delta = self.ops.linear(self.ops.swiglu(g, u), L.down)
        out, _ = self.ops.add_rms_norm(delta, residual, self.w_final, eps)
        return out

    # ------------------------------------------------------------------ entry points
    @torch.no_grad()
    def forward_packed(self, ids: Tensor, lengths: Sequence[int]) -> Tensor:
        """Eager, padding-free. ids: (T,) device ids back to back; lengths: host ints (all > 0).
        Returns the final-normed hidden states (T, hidden)."""
        with torch.cuda.device(self.device):
            self.sync_norms()
            cu, pos = packed_layout_host(lengths)
            d_pos, d_cu = self._stager.put([pos, cu])
            x = F.embedding(ids.reshape(-1), self.embed)
            cos, sin = self._rope(d_pos)
            lay = _Layout(shape=(1, x.shape[0]), cos=cos, sin=sin, cu=d_cu, cu_cpu=cu, cu32=d_cu.to(torch.int32),
                          max_len=int(max(lengths)), lengths=list(lengths), pos=d_pos)
            with fla_tensor_cache():
                return self._trunk(x, lay)

    def plan_sharing(self, ids: Tensor, lengths: Sequence[int]) -> SharedPlan | None:
        """A plan for `forward_shared` when rows share at least `min_shared_prefix` leading tokens."""
        if not self.min_shared_prefix:
            return None
        return plan_shared_prefixes(ids.reshape(-1), lengths, min_prefix=self.min_shared_prefix,
                                    conv_width=self.conv_width)

    def prepare_prefix_layout(self, lengths, prefix, *, write=False):
        """Build host metadata and stable device buffers outside a captured forward."""
        cu, pos = packed_layout_host(lengths)
        d_cu, d_pos = self._stager.put([cu, pos])
        cos, sin = self._rope(d_pos if write else d_pos + prefix.num_tokens)
        meta = None
        if not write:
            width, P = self.conv_width, prefix.num_tokens
            fix_rows, fix_taps, kv = [], [], []
            for start, n in zip(cu[:-1].tolist(), lengths):
                kv.extend([torch.arange(P), torch.arange(P + start, P + start + n)])
                for i in range(min(width - 1, n)):
                    fix_rows.append(start + i)
                    fix_taps.append([start + i - lag if i >= lag else i - lag
                                     for lag in range(width - 1, -1, -1)])
            kv_lengths = [P + n for n in lengths]
            cu_k, _ = packed_layout_host(kv_lengths)
            rows, taps, idx, d_ck = self._stager.put([
                torch.tensor(fix_rows, dtype=torch.long), torch.tensor(fix_taps, dtype=torch.long).view(-1, width),
                torch.cat(kv), cu_k.to(torch.int32)])
            meta = SimpleNamespace(fix_rows=rows, fix_taps=taps.view(-1, width), kv_idx=idx,
                                   cu_k32=d_ck.to(torch.int32), kv_lengths=kv_lengths)
        lay = _PrefixLayout(shape=(1, sum(lengths)), cos=cos, sin=sin, cu=d_cu, cu_cpu=cu,
                      cu32=d_cu.to(torch.int32), max_len=max(lengths), lengths=lengths, pos=d_pos,
                      prefix=prefix, prefix_write=write, prefix_meta=meta)
        return lay

    def prefix_core(self, ids, lay):
        with fla_tensor_cache():
            return self._trunk(F.embedding(ids, self.embed), lay)

    @torch.no_grad()
    def forward_prefix(self, ids, lengths, prefix, *, write=False):
        """Build a persistent prefix or run packed continuations from its layer states."""
        with torch.cuda.device(self.device):
            self.sync_norms()
            lay = self.prepare_prefix_layout(lengths, prefix, write=write)
            return self.prefix_core(ids, lay)

    def prepare_shared_layout(self, plan):
        """Stage one forest outside capture; host lengths determine kernel launch geometry."""
        n, R = plan.n_roots, plan.root_tokens
        cu, _ = packed_layout_host(plan.lengths)
        cu_k, _ = packed_layout_host(plan.kv_lengths)
        cu_roots, cu_kids = cu[: n + 1], cu[n:] - R
        src, out, rope, pos, d_cu, d_cu_k, d_roots, d_kids, parent, fix_rows, fix_taps, kv_idx = self._stager.put(
            [plan.src, plan.out, plan.rope_pos, plan.conv_pos, cu, cu_k, cu_roots, cu_kids, plan.parent,
             plan.fix_rows, plan.fix_taps, plan.kv_idx])
        cos, sin = self._rope(rope)
        share = _Shared(root_tokens=R, cu_roots=d_roots, cu_roots_cpu=cu_roots, cu_kids=d_kids, cu_kids_cpu=cu_kids,
                        parent=parent, fix_rows=fix_rows, fix_taps=fix_taps.view(plan.fix_taps.shape), kv_idx=kv_idx,
                        cu_k32=d_cu_k.to(torch.int32), max_k=max(plan.kv_lengths), kv_lengths=plan.kv_lengths)
        lay = _PrefixLayout(shape=(1, plan.src.numel()), cos=cos, sin=sin, cu=d_cu, cu_cpu=cu, cu32=d_cu.to(torch.int32),
                      max_len=max(plan.lengths), lengths=plan.lengths, pos=pos, share=share)
        return SimpleNamespace(src=src, out=out, layout=lay)

    def shared_core(self, ids, static):
        with fla_tensor_cache():
            x = F.embedding(ids.reshape(-1).index_select(0, static.src), self.embed)
            return self._trunk(x, static.layout).index_select(0, static.out)

    @torch.no_grad()
    def forward_shared(self, ids: Tensor, plan: SharedPlan) -> Tensor:
        """Run each shared prefix once, returning hidden states in the caller's layout."""
        with torch.cuda.device(self.device):
            self.sync_norms()
            return self.shared_core(ids, self.prepare_shared_layout(plan))

    def _use_recurrent(self, lay: _Layout) -> bool:
        return lay.static is None and lay.max_len <= self.gdn.recurrent_max_len     # eager only; see RECURRENT_MAX_LEN

    def prepare_static(self, static: PaddedStatic) -> None:
        pos = torch.arange(static.seq, device=self.device).repeat(static.rows)
        static.extras["cos"], static.extras["sin"] = self._rope(pos)
        static.extras["pos"] = pos

    def padded_core(self, static: PaddedStatic) -> Tensor:
        x = F.embedding(static.ids.view(-1), self.embed)
        lay = _Layout(shape=(static.rows, static.seq), cos=static.extras["cos"], sin=static.extras["sin"],
                      max_len=static.seq, static=static, pos=static.extras["pos"])
        return self._trunk(x, lay)
