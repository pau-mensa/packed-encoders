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

Two entry points share the layer code: `forward_packed` (eager, padding-free, any batch)
and `padded_core` (the CUDA-graph region, right-padded `(rows, S)`; exact because the
mixers are causal and attention sees `[real | pad]` segments — see runtime.graphs).
"""

from __future__ import annotations

import contextlib
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


def _require_plain(linears: Sequence[nn.Module]) -> None:
    """The engine reads `.weight` and nothing else. A peft tuner layer exposes its *base* weight
    there (BaseTunerLayer.weight), so an unmerged adapter would be silently dropped; refuse it,
    and a bias, which would be dropped too."""
    for lin in linears:
        if hasattr(lin, "lora_A") or hasattr(lin, "base_layer") or getattr(lin, "bias", None) is not None:
            raise UnsupportedTargetError(
                f"{type(lin).__name__} carries a bias or an unmerged LoRA adapter; merge adapters "
                "first (peft: model = model.merge_and_unload()) — packing reads plain dense weights"
            )


def _share_rows(linears: Sequence[nn.Linear], registry: list[nn.Linear]) -> Tensor:
    """Concatenate the weights row-wise and re-point every parameter at its slice of the
    result, so the merged GEMM weight and the HF parameters are one storage. Re-pointed
    layers are appended to `registry` for `unshare_rows`."""
    _require_plain(linears)
    merged = torch.cat([lin.weight.detach() for lin in linears], 0)
    off = 0
    for lin in linears:
        n = lin.weight.shape[0]
        lin.weight.data = merged[off:off + n]
        registry.append(lin)
        off += n
    return merged


def unshare_rows(registry: list[nn.Linear], *, rollback: bool = False) -> None:
    """Restore independent parameter storage, retaining live values and identities.

    Shared source parameters are rejected before preparation. Restoring independent
    storage therefore restores the accepted aliasing contract too. Addresses can change;
    callers must not retain pre-pack tensor views or external graphs across pack/unpack.
    """
    for lin in registry:
        lin.weight.data = lin.weight.data.clone()
    registry.clear()


def _require_independent_parameters(model: nn.Module) -> None:
    """Reject aliasing before any re-pointing; merged projections must be independent."""
    seen = set()
    for name, param in model.named_parameters(remove_duplicate=False):
        key = (param.device, param.untyped_storage().data_ptr())
        if key in seen:
            raise UnsupportedTargetError(f"Qwen3.5 requires independent parameter storage; {name} is tied or aliased")
        seen.add(key)


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
        try:
            # Triton launches and CUDA graphs use the *current* device, not the tensors': pin it to the
            # weights' for every launch (here, `forward_packed`, the graph runner, the stager).
            with torch.cuda.device(self.device):
                self.layers = [self._prepare(layer, kind) for layer, kind in zip(text_model.layers[:n], kinds[:n])]
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
                self._stager = PinnedStager(self.device)
        except BaseException:
            unshare_rows(self.shared, rollback=True)            # leave the model exactly as we found it
            raise

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

    def _rope(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        probe = torch.empty(1, device=self.device, dtype=self.dtype)
        cos, sin = self.tm.rotary_emb(probe, pos[None])
        return cos[0], sin[0]

    # ------------------------------------------------------------------ layers
    def _gdn(self, L: _Layer, h: Tensor, lay: _Layout) -> Tensor:
        if not self.fused:
            return self._gdn_unfused(L, h, lay)
        B, S = lay.shape
        proj = self.ops.linear(h, L.in_proj)
        # one launch: q|k|v conv + SiLU, and b|a copied out contiguous (fla would copy each)
        q, k, v, b, a = self.ops.conv_split(proj, L.conv_w, L.conv_b, lay.pos, L.kd, L.vd, L.gate_off, L.nv, L.nv, True)
        q, k, v = q.view(B, S, L.nk, L.hk), k.view(B, S, L.nk, L.hk), v.view(B, S, L.nv, L.hv)
        if self.gdn.expand_gva:
            q, k = q.repeat_interleave(L.nv // L.nk, dim=2), k.repeat_interleave(L.nv // L.nk, dim=2)
        kernel = self.ops.gdn_recurrent if self._use_recurrent(lay) else self.ops.gdn_chunk
        o = kernel(q, k, v, a.view(B, S, L.nv), b.view(B, S, L.nv), L.A_log, L.dt_bias, lay.cu, lay.cu_cpu)
        z = proj[:, L.bounds[-1]: L.bounds[-1] + L.nv * L.hv].view(-1, L.nv, L.hv)     # read in place
        return self.ops.linear(self.ops.gated_rms_norm(o.view(-1, L.nv, L.hv), z, L.gn_w, L.gn_eps), L.out)

    def _gdn_unfused(self, L: _Layer, h: Tensor, lay: _Layout) -> Tensor:
        B, S = lay.shape
        qkv, z, b, a = self.ops.linear(h, L.in_proj).split(L.split_in, dim=-1)
        qkv = qkv.view(B, S, -1)
        q, k, v = (causal_conv1d(x=qkv[..., i:j], weight=w, bias=bias, activation=L.act, cu_seqlens=lay.cu,
                                 cu_seqlens_cpu=lay.cu_cpu)[0]
                   for (w, bias), i, j in zip(L.conv, L.bounds, L.bounds[1:]))
        q, k, v = q.reshape(B, S, L.nk, L.hk), k.reshape(B, S, L.nk, L.hk), v.reshape(B, S, L.nv, L.hv)
        if self.gdn.expand_gva:                       # this fla lacks grouped value heads: expand as the model does
            q, k = q.repeat_interleave(L.nv // L.nk, dim=2), k.repeat_interleave(L.nv // L.nk, dim=2)
        kernel = self.ops.gdn_recurrent if self._use_recurrent(lay) else self.ops.gdn_chunk
        o = kernel(q, k, v, a.reshape(B, S, L.nv), b.reshape(B, S, L.nv), L.A_log, L.dt_bias, lay.cu, lay.cu_cpu)
        o = L.gnorm(o.reshape(-1, L.hv), z.reshape(-1, L.hv))
        return self.ops.linear(o.reshape(h.shape[0], -1), L.out)

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
        if lay.static is None:
            o = self.ops.attention_packed(self.attention, q, k, v, lay.cu32, lay.max_len, lay.lengths, self.causal)
        else:
            o = self.ops.attention_padded(self.attention, q, k, v, lay.static, self.causal)
        if gate is None:
            return self.ops.linear(o.reshape(N, nq * hd), L.o)
        if self.fused:
            return self.ops.linear(self.ops.sigmoid_gate(o.view(N, nq, hd), gate), L.o)
        return self.ops.linear((o * torch.sigmoid(gate)).reshape(N, nq * hd), L.o)

    def _trunk(self, x: Tensor, lay: _Layout) -> Tensor:
        residual, delta, eps = x, None, self.eps
        for L in self.layers:
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
