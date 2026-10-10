"""Exact fast forward for an LFM2 text backbone (gated short convolutions + GQA softmax attention).

Same math as the model, fewer and fatter kernels:

- merged projections: attention q|k|v and MLP w1|w3, one GEMM each. As in the Qwen3.5 engine the
  merged weight *is* the HF weight (each Linear is re-pointed at its rows; `release()` separates them).
- the gated short conv `C * conv(B * x)` in one Triton launch reading the in-projection in place,
  positions restarting per sequence (`pieces.hybrid.SHORT_CONV`), so packed rows never mix.
- q/k RMSNorm + RoPE in one launch reading the q|k|v projection in place; varlen attention chosen
  by probe (runtime.attention); fla residual-add + RMSNorm and a Triton SwiGLU.

LFM2's RMSNorm scales by `weight` (no `1 +`); the fused norms take fp32 copies of the weights,
tracked by `sync_norms` like the Qwen3.5 engine's.

Three entry points share the layer code: `forward_packed` (eager, padding-free, token ids or
input embeddings), `padded_core` (the CUDA-graph region, right-padded `(rows, S)`; exact
because the conv and the attention are causal — see runtime.graphs), and `forward_shared`
(opt in: rows that start with the same tokens run that prefix once — see runtime.sharing;
`prepare_shared_layout` + `shared_core` are its graph-capturable halves, runtime.shared_graphs).
With no recurrent state to resume, the whole forest is one pass: a child's first conv taps are
recomputed over its root's last tokens, and its attention reads the root's keys and values
through the prefixed varlen kernel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from packed_encoders.arch.weights import require_independent_parameters, require_plain, share_rows, unshare_rows
from packed_encoders.errors import UnsupportedTargetError, ValidationError
from packed_encoders.runtime.attention import AttentionChoice, select_attention
from packed_encoders.runtime.graphs import PaddedStatic
from packed_encoders.runtime.sharing import SharedPlan, plan_shared_prefixes
from packed_encoders.runtime.staging import PinnedStager, packed_layout_host

KERNEL_TOLERANCE = 2e-2        # max |fused - model's module|, relative to the module's max |output| (bf16: a few ulp)
# Shared prefixes are opt in (`min_shared_prefix` starts at 0). validate() checks the shared pass with
# prefixes of at least this many tokens when the caller hasn't set a threshold.
SHARED_PREFIX_MIN = 64


class _Layer:
    pass


@dataclass
class _Layout:
    pos: Tensor                        # (T,) position of each token in its segment (the conv restarts there)
    cos: Tensor | None = None          # RoPE tables at each token's position in its row
    sin: Tensor | None = None
    cu32: Tensor | None = None         # packed: int32 cu_seqlens (attention)
    max_len: int = 0
    lengths: list[int] | None = None
    static: PaddedStatic | None = None
    share: _Shared | None = None       # forward_shared: segments are roots, then children


@dataclass
class _Shared:
    """Device side of a SharedPlan (runtime.sharing)."""

    fix_rows: Tensor                   # child tokens whose conv taps reach back into the root
    fix_taps: Tensor                   # [F, K] rows under those taps, oldest first
    kv_idx: Tensor                     # rows forming each segment's keys and values
    cu_k32: Tensor
    max_k: int
    kv_lengths: list[int]


class Lfm2Engine:
    def __init__(self, text_model: nn.Module, *, causal: bool = True, attention_order: Sequence[str] | None = None,
                 pieces=None):
        from packed_encoders.arch.lfm2.pieces import default_pieces

        if not causal:
            # A bidirectional LFM2 also changes the convolution (centred taps), not just the mask.
            raise UnsupportedTargetError("the LFM2 engine runs causal backbones only")
        self.composition = pieces or default_pieces()
        self.ops = self.composition.bind()
        require_independent_parameters(text_model, "LFM2")
        cfg = text_model.config
        self.tm, self.cfg, self.causal = text_model, cfg, causal
        self.embed = text_model.embed_tokens.weight
        self.device, self.dtype = self.embed.device, self.embed.dtype
        if self.device.type != "cuda":
            raise UnsupportedTargetError(f"the LFM2 engine needs CUDA weights; model is on {self.device}")
        self.hidden_size = cfg.hidden_size
        self.eps = cfg.norm_eps
        self._norms: list[list] = []            # [parameter (detached), fp32 copy, version seen]; see sync_norms
        self.shared: list[nn.Linear] = []
        self.layers: list[_Layer] = []
        n = cfg.num_hidden_layers
        kinds = list(getattr(cfg, "layer_types", None) or [])
        if len(kinds) < n:
            raise UnsupportedTargetError("config.layer_types is missing or shorter than num_hidden_layers")
        for layer, kind in zip(text_model.layers[:n], kinds[:n]):
            _require_supported(layer, kind)          # every layer before any re-pointing
        try:
            with torch.cuda.device(self.device):
                self.w_final = self._scale(text_model.embedding_norm.weight)
                for layer, kind in zip(text_model.layers[:n], kinds[:n]):
                    self.layers.append(self._prepare(layer, kind))
                attn = next((L for L in self.layers if L.attn), None)
                self.attention: AttentionChoice | None = None
                self.kernel_errors: dict[str, float] = {}
                if attn is not None:
                    self.attention = select_attention(n_heads=attn.nq, n_kv_heads=attn.nkv, head_dim=attn.hd,
                                                      causal=causal, device=self.device, dtype=self.dtype,
                                                      order=attention_order)
                    self.kernel_errors["qk_norm_rope"] = self._check_qk(attn)
                conv = next((L for L in self.layers if not L.attn), None)
                if conv is not None:
                    self.kernel_errors["short_conv"] = self._check_conv(conv)
                self.conv_width = conv.conv_w.shape[1] if conv is not None else 1
                self.min_shared_prefix = 0                          # off until the caller opts in
                self.share_rejected: str | None = None              # None: forward_shared can run here
                if attn is not None and self.attention.prefixed is None:
                    self.share_rejected = "no prefixed attention kernel"
                self._stager = PinnedStager(self.device)
        except BaseException:
            self.release(rollback=True)                          # leave the model exactly as we found it
            raise

    def release(self, *, rollback: bool = False) -> None:
        """Drop the merged weights, then give every projection its own storage back."""
        self.layers.clear()
        self._norms.clear()
        self.w_final = None
        unshare_rows(self.shared, rollback=rollback)

    # ------------------------------------------------------------------ weights
    def _prepare(self, layer: nn.Module, kind: str) -> _Layer:
        L = _Layer()
        L.attn = kind == "full_attention"
        L.w_op = self._scale(layer.operator_norm.weight)
        L.w_ffn = self._scale(layer.ffn_norm.weight)
        mlp = layer.feed_forward
        L.w13 = share_rows([mlp.w1, mlp.w3], self.shared)
        L.inter = mlp.w1.weight.shape[0]
        L.w2 = mlp.w2.weight
        if L.attn:
            sa = layer.self_attn
            L.module = sa
            L.hd = sa.head_dim
            L.nq, L.nkv = sa.q_proj.weight.shape[0] // L.hd, sa.k_proj.weight.shape[0] // L.hd
            L.qkv = share_rows([sa.q_proj, sa.k_proj, sa.v_proj], self.shared)
            L.wqk = torch.empty((2, L.hd), device=self.device, dtype=torch.float32)   # q row, k row
            self._scale(sa.q_layernorm.weight, out=L.wqk[0])
            self._scale(sa.k_layernorm.weight, out=L.wqk[1])
            L.qk_eps = sa.q_layernorm.variance_epsilon
            L.out = sa.out_proj.weight
        else:
            sc = layer.conv
            L.module = sc
            L.in_proj, L.out = sc.in_proj.weight, sc.out_proj.weight
            L.conv_w = sc.conv.weight.detach().squeeze(1)          # (D, K) taps, oldest first; the parameter's storage
            if not L.conv_w.is_contiguous():
                raise UnsupportedTargetError("LFM2 convolution weights must be contiguous")
        return L

    def _scale(self, weight: Tensor, out: Tensor | None = None) -> Tensor:
        """fp32 copy of a norm weight for the fused norm kernels; tracked (sync_norms)."""
        src = weight.detach()                    # shares the parameter's storage and version counter
        if out is None:
            out = torch.empty(src.shape, device=src.device, dtype=torch.float32)
        out.copy_(src)
        self._norms.append([src, out, src._version])
        return out

    def sync_norms(self) -> None:
        """Re-copy every norm weight whose parameter changed since it was copied (an optimizer step,
        `load_state_dict`). In place, so captured graphs stay valid. Writes through `.data` bypass the
        version counter: re-pack after those."""
        for entry in self._norms:
            src, out, seen = entry
            if src._version != seen:
                out.copy_(src)
                entry[2] = src._version

    def _check_qk(self, L: _Layer) -> float:
        """The fused q/k norm + RoPE against the model's own RMSNorms + apply_rotary_pos_emb."""
        from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb

        g = torch.Generator(device=self.device).manual_seed(0)
        T = 37
        qw = L.nq * L.hd
        proj = torch.randn(T, qw + 2 * L.nkv * L.hd, device=self.device, dtype=self.dtype, generator=g)
        cos, sin = self._rope(torch.arange(T, device=self.device) * 7)
        with torch.no_grad():
            qx, kx = proj[:, :qw].view(T, L.nq, L.hd), proj[:, qw: qw + L.nkv * L.hd].view(T, L.nkv, L.hd)
            rq, rk = apply_rotary_pos_emb(L.module.q_layernorm(qx), L.module.k_layernorm(kx), cos, sin, unsqueeze_dim=1)
            pq, pk = self.ops.qk_norm_rope_pair(proj, 0, L.hd, L.nq, qw, L.hd, L.nkv, L.wqk, cos, sin, L.qk_eps)
            err = max(_rel(pq, rq), _rel(pk, rk))
        if not err < KERNEL_TOLERANCE:
            raise ValidationError(f"fused q/k norm + RoPE mismatch vs the model's modules: relative error {err:.2e}")
        return err

    def _check_conv(self, L: _Layer) -> float:
        """The packed short conv against the model's own module, run per sequence."""
        g = torch.Generator(device=self.device).manual_seed(0)
        lengths = [5, 20, 1, 12]
        h = torch.randn(sum(lengths), self.hidden_size, device=self.device, dtype=self.dtype, generator=g)
        _, pos = packed_layout_host(lengths)
        with torch.no_grad():
            got = self._conv(L, h, _Layout(pos=pos.to(self.device)))
            ref = torch.cat([L.module(x[None])[0] for x in h.split(lengths)])
        err = _rel(got, ref)
        if not err < KERNEL_TOLERANCE:
            raise ValidationError(f"packed short convolution mismatch vs the model's module: relative error {err:.2e}")
        return err

    def _rope(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        probe = torch.empty(1, device=self.device, dtype=self.dtype)
        cos, sin = self.tm.rotary_emb(probe, pos[None])
        return cos[0], sin[0]

    # ------------------------------------------------------------------ layers
    def _conv(self, L: _Layer, h: Tensor, lay: _Layout) -> Tensor:
        proj = self.ops.linear(h, L.in_proj)
        y = self.ops.short_conv(proj, L.conv_w, lay.pos)
        if lay.share is not None:
            self._continue_conv(L, proj, y, lay.share)
        return self.ops.linear(y, L.out)

    def _continue_conv(self, L: _Layer, proj: Tensor, y: Tensor, sh: _Shared) -> None:
        """The packed conv restarts at each segment; a child's first K-1 tokens redo theirs over
        their root's last tokens, rounding where the kernel does (B * x, the conv, the gate), in place."""
        D = L.conv_w.shape[0]
        taps = proj.index_select(0, sh.fix_taps.view(-1)).view(*sh.fix_taps.shape, -1)
        bx = (taps[..., :D].float() * taps[..., 2 * D: 3 * D].float()).to(y.dtype).float()
        conv = torch.einsum("fkc,ck->fc", bx, L.conv_w.float()).to(y.dtype).float()
        c = proj.index_select(0, sh.fix_rows)[:, D: 2 * D].float()
        y.index_copy_(0, sh.fix_rows, (c * conv).to(y.dtype))

    def _attn(self, L: _Layer, h: Tensor, lay: _Layout) -> Tensor:
        N, nq, nkv, hd = h.shape[0], L.nq, L.nkv, L.hd
        out = self.ops.linear(h, L.qkv)
        qw = nq * hd
        q, k = self.ops.qk_norm_rope_pair(out, 0, hd, nq, qw, hd, nkv, L.wqk, lay.cos, lay.sin, L.qk_eps)
        v = out[:, qw + nkv * hd:].view(N, nkv, hd)
        if lay.share is not None:          # each child's keys: its root's prefix, then its own
            sh = lay.share
            o = self.ops.attention_prefixed(self.attention, q, k.index_select(0, sh.kv_idx), v.index_select(0, sh.kv_idx),
                                            lay.cu32, sh.cu_k32, lay.max_len, sh.max_k, lay.lengths, sh.kv_lengths,
                                            self.causal)
        elif lay.static is None:
            o = self.ops.attention_packed(self.attention, q, k, v, lay.cu32, lay.max_len, lay.lengths, self.causal)
        else:
            o = self.ops.attention_padded(self.attention, q, k, v, lay.static, self.causal)
        return self.ops.linear(o.reshape(N, nq * hd), L.out)

    def _trunk(self, x: Tensor, lay: _Layout) -> Tensor:
        residual, delta, eps = x, None, self.eps
        for L in self.layers:
            if delta is None:
                h = self.ops.rms_norm(residual, L.w_op, eps)
            else:
                h, residual = self.ops.add_rms_norm(delta, residual, L.w_op, eps)
            mix = self._attn(L, h, lay) if L.attn else self._conv(L, h, lay)
            h, residual = self.ops.add_rms_norm(mix, residual, L.w_ffn, eps)
            g, u = self.ops.linear(h, L.w13).split(L.inter, dim=-1)
            delta = self.ops.linear(self.ops.swiglu(g, u), L.w2)
        out, _ = self.ops.add_rms_norm(delta, residual, self.w_final, eps)
        return out

    # ------------------------------------------------------------------ entry points
    @torch.no_grad()
    def forward_packed(self, ids: Tensor | None, lengths: Sequence[int], *, embeds: Tensor | None = None) -> Tensor:
        """Eager, padding-free. ids: (T,) device ids back to back, or embeds: (T, hidden) input
        embeddings in their place; lengths: host ints (all > 0). Returns the final-normed hidden
        states (T, hidden)."""
        with torch.cuda.device(self.device):
            self.sync_norms()
            cu, pos = packed_layout_host(lengths)
            d_pos, d_cu = self._stager.put([pos, cu])
            x = F.embedding(ids.reshape(-1), self.embed) if embeds is None else embeds.to(self.dtype)
            cos, sin = self._rope(d_pos)
            lay = _Layout(cos=cos, sin=sin, pos=d_pos, cu32=d_cu.to(torch.int32), max_len=int(max(lengths)),
                          lengths=list(lengths))
            return self._trunk(x, lay)

    def plan_sharing(self, ids: Tensor | None, lengths: Sequence[int], *, embeds: Tensor | None = None) -> SharedPlan | None:
        """A plan for `forward_shared` when rows share at least `min_shared_prefix` leading tokens.
        Input embeddings are matched row for row: equal rows are the same token (or the same image
        feature), so a wrapper that merges images into the embeddings shares those too."""
        if not self.min_shared_prefix or len(lengths) < 2:
            return None
        if ids is None:
            ids = _row_hashes(embeds)
        plan = plan_shared_prefixes(ids.reshape(-1), lengths, min_prefix=self.min_shared_prefix,
                                    conv_width=self.conv_width)
        if plan is not None and embeds is not None:
            # Hashes can collide: every token must take the forward row of an identical embedding.
            d_src, d_out = self._stager.put([plan.src, plan.out])
            if not torch.equal(embeds.index_select(0, d_src.index_select(0, d_out)), embeds):
                return None
        return plan

    def prepare_shared_layout(self, plan: SharedPlan) -> SimpleNamespace:
        """Stage one forest outside capture (runtime.shared_graphs); host lengths fix the launch geometry."""
        cu, _ = packed_layout_host(plan.lengths)
        cu_k, _ = packed_layout_host(plan.kv_lengths)
        src, out, rope, pos, d_cu, d_cu_k, fix_rows, fix_taps, kv_idx = self._stager.put(
            [plan.src, plan.out, plan.rope_pos, plan.conv_pos, cu, cu_k, plan.fix_rows, plan.fix_taps, plan.kv_idx])
        cos, sin = self._rope(rope)
        share = _Shared(fix_rows=fix_rows, fix_taps=fix_taps.view(plan.fix_taps.shape), kv_idx=kv_idx,
                        cu_k32=d_cu_k.to(torch.int32), max_k=max(plan.kv_lengths), kv_lengths=plan.kv_lengths)
        lay = _Layout(cos=cos, sin=sin, pos=pos, cu32=d_cu.to(torch.int32), max_len=max(plan.lengths),
                      lengths=plan.lengths, share=share)
        return SimpleNamespace(src=src, out=out, layout=lay)

    def shared_core(self, ids: Tensor | None, static: SimpleNamespace, *, embeds: Tensor | None = None) -> Tensor:
        if embeds is None:
            x = F.embedding(ids.reshape(-1).index_select(0, static.src), self.embed)
        else:
            x = embeds.to(self.dtype).index_select(0, static.src)
        return self._trunk(x, static.layout).index_select(0, static.out)

    @torch.no_grad()
    def forward_shared(self, ids: Tensor | None, plan: SharedPlan, *, embeds: Tensor | None = None) -> Tensor:
        """Run each shared prefix once; hidden states (T, hidden) in the caller's layout."""
        with torch.cuda.device(self.device):
            self.sync_norms()
            return self.shared_core(ids, self.prepare_shared_layout(plan), embeds=embeds)

    def prepare_static(self, static: PaddedStatic) -> None:
        pos = torch.arange(static.seq, device=self.device).repeat(static.rows)
        static.extras["cos"], static.extras["sin"] = self._rope(pos)
        static.extras["pos"] = pos

    def padded_core(self, static: PaddedStatic) -> Tensor:
        x = F.embedding(static.ids.view(-1), self.embed)
        lay = _Layout(cos=static.extras["cos"], sin=static.extras["sin"], pos=static.extras["pos"],
                      max_len=static.seq, static=static)
        return self._trunk(x, lay)


def _require_supported(layer: nn.Module, kind: str) -> None:
    """Everything `_prepare` reads must be a plain dense weight; checked for all layers first, so a
    refusal leaves every parameter's storage untouched."""
    if kind not in ("conv", "full_attention"):
        raise UnsupportedTargetError(f"unsupported LFM2 layer type {kind!r}")
    mlp = layer.feed_forward
    if not all(hasattr(mlp, a) for a in ("w1", "w2", "w3")):
        raise UnsupportedTargetError("only the dense SwiGLU MLP is supported (no MoE layers)")
    linears = [mlp.w1, mlp.w2, mlp.w3]
    if kind == "full_attention":
        sa = layer.self_attn
        linears += [sa.q_proj, sa.k_proj, sa.v_proj, sa.out_proj]
    else:
        if layer.conv.conv.bias is not None:
            raise UnsupportedTargetError("LFM2 convolutions with a bias are not supported")
        linears += [layer.conv.in_proj, layer.conv.out_proj]
    require_plain(linears)


def _row_hashes(embeds: Tensor) -> Tensor:
    """A 64-bit hash of each embedding row's bits (one GEMV-sized pass; int64 products wrap).
    `plan_sharing` checks the rows a plan pairs are bit-identical, so a collision costs only sharing."""
    bits = embeds.reshape(embeds.shape[0], -1)
    bits = (bits.view(torch.int16) if bits.element_size() == 2 else bits.view(torch.int32)).long()
    g = torch.Generator(device=bits.device).manual_seed(0x5EED)
    mult = torch.randint(-(2 ** 62), 2 ** 62, (bits.shape[1],), device=bits.device, generator=g) | 1
    return (bits * mult).sum(1)


def _rel(got: Tensor, ref: Tensor) -> float:
    ref = ref.float()
    return (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
