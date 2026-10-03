"""Qwen3.5 hybrid (GatedDeltaNet + gated attention) — kernels, attention probe, engine, topk entry.

    pytest tests/test_qwen35.py -q                  # kernels + a tiny random Qwen3.5 (no download)
    PE_TEST_TOPK=1 pytest tests/test_qwen35.py -q   # + topk-embed-v1-xsmall end to end (downloads ~1.7 GB)

The engine tests build a random 4-layer Qwen3.5 text model and compare the engine (causal, as
the plain HF model is) with HF's own forward per sequence, packed and graphed: that covers the
architecture path any Qwen3.5-based model takes, not just topk's wrapper.
"""

from __future__ import annotations

import copy
import inspect
import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the Qwen3.5 engine needs an sm_80+ CUDA GPU",
)
fla = pytest.importorskip("fla")


def _shims():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    if "block_type" in inspect.getsource(mq.Qwen3_5DecoderLayer.__init__):
        mq.Qwen3_5DecoderLayer.layer_type = property(lambda self: self.block_type)
    if not hasattr(mq.Qwen3_5GatedDeltaNet, "chunk_gated_delta_rule"):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        mq.Qwen3_5GatedDeltaNet.chunk_gated_delta_rule = staticmethod(chunk_gated_delta_rule)


# ---------------------------------------------------------------------------- kernels


def _ref_norm_rope(x, w, cos, sin, eps):
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype).float()
    r = cos.shape[-1]
    half = r // 2
    yr, yp = y[..., :r], y[..., r:]
    rot = torch.cat([-yr[..., half:], yr[..., :half]], -1)
    out = yr * cos[:, None].float() + rot * sin[:, None].float()
    return torch.cat([out, yp], -1).to(x.dtype)


@pytest.mark.parametrize("heads,hd,rot", [(8, 256, 64), (2, 256, 64), (4, 128, 128)])
def test_qk_norm_rope_matches_reference(heads, hd, rot):
    from packed_encoders.arch.qwen3_5.kernels import qk_norm_rope

    g = torch.Generator(device="cuda").manual_seed(0)
    T = 41
    x = torch.randn(T, heads, 2 * hd, device="cuda", dtype=torch.bfloat16, generator=g)[..., :hd]  # strided, as in use
    w = 1 + 0.1 * torch.randn(hd, device="cuda", generator=g)
    ang = torch.randn(T, rot // 2, device="cuda", generator=g)
    cos, sin = torch.cat([ang.cos()] * 2, -1).bfloat16(), torch.cat([ang.sin()] * 2, -1).bfloat16()
    got = qk_norm_rope(x, w, cos, sin, 1e-6).float()
    ref = _ref_norm_rope(x, w, cos, sin, 1e-6).float()
    assert (got - ref).abs().max().item() < 2e-2


def test_conv_split_matches_fla_per_sequence():
    from fla.modules.convolution import causal_conv1d

    from packed_encoders.arch.qwen3_5.kernels import conv_split
    from packed_encoders.runtime.staging import packed_layout_host

    g = torch.Generator(device="cuda").manual_seed(0)
    kd, vd, zd, nb = 256, 512, 512, 8
    lengths = [3, 70, 1, 17, 40]
    N = sum(lengths)
    proj = torch.randn(N, 2 * kd + vd + zd + 2 * nb, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(2 * kd + vd, 4, device="cuda", dtype=torch.bfloat16, generator=g)
    cu, pos = packed_layout_host(lengths)
    q, k, v, b, a = conv_split(proj, w, None, pos.cuda(), kd, vd, 2 * kd + vd + zd, nb, nb, True)
    ref = causal_conv1d(x=proj[None, :, : 2 * kd + vd], weight=w, activation="silu", cu_seqlens=cu.cuda(),
                        cu_seqlens_cpu=cu)[0][0]
    assert all(t.is_contiguous() for t in (q, k, v, b, a))
    assert (torch.cat([q, k, v], 1).float() - ref.float()).abs().max().item() < 1e-2
    assert torch.equal(b, proj[:, -2 * nb: -nb]) and torch.equal(a, proj[:, -nb:])
    # padded rows: positions restart per row; pads (after each row's real tokens) never reach a real token
    rows, S = 3, 24
    lens = [24, 5, 13]
    x = torch.randn(rows * S, proj.shape[1], device="cuda", dtype=torch.bfloat16, generator=g)
    qp = conv_split(x, w, None, torch.arange(S, device="cuda").repeat(rows), kd, vd, 2 * kd + vd + zd, nb, nb, True)[0]
    for i, n in enumerate(lens):
        one = causal_conv1d(x=x[None, i * S: i * S + n, :kd], weight=w[:kd], activation="silu")[0][0]
        assert (qp[i * S: i * S + n].float() - one.float()).abs().max().item() < 1e-2


def test_gated_norm_and_output_gate_match_unfused():
    from fla.modules import FusedRMSNormGated

    from packed_encoders.arch.qwen3_5.kernels import gated_rms_norm, sigmoid_gate

    g = torch.Generator(device="cuda").manual_seed(0)
    N, H, D, width = 37, 4, 128, 1000
    proj = torch.randn(N, width, device="cuda", dtype=torch.bfloat16, generator=g)
    z = proj[:, 100: 100 + H * D].view(N, H, D)                  # strided, as read from the projection
    o = torch.randn(N, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    norm = FusedRMSNormGated(D, eps=1e-6, device="cuda", dtype=torch.bfloat16)
    norm.weight.data = 1 + 0.1 * torch.randn(D, device="cuda", generator=g).bfloat16()
    ref = norm(o.reshape(-1, D), z.reshape(-1, D)).view(N, -1).float()
    got = gated_rms_norm(o, z, norm.weight, 1e-6).float()
    assert (got - ref).abs().max().item() <= 2e-2 * ref.abs().max().item()
    gate = torch.randn(N, H, 2 * D, device="cuda", dtype=torch.bfloat16, generator=g)[..., D:]
    ref = (o * torch.sigmoid(gate)).reshape(N, -1).float()       # Triton's sigmoid may differ in the last fp32 bit
    assert (sigmoid_gate(o, gate).float() - ref).abs().max().item() <= 1e-2 * ref.abs().max().item()


@pytest.mark.parametrize("gated", [True, False])
def test_qk_pair_matches_two_launches(gated):
    from packed_encoders.arch.qwen3_5.kernels import qk_norm_rope, qk_norm_rope_pair

    g = torch.Generator(device="cuda").manual_seed(0)
    T, nq, nk, hd, rot = 29, 8, 2, 256, 64
    qw = nq * (2 * hd if gated else hd)
    proj = torch.randn(T, qw + 2 * nk * hd, device="cuda", dtype=torch.bfloat16, generator=g)
    q = proj[:, :qw].view(T, nq, -1)[..., :hd]
    k = proj[:, qw: qw + nk * hd].view(T, nk, hd)
    wq, wk = (1 + 0.1 * torch.randn(hd, device="cuda", generator=g) for _ in range(2))
    ang = torch.randn(T, rot // 2, device="cuda", generator=g)
    cos, sin = torch.cat([ang.cos()] * 2, -1).bfloat16(), torch.cat([ang.sin()] * 2, -1).bfloat16()
    pq, pk = qk_norm_rope_pair(proj, 0, q.stride(1), nq, qw, hd, nk, torch.stack([wq, wk]), cos, sin, 1e-6)
    assert pq.is_contiguous() and pk.is_contiguous()
    assert torch.equal(pq, qk_norm_rope(q, wq, cos, sin, 1e-6)) and torch.equal(pk, qk_norm_rope(k, wk, cos, sin, 1e-6))


@pytest.mark.parametrize("causal", [False, True])
def test_attention_probe_selects_a_passing_kernel(causal):
    from packed_encoders.runtime.attention import PROBE_TOLERANCE, select_attention

    choice = select_attention(n_heads=8, n_kv_heads=2, head_dim=256, causal=causal, device=torch.device("cuda"))
    assert choice.max_abs_err < PROBE_TOLERANCE
    sdpa = select_attention(n_heads=8, n_kv_heads=2, head_dim=256, causal=causal, device=torch.device("cuda"),
                            order=("sdpa",))
    assert sdpa.name == "sdpa" and sdpa.max_abs_err < PROBE_TOLERANCE


# ---------------------------------------------------------------------------- engine on a tiny Qwen3.5


@pytest.fixture(scope="module")
def tiny():
    _shims()
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    cfg = Qwen3_5TextConfig(
        vocab_size=1024, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=256,
        linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=32, linear_value_head_dim=32,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    model = Qwen3_5TextModel(cfg).to("cuda", torch.bfloat16).eval()
    for p in model.parameters():                     # random init is too small to exercise the kernels
        p.data.normal_(0, 0.05) if p.dim() > 1 else None
    return model


def _hf_hidden(model, ids):
    with torch.no_grad():
        return model(input_ids=ids[None]).last_hidden_state[0].float()


def test_fusions_pass_their_probe_and_match_the_unfused_engine(tiny):
    from packed_encoders.arch.qwen3_5.engine import FUSION_TOLERANCE, Qwen35Engine, unshare_rows

    eng = Qwen35Engine(tiny, causal=True)
    try:
        assert eng.fused, eng.fusion_rejected
        assert {"conv", "gated_norm", "output_gate"} <= set(eng.fusion_errors)
        g = torch.Generator().manual_seed(3)
        lengths = [9, 70, 1, 33]
        ids = torch.randint(0, 1024, (sum(lengths),), generator=g).cuda()
        fused = eng.forward_packed(ids, lengths).float()
        eng.fused = False
        unfused = eng.forward_packed(ids, lengths).float()
        cos = F.cosine_similarity(fused, unfused, dim=-1)
        assert cos.min().item() > 0.999, cos.min().item()
    finally:
        unshare_rows(eng.shared)
    assert FUSION_TOLERANCE < 0.05


def test_gdn_kernels_match_the_fp32_recurrence(tiny):
    from packed_encoders.arch.qwen3_5.engine import GDN_PROBE_TOLERANCE, Qwen35Engine, unshare_rows

    eng = Qwen35Engine(tiny, causal=True)            # 4 key / 8 value heads: grouped values
    try:
        assert eng.gdn.rejected == {}, eng.gdn.rejected
        assert set(eng.gdn.errors) == {"chunk", "recurrent"}
        assert max(eng.gdn.errors.values()) < GDN_PROBE_TOLERANCE
        assert eng.gdn.recurrent_max_len > 0
    finally:
        unshare_rows(eng.shared)


# Eager, rows <= RECURRENT_MAX_LEN (64) run the recurrent GatedDeltaNet kernel, longer ones the chunked one;
# graphs always run the chunked one.
@pytest.mark.parametrize("lengths", [[9, 70, 1, 33], [9, 1, 17, 32]], ids=["chunked", "recurrent"])
def test_engine_matches_hf_per_sequence_packed_and_graphed(tiny, lengths):
    from packed_encoders.arch.qwen3_5.engine import Qwen35Engine, unshare_rows
    from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner

    eng = Qwen35Engine(tiny, causal=True)            # plain HF Qwen3.5 is a causal LM
    try:
        g = torch.Generator().manual_seed(1)
        seqs = [torch.randint(0, 1024, (n,), generator=g).cuda() for n in lengths]
        ids = torch.cat(seqs)
        ref = torch.cat([_hf_hidden(tiny, s) for s in seqs])
        packed = eng.forward_packed(ids, lengths).float()
        cos = F.cosine_similarity(packed, ref, dim=-1)
        assert cos.mean().item() > 0.999 and cos.min().item() > 0.99, (cos.mean().item(), cos.min().item())

        runner = PaddedGraphRunner(eng, PaddedGraphConfig(row_buckets=(4, 8), max_seq=128, max_tokens=1024))
        graphed = runner(ids, lengths).float()
        again = runner(ids, lengths).float()          # replay, not capture
        cos = F.cosine_similarity(graphed, packed, dim=-1)
        assert cos.mean().item() > 0.9995 and cos.min().item() > 0.99
        assert torch.equal(graphed, again)
        assert runner.plan([200]) is None             # beyond max_seq -> caller runs eager
        assert runner.plan_groups(lengths * 3) is None      # more rows than the largest bucket -> eager
        if max(lengths) <= eng.gdn.recurrent_max_len:
            # Recurrent and chunked differ by bf16 rounding (fp32 state vs bf16 chunk operands), so hold
            # both to the fp32 model: each must be at least as close to it as HF's own bf16 forward is.
            eng.gdn.recurrent_max_len, saved = 0, eng.gdn.recurrent_max_len
            chunked = eng.forward_packed(ids, lengths).float()
            eng.gdn.recurrent_max_len = saved
            fp32 = copy.deepcopy(tiny).float()
            truth = torch.cat([_hf_hidden(fp32, s) for s in seqs])
            hf_bf16 = F.cosine_similarity(ref, truth, dim=-1).mean().item()
            for x in (packed, chunked):
                assert F.cosine_similarity(x, truth, dim=-1).mean().item() > hf_bf16 - 1e-4
    finally:
        unshare_rows(eng.shared)


def test_share_and_unshare_restore_independent_weights(tiny):
    from packed_encoders.arch.qwen3_5.engine import Qwen35Engine, unshare_rows

    before = {n: p.detach().clone() for n, p in tiny.named_parameters()}
    eng = Qwen35Engine(tiny, causal=True)
    q = tiny.layers[3].self_attn.q_proj.weight
    assert q.data.untyped_storage().data_ptr() == eng.layers[3].qkv.untyped_storage().data_ptr()  # shared, no copy
    unshare_rows(eng.shared)
    assert q.data.untyped_storage().data_ptr() != eng.layers[3].qkv.untyped_storage().data_ptr()
    for n, p in tiny.named_parameters():
        assert torch.equal(p.detach(), before[n]), n


class _Adapted(torch.nn.Module):
    """What peft puts in place of a Linear: `.weight` answers the *base* weight, so a reader of
    `.weight` alone would drop the adapter."""

    def __init__(self, base):
        super().__init__()
        self.base_layer = base

    @property
    def weight(self):
        return self.base_layer.weight


@pytest.mark.parametrize("layer,where", [(0, "mlp.down_proj"), (0, "linear_attn.out_proj"), (3, "self_attn.o_proj"),
                                         (3, "self_attn.k_proj")])
def test_unmerged_adapter_on_any_projection_is_refused(tiny, layer, where):
    from packed_encoders.arch.qwen3_5.engine import Qwen35Engine
    from packed_encoders.errors import UnsupportedTargetError

    parent_name, attr = where.split(".")
    parent = getattr(tiny.layers[layer], parent_name)
    base = getattr(parent, attr)
    before = {n: p.detach().clone() for n, p in tiny.named_parameters()}
    setattr(parent, attr, _Adapted(base))
    try:
        with pytest.raises(UnsupportedTargetError, match="LoRA"):
            Qwen35Engine(tiny, causal=True)
    finally:
        setattr(parent, attr, base)
    for n, p in tiny.named_parameters():             # unchanged, and every merged row has its own storage back
        assert torch.equal(p.detach(), before[n]), n
    mlp = tiny.layers[0].mlp
    assert mlp.gate_proj.weight.untyped_storage().data_ptr() != mlp.up_proj.weight.untyped_storage().data_ptr()


def test_norms_trained_after_packing_reach_eager_and_captured_graphs(tiny):
    from packed_encoders.arch.qwen3_5.engine import Qwen35Engine, unshare_rows
    from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner

    eng = Qwen35Engine(tiny, causal=True)
    norms = [tiny.norm.weight, tiny.layers[0].input_layernorm.weight, tiny.layers[3].post_attention_layernorm.weight,
             tiny.layers[3].self_attn.q_norm.weight, tiny.layers[3].self_attn.k_norm.weight]
    saved = [w.detach().clone() for w in norms]
    try:
        lengths = [9, 70, 1, 33]
        g = torch.Generator().manual_seed(2)
        seqs = [torch.randint(0, 1024, (n,), generator=g).cuda() for n in lengths]
        ids = torch.cat(seqs)
        runner = PaddedGraphRunner(eng, PaddedGraphConfig(row_buckets=(4, 8), max_seq=128, max_tokens=1024))
        stale = runner(ids, lengths).float()                         # captured with the packing-time norms
        with torch.no_grad():                                        # what an optimizer step does: in place
            for w in norms:
                w.add_(torch.randn(w.shape, generator=g).to(w) * 0.3)
        ref = torch.cat([_hf_hidden(tiny, s) for s in seqs])
        assert F.cosine_similarity(stale, ref, dim=-1).mean().item() < 0.999     # the update matters
        eng.sync_norms()                                             # the topk forward does this per call
        for out in (runner(ids, lengths).float(), eng.forward_packed(ids, lengths).float()):
            cos = F.cosine_similarity(out, ref, dim=-1)
            assert cos.mean().item() > 0.999 and cos.min().item() > 0.99, (cos.mean().item(), cos.min().item())
        assert runner.num_graphs == 1                                # same graph, refreshed in place
    finally:
        with torch.no_grad():
            for w, s in zip(norms, saved):
                w.copy_(s)
        unshare_rows(eng.shared)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_engine_runs_on_its_weights_device_not_the_current_one(tiny):
    from packed_encoders.arch.qwen3_5.engine import Qwen35Engine, unshare_rows
    from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner

    model = copy.deepcopy(tiny).to("cuda:1")
    torch.cuda.set_device(0)
    lengths = [9, 70, 1, 33]
    g = torch.Generator().manual_seed(3)
    seqs = [torch.randint(0, 1024, (n,), generator=g).to("cuda:1") for n in lengths]
    ids = torch.cat(seqs)
    with torch.cuda.device(1):                                       # the reference, run the safe way
        ref = torch.cat([_hf_hidden(model, s) for s in seqs])
    eng = Qwen35Engine(model, causal=True)                           # probes, current device still 0
    try:
        runner = PaddedGraphRunner(eng, PaddedGraphConfig(row_buckets=(4, 8), max_seq=128, max_tokens=1024))
        for out in (eng.forward_packed(ids, lengths), runner(ids, lengths), runner(ids, lengths)):
            assert out.device == torch.device("cuda:1")
            cos = F.cosine_similarity(out.float(), ref, dim=-1)
            assert cos.mean().item() > 0.999 and cos.min().item() > 0.99, (cos.mean().item(), cos.min().item())
        torch.cuda.synchronize(1)
        assert torch.cuda.current_device() == 0
    finally:
        unshare_rows(eng.shared)


# ---------------------------------------------------------------------------- topk end to end


@pytest.mark.skipif(os.environ.get("PE_TEST_TOPK") != "1", reason="set PE_TEST_TOPK=1 (downloads the model)")
def test_topk_pack_encode_unpack():
    _shims()
    import packed_encoders as pe
    from packed_encoders.state import ATTR
    from sentence_transformers.multi_vector_encoder import MultiVectorEncoder

    model = MultiVectorEncoder("topk-io/topk-embed-v1-xsmall", model_kwargs={"torch_dtype": torch.bfloat16},
                               trust_remote_code=True, device="cuda").eval()
    net = model[0].auto_model
    docs = ["Packed encoders remove padding from encoder inference.",
            "Gated DeltaNet layers are causal linear-attention mixers with a decay gate. " * 12,
            "short", "CUDA graphs replay a captured sequence of kernels with a single launch."]
    queries = ["what is a CUDA graph", "gated delta rule"]

    def enc():
        d = model.encode(docs, task="document", batch_size=8, convert_to_numpy=False)
        q = model.encode(queries, task="query", batch_size=8, convert_to_numpy=False)
        return [x.float() for x in d + q]

    stock = enc()  # An explicitly requested end-to-end check must fail if its oracle fails.
    params_before = {n: p.detach().clone() for n, p in net.named_parameters() if "language_model" in n}
    pe.pack(model)
    state = getattr(net, ATTR)
    assert state.report.eager_cos_mean > 0.999
    packed = enc()
    for a, b in zip(packed, stock):
        assert a.shape == b.shape
        cos = F.cosine_similarity(a, b, dim=-1)
        assert cos.mean().item() > 0.999 and cos.min().item() > 0.98
    assert state.runner.num_graphs > 0

    with pe.no_cuda_graph(model):
        eager = enc()
    for a, b in zip(eager, stock):
        assert F.cosine_similarity(a, b, dim=-1).mean().item() > 0.999

    pe.pack(model)                                    # idempotent
    pe.unpack(model)
    assert getattr(net, ATTR, None) is None
    for n, p in net.named_parameters():
        if n in params_before:
            assert torch.equal(p.detach(), params_before[n]), n
    restored = enc()
    for a, b in zip(restored, stock):
        assert torch.allclose(a, b, atol=1e-3)


@pytest.fixture
def topk_tiny(tiny):
    """The topk entry contract with a local random backbone; no download."""
    from types import SimpleNamespace

    class Topk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="topk_embed", output_dim=32, dim=64, normalize=True)
            self.model = torch.nn.Module()
            self.model.language_model = copy.deepcopy(tiny)
            self.model.language_model.document_causal = True
            self.head = torch.nn.Linear(256, 64, bias=False, device="cuda", dtype=torch.bfloat16)

        def forward(self, input_ids, attention_mask, packed_ids, position_ids, cu_seqlens, seq_idx,
                    pixel_values=None, image_scatter_index=None, vision_inputs=None, return_dict=True):
            lengths = cu_seqlens.diff().tolist()
            hidden = torch.cat([self.model.language_model(input_ids=ids[None]).last_hidden_state[0]
                                for ids in packed_ids.flatten().split(lengths)])
            vectors = F.normalize(self.head(hidden).float()[:, :32], dim=-1)
            out = vectors.new_zeros((*input_ids.shape, 32))
            out[attention_mask.bool()] = vectors
            return out

    model = Topk().eval()
    yield model
    import packed_encoders as pe
    pe.unpack(model)


def test_qwen_prepared_boundary_pieces_controls_and_teardown(topk_tiny):
    from collections import Counter
    from dataclasses import replace
    import packed_encoders as pe
    from packed_encoders.arch.qwen3_5 import Qwen35Hybrid
    from packed_encoders.arch.qwen3_5.pieces import default_pieces
    from packed_encoders.errors import PackedEncodersError
    from packed_encoders.runtime.graphs import PaddedGraphConfig

    model = topk_tiny
    calls = Counter()
    pieces = default_pieces()
    def counted(name, fn):
        def call(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)
        return call
    selected = replace(pieces, **{name: replace(getattr(pieces, name), execute=counted(name, getattr(pieces, name).execute))
                                 for name in pieces.__dataclass_fields__})
    owner = Qwen35Hybrid(pieces=selected)
    original = model.forward
    param = model.model.language_model.layers[0].mlp.gate_proj.weight
    pe.pack(model, engine=owner, validate=False, cuda_graph=False, attention_backend="sdpa")
    packed = pe.get_engine(model)
    assert packed.binding.weight_source is model.model.language_model
    assert packed.composition is selected and packed.pieces == selected.all()
    assert packed.capabilities.original_forward_fallback and not packed.capabilities.training
    batch = pe.PackedBatch(torch.randint(0, 1024, (29,), device="cuda"), host_lengths=(7, 22))
    with pytest.raises(PackedEncodersError, match="no_grad"):
        packed.forward_packed(batch)
    with torch.no_grad():
        for malformed in (replace(batch, host_lengths=(0, 29)),
                          replace(batch, position_ids=torch.arange(29, device="cuda")),
                          replace(batch, input_ids=batch.input_ids.int())):
            with pytest.raises(PackedEncodersError):
                packed.forward_packed(malformed)
        eager = packed.forward_packed(batch)
        reference = torch.cat([_hf_hidden(model.model.language_model, ids) for ids in batch.input_ids.split([7, 22])])
        assert eager.shape == (29, 256)  # hidden states, not projected vectors
        assert F.cosine_similarity(eager.float(), reference, dim=-1).min() > 0.99
        pe.set_cuda_graph(model, True, config=PaddedGraphConfig(row_buckets=(2,), max_seq=64, max_tokens=128))
        graphed = packed.forward_packed(batch)
        assert F.cosine_similarity(graphed.float(), eager.float(), dim=-1).min() > 0.99
        before = calls.copy()
        packed.forward_packed(batch)
        assert calls == before  # replay bypasses the Python schedule
        with pe.no_cuda_graph(model):
            packed.forward_packed(batch)
        assert calls["linear"] > before["linear"]
        for slot in ("rms_norm", "add_rms_norm", "linear", "swiglu", "conv_split", "gated_rms_norm", "qk_norm_rope_pair", "sigmoid_gate", "gdn_chunk", "gdn_recurrent", "attention_packed", "attention_padded"):
            assert calls[slot] > 0, slot
        report = packed.validate(batches=((7, 70),), graphs=True)
        assert report.engine == "qwen3_5" and report.details.eager_cos_mean > 0.999
        assert len(report.details.pieces) == 13
        param.add_(0.01)
        updated = param.clone()
    with pytest.raises(PackedEncodersError, match="unpack"):
        pe.pack(model, attention_backend="flash")
    with pytest.raises(PackedEncodersError, match="training graphs"):
        pe.set_train_cuda_graph(model, True)
    pe.pack(model, cuda_graph=False)
    assert not packed.graph_enabled
    pe.unpack(model)
    assert model.forward == original and "forward" not in model.__dict__
    torch.testing.assert_close(param, updated, rtol=0, atol=0)
    assert packed.state.runner is None
    with pytest.raises(PackedEncodersError, match="closed"):
        packed.forward_packed(batch)


@pytest.mark.parametrize("failure", ["piece", "install"])
def test_qwen_failure_restores_storage_and_forward(topk_tiny, monkeypatch, failure):
    from dataclasses import replace
    import packed_encoders as pe
    from packed_encoders.arch.qwen3_5 import Qwen35Hybrid, TopkEmbedAdapter
    from packed_encoders.arch.qwen3_5.pieces import default_pieces
    from packed_encoders.errors import ValidationError
    from packed_encoders.state import ATTR, INSTALL_ATTR

    model = topk_tiny
    original = model.forward
    params = [(p, p.data_ptr(), p.detach().clone()) for p in model.parameters()]
    pieces = default_pieces()
    if failure == "piece":
        pieces = replace(pieces, swiglu=replace(pieces.swiglu, execute=lambda g, u: torch.zeros_like(g)))
    else:
        def fail_install(self, binding, packed):
            setattr(binding.patch_target, ATTR, packed.state)
            binding.patch_target.forward = lambda *a: None
            raise RuntimeError("install failed")
        monkeypatch.setattr(TopkEmbedAdapter, "install", fail_install)
    with pytest.raises((ValidationError, RuntimeError)):
        pe.pack(model, engine=Qwen35Hybrid(pieces=pieces), cuda_graph=False, validate=failure == "piece")
    assert model.forward == original and "forward" not in model.__dict__
    assert not hasattr(model, ATTR) and not hasattr(model, INSTALL_ATTR)
    assert len({p.untyped_storage().data_ptr() for p, _, _ in params}) == len(params)
    for param, _, value in params:
        torch.testing.assert_close(param, value, rtol=0, atol=0)


def test_qwen_rejects_aliased_parameters_before_mutation(topk_tiny):
    import packed_encoders as pe
    from packed_encoders.errors import UnsupportedTargetError
    model = topk_tiny
    mlp = model.model.language_model.layers[0].mlp
    mlp.up_proj.weight = mlp.gate_proj.weight
    ptr = mlp.gate_proj.weight.data_ptr()
    with pytest.raises(UnsupportedTargetError, match="tied or aliased"):
        pe.pack(model, validate=False)
    assert mlp.up_proj.weight is mlp.gate_proj.weight
    assert mlp.gate_proj.weight.data_ptr() == ptr


def test_topk_adapter_delegates_grad_and_all_image_inputs(topk_tiny):
    import packed_encoders as pe
    model = topk_tiny
    calls = []
    sentinel = object()
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel
    model.forward = original
    pe.pack(model, validate=False, cuda_graph=False)
    args = (None,) * 6
    with pytest.warns(UserWarning, match="original forward"):
        assert model(*args) is sentinel  # grad enabled
    with torch.no_grad():
        for keyword in ("pixel_values", "image_scatter_index", "vision_inputs"):
            marker = object()
            assert model(*args, **{keyword: marker}) is sentinel
            assert calls[-1][1][keyword] is marker
    assert len(calls) == 4
    pe.unpack(model)
    assert model.forward is original


# Adapted from e97d004: use #5's topk adapter and prepared packed entry instead
# of the stock HF entry introduced separately in #7.
@pytest.mark.parametrize("entry,failure", [
    ("packed", "capture"), ("topk", "capture"),
    ("packed", "eager"), ("topk", "eager"), ("topk", "projection"),
])
def test_out_of_memory_with_graphs_drops_them_and_continues_eager(topk_tiny, monkeypatch, entry, failure):
    import weakref
    import packed_encoders as pe
    import packed_encoders.arch.qwen3_5 as qwen
    from packed_encoders.runtime.graphs import PaddedGraphConfig

    model = topk_tiny
    g = torch.Generator().manual_seed(6)
    ids = torch.randint(0, 1024, (3, 40), generator=g).cuda()
    feats = dict(input_ids=ids, attention_mask=torch.ones_like(ids), packed_ids=ids.flatten(),
                 position_ids=torch.arange(40, device=ids.device).repeat(3),
                 cu_seqlens=torch.arange(4, device=ids.device, dtype=torch.int32) * 40,
                 seq_idx=torch.arange(3, device=ids.device).repeat_interleave(40))
    batch = pe.PackedBatch(ids.flatten(), host_lengths=(40, 40, 40))
    with torch.no_grad():
        stock = (model(**feats) if entry == "topk" else
                 torch.cat([_hf_hidden(model.model.language_model, row) for row in ids]))
    cfg = PaddedGraphConfig(row_buckets=(3,), max_seq=64, max_tokens=192)
    pe.pack(model, cuda_graph=cfg, validate=False)
    packed = pe.get_engine(model)
    state = packed.state

    def run():
        return model(**feats) if entry == "topk" else packed.forward_packed(batch)

    # Hold a real graph before injecting OOM; the failing capture uses a new bucket.
    with torch.no_grad():
        packed.forward_packed(pe.PackedBatch(ids[:, :8].flatten(), host_lengths=(8, 8, 8)))
    assert state.runner.num_graphs == 1
    runner_ref = weakref.ref(state.runner)
    failed_tensor = []
    calls = []

    def oom(*a, **k):
        calls.append(1)
        temporary = torch.empty(16, device=ids.device)
        failed_tensor.append(weakref.ref(temporary))
        raise torch.OutOfMemoryError("CUDA out of memory. (simulated)")

    if failure == "capture":
        state.runner._capture = oom
    else:
        original = state.engine.forward_packed if failure == "eager" else qwen._vectors

        def fail_once(*a, **k):
            if not calls:
                return oom(*a, **k)
            # The traceback and graph runner must be gone before eager retry.
            assert runner_ref() is None and failed_tensor[0]() is None
            return original(*a, **k)

        if failure == "eager":
            # Disabled graphs still hold memory that the eager path may need.
            pe.set_cuda_graph(model, False)
            monkeypatch.setattr(state.engine, "forward_packed", fail_once)
        else:
            monkeypatch.setattr(qwen, "_vectors", fail_once)

    with pytest.warns(UserWarning, match="dropped them"), torch.no_grad():
        out = run()
    assert len(calls) == 1
    assert state.runner is None and not packed.graph_enabled
    assert runner_ref() is None and failed_tensor[0]() is None
    cos = F.cosine_similarity(out.float(), stock.float(), dim=-1)
    assert cos.mean().item() > 0.999

    with torch.no_grad():
        again = run()
    torch.testing.assert_close(again, out)
    assert state.runner is None
    with monkeypatch.context() as patch:
        patch.setattr(state.engine, "forward_packed", oom)
        with pytest.raises(torch.OutOfMemoryError), torch.no_grad():
            run()                   # no graphs left: a real OOM propagates

    pe.set_cuda_graph(model, True, config=cfg)
    with torch.no_grad():
        restored = run()
    assert packed.graph_enabled and state.runner.num_graphs > 0
    assert F.cosine_similarity(restored.float(), stock.float(), dim=-1).mean() > 0.999
    pe.unpack(model)


def test_graph_recovery_does_not_swallow_errors_or_repeat_eager_retry(topk_tiny, monkeypatch):
    import packed_encoders as pe
    from packed_encoders.runtime.graphs import PaddedGraphConfig

    pe.pack(topk_tiny, validate=False, cuda_graph=PaddedGraphConfig(max_tokens=128))
    packed = pe.get_engine(topk_tiny)
    batch = pe.PackedBatch(torch.arange(8, device="cuda"), host_lengths=(8,))

    def other_error(*a, **k):
        raise RuntimeError("unrelated failure")

    packed.state.runner._capture = other_error
    with pytest.raises(RuntimeError, match="unrelated failure"), torch.no_grad():
        packed.forward_packed(batch)
    assert packed.graph_enabled and packed.state.runner is not None

    calls = []
    def oom(*a, **k):
        calls.append(1)
        raise torch.OutOfMemoryError("simulated")

    packed.state.runner._capture = oom
    monkeypatch.setattr(packed.state.engine, "forward_packed", oom)
    with pytest.warns(UserWarning, match="dropped them"), pytest.raises(torch.OutOfMemoryError), torch.no_grad():
        packed.forward_packed(batch)
    assert len(calls) == 2           # one failed capture and one failed eager attempt
    assert packed.state.runner is None and not packed.graph_enabled
