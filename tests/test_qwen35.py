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

    try:
        stock = enc()
    except Exception as exc:  # noqa: BLE001 — topk's own compiled flex_attention, before any packing
        pytest.skip(f"topk's shipped forward fails on torch {torch.__version__} (it pins 2.11), so there is no "
                    f"reference here: {type(exc).__name__}")
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
