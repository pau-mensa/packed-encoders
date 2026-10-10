"""LFM2 hybrid (gated short convolutions + GQA attention) — kernel, engine, HF entry.

    pytest tests/test_lfm2.py -q        # a tiny random LFM2 (no download)

The engine tests build a random 6-layer LFM2 text model and compare the engine with HF's own
forward per sequence, packed and graphed; the entry tests pack stock models (the text backbone,
a causal LM over it, and a wrapper that feeds it input embeddings) and keep their contract.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the LFM2 engine needs an sm_80+ CUDA GPU",
)
fla = pytest.importorskip("fla")
lfm2 = pytest.importorskip("transformers.models.lfm2.modeling_lfm2")

VOCAB = 1024


def _config(**overrides):
    from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

    cfg = dict(vocab_size=VOCAB, hidden_size=256, intermediate_size=768, num_hidden_layers=6,
               num_attention_heads=4, num_key_value_heads=2, conv_L_cache=3, norm_eps=1e-5,
               layer_types=["conv", "conv", "full_attention", "conv", "full_attention", "conv"],
               max_position_embeddings=4096, use_cache=False)
    cfg.update(overrides)
    return Lfm2Config(**cfg)


def _randomize(model):
    torch.manual_seed(0)
    for p in model.parameters():                     # random init is too small to exercise the kernels
        if p.dim() > 1:
            p.data.normal_(0, 0.05)
        else:
            p.data.uniform_(0.5, 1.5)                # norm weights away from 1: catches a missing scale
    return model


@pytest.fixture(scope="module")
def tiny():
    return _randomize(lfm2.Lfm2Model(_config())).to("cuda", torch.bfloat16).eval()


def _hf_hidden(model, ids):
    with torch.no_grad():
        return model(input_ids=ids[None], use_cache=False).last_hidden_state[0].float()


# ---------------------------------------------------------------------------- kernel


@pytest.mark.parametrize("lengths", [[3, 70, 1, 17, 40], [1500, 900, 2]], ids=["short", "long"])
def test_short_conv_matches_reference_and_the_model_per_sequence(tiny, lengths):
    from packed_encoders._kernels.short_conv import short_conv
    from packed_encoders.pieces.hybrid import ref_short_conv
    from packed_encoders.runtime.staging import packed_layout_host

    sc = tiny.layers[0].conv
    D = tiny.config.hidden_size
    g = torch.Generator(device="cuda").manual_seed(0)
    _, pos = packed_layout_host(lengths)
    pos = pos.cuda()
    h = torch.randn(sum(lengths), D, device="cuda", dtype=torch.bfloat16, generator=g)
    w = sc.conv.weight.detach().squeeze(1)
    with torch.no_grad():
        proj = F.linear(h, sc.in_proj.weight)
        got = short_conv(proj, w, pos)
        ref = ref_short_conv(proj, w, pos).float()
        assert (got.float() - ref).abs().max().item() / ref.abs().max().item() < 1e-2
        model = torch.cat([sc(x[None])[0] for x in h.split(lengths)]).float()
        mine = F.linear(got, sc.out_proj.weight).float()
    assert (mine - model).abs().max().item() / model.abs().max().item() < 2e-2


# ---------------------------------------------------------------------------- engine


@torch.no_grad()
@pytest.mark.parametrize("lengths", [[9, 70, 1, 33], [200, 5, 64]])
def test_engine_matches_hf_per_sequence_packed_and_graphed(tiny, lengths):
    from packed_encoders.arch.lfm2.engine import Lfm2Engine
    from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner

    eng = Lfm2Engine(tiny)
    try:
        g = torch.Generator().manual_seed(1)
        seqs = [torch.randint(0, VOCAB, (n,), generator=g).cuda() for n in lengths]
        ids = torch.cat(seqs)
        ref = torch.cat([_hf_hidden(tiny, s) for s in seqs])
        packed = eng.forward_packed(ids, lengths).float()
        cos = F.cosine_similarity(packed, ref, dim=-1)
        assert cos.mean().item() > 0.999 and cos.min().item() > 0.99, (cos.mean().item(), cos.min().item())
        via_embeds = eng.forward_packed(None, lengths, embeds=F.embedding(ids, eng.embed)).float()
        assert torch.equal(via_embeds, packed)

        runner = PaddedGraphRunner(eng, PaddedGraphConfig(row_buckets=(4, 8), max_seq=256, max_tokens=2048))
        graphed = runner(ids, lengths).float()
        again = runner(ids, lengths).float()          # replay, not capture
        cos = F.cosine_similarity(graphed, packed, dim=-1)
        assert cos.mean().item() > 0.9995 and cos.min().item() > 0.99
        assert torch.equal(graphed, again)

        # Both must be at least as close to the fp32 model as HF's own bf16 forward is.
        fp32 = copy.deepcopy(tiny).float()
        truth = torch.cat([_hf_hidden(fp32, s) for s in seqs])
        hf_bf16 = F.cosine_similarity(ref, truth, dim=-1).mean().item()
        for x in (packed, graphed):
            assert F.cosine_similarity(x, truth, dim=-1).mean().item() > hf_bf16 - 1e-4
    finally:
        eng.release()


def test_release_restores_independent_weights(tiny):
    from packed_encoders.arch.lfm2.engine import Lfm2Engine

    model = copy.deepcopy(tiny)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    eng = Lfm2Engine(model)
    sa = model.layers[2].self_attn
    assert sa.q_proj.weight.untyped_storage().data_ptr() == sa.k_proj.weight.untyped_storage().data_ptr()
    eng.release()
    assert sa.q_proj.weight.untyped_storage().data_ptr() != sa.k_proj.weight.untyped_storage().data_ptr()
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), before[n]), n


@pytest.mark.parametrize("where", ["conv_bias", "lora"])
def test_unsupported_weights_are_refused_before_mutation(tiny, where):
    from packed_encoders.arch.lfm2.engine import Lfm2Engine
    from packed_encoders.errors import UnsupportedTargetError

    model = copy.deepcopy(tiny)
    if where == "conv_bias":
        conv = model.layers[3].conv.conv
        conv.bias = nn.Parameter(torch.zeros(conv.out_channels, device="cuda", dtype=torch.bfloat16))
    else:
        model.layers[4].feed_forward.w2.lora_A = nn.Identity()
    ptr = model.layers[0].feed_forward.w1.weight.untyped_storage().data_ptr()
    with pytest.raises(UnsupportedTargetError):
        Lfm2Engine(model)
    assert model.layers[0].feed_forward.w1.weight.untyped_storage().data_ptr() == ptr


@torch.no_grad()
def test_norms_trained_after_packing_reach_eager_and_captured_graphs(tiny):
    from packed_encoders.arch.lfm2.engine import Lfm2Engine
    from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner

    model = copy.deepcopy(tiny)
    eng = Lfm2Engine(model)
    try:
        lengths = [12, 30]
        ids = torch.randint(0, VOCAB, (sum(lengths),), generator=torch.Generator().manual_seed(2)).cuda()
        runner = PaddedGraphRunner(eng, PaddedGraphConfig(row_buckets=(2,), max_seq=64, max_tokens=256))
        runner(ids, lengths)                                     # captured with the old norms
        for layer in model.layers:
            layer.ffn_norm.weight.mul_(1.7)
            if layer.is_attention_layer:
                layer.self_attn.k_layernorm.weight.mul_(0.5)
        model.embedding_norm.weight.add_(0.3)
        ref = torch.cat([_hf_hidden(model, s) for s in ids.split(lengths)])
        eager = eng.forward_packed(ids, lengths).float()
        eng.sync_norms()                                         # the HF entry syncs before each replay
        graphed = runner(ids, lengths).float()
        for x in (eager, graphed):
            assert F.cosine_similarity(x, ref, dim=-1).min().item() > 0.99
    finally:
        eng.release()


# ---------------------------------------------------------------------------- HF entry


class _EmbedsWrapper(nn.Module):
    """A multimodal-style wrapper: embeds the tokens itself and calls the backbone with inputs_embeds."""

    def __init__(self, lm):
        super().__init__()
        self.language_model = lm
        self.config = lm.config

    def forward(self, input_ids, attention_mask=None, **kwargs):
        embeds = self.language_model.embed_tokens(input_ids)
        return self.language_model(inputs_embeds=embeds, attention_mask=attention_mask, position_ids=None,
                                   past_key_values=None, **kwargs)


def _stock(tiny, kind):
    if kind == "text_model":
        return copy.deepcopy(tiny)
    if kind == "causal_lm":
        model = lfm2.Lfm2ForCausalLM(tiny.config).to("cuda", torch.bfloat16).eval()
        model.model.load_state_dict(tiny.state_dict())
        return model
    return _EmbedsWrapper(copy.deepcopy(tiny))


@pytest.mark.parametrize("kind", ["text_model", "causal_lm", "embeds_wrapper"])
def test_hf_entry_packs_stock_models_and_keeps_their_contract(tiny, kind):
    import packed_encoders as pe
    from packed_encoders.state import ATTR

    model = _stock(tiny, kind)
    backbone = model.model if kind == "causal_lm" else getattr(model, "language_model", model)
    g = torch.Generator().manual_seed(5)
    lengths = [9, 70, 1, 33]
    S = max(lengths)
    ids = torch.randint(0, VOCAB, (len(lengths), S), generator=g).cuda()
    right = (torch.arange(S)[None] < torch.tensor(lengths)[:, None]).long().cuda()
    left = right.flip(1)

    def run(m=right):
        with torch.no_grad():
            return backbone_out(model, ids, m)

    stock = {"right": run(right), "left": run(left)}
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}

    pe.pack(model)
    state = getattr(backbone, ATTR)
    assert state.report.eager_cos_mean > 0.999 and state.report.embeds_cos_mean > 0.999

    for side, m in (("right", right), ("left", left)):
        out = run(m)
        real = m.bool()
        assert out.shape == stock[side].shape and out.dtype == stock[side].dtype
        cos = F.cosine_similarity(out[real].float(), stock[side][real].float(), dim=-1)
        assert cos.mean().item() > 0.999 and cos.min().item() > 0.99, (side, cos.mean().item(), cos.min().item())
        assert not out[~real].any()                            # pads come back as zeros
    if kind != "embeds_wrapper":
        assert state.runner.num_graphs > 0

    # Calls the engine can't serve run the model's own forward, unchanged.
    calls, original = [], state.original_forward
    state.original_forward = lambda *a, **k: calls.append(1) or original(*a, **k)
    holes = right.clone()
    holes[1, 3] = 0
    with pytest.warns(UserWarning, match="original forward"):
        run(holes)
    with torch.enable_grad():
        backbone_out(model, ids, right)
    with torch.no_grad():
        backbone(input_ids=ids, attention_mask=right, use_cache=True)
    assert len(calls) == 3
    state.original_forward = original

    pe.unpack(model)
    assert getattr(backbone, ATTR, None) is None
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), params_before[n]), n
    assert torch.allclose(run(right), stock["right"], atol=1e-3)


def backbone_out(model, ids, mask):
    if isinstance(model, lfm2.Lfm2ForCausalLM):
        return model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    return model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state


# ---------------------------------------------------------------------------- shared prefixes


def _shared_rows(g, prefix=90):
    """Two prefixes, children of 1 to 25 own tokens (1 and 2 are shorter than the conv's taps),
    and rows that share nothing."""
    a, b = (torch.randint(0, VOCAB, (n,), generator=g) for n in (prefix, prefix + 13))
    own = lambda n: torch.randint(0, VOCAB, (n,), generator=g)  # noqa: E731
    return [torch.cat([a, own(1)]), own(40), torch.cat([b, own(2)]), torch.cat([a, own(25)]),
            torch.cat([b, own(3)]), own(prefix + 20), torch.cat([a, own(2)])]


@torch.no_grad()
def test_shared_prefixes_run_once_and_match_full_rows(tiny):
    from packed_encoders.arch.lfm2.engine import Lfm2Engine

    eng = Lfm2Engine(tiny)
    try:
        assert eng.share_rejected is None
        seqs = [s.cuda() for s in _shared_rows(torch.Generator().manual_seed(3))]
        lengths = [len(s) for s in seqs]
        ids = torch.cat(seqs)
        ref = torch.cat([_hf_hidden(tiny, s) for s in seqs])
        assert eng.plan_sharing(ids, lengths) is None                 # off until opted in
        eng.min_shared_prefix = 64
        full = eng.forward_packed(ids, lengths).float()
        embeds = F.embedding(ids, eng.embed)
        for plan_ids, emb in ((ids, None), (None, embeds)):
            plan = eng.plan_sharing(plan_ids, lengths, embeds=emb)
            assert plan is not None and plan.n_roots == 4 and plan.saved_tokens == 2 * 90 + (90 + 13)
            shared = eng.forward_shared(plan_ids, plan, embeds=emb).float()
            cos = F.cosine_similarity(shared, ref, dim=-1)
            assert cos.mean().item() > 0.999 and cos.min().item() > 0.99, (cos.mean().item(), cos.min().item())
            # as close to HF as the full-row pass, token for token
            assert F.cosine_similarity(shared, full, dim=-1).min().item() > 0.995
    finally:
        eng.release()


@torch.no_grad()
@pytest.mark.parametrize("kind", ["text_model", "embeds_wrapper"])
def test_hf_entry_runs_each_shared_prefix_once(tiny, kind, monkeypatch):
    import packed_encoders as pe
    from packed_encoders.runtime.shared_graphs import SharedGraphRunner
    from packed_encoders.state import ATTR

    model = _stock(tiny, kind)
    backbone = getattr(model, "language_model", model)
    seqs = _shared_rows(torch.Generator().manual_seed(6))
    S = max(map(len, seqs))
    ids = torch.zeros(len(seqs), S, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):                                     # left padded
        ids[i, S - len(s):], mask[i, S - len(s):] = s, 1
    ids, mask = ids.cuda(), mask.cuda()
    stock = backbone_out(model, ids, mask)

    pe.pack(model)
    state = getattr(backbone, ATTR)
    assert state.report.shared_prefix_rejected is None and state.report.shared_cos_mean > 0.999
    pe.get_engine(model).min_shared_prefix = 64
    calls = []
    real_shared, real_graph = state.engine.forward_shared, SharedGraphRunner.__call__
    monkeypatch.setattr(state.engine, "forward_shared", lambda *a, **k: calls.append("eager") or real_shared(*a, **k))
    monkeypatch.setattr(SharedGraphRunner, "__call__", lambda self, *a: calls.append("graph") or real_graph(self, *a))
    out = backbone_out(model, ids, mask)
    assert calls == (["eager"] if kind == "embeds_wrapper" else ["graph"])    # token ids replay a graph per plan
    real = mask.bool()
    cos = F.cosine_similarity(out[real].float(), stock[real].float(), dim=-1)
    assert cos.mean().item() > 0.999 and cos.min().item() > 0.99, (cos.mean().item(), cos.min().item())
    assert not out[~real].any()
    pe.get_engine(model).min_shared_prefix = 0
    backbone_out(model, ids, mask)
    assert len(calls) == 1                                           # off again: rows run in full
    pe.unpack(model)


@torch.no_grad()
def test_shared_graphs_reuse_a_plan_across_token_values_and_match_eager(tiny):
    from packed_encoders.arch.lfm2.engine import Lfm2Engine
    from packed_encoders.runtime.graphs import PaddedGraphConfig
    from packed_encoders.runtime.shared_graphs import SharedGraphRunner

    eng = Lfm2Engine(tiny)
    try:
        eng.min_shared_prefix = 64
        runner = SharedGraphRunner(eng, PaddedGraphConfig(max_graphs=2))
        previous = saved = None
        for seed, order in [(0, None), (1, None), (2, [0, 2, 1, 3, 4, 5, 6]), (3, [6, 5, 4, 3, 2, 1, 0]), (4, None)]:
            seqs = _shared_rows(torch.Generator().manual_seed(seed))
            if order:
                seqs = [seqs[i] for i in order]
            ids, lengths = torch.cat(seqs).cuda(), [len(s) for s in seqs]
            plan = eng.plan_sharing(ids, lengths)
            expected = eng.forward_shared(ids, plan).float()
            got = runner(ids, plan)
            assert F.cosine_similarity(got.float(), expected, dim=-1).min().item() > 0.999
            assert runner.num_graphs <= 2
            if seed == 1:
                assert runner.num_graphs == 1                        # new token values, the same plan
            if previous is not None:
                assert torch.equal(previous, saved)                  # outputs own their storage
            previous, saved = got, got.clone()
        runner.config = PaddedGraphConfig(max_tokens=1)
        assert runner(ids, plan) is None                             # out of bounds: the caller runs eagerly
        del runner
    finally:
        eng.release()


@torch.no_grad()
def test_embedding_hash_collisions_never_share_different_rows(tiny, monkeypatch):
    from packed_encoders.arch.lfm2 import engine as lfm2_engine

    eng = lfm2_engine.Lfm2Engine(tiny)
    try:
        eng.min_shared_prefix = 64
        g = torch.Generator().manual_seed(4)
        seqs = [torch.randint(0, VOCAB, (100,), generator=g).cuda() for _ in range(3)]       # nothing shared
        embeds = F.embedding(torch.cat(seqs), eng.embed)
        monkeypatch.setattr(lfm2_engine, "_row_hashes", lambda e: torch.zeros(e.shape[0], dtype=torch.long, device=e.device))
        assert eng.plan_sharing(None, [100, 100, 100], embeds=embeds) is None              # all collide: refused
    finally:
        eng.release()
