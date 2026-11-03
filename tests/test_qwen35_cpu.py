"""Qwen3.5 CPU lifecycle and numerical parity, including native recurrence and INT8."""
import importlib.util

import pytest
import torch

import packed_encoders as pe
from packed_encoders.cpu.int8 import sme2_available, vnni_available
from packed_encoders.errors import PackedEncodersError

pytestmark = pytest.mark.skipif(importlib.util.find_spec("transformers.models.qwen3_5") is None,
                                reason="Transformers with Qwen3.5")


@pytest.fixture
def model():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["linear_attention", "full_attention"], max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .5,
                         "mrope_section": [2, 1, 1]}, use_cache=False)
    torch.manual_seed(13)
    m = Qwen3_5TextModel(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim > 1:
                p.mul_(3.)
    return m


def inputs(lengths=(7, 65)):
    ids = torch.arange(len(lengths) * max(lengths)).reshape(len(lengths), -1) % 127
    mask = torch.arange(ids.shape[1])[None] < torch.tensor(lengths)[:, None]
    return ids, mask


@pytest.mark.parametrize("hd,rot", [(7, 4), (16, 16), (256, 64)])
@torch.no_grad()
def test_native_qk_norm_rope(model, hd, rot):
    from types import SimpleNamespace
    from packed_encoders.cpu.qwen35 import GatedDeltaNet
    norm_type = type(model.norm)
    a = SimpleNamespace(head_dim=hd, q_norm=norm_type(hd, eps=1e-5), k_norm=norm_type(hd, eps=1e-4))
    a.q_norm.weight.copy_(torch.randn(hd))
    a.k_norm.weight.copy_(torch.randn(hd))
    nq, nk = 3, 1
    native = GatedDeltaNet(2)
    def rope(t, cos, sin):
        prefix = t[..., :rot]
        turn = torch.cat((-prefix[..., rot // 2:], prefix[..., :rot // 2]), -1)
        return torch.cat((prefix * cos[:, None] + turn * sin[:, None], t[..., rot:]), -1)
    try:
        for rows in (1, 7, 3):
            p = torch.randn(rows, 2 * (nq + nk) * hd + 5)
            before = p.clone()
            # Distinct entries in both halves also exercise scaled/custom tables.
            cos, sin = torch.randn(rows, rot), torch.randn(rows, rot)
            q = a.q_norm(p[:, :2 * nq * hd].reshape(rows, nq, 2 * hd)[..., :hd])
            k = a.k_norm(p[:, 2 * nq * hd:(2 * nq + nk) * hd].reshape(rows, nk, hd))
            got_q, got_k = native.qk_rope(p, a, cos, sin, nq, nk)
            torch.testing.assert_close(got_q, rope(q, cos, sin), atol=3e-6, rtol=3e-6)
            torch.testing.assert_close(got_k, rope(k, cos, sin), atol=3e-6, rtol=3e-6)
            torch.testing.assert_close(p, before, atol=0, rtol=0)
            a.q_norm.weight.add_(.25)
            a.k_norm.weight.sub_(.25)
    finally:
        native.close()
    assert native.q_buffer is native.k_buffer is None


@torch.no_grad()
def test_qk_norm_rope_fallbacks(model, monkeypatch):
    pe.pack(model, engine=pe.Qwen35CPU(threads=1), attention_backend="native", validate=False)
    engine = pe.get_engine(model)
    attention = engine.layers[1].mixer
    p = torch.randn(3, engine.layers[1].mixed.shape[0])
    cos, sin = torch.randn(3, 32)[:, ::2], torch.randn(3, 32)[:, ::2]
    try:
        want = engine._qk_rope(p, attention, cos.contiguous(), sin.contiguous())
        want = tuple(t.clone() for t in want)
        def unexpected(*args, **kwargs):
            raise AssertionError("unsupported input reached fused Q/K norm/RoPE")
        monkeypatch.setattr(engine.native, "qk_rope", unexpected)
        got = engine._qk_rope(p, attention, cos, sin)
        for actual, expected in zip(got, want):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        # Large packed batches retain the eager path even with contiguous tables.
        large = p[:1].expand(129, -1).contiguous()
        large_cos = cos[:1].expand(129, -1).contiguous()
        large_sin = sin[:1].expand(129, -1).contiguous()
        got = engine._qk_rope(large, attention, large_cos, large_sin)
        for actual, expected in zip(got, want):
            torch.testing.assert_close(actual, expected[:1].expand_as(actual), atol=3e-6, rtol=3e-6)
        calls = []
        original = attention.q_norm.forward
        def custom(x):
            calls.append(True)
            return original(x)
        monkeypatch.setattr(attention.q_norm, "forward", custom)
        engine._qk_rope(p, attention, cos.contiguous(), sin.contiguous())
        assert calls == [True]
    finally:
        pe.unpack(model)


@pytest.mark.parametrize("width,padding", [(1, 0), (15, 34), (128, 0), (3584, 0)])
@pytest.mark.parametrize("threads", [1, 3])
@torch.no_grad()
def test_native_swiglu(width, padding, threads):
    from packed_encoders.cpu.qwen35 import GatedDeltaNet
    native = GatedDeltaNet(threads)
    try:
        for rows in (1, 7, 3):
            x = torch.randn(rows, 2 * width + padding)
            x[:, :width] *= 30  # Includes sigmoid saturation and exp overflow.
            before = x.clone()
            want = torch.nn.functional.silu(x[:, :width]) * x[:, width:2 * width]
            got = native.swiglu(x, width)
            assert got.is_contiguous()
            torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(x, before, atol=0, rtol=0)
    finally:
        native.close()
    assert native.swiglu_buffer is None


@pytest.mark.parametrize("width", [1, 3, 15, 64, 1024])
@torch.no_grad()
def test_native_add_rmsnorm(model, width):
    from packed_encoders.cpu.qwen35 import GatedDeltaNet
    norm = type(model.norm)(width, eps=1e-5)
    norm.weight.copy_(torch.randn(width))
    native = GatedDeltaNet(2)
    try:
        x, branch = torch.randn(3, width), torch.randn(3, width)
        branch_before = branch.clone()
        residual = x + branch
        want = norm(residual)
        updated, out = native.add_norm(x, branch, norm, owned=True)
        assert updated is x
        torch.testing.assert_close(x, residual, atol=0, rtol=0)
        torch.testing.assert_close(out, want, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(branch, branch_before, atol=0, rtol=0)
        saved = out.clone()
        # Live norm updates and scratch reuse must not alter earlier owned outputs.
        norm.weight.add_(.5)
        for rows in (3, 1, 3):
            x = torch.randn(rows, width)
            b = branch[:rows]
            want = norm(x + b)
            _, scratch = native.add_norm(x, b, norm)
            torch.testing.assert_close(scratch, want, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(out, saved, atol=0, rtol=0)
    finally:
        native.close()
    assert native.norm_buffer is None


@torch.no_grad()
def test_add_rmsnorm_fallbacks(model, monkeypatch):
    pe.pack(model, engine=pe.Qwen35CPU(threads=1), attention_backend="native", validate=False)
    engine = pe.get_engine(model)
    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported input reached fused norm")
    monkeypatch.setattr(engine.native, "add_norm", unexpected)
    try:
        x = torch.randn(3, 64)
        branch = torch.randn(3, 128)[:, ::2]
        want = model.norm(x + branch)
        _, out = engine._add_norm(x, branch, model.norm)
        torch.testing.assert_close(out, want)
        branch = branch.contiguous()
        monkeypatch.setattr(model.norm, "forward", lambda x: x * 2)
        _, out = engine._add_norm(x, branch, model.norm)
        torch.testing.assert_close(out, (x + branch) * 2)
    finally:
        pe.unpack(model)


@pytest.mark.parametrize("lengths", [(1, 1), (7, 7, 7), (7, 3)])
@pytest.mark.parametrize("causal", [True, False])
@torch.no_grad()
def test_attention_batch_boundaries(model, lengths, causal):
    model.config.is_causal = causal
    ids, mask = inputs(lengths)
    want = model(ids, attention_mask=mask).last_hidden_state[mask]
    pe.pack(model, engine=pe.Qwen35CPU(threads=2), validate=False)
    try:
        got = model(ids, attention_mask=mask).last_hidden_state[mask]
        torch.testing.assert_close(got, want, atol=1e-5, rtol=2e-5)
        # Independent requests must agree with batching: neither attention nor
        # recurrence may carry information across a sequence boundary.
        separate = torch.cat([model(ids[i:i + 1, :n]).last_hidden_state[0]
                              for i, n in enumerate(lengths)])
        torch.testing.assert_close(got, separate, atol=1e-5, rtol=2e-5)
    finally:
        pe.unpack(model)


@pytest.mark.parametrize("backend", ["auto", "native", "sdpa"])
@torch.no_grad()
def test_parity(model, backend):
    ids, mask = inputs()
    want = model(ids, attention_mask=mask).last_hidden_state
    before = {name: (p.data_ptr(), p.clone()) for name, p in model.named_parameters()}
    threads = torch.get_num_threads()
    pe.pack(model, engine=pe.Qwen35CPU(threads=2), attention_backend=backend)
    engine = pe.get_engine(model)
    assert engine.precision == "fp32" and torch.get_num_threads() == threads
    got = model(ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(got[mask], want[mask], atol=1e-5, rtol=2e-5)
    assert not got[~mask].any()
    batch = pe.PackedBatch(ids[mask], host_lengths=(7, 65))
    torch.testing.assert_close(engine.forward_packed(batch), got[mask], atol=0, rtol=0)
    saved = got.clone()
    model(ids.flip(1), attention_mask=mask)
    assert torch.equal(got, saved)
    assert model(ids, attention_mask=mask, return_dict=False)[0].shape == want.shape
    pe.unpack(model)
    assert all(p.data_ptr() == before[name][0] and torch.equal(p, before[name][1]) for name, p in model.named_parameters())
    with pytest.raises(PackedEncodersError, match="closed"):
        engine.forward_packed(batch)


@torch.no_grad()
def test_default_selection_microbatches_and_positions(model):
    ids, mask = inputs((5, 3, 9))
    want = model(ids, attention_mask=mask).last_hidden_state
    pe.pack(model, engine=pe.Qwen35CPU(max_tokens=9, prepack_weights=False))
    engine = pe.get_engine(model)
    batch = pe.PackedBatch(ids[mask], cu_seqlens=torch.tensor([0, 5, 8, 17], dtype=torch.int32))
    torch.testing.assert_close(engine.forward_packed(batch), want[mask], atol=1e-5, rtol=2e-5)
    pe.unpack(model)
    pe.pack(model)  # CPU is selected automatically through the Qwen architecture.
    assert pe.get_engine(model).state.gdn_backend == "native"
    torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state[mask], want[mask], atol=1e-5, rtol=2e-5)
    positions = torch.arange(ids.shape[1])[None] + 11
    original = pe.get_engine(model).original_forward
    want = original(ids[:1], position_ids=positions, use_cache=False).last_hidden_state[0]
    got = pe.get_engine(model).forward_packed(pe.PackedBatch(ids[0], host_lengths=(9,), position_ids=positions[0]))
    torch.testing.assert_close(got, want, atol=1e-5, rtol=2e-5)


def test_fallbacks_and_updates(model):
    ids, mask = inputs((5, 3))
    original = model.forward
    pe.pack(model, engine=pe.Qwen35CPU(threads=1))
    engine = pe.get_engine(model)
    model.train()
    got = model(ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(got, original(ids, attention_mask=mask).last_hidden_state, atol=0, rtol=0)
    got.sum().backward()
    assert model.embed_tokens.weight.grad is not None
    model.eval()
    with torch.no_grad():
        for kwargs in ({"use_cache": True}, {"output_hidden_states": True}, {"position_ids": torch.arange(5)[None]}):
            actual = model(ids, attention_mask=mask, **kwargs)
            expected = original(ids, attention_mask=mask, **kwargs)
            torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=0, rtol=0)
        # Unsupported masks keep their exact HF behavior.
        for m in (mask.flip(1), torch.tensor([[1, 0, 1, 0, 1], [1, 1, 0, 0, 0]])):
            torch.testing.assert_close(model(ids, attention_mask=m).last_hidden_state,
                                       original(ids, attention_mask=m).last_hidden_state, atol=0, rtol=0)
        model.layers[0].mlp.gate_proj.weight.mul_(.9)
        model.layers[0].linear_attn.in_proj_qkv.weight.mul_(1.1)
        model.layers[1].self_attn.q_norm.weight.add_(.2)
        torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state[mask],
                                   original(ids, attention_mask=mask).last_hidden_state[mask], atol=1e-5, rtol=2e-5)
    with pytest.raises(PackedEncodersError, match="no_grad"):
        engine.forward_packed(pe.PackedBatch(ids[mask], host_lengths=(5, 3)))


def test_pack_inside_inference_mode_keeps_fp32_updates_live(model):
    ids, mask = inputs((5, 3))
    original = model.forward
    with torch.inference_mode():
        pe.pack(model)
        model(ids, attention_mask=mask)
    with torch.no_grad():
        model.layers[0].mlp.gate_proj.weight.mul_(.9)
        torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state[mask],
                                   original(ids, attention_mask=mask).last_hidden_state[mask], atol=1e-5, rtol=2e-5)


@torch.no_grad()
def test_invalid_options_and_metadata(model):
    for kw in ({"threads": 0}, {"max_tokens": 0}, {"prepack_weights": 1}):
        with pytest.raises(ValueError):
            pe.Qwen35CPU(**kw)
    for kw in ({"precision": "fp16"}, {"cuda_graph": True}, {"attention_backend": "flash"}):
        with pytest.raises(PackedEncodersError):
            pe.pack(model, **kw)
    pe.pack(model, engine=pe.Qwen35CPU(max_tokens=16))
    engine = pe.get_engine(model)
    for b in (pe.PackedBatch(torch.tensor([1, 2]), host_lengths=(1,)),
              pe.PackedBatch(torch.tensor([1, 2]), host_lengths=(2,), max_seqlen=1),
              pe.PackedBatch(torch.tensor([1, 2]), host_lengths=(2,), position_ids=torch.tensor([-1, 0])),
              pe.PackedBatch(torch.tensor([1, 2]), host_lengths=(2,), cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32)),
              pe.PackedBatch(torch.tensor([128]), host_lengths=(1,))):
        with pytest.raises(PackedEncodersError):
            engine.forward_packed(b)
    with pytest.raises(PackedEncodersError, match="unpack"):
        pe.pack(model, precision="int8")
    with pytest.raises(PackedEncodersError, match="CUDA"):
        pe.set_cuda_graph(model, True)


@pytest.mark.skipif(not (sme2_available() or vnni_available()), reason="SME2 or AVX512-VNNI")
@torch.no_grad()
def test_int8(model):
    ids, mask = inputs((5, 9))
    want = model(ids, attention_mask=mask).last_hidden_state[mask]
    pe.pack(model, engine=pe.Qwen35CPU(threads=2), precision="int8")
    engine = pe.get_engine(model)
    got = model(ids, attention_mask=mask).last_hidden_state[mask]
    assert engine.precision == engine.state.validation_report.precision == "int8"
    rel = float((got - want).norm() / want.norm())
    assert 1e-5 < rel < .05
    model.layers[0].mlp.down_proj.weight.mul_(.9)
    with pytest.raises(PackedEncodersError, match="repack"):
        model(ids, attention_mask=mask)


@pytest.mark.skipif(not (sme2_available() or vnni_available()), reason="SME2 or AVX512-VNNI")
@torch.no_grad()
@pytest.mark.parametrize("rows", [1, 17, 37])
def test_int8_projection_matches_quantized_math(rows):
    from packed_encoders.arch.qwen3_5.cpu import Projection
    from packed_encoders.cpu.qwen35_int8 import Int8Projections
    from packed_encoders.cpu.int8 import rotate
    torch.manual_seed(3)
    p = Projection((torch.nn.Linear(192, 128, bias=False),))
    q8 = Int8Projections([p], threads=3)
    x = torch.randn(rows, 192)
    x[0] = 0
    got = q8.linear(x, p)
    w, ws, _, block = q8.prepared[id(p)]
    # Quantize in float64 independently of the native implementation. FP32 rounding
    # may move a quantized value by one at a half-integer boundary.
    rotated = rotate(x, 192, block)
    scale = rotated.abs().amax(1).clamp_min(1e-30) / 127
    a = (rotated / scale[:, None]).round().clamp(-127, 127)
    qw = w.transpose(-3, -2).reshape(128, 192).double()
    want = ((a @ qw.T) * scale[:, None] * ws.double()).float()
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
    assert not got[0].any()
    q8.close()


@torch.no_grad()
@pytest.mark.parametrize("seq", [1, 7, 32, 65, 129])
def test_onnx_baseline(model, tmp_path, seq):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from benchmarks.cpu.qwen35_onnx import export
    path = export(model, tmp_path, seq)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for lengths in ((seq,), (max(1, seq // 2), seq)):
        ids, mask = inputs(lengths)
        want = model(ids, attention_mask=mask).last_hidden_state[mask]
        got = torch.from_numpy(session.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.long().numpy()})[0])[mask]
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("head_dims", [(16, 15), (16, 16), (16, 32), (128, 128)])
@torch.no_grad()
def test_native_grouped_gdn_with_conv_bias_and_long_sequences(model, head_dims):
    from packed_encoders.cpu.qwen35 import GatedDeltaNet, reference_gdn
    from types import SimpleNamespace
    hk, hv = head_dims
    kd, vd = 2 * hk, 4 * hv
    conv_dim = 2 * kd + vd
    norm_type = type(model.layers[0].linear_attn.norm)
    g = SimpleNamespace(num_k_heads=2, num_v_heads=4, head_k_dim=hk, head_v_dim=hv,
                        key_dim=kd, value_dim=vd, conv_kernel_size=4,
                        conv1d=torch.nn.Conv1d(conv_dim, conv_dim, 4, groups=conv_dim),
                        A_log=torch.tensor([-3., 0., 2., 4.]), dt_bias=torch.randn(4),
                        norm=norm_type(hv, eps=1e-6))
    g.conv1d.bias.copy_(torch.linspace(-2, 2, conv_dim))
    lengths = (1, 3, 65, 129)
    projection = torch.randn(sum(lengths), 2 * g.key_dim + 2 * g.value_dim + 2 * g.num_v_heads) * 2
    layer = SimpleNamespace(mixer=g)
    cu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    want = reference_gdn(projection, layer, lengths)
    native = GatedDeltaNet(3)
    try:
        torch.testing.assert_close(native(projection, layer, cu), want, atol=3e-5, rtol=3e-5)
    finally:
        native.close()


@torch.no_grad()
def test_compiler_fallback_and_failure_atomicity(model, monkeypatch):
    import packed_encoders.arch.qwen3_5.cpu as cpu
    original = model.forward
    def unavailable(*args):
        raise PackedEncodersError("no compiler")
    monkeypatch.setattr(cpu, "GatedDeltaNet", unavailable)
    with pytest.raises(PackedEncodersError, match="no compiler"):
        pe.pack(model, attention_backend="native")
    assert model.forward == original
    with pytest.warns(UserWarning, match="using torch"):
        pe.pack(model, engine=pe.Qwen35CPU(prepack_weights=False))
    assert pe.get_engine(model).state.gdn_backend == "torch"
    pe.unpack(model)
    assert model.forward == original


@torch.no_grad()
def test_epilogue_microbatches(model):
    from packed_encoders import epilogues
    ids, mask = inputs((5, 3, 9))
    pe.pack(model, engine=pe.Qwen35CPU(max_tokens=9))
    engine = pe.get_engine(model)
    projection = torch.nn.Linear(64, 11)
    class MetadataProjection:
        def prepare(self, ops):
            project = epilogues.TokenProjection(projection, normalize=True).prepare(ops)
            def forward(hidden, batch):
                assert batch.cu_seqlens.dtype == torch.int32
                assert batch.cu_seqlens.diff().tolist() == list(batch.host_lengths)
                assert batch.max_seqlen == max(batch.host_lengths)
                assert batch.position_ids.shape == batch.input_ids.shape
                return project(hidden, batch)
            return forward
    head = engine.with_epilogue(MetadataProjection())
    batch = pe.PackedBatch(ids[mask], host_lengths=(5, 3, 9))
    want = torch.nn.functional.normalize(projection(engine.forward_packed(batch)), dim=-1)
    torch.testing.assert_close(head.forward_packed(batch), want, atol=1e-6, rtol=1e-6)
    pe.unpack(model)
    with pytest.raises(PackedEncodersError, match="closed"):
        head.forward_packed(batch)


@torch.no_grad()
def test_topk_adapter_without_cuda_reference(model):
    from types import SimpleNamespace
    import copy
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    # Reproduce topk's packed-only backbone signature, including its noncausal
    # attention. The CPU engine must not invoke this CUDA-only forward at pack time.
    class PackedText(Qwen3_5TextModel):
        def forward(self, inputs_embeds, position_ids, *, cu_seqlens, seq_idx):
            raise AssertionError("CUDA-only topk backbone was called")
    expected_model = copy.deepcopy(model)
    expected_model.config.is_causal = False
    model.__class__ = PackedText
    model.document_causal = False
    class Topk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="topk_embed", output_dim=7, dim=11, normalize=True)
            self.model = torch.nn.Module()
            self.model.language_model = model
            self.head = torch.nn.Linear(64, 11, bias=False)
        def forward(self, *args, **kwargs):
            raise AssertionError("topk original forward was called")
    wrapper = Topk().eval()
    ids, mask = inputs((5, 3))
    expected = expected_model(ids, attention_mask=mask).last_hidden_state
    want = torch.nn.functional.normalize(wrapper.head(expected)[..., :7], dim=-1)
    pe.pack(wrapper)
    cu = torch.tensor([0, 5, 8], dtype=torch.int32)
    got = wrapper(ids, mask, ids[mask][None], torch.cat((torch.arange(5), torch.arange(3)))[None], cu,
                  torch.tensor([[0] * 5 + [1] * 3]))
    torch.testing.assert_close(got[mask], want[mask], atol=1e-5, rtol=2e-5)
    assert not got[~mask].any()
    pe.unpack(wrapper)


@torch.no_grad()
@pytest.mark.parametrize("variant", ["int8_dynamic", "int8_w8a8"])
def test_onnx_quantized_weight_reuse(model, tmp_path, variant):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from benchmarks.cpu.qwen35_onnx import export, quantize, reuse_quantized_weights
    template = export(model, tmp_path, 65)
    src = export(model, tmp_path, 7)
    qtemplate = tmp_path / f"{variant}_65.onnx"
    direct = tmp_path / f"{variant}_7_direct.onnx"
    reused = tmp_path / f"{variant}_7_reused.onnx"
    quantize(template, qtemplate, variant)
    quantize(src, direct, variant)
    reuse_quantized_weights(template, qtemplate, src, reused)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    ids, mask = inputs((7, 3))
    feed = {"input_ids": ids.numpy(), "attention_mask": mask.long().numpy()}
    a = ort.InferenceSession(str(direct), options, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    b = ort.InferenceSession(str(reused), options, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    assert (a == b).all()
