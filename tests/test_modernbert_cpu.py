"""Small full-model CPU checks; no checkpoint downloads or CUDA packages."""
import sys

import pytest
import torch
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertModel

import packed_encoders as pe
from packed_encoders.errors import PackedEncodersError, ValidationError


def _native_kernels():
    # What this machine builds: Apple's Accelerate library, or the gcc library on Linux x86-64.
    from packed_encoders.cpu.attention import CPUAttention
    attention = CPUAttention("auto", 64, 64, 1)
    return attention.rows is not None, attention.library is not None, attention.native_linear


ROWS, NATIVE, LINEAR = _native_kernels()
native_rows = pytest.mark.skipif(not ROWS, reason="native row kernels (Apple, or Linux x86-64 with gcc)")
native_attention = pytest.mark.skipif(not NATIVE, reason="native attention (Apple, or Linux x86-64 with gcc)")
native_linear = pytest.mark.skipif(not LINEAR, reason="native FP32 projections (Linux x86-64 with gcc)")


@pytest.fixture
def model():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = ModernBertModel(ModernBertConfig(
            vocab_size=64, hidden_size=128, intermediate_size=192,
            num_attention_heads=2, num_hidden_layers=3, max_position_embeddings=256,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
        )).eval()
    yield model
    pe.unpack(model)
    torch.set_num_threads(threads)


def inputs():
    ids = torch.arange(2 * 129).reshape(2, 129) % 61 + 3
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[0, 7:] = False
    return ids, mask


@pytest.mark.parametrize('workers,expected', [(None, 1), (2, 2)])
def test_default_worker_budget_reaches_engine_without_resizing_torch(model, monkeypatch, workers, expected):
    from packed_encoders.cpu import topology

    monkeypatch.setattr(topology, 'default_threads', lambda: 1)
    before = torch.get_num_threads()
    pe.pack(model, engine=pe.ModernBertCPU(threads=workers))
    assert pe.get_engine(model).attention.threads == expected
    assert torch.get_num_threads() == before


@pytest.mark.parametrize("backend", ["sdpa", "auto", "native"])
@torch.no_grad()
def test_parity_and_owned_outputs(model, backend):
    if backend == "native" and not NATIVE:
        pytest.skip("native attention")
    ids, mask = inputs()
    original = model.forward
    storage = {k: v.data_ptr() for k, v in model.state_dict().items()}
    ref = model(ids, attention_mask=mask).last_hidden_state
    pe.pack(model, attention_backend=backend)
    packed = pe.get_engine(model)
    assert packed.state.validation_report.min_cosine > .99999
    out = model(ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(out[mask], ref[mask], atol=2e-5, rtol=2e-5)
    assert not out[~mask].any()
    saved = out.clone()
    scratch = packed.attention.scratch
    model(ids.flip(1), attention_mask=mask)
    torch.testing.assert_close(out, saved, atol=0, rtol=0)
    assert scratch is packed.attention.scratch
    assert storage == {k: v.data_ptr() for k, v in model.state_dict().items()}
    pe.unpack(model)
    assert model.forward == original
    assert packed.attention.scratch is None
    with pytest.raises(PackedEncodersError, match="closed"):
        packed.forward_packed(pe.PackedBatch(ids[mask], host_lengths=(7, 129)))


@torch.no_grad()
@pytest.mark.parametrize("backend", ["sdpa", "native"])
def test_direct_packed_microbatches_and_left_padding(model, backend):
    if backend == "native" and not NATIVE:
        pytest.skip("native attention")
    ids, mask = inputs()
    mask[0] = mask[0].flip(0)
    ref = model(ids, attention_mask=mask).last_hidden_state[mask]
    pe.pack(model, engine=pe.ModernBertCPU(max_tokens=129), attention_backend=backend)
    out = model(ids, attention_mask=mask, return_dict=False)[0]
    torch.testing.assert_close(out[mask], ref, atol=2e-5, rtol=2e-5)
    batch = pe.PackedBatch(ids[mask], host_lengths=(7, 129),
                           cu_seqlens=torch.tensor([0, 7, 136], dtype=torch.int32),
                           position_ids=mask.nonzero()[:, 1])
    torch.testing.assert_close(pe.get_engine(model).forward_packed(batch), ref, atol=2e-5, rtol=2e-5)
    with pytest.raises(PackedEncodersError, match="disagree"):
        pe.get_engine(model).forward_packed(pe.PackedBatch(ids[mask], host_lengths=(8, 128), cu_seqlens=batch.cu_seqlens))


def test_fallbacks_and_live_weight_updates(model):
    ids, mask = inputs()
    ids, mask = ids[:, :8], mask[:, :8]
    original = model.forward
    pe.pack(model, attention_backend="sdpa")
    # Grad-enabled inference and training retain HF autograd.
    model.train()
    model(ids, attention_mask=mask).last_hidden_state.square().sum().backward()
    assert model.embeddings.tok_embeddings.weight.grad is not None
    model.eval()
    with torch.no_grad():
        model.layers[0].mlp.Wo.weight.mul_(.9)
        ref = original(ids, attention_mask=mask).last_hidden_state
        out = model(ids, attention_mask=mask).last_hidden_state
        torch.testing.assert_close(out[mask], ref[mask], atol=2e-5, rtol=2e-5)
        hidden = model(ids, attention_mask=mask, output_hidden_states=True)
        assert hidden.hidden_states is not None
        mask[0, 2] = False
        torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state,
                                   original(ids, attention_mask=mask).last_hidden_state)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state,
                                       original(ids, attention_mask=mask).last_hidden_state)


@torch.no_grad()
def test_mask_paths(model, monkeypatch):
    ids = torch.arange(3 * 9).reshape(3, 9) % 61 + 3
    pe.pack(model)
    engine = pe.get_engine(model)
    delegated = []
    hf = engine.original_forward
    monkeypatch.setattr(engine, "original_forward", lambda *a, **k: delegated.append(1) or hf(*a, **k))
    # Unpadded and right-padded rows take the fast path, any other contiguous rows the
    # general one; both give the same bits.
    right = torch.arange(9)[None] < torch.tensor([[9], [4], [1]])
    for mask in (None, torch.ones(3, 9, dtype=torch.long), right, right.long(), right.flip(1), right[1:2]):
        rows = ids[:len(mask)] if mask is not None else ids
        outputs = []
        for fast in (True, False):
            engine.fast_masks = fast
            outputs.append(model(rows, attention_mask=mask).last_hidden_state)
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        assert mask is None or not outputs[0][~mask.bool()].any()
    assert not delegated
    # Values other than 0/1, and empty rows, delegate to HF on both paths.
    for mask in (right.long() * 2, right & torch.tensor([[False], [True], [True]])):
        for fast in (True, False):
            engine.fast_masks = fast
            model(ids, attention_mask=mask)
    assert len(delegated) == 4


@torch.no_grad()
def test_hf_inputs_take_the_packed_path(model, monkeypatch):
    # Float masks and int32 IDs, which HF accepts, stay on the engine and match HF.
    ids, mask = inputs()
    pe.pack(model)
    engine = pe.get_engine(model)
    delegated = []
    hf = engine.original_forward
    monkeypatch.setattr(engine, "original_forward", lambda *a, **k: delegated.append(1) or hf(*a, **k))
    for call_ids, call_mask in ((ids, mask.float()), (ids.int(), mask), (ids.int(), mask.flip(1).float())):
        real = call_mask.bool()
        expected = hf(call_ids.long(), attention_mask=real).last_hidden_state[real]
        out = model(call_ids, attention_mask=call_mask).last_hidden_state
        torch.testing.assert_close(out[real], expected, atol=2e-5, rtol=2e-5)
    assert not delegated


@pytest.mark.parametrize("task", ["sequence_classification", "masked_lm", "token_classification"])
@torch.no_grad()
def test_hf_task_heads_take_the_packed_path(model, monkeypatch, task):
    # HF heads call the backbone with position_ids=None and inputs_embeds=None.
    import transformers
    cls = {"sequence_classification": transformers.ModernBertForSequenceClassification,
           "masked_lm": transformers.ModernBertForMaskedLM,
           "token_classification": transformers.ModernBertForTokenClassification}[task]
    with torch.random.fork_rng():
        torch.manual_seed(42)
        head = cls(model.config).eval()
    ids, mask = inputs()
    ref = head(ids, attention_mask=mask).logits
    pe.pack(head)
    try:
        engine = pe.get_engine(head)
        delegated = []
        hf = engine.original_forward
        monkeypatch.setattr(engine, "original_forward", lambda *a, **k: delegated.append(1) or hf(*a, **k))
        out = head(ids, attention_mask=mask).logits
        assert not delegated
        if task != "sequence_classification":
            out, ref = out[mask], ref[mask]  # padded positions hold zeros, not HF's values
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)
    finally:
        pe.unpack(head)


def test_validate_options(model):
    # pe.validate's ModernBERT options; validating without packing leaves the weights in place.
    storage = [p.data_ptr() for p in model.parameters()]
    report = pe.validate(model, engine=pe.ModernBertCPU(rehome_weights=True), seq_lens=(5, 40))
    assert report.sequence_lengths[:2] == (5, 40) and report.min_cosine > .99999
    assert storage == [p.data_ptr() for p in model.parameters()]
    pe.pack(model)
    assert pe.validate(model, seq_lens=(3, 9, 256), cos_threshold=.999).sequence_lengths[:3] == (3, 9, 256)
    for lengths in ((), (0,), (257,), (4.0,)):
        with pytest.raises(PackedEncodersError, match="seq_lens"):
            pe.validate(model, seq_lens=lengths)
    with pytest.raises(ValidationError):
        pe.validate(model, cos_threshold=1.01)


@pytest.mark.parametrize("change", ["parameter", "storage", "dtype", "submodule", "forward_hook", "pre_hook",
                                    "config"])
@torch.no_grad()
def test_structural_changes_are_not_tracked(model, change):
    # Such changes need a repack. Until then the engine keeps computing with the tensors it
    # resolved at packing, so a swapped or freed storage never reaches native code.
    ids, mask = inputs()
    pe.pack(model)
    # Hooks on the patched root still run around the packed forward.
    calls = []
    model.register_forward_hook(lambda *args: calls.append(1))
    before = model(ids, attention_mask=mask).last_hidden_state
    assert calls
    layer = model.layers[1]
    if change == "parameter":
        layer.mlp.Wo.weight = torch.nn.Parameter(layer.mlp.Wo.weight * 2)
    elif change == "storage":
        layer.mlp.Wo.weight.data = layer.mlp.Wo.weight.data * 2
        layer.mlp_norm.weight.data = layer.mlp_norm.weight.data * 2
    elif change == "dtype":
        layer.mlp_norm.double()
    elif change == "submodule":
        layer.mlp.Wo = torch.nn.Linear(layer.mlp.Wo.in_features, layer.mlp.Wo.out_features, bias=False)
    elif change == "forward_hook":
        layer.attn.register_forward_hook(lambda *args: calls.append(2))
    elif change == "pre_hook":
        layer.mlp_norm.register_forward_pre_hook(lambda *args: calls.append(2))
    else:
        model.config.local_attention = 64
    torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state, before, atol=0, rtol=0)
    assert 2 not in calls


def test_failed_validation_restores_model(model, monkeypatch):
    from packed_encoders.arch.modernbert_cpu import PackedModernBertCPU
    original = model.forward
    model.train()
    core = PackedModernBertCPU._core
    monkeypatch.setattr(PackedModernBertCPU, "_core", lambda self, *args: core(self, *args) + 1)
    with pytest.raises(ValidationError):
        pe.pack(model, attention_backend="sdpa")
    assert model.forward == original
    assert all(m.training for m in model.modules())
    assert not hasattr(model, "_packed_encoders_installation")


def test_cpu_imports_do_not_load_cuda(model):
    # Run this module alone to also enforce the CPU-only import boundary.
    before = set(sys.modules)
    pe.pack(model, attention_backend="sdpa")
    added = set(sys.modules) - before
    assert not any(n.split('.')[0] in {"cutlass", "triton", "fla"} for n in added)


@native_rows
@torch.no_grad()
def test_fused_rope_layout():
    from packed_encoders.cpu.attention import CPUAttention
    rng = torch.Generator().manual_seed(17)
    # 67 tokens end in a partial native 32-token block, split across one or
    # three workers; 7 tokens clamp the workers to one block. Distinct angles
    # exercise both halves, repeated/nonmonotonic positions and several heads.
    for threads in (1, 3):
        attention = CPUAttention("auto", 64, 64, threads)
        for tokens in (67, 7):
            qkv = torch.randn(1, tokens, 3 * 3 * 64, generator=rng)
            positions = torch.randint(0, 129, (tokens,), generator=rng)
            angles = torch.randn(129, 64, generator=rng)[positions][None, None]
            cos, sin = angles.cos(), angles.sin()
            attention.fused_rope = False
            expected = attention.rope_layout(qkv, cos, sin, 3, 64)
            attention.fused_rope = True
            actual = attention.rope_layout(qkv, cos, sin, 3, 64)
            for a, b in zip(actual, expected):
                assert a.is_contiguous()
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        attention.close()
        assert attention.rope_buffer is None


@native_attention
@torch.no_grad()
def test_native_local_attention():
    from packed_encoders.cpu.attention import CPUAttention
    rng = torch.Generator().manual_seed(31)
    # Lengths straddle the +/-64 window (65/66), the 64-row tile and several
    # tiles with clipped windows on both sides, packed with short sequences.
    lengths = (7, 65, 66, 1, 130, 200)
    q, k, v = [torch.randn(1, 3, sum(lengths), 64, generator=rng) * 2 for _ in range(3)]
    boundaries = torch.tensor(lengths, dtype=torch.int32)
    for threads in (1, 3):
        attention = CPUAttention("native", 64, 64, threads)
        expected = attention._sdpa(q, k, v, lengths, local=True)
        actual = attention.local(q, k, v, lengths, boundaries).clone()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        # Token-major context is the same values stored for the next projection.
        token_major = attention.local(q, k, v, lengths, boundaries, token_major=True)
        assert token_major is attention.output and token_major.shape == (1, sum(lengths), 3 * 64)
        torch.testing.assert_close(token_major, actual.transpose(1, 2).flatten(2), atol=0, rtol=0)
        attention.close()


@native_rows
@torch.no_grad()
def test_fused_geglu():
    from packed_encoders.cpu.attention import CPUAttention
    rng = torch.Generator().manual_seed(23)
    # Odd width leaves a vector remainder; 67 rows end in a partial 32-row block
    # split across workers. Edge activations cover zeros, tiny values and both
    # saturated erf tails. Native and PyTorch GELU must meet the same FP64 bound.
    edges = torch.tensor([0., -0., 1e-30, -1e-30, 4 * 2 ** .5, -4 * 2 ** .5, 6., -6., 30., -30.])
    for threads in (1, 3):
        attention = CPUAttention("auto", 64, 64, threads)
        for rows, width in ((67, 37), (7, 1152)):
            hidden = torch.randn(1, rows, 2 * width, generator=rng) * 3
            hidden.view(-1)[:len(edges)] = edges
            a, gate = hidden.double().chunk(2, -1)
            expected = (0.5 * a * (1 + torch.erf(a / 2 ** .5)) * gate).float()
            attention.fused_geglu = False
            torch.testing.assert_close(attention.geglu(hidden), expected, atol=2e-5, rtol=1e-6)
            attention.fused_geglu = True
            actual = attention.geglu(hidden)
            assert actual.shape == (1, rows, width) and actual.is_contiguous()
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-6)
        attention.close()
        assert attention.geglu_buffer is None


@native_rows
@torch.no_grad()
def test_fused_add_norm():
    from packed_encoders.cpu.attention import CPUAttention
    rng = torch.Generator().manual_seed(31)
    # 67 rows end in a partial 32-row block split across workers. Rows carry
    # large offsets and an outlier channel, as ModernBERT's residual stream does;
    # one row cancels to zero and one is constant (zero variance).
    for threads in (1, 3):
        attention = CPUAttention("auto", 64, 64, threads)
        for rows, width, bias in ((67, 64, True), (7, 768, False)):
            norm = torch.nn.LayerNorm(width, bias=bias)
            norm.weight.copy_(torch.randn(width, generator=rng) * 0.2 + 1)
            if bias:
                norm.bias.copy_(torch.randn(width, generator=rng) * 0.1)
            x = torch.randn(1, rows, width, generator=rng) * 2 + torch.randn(width, generator=rng) * 5
            x[..., 3] += 300
            branch = torch.randn(1, rows, width, generator=rng)
            x[0, 1], x[0, 2], branch[0, 2] = -branch[0, 1], 7, 0
            stream = x + branch
            expected = F.layer_norm(stream.double(), (width,), norm.weight.double(),
                                    norm.bias.double() if bias else None, norm.eps).float()
            for fused in (False, True):
                attention.fused_norm = fused
                inputs = x.clone()
                actual_stream, actual = attention.add_norm(inputs, branch, norm)
                assert torch.equal(actual_stream, stream) and (actual_stream is inputs) == fused
                assert (actual is attention.norm_buffer) == fused and actual.shape == x.shape
                torch.testing.assert_close(actual, expected, atol=5e-7, rtol=1e-6)
            owned = attention.add_norm(x.clone(), branch, norm, owned=True)[1]
            assert owned is not attention.norm_buffer and torch.equal(owned, actual)
            identity = attention.add_norm(x, branch, torch.nn.Identity())
            assert torch.equal(identity[0], stream) and identity[1] is identity[0]
        attention.close()
        assert attention.norm_buffer is None


@native_linear
@torch.no_grad()
def test_native_linear():
    from packed_encoders.cpu.attention import CPUAttention
    rng = torch.Generator().manual_seed(37)
    # Remainders in every dimension: activation rows (3/4-row tiles), weight rows (8/6-row blocks split
    # across workers) and K (16-float steps, masked tail). Every output must be the same bits whatever
    # the thread count or the batch its row shares (one accumulation order and reduction tree).
    attentions = [CPUAttention("auto", 64, 64, threads) for threads in (1, 3, 8)]
    for rows, n, k, bias in ((0, 8, 16, False), (1, 1, 1, True), (7, 9, 17, False), (19, 37, 33, True),
                             (5, 768, 1152, False), (32, 2304, 768, True), (16, 23, 768, False),
                             (7, 13, 1152, True), (32, 768, 1152, False), (256, 17, 1152, True)):
        x = torch.randn(1, rows, k, generator=rng)
        w = torch.randn(n, k, generator=rng) * 0.05
        b = torch.randn(n, generator=rng) if bias else None
        outs = [attention.linear(x, w, b).clone() for attention in attentions]
        assert outs[0].shape == (1, rows, n)
        assert all(torch.equal(outs[0], out) for out in outs[1:])
        if rows:
            expected = F.linear(x.double(), w.double(), None if b is None else b.double())
            assert (outs[0].double() - expected).norm() / expected.norm() < 1e-6
        for row in range(rows):
            assert torch.equal(attentions[1].linear(x[:, row:row + 1].contiguous(), w, b)[0, 0], outs[0][0, row])
    attention = attentions[0]
    # One borrowed buffer per output width; inputs that are not contiguous FP32 take F.linear.
    x, w = torch.randn(1, 6, 40, generator=rng), torch.randn(24, 40, generator=rng)
    first = attention.linear(x, w)
    assert attention.linear(x + 1, w) is first is attention.linear_buffers[24]
    assert torch.equal(attention.linear(x.transpose(1, 2), w[:, :6]), F.linear(x.transpose(1, 2), w[:, :6]))
    assert torch.equal(attention.linear(x.double(), w.double()), F.linear(x.double(), w.double()))
    with pytest.raises(RuntimeError):  # mismatched K reaches F.linear's error, never native code
        attention.linear(torch.randn(1, 6, 80), w)
    attention.native_linear = False
    assert torch.equal(attention.linear(x, w), F.linear(x, w))
    for attention in attentions:
        attention.close()
        assert attention.linear_buffers == {}


@native_linear
@pytest.mark.parametrize("backend", ["native", "auto"])
@torch.no_grad()
def test_avx2_build_keeps_native_rows_and_attention(model, monkeypatch, backend):
    monkeypatch.setenv('PACKED_ENCODERS_CPU_CACHE_DIR', '')
    import packed_encoders.cpu.attention as cpu

    # Compile and execute the AVX2 target on this AVX-512 host. The optional GEMM
    # must not disable the existing row/attention kernels when its ISA is absent.
    run = cpu.subprocess.run

    def avx2(command, **kwargs):
        command = ["-march=x86-64-v3" if arg == "-march=native" else arg for arg in command]
        return run(command, **kwargs)

    monkeypatch.setattr(cpu.subprocess, "run", avx2)
    library, directory = cpu._x86_library.__wrapped__()
    monkeypatch.setattr(cpu, "_x86_library", lambda: (library, directory))
    assert not hasattr(library, "pe_linear")
    pe.pack(model, attention_backend=backend)  # also validates full-model FP32 parity
    attention = pe.get_engine(model).attention
    assert attention.rows is library and attention.library is library
    assert attention.fused_rope and attention.fused_geglu and attention.fused_norm
    assert not attention.native_linear
    assert bool(attention.attention_tuning) == (backend == "auto")
    # The short attention specializations also work when v16 uses two AVX2 halves.
    rng = torch.Generator().manual_seed(43)
    lengths = (16, 32)
    q, k, v = [torch.randn(1, 2, sum(lengths), 64, generator=rng) for _ in range(3)]
    boundaries = torch.tensor(lengths, dtype=torch.int32)
    expected = attention._sdpa(q, k, v, lengths, local=False)
    actual = attention.global_(q, k, v, lengths, boundaries).clone()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    attention.tile = 4
    general = attention.global_(q, k, v, lengths, boundaries)
    assert torch.equal(actual, general)


@native_linear
@torch.no_grad()
def test_pack_time_gemm_tuning(model):
    with pytest.raises(ValueError, match="tune_gemm"):
        pe.ModernBertCPU(tune_gemm=1)
    ids, mask = inputs()
    ids, mask = ids[:, :16], mask[:, :16]
    expected = model(ids, attention_mask=mask).last_hidden_state
    pe.pack(model, engine=pe.ModernBertCPU(threads=1, max_tokens=32, tune_gemm=True))
    attention = pe.get_engine(model).attention
    assert attention.linear_tuning and {key[0] for key in attention.linear_tuning} == {16, 32}
    torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state[mask], expected[mask],
                               atol=2e-5, rtol=2e-5)
    pe.unpack(model)
    assert not attention.linear_tiles and not attention.linear_tuning
    pe.pack(model, engine=pe.ModernBertCPU(tune_gemm=True), attention_backend="sdpa")
    assert not pe.get_engine(model).attention.linear_tuning


@native_linear
@torch.no_grad()
def test_native_linear_reads_live_weights(model):
    ids, mask = inputs()
    original = model.forward
    pe.pack(model)
    assert pe.get_engine(model).attention.native_linear
    for layer in model.layers:
        layer.attn.Wqkv.weight.mul_(1.05)
        layer.mlp.Wo.weight.mul_(.9)
    torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state[mask],
                               original(ids, attention_mask=mask).last_hidden_state[mask], atol=2e-5, rtol=2e-5)


@torch.no_grad()
def test_rehome_weights(model, tmp_path):
    from packed_encoders.cpu.memory import rehome
    with pytest.raises(ValueError):
        pe.ModernBertCPU(rehome_weights=1)
    ids, mask = inputs()
    expected = model(ids, attention_mask=mask).last_hidden_state
    projections = [p for layer in model.layers for p in (layer.attn.Wqkv.weight, layer.attn.Wo.weight,
                                                         layer.mlp.Wi.weight, layer.mlp.Wo.weight)]
    before = [(p, p.data_ptr(), p.detach().clone()) for p in projections]
    others = {name: p.data_ptr() for name, p in model.named_parameters() if not any(p is q for q in projections)}
    pe.pack(model, engine=pe.ModernBertCPU(rehome_weights=True))
    for (param, pointer, values), now in zip(before, projections):
        assert now is param and param.data_ptr() != pointer and torch.equal(param, values)
    assert all(p.data_ptr() == others[name] for name, p in model.named_parameters() if name in others)
    torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state[mask], expected[mask],
                               atol=2e-5, rtol=2e-5)
    model.layers[0].mlp.Wo.weight.mul_(.9)  # in-place updates reach the moved storage
    assert torch.equal(pe.get_engine(model).plan[0][4][0], model.layers[0].mlp.Wo.weight)
    # A private file mapping, as from_pretrained leaves safetensors weights: the pages left
    # behind are paged out, which must not change what the old storage reads.
    path = tmp_path / "weights.bin"
    data = torch.randn(3 * 4096)
    data.numpy().tofile(path)
    mapped = torch.from_file(str(path), shared=False, size=data.numel(), dtype=torch.float32)
    param = torch.nn.Parameter(mapped, requires_grad=False)
    rehome([param])
    assert param.data_ptr() != mapped.data_ptr() and torch.equal(param, data) and torch.equal(mapped, data)


@native_attention
@torch.no_grad()
def test_native_global_attention():
    from packed_encoders.cpu.attention import CPUAttention
    rng = torch.Generator().manual_seed(29)
    # Batches up to 128 tokens (crossing the 64-row tile) use the native kernel
    # with a full window; any longer sequence keeps per-sequence SDPA.
    for lengths in ((17, 1, 8, 3, 65, 2, 128), (7,), (129, 1, 2)):
        q, k, v = [torch.randn(1, 3, sum(lengths), 64, generator=rng) * 2 for _ in range(3)]
        boundaries = torch.tensor(lengths, dtype=torch.int32)
        for threads in (1, 3):
            attention = CPUAttention("native", 64, 64, threads)
            expected = attention._sdpa(q, k, v, lengths, local=False)
            actual = attention.global_(q, k, v, lengths, boundaries)
            assert (actual is attention.output) == (max(lengths) <= 128)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            actual = actual.clone()
            for direct in (True, False):  # native token-major stores, or a transposed copy
                attention.token_major = direct
                token_major = attention.global_(q, k, v, lengths, boundaries, token_major=True)
                assert token_major.shape == (1, sum(lengths), 3 * 64) and token_major.is_contiguous()
                assert (token_major is attention.output) == (direct and max(lengths) <= 128)
                torch.testing.assert_close(token_major, actual.transpose(1, 2).flatten(2), atol=0, rtol=0)
            attention.close()


@native_attention
@torch.no_grad()
def test_native_global_full_model(model):
    lengths = torch.tensor([31, 1, 65, 7, 17])
    ids = torch.arange(5 * 65).reshape(5, 65) % 61 + 3
    mask = torch.arange(65)[None] < lengths[:, None]
    mask[0] = mask[0].flip(0)
    reference = model(ids, attention_mask=mask).last_hidden_state[mask]
    pe.pack(model, attention_backend="native")
    actual = model(ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(actual[mask], reference, atol=2e-5, rtol=2e-5)
    assert not actual[~mask].any()


def _int8():
    from packed_encoders.cpu.int8 import sme2_available, vnni_available
    return sme2_available() or vnni_available()


int8_kernels = pytest.mark.skipif(not _int8(), reason="Apple SME2 or x86 AVX512-VNNI")


def scale_projections(model, factor=4.):
    # Random-init branches are tiny next to the residual stream, which would hide int8
    # errors; scaled projections make them visible without leaving int8's working range.
    with torch.no_grad():
        for layer in model.layers:
            for linear in (layer.attn.Wqkv, layer.attn.Wo, layer.mlp.Wi, layer.mlp.Wo):
                linear.weight.mul_(factor)


def emulate_int8(engine, ids, lengths, positions, kernel_quantize=False):
    """The engine's FP32 loop with every projection replaced by the int8 scheme in float64:
    rotate, quantize per row, multiply the prepared integer weights, dequantize. With
    kernel_quantize, the kernel's own FP32 rotate + quantize and FP32 dequantization replace the
    float64 ones, so only the pipeline around them is emulated: the results should be exact.
    Otherwise FP32 rounding flips a few int8 values (each moving the output ~1e-3)."""
    from packed_encoders.cpu.int8 import rotation, vnni_available
    q8, g = engine.int8, engine.int8.geometry
    hidden, width = rotation(g.hidden, g.block_hidden), rotation(g.width, g.block_width)
    rotations = (hidden, None, hidden, width)  # the context arrives rotated through Wv
    signs = ((q8.signs[0], g.block_hidden), (None, 0), (q8.signs[0], g.block_hidden), (q8.signs[1], g.block_width))
    index = {id(weight): divmod(n, 4) for n, weight in enumerate(q8.weights)}

    def quantize(rows, kind):
        (s, block), (m, k) = signs[kind], rows.shape
        blocks = (m + 15) // 16
        out, scale, tmp = torch.empty(blocks * 16 * k, dtype=torch.int8), torch.empty(m), torch.empty(k)
        q8.library.pe_q8_quantize(rows.data_ptr(), m, k, None if s is None else s.data_ptr(), block,
                                  tmp.data_ptr(), out.data_ptr(), scale.data_ptr())
        if vnni_available():  # u8 rows
            return out.view(torch.uint8)[:m * k].view(m, k).double() - 128, scale
        return out.view(blocks, k // 4, 16, 4).transpose(1, 2).reshape(blocks * 16, k)[:m].double(), scale

    def linear(x, weight, bias=None):
        layer, kind = index[id(weight)]
        packed, scales, _ = q8.prepared[kind]
        n, k = weight.shape
        w = packed[layer].transpose(-3, -2).reshape(n, k).double()
        if kernel_quantize:
            q, scale = quantize(x.reshape(-1, k).contiguous(), kind)
            return ((q @ w.T).float() * (scales[layer] * scale[:, None])).reshape(*x.shape[:-1], n)
        rows = x.reshape(-1, k).double()
        if rotations[kind] is not None:
            rows = rows @ rotations[kind]
        amax = rows.abs().amax(1, keepdim=True).clamp_min(1e-30)
        q = torch.round(rows * 127 / amax).clamp(-127, 127)
        return ((q @ w.T) * (amax / 127) * scales[layer].double()).float().reshape(*x.shape[:-1], n)

    return engine._core(ids, lengths, positions, linear=linear)


@torch.no_grad()
def test_precision_option(model, monkeypatch):
    import packed_encoders.cpu.int8 as int8
    ids, mask = inputs()
    reference = model(ids, attention_mask=mask).last_hidden_state[mask]
    with pytest.raises(PackedEncodersError, match="precision"):
        pe.pack(model, precision="int4")
    # Machines without SME2 (M1-M3) or AVX512-VNNI, or a compiler for them, warn and stay FP32.
    monkeypatch.setattr(int8, "sme2_available", lambda: False)
    monkeypatch.setattr(int8, "vnni_available", lambda: False)
    with pytest.warns(UserWarning, match="SME2.*using exact FP32"):
        pe.pack(model, precision="int8", attention_backend="sdpa")
    engine = pe.get_engine(model)
    assert engine.state.precision == engine.state.validation_report.precision == "fp32"
    torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state[mask], reference, atol=2e-5, rtol=2e-5)
    pe.pack(model, precision="int8")  # repeating the request or the precision in effect is fine
    pe.pack(model, precision="fp32")
    pe.unpack(model)
    monkeypatch.setattr(int8, "sme2_available", lambda: True)
    monkeypatch.setattr(int8, "vnni_available", lambda: False)
    monkeypatch.setattr(int8, "library", lambda: (_ for _ in ()).throw(int8.Int8Unavailable("no compiler")))
    with pytest.warns(UserWarning, match="no compiler; using exact FP32"):
        pe.pack(model, precision="int8", attention_backend="sdpa")
    pe.unpack(model)
    pe.pack(model, attention_backend="sdpa", precision="fp32")
    assert pe.get_engine(model).state.precision == "fp32"
    with pytest.raises(PackedEncodersError, match="unpack before changing the CPU precision"):
        pe.pack(model, precision="int8")


@int8_kernels
@torch.no_grad()
def test_int8_kernels():
    from packed_encoders.cpu.int8 import library, pack_weights, rotation, signs, vnni_available
    lib, _ = library()
    x86 = vnni_available()
    rng = torch.Generator().manual_seed(3)
    # SME: 7 and 16 rows take the single-block kernel; 45 rows leave an odd block for 2x2 tiles.
    # VNNI: 7 and 45 rows leave partial register blocks, and 8 vectors a narrower last panel.
    for m in (7, 16, 45):
        n, k, blocks = 128, 192, (m + 15) // 16
        a = torch.randint(-127, 128, (m, k), generator=rng, dtype=torch.int8)
        w = torch.randint(-127, 128, (n, k), generator=rng, dtype=torch.int8)
        c = torch.empty(m, n, dtype=torch.int32)
        if x86:  # u8 = a + 128 rows: the products carry 128 * sum_k w
            rows, weights = (a.int() + 128).to(torch.uint8), pack_weights(w, 16)
            lib.pe_q8_gemm(rows.data_ptr(), weights.data_ptr(), c.data_ptr(), m, n, k)
            c -= 128 * w.int().sum(1)
        else:
            rows = torch.zeros(blocks * 16, k, dtype=torch.int8)
            rows[:m] = a
            packed, weights = rows.view(blocks, 16, k // 4, 4).transpose(1, 2).contiguous(), pack_weights(w)
            lib.pe_q8_gemm(packed.data_ptr(), weights.data_ptr(), c.data_ptr(), m, n, k)
        assert torch.equal(c.long(), a.long() @ w.long().T)
    # Per-row quantization after 128- and 64-point rotations, or none; a zero row stays zero.
    for k, block in ((256, 128), (192, 64), (192, 0)):
        x = torch.randn(37, k, generator=rng) * torch.rand(k, generator=rng) * 4
        x[3] = 0
        s = signs(k, block).float() if block else None
        packed, scale, scratch = torch.full((48 * k,), 9, dtype=torch.int8), torch.empty(37), torch.empty(k)
        lib.pe_q8_quantize(x.data_ptr(), 37, k, None if s is None else s.data_ptr(), block,
                           scratch.data_ptr(), packed.data_ptr(), scale.data_ptr())
        if x86:  # u8 rows
            q = packed.view(torch.uint8)[:37 * k].view(37, k).int() - 128
        else:
            q = packed.view(3, k // 4, 16, 4).transpose(1, 2).reshape(48, k)
            assert not q[37:].any()
        rotated = x.double() @ rotation(k, block) if block else x.double()
        torch.testing.assert_close(scale.double(), rotated.abs().amax(1) / 127, rtol=1e-6, atol=0)
        expected = torch.round(rotated / torch.where(scale > 0, scale, 1).double()[:, None])
        difference = (q[:37].double() - expected).abs()
        assert difference.max() <= 1 and (difference == 0).double().mean() > .999


@int8_kernels
@torch.no_grad()
def test_int8_matches_its_emulation(model):
    scale_projections(model)
    ids, mask = inputs()
    reference = model(ids, attention_mask=mask).last_hidden_state[mask]
    pe.pack(model, precision="int8", attention_backend="native")
    engine = pe.get_engine(model)
    assert engine.state.precision == engine.state.validation_report.precision == "int8"
    out = model(ids, attention_mask=mask).last_hidden_state
    assert not out[~mask].any()
    relative = lambda a, b: float((a - b).norm() / b.norm())
    # int8 is active and close to FP32; the pipeline around the kernel's rotate + quantize is
    # exact; and the scheme is the intended float64 one up to int8 values flipped by FP32 rounding.
    assert 1e-3 < relative(out[mask], reference) < 2e-2
    positions = mask.nonzero()[:, 1]
    assert relative(out[mask], emulate_int8(engine, ids[mask], (7, 129), positions, kernel_quantize=True)) < 1e-6
    assert relative(out[mask], emulate_int8(engine, ids[mask], (7, 129), positions)) < 5e-3
    saved = out.clone()
    model(ids.flip(1), attention_mask=mask)
    torch.testing.assert_close(out, saved, atol=0, rtol=0)
    batch = pe.PackedBatch(ids[mask], host_lengths=(7, 129))
    torch.testing.assert_close(engine.forward_packed(batch), out[mask], atol=0, rtol=0)
    # One worker and microbatches at sequence boundaries; repacking reproduces the weights.
    pe.unpack(model)
    assert engine.int8 is None
    pe.pack(model, engine=pe.ModernBertCPU(threads=1), precision="int8", attention_backend="native")
    torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state, out, atol=0, rtol=0)
    pe.unpack(model)
    pe.pack(model, engine=pe.ModernBertCPU(max_tokens=129), precision="int8", attention_backend="native")
    torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state[mask], out[mask], atol=2e-5, rtol=2e-5)


@int8_kernels
@pytest.mark.parametrize("threads", [1, 8])
@torch.no_grad()
def test_int8_prepared_attention_reuses_current_buffers(model, monkeypatch, threads):
    from packed_encoders.cpu.int8 import vnni_available
    if not vnni_available():
        pytest.skip("x86 native attention call preparation")
    scale_projections(model)
    pe.pack(model, engine=pe.ModernBertCPU(threads=threads), precision="int8", attention_backend="native")
    engine = pe.get_engine(model)
    att = engine.attention
    prepare = att._native_args
    calls = []

    def record(*args, **kwargs):
        result = prepare(*args, **kwargs)
        calls.append(result[1])
        return result

    monkeypatch.setattr(att, "_native_args", record)
    # Grow, shrink, cross the SDPA cutoff, then reuse larger allocations. 128-token
    # sequences exercise distinct local/global windows sharing the same scratch.
    saved = None
    for lengths in ((16,), (32,), (32,) * 8, (3, 17, 32), (128,), (129,), (16,)):
        ids = torch.arange(sum(lengths)) % 61 + 3
        positions = torch.cat([torch.arange(n) for n in lengths])
        calls.clear()
        actual = engine.forward_packed(pe.PackedBatch(ids, host_lengths=lengths)).clone()
        if max(lengths) <= 128:
            assert len(calls) == 2  # once per window kind, rather than per layer
            assert calls[0][3] == calls[1][3] == att.output.data_ptr()
            assert calls[0][-2] == calls[1][-2] == att.scratch.data_ptr()
        expected = emulate_int8(engine, ids, lengths, positions, kernel_quantize=True)
        assert float((actual - expected).norm() / expected.norm()) < 1e-6
        if lengths == (16,):
            if saved is None:
                saved = actual
            else:
                assert torch.equal(actual, saved)

    # Disabling token-major native output must retain the layout-conversion path.
    ids = torch.arange(32) % 61 + 3
    batch = pe.PackedBatch(ids, host_lengths=(32,))
    expected = engine.forward_packed(batch).clone()
    att.token_major = False
    actual = engine.forward_packed(batch)
    assert torch.equal(actual, expected)


@int8_kernels
@torch.no_grad()
def test_int8_attention_respects_tuned_sdpa_and_mixed_choices(model):
    scale_projections(model)
    pe.pack(model, engine=pe.ModernBertCPU(threads=1), precision="int8", attention_backend="native")
    engine = pe.get_engine(model)
    att = engine.attention
    ids = torch.arange(32) % 61 + 3
    batch = pe.PackedBatch(ids, host_lengths=(32,))
    expected = engine.forward_packed(batch).clone()
    for local, global_ in (("sdpa", "sdpa"), ("native", "sdpa"), ("sdpa", "native")):
        att.attention_tuning = {(True, 32, 1): {"selected": local},
                                (False, 32, 1): {"selected": global_}}
        torch.testing.assert_close(engine.forward_packed(batch), expected, atol=2e-5, rtol=2e-5)


@int8_kernels
def test_int8_fallbacks_and_weight_updates(model):
    ids, mask = inputs()
    ids, mask = ids[:, :8], mask[:, :8]
    original = model.forward
    pe.pack(model, precision="int8", attention_backend="sdpa")
    # Training and grad-enabled calls run the original FP32 forward.
    model.train()
    out = model(ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(out, original(ids, attention_mask=mask).last_hidden_state, atol=0, rtol=0)
    model.eval()
    with torch.no_grad():
        model(ids, attention_mask=mask)
        # Prepared int8 weights would go stale: in-place updates require a repack.
        model.layers[0].mlp.Wo.weight.mul_(.9)
        with pytest.raises(PackedEncodersError, match="repack"):
            model(ids, attention_mask=mask)
