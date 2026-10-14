"""Small full-model CPU checks; no checkpoint downloads or CUDA packages."""
import platform
import sys

import pytest
import torch
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertModel

import packed_encoders as pe
from packed_encoders.errors import PackedEncodersError, ValidationError


def _rows_backend():
    # The backend whose CPUAttention carries native row kernels here: Apple's Accelerate
    # library, or rows.c on Linux x86-64 when gcc is available.
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "native"
    from packed_encoders.cpu.attention import CPUAttention
    return "auto" if CPUAttention("auto", 64, 64, 1).rows is not None else None


ROWS = _rows_backend()
native_rows = pytest.mark.skipif(ROWS is None, reason="native row kernels (Apple, or Linux x86-64 with gcc)")


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


@pytest.mark.parametrize("backend", ["sdpa", "auto", "native"])
@torch.no_grad()
def test_parity_and_owned_outputs(model, backend):
    if backend == "native" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        pytest.skip("Apple Accelerate")
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
    if backend == "native" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        pytest.skip("Apple Accelerate")
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
    model.layers[0].mlp.Wo.weight = torch.nn.Parameter(model.layers[0].mlp.Wo.weight.clone())
    with pytest.raises(PackedEncodersError, match="repack"):
        model(ids)


@pytest.mark.parametrize("change", ["parameter", "storage", "submodule", "forward_hook", "pre_hook", "config"])
@torch.no_grad()
def test_structural_changes_require_repack(model, change):
    ids, mask = inputs()
    pe.pack(model, attention_backend="sdpa")
    # Hooks on the patched root still run around the packed forward.
    calls = []
    model.register_forward_hook(lambda *args: calls.append(1))
    model(ids, attention_mask=mask)
    assert calls
    mlp = model.layers[1].mlp
    if change == "parameter":
        mlp.Wo.weight = torch.nn.Parameter(mlp.Wo.weight.clone())
    elif change == "storage":
        mlp.Wo.weight.data = mlp.Wo.weight.data.clone()
    elif change == "submodule":
        mlp.Wo = torch.nn.Linear(mlp.Wo.in_features, mlp.Wo.out_features, bias=False)
    elif change == "forward_hook":
        model.layers[1].attn.register_forward_hook(lambda *args: None)
    elif change == "pre_hook":
        model.layers[1].mlp_norm.register_forward_pre_hook(lambda *args: None)
    else:
        model.config.local_attention = 64
    with pytest.raises(PackedEncodersError, match="repack"):
        model(ids, attention_mask=mask)


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
        attention = CPUAttention(ROWS, 64, 64, threads)
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


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64", reason="Apple Accelerate")
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
        actual = attention.local(q, k, v, lengths, boundaries)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
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
        attention = CPUAttention(ROWS, 64, 64, threads)
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
        attention = CPUAttention(ROWS, 64, 64, threads)
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


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64", reason="Apple Accelerate")
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
            attention.close()


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64", reason="Apple Accelerate")
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


def _sme2():
    from packed_encoders.cpu.int8 import sme2_available
    return sme2_available()


sme2 = pytest.mark.skipif(not _sme2(), reason="Apple SME2")


def scale_projections(model, factor=4.):
    # Random-init branches are tiny next to the residual stream, which would hide int8
    # errors; scaled projections make them visible without leaving int8's working range.
    with torch.no_grad():
        for layer in model.layers:
            for linear in (layer.attn.Wqkv, layer.attn.Wo, layer.mlp.Wi, layer.mlp.Wo):
                linear.weight.mul_(factor)


def emulate_int8(engine, ids, lengths, positions):
    """The engine's FP32 loop with every projection replaced by the int8 scheme in float64:
    rotate, quantize per row, multiply the prepared integer weights, dequantize."""
    from packed_encoders.cpu.int8 import rotation
    q8, g = engine.int8, engine.int8.geometry
    hidden, width = rotation(g.hidden, g.block_hidden), rotation(g.width, g.block_width)
    rotations = (hidden, None, hidden, width)  # the context arrives rotated through Wv
    index = {id(weight): divmod(n, 4) for n, weight in enumerate(q8.weights)}

    def linear(x, weight, bias=None):
        layer, kind = index[id(weight)]
        packed, scales = q8.prepared[kind]
        n, k = weight.shape
        rows = x.reshape(-1, k).double()
        if rotations[kind] is not None:
            rows = rows @ rotations[kind]
        amax = rows.abs().amax(1, keepdim=True).clamp_min(1e-30)
        q = torch.round(rows * 127 / amax).clamp(-127, 127)
        w = packed[layer].transpose(-3, -2).reshape(n, k).double()
        return ((q @ w.T) * (amax / 127) * scales[layer].double()).float().reshape(*x.shape[:-1], n)

    return engine._core(ids, lengths, positions, linear=linear)


@torch.no_grad()
def test_precision_option(model, monkeypatch):
    import packed_encoders.cpu.int8 as int8
    ids, mask = inputs()
    reference = model(ids, attention_mask=mask).last_hidden_state[mask]
    with pytest.raises(PackedEncodersError, match="precision"):
        pe.pack(model, precision="int4")
    # Machines without SME2 (M1-M3, other CPUs) or a compiler for it warn and stay FP32.
    monkeypatch.setattr(int8, "sme2_available", lambda: False)
    with pytest.warns(UserWarning, match="SME2.*using exact FP32"):
        pe.pack(model, precision="int8", attention_backend="sdpa")
    engine = pe.get_engine(model)
    assert engine.state.precision == engine.state.validation_report.precision == "fp32"
    torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state[mask], reference, atol=2e-5, rtol=2e-5)
    pe.pack(model, precision="int8")  # repeating the request or the precision in effect is fine
    pe.pack(model, precision="fp32")
    pe.unpack(model)
    monkeypatch.setattr(int8, "sme2_available", lambda: True)
    monkeypatch.setattr(int8, "library", lambda: (_ for _ in ()).throw(int8.Int8Unavailable("no compiler")))
    with pytest.warns(UserWarning, match="no compiler; using exact FP32"):
        pe.pack(model, precision="int8", attention_backend="sdpa")
    pe.unpack(model)
    pe.pack(model, attention_backend="sdpa", precision="fp32")
    assert pe.get_engine(model).state.precision == "fp32"
    with pytest.raises(PackedEncodersError, match="unpack before changing the CPU precision"):
        pe.pack(model, precision="int8")


@sme2
@torch.no_grad()
def test_int8_kernels():
    from packed_encoders.cpu.int8 import library, pack_weights, rotation, signs
    lib, _ = library()
    rng = torch.Generator().manual_seed(3)
    # 7 and 16 rows take the single-block kernel; 45 rows leave an odd block for 2x2 tiles.
    for m in (7, 16, 45):
        n, k, blocks = 128, 192, (m + 15) // 16
        a = torch.randint(-127, 128, (m, k), generator=rng, dtype=torch.int8)
        w = torch.randint(-127, 128, (n, k), generator=rng, dtype=torch.int8)
        rows = torch.zeros(blocks * 16, k, dtype=torch.int8)
        rows[:m] = a
        packed, weights = rows.view(blocks, 16, k // 4, 4).transpose(1, 2).contiguous(), pack_weights(w)
        c = torch.empty(m, n, dtype=torch.int32)
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
        q = packed.view(3, k // 4, 16, 4).transpose(1, 2).reshape(48, k)
        assert not q[37:].any()
        rotated = x.double() @ rotation(k, block) if block else x.double()
        torch.testing.assert_close(scale.double(), rotated.abs().amax(1) / 127, rtol=1e-6, atol=0)
        expected = torch.round(rotated / torch.where(scale > 0, scale, 1).double()[:, None])
        difference = (q[:37].double() - expected).abs()
        assert difference.max() <= 1 and (difference == 0).double().mean() > .999


@sme2
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
    # int8 is active, close to FP32, and exactly the intended scheme up to FP32 rounding.
    assert 1e-3 < relative(out[mask], reference) < 2e-2
    assert relative(out[mask], emulate_int8(engine, ids[mask], (7, 129), mask.nonzero()[:, 1])) < 1e-5
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


@sme2
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
