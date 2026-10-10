"""Small full-model CPU checks; no checkpoint downloads or CUDA packages."""
import platform
import sys

import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

import packed_encoders as pe
from packed_encoders.errors import PackedEncodersError, ValidationError


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


@pytest.mark.parametrize("backend", ["sdpa", "native"])
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


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64", reason="Apple Accelerate")
@torch.no_grad()
def test_fused_rope_layout():
    from packed_encoders.cpu.attention import CPUAttention
    attention = CPUAttention("native", 64, 64, 2)
    rng = torch.Generator().manual_seed(17)
    # Odd token count crosses the native 64-token work tile; distinct angles
    # exercise both halves, repeated/nonmonotonic positions and several heads.
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
