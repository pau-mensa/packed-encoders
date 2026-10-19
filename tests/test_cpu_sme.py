"""SME2 FP32 GEMM numerics, bounded native accesses, and prepared-weight lifecycle."""
import ctypes
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

import packed_encoders as pe
from packed_encoders.cpu import sme
from packed_encoders.cpu.attention import CPUAttention
from transformers import ModernBertConfig, ModernBertModel


def test_hardware_gate_checks_capability_instead_of_chip_name(monkeypatch):
    calls = []

    def sysctl(name, value, size, new, new_size):
        calls.append(name)
        assert name == b"hw.optional.arm.FEAT_SME2"
        ctypes.cast(value, ctypes.POINTER(ctypes.c_int)).contents.value = 1
        return 0

    monkeypatch.setattr(sme.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sme.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(sme.ctypes, "CDLL", lambda _: SimpleNamespace(sysctlbyname=sysctl))
    assert sme.sme2_available()
    assert calls == [b"hw.optional.arm.FEAT_SME2"]


def test_unsupported_streaming_width_reports_accelerate_fallback(monkeypatch):
    lib = SimpleNamespace(pe_sme_fp32_available=lambda: 0)
    monkeypatch.setattr(sme.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(sme.ctypes, "CDLL", lambda _: lib)
    with pytest.raises(pe.PackedEncodersError, match="512-bit streaming vectors; using Accelerate"):
        sme.library.__wrapped__()


@pytest.fixture
def native():
    if not sme.sme2_available():
        pytest.skip("Apple SME2 required")
    try:
        lib, _ = sme.library()
    except pe.PackedEncodersError as exc:
        pytest.skip(str(exc))
    return lib


@torch.no_grad()
def test_fp32_tiles_workers_and_row_boundaries(native):
    rng = torch.Generator().manual_seed(713)
    for m, n, k in ((1, 64, 4), (7, 128, 68), (15, 192, 128), (16, 768, 1152),
                    (17, 128, 128), (31, 768, 768), (32, 2304, 768), (33, 128, 128),
                    (48, 768, 128), (64, 768, 768), (65, 128, 128), (128, 768, 1152), (256, 128, 128)):
        x, w = torch.randn(m, k, generator=rng), torch.randn(n, k, generator=rng) * .05
        projections = sme.FP32Projections([(w, None)])
        projections.min_rows = 1  # Exercise the native body even though runtime dispatch prefers Accelerate at M=1.
        projections.worker_limit = 8
        projections.small_worker_limit = 8
        projections.max_rows = 4096  # Cover the tiled native body beyond the conservative dispatch limit.
        reference = F.linear(x.double(), w.double())
        first = None
        for workers in (1, 2, 4, 8):
            out = projections.linear(x, w, None, workers).clone()
            assert (out.double() - reference).norm() / reference.norm() < 1e-6
            if first is None:
                first = out
            else:
                assert torch.equal(out, first)
            # One output row retains its accumulation order when separated from its batch.
            row = m // 2
            assert torch.equal(projections.linear(x[row:row + 1], w, None, workers)[0], first[row])
        projections.close()


@torch.no_grad()
def test_native_rejects_invalid_metadata_and_preserves_guards(native):
    fn = native.pe_sme_fp32_linear
    for m, n, k, workers in ((0, 64, 4, 1), (4097, 64, 4, 1), (1, 63, 4, 1), (1, 64, 5, 1), (1, 64, 4, 0)):
        assert fn(None, None, None, None, m, n, k, workers, 0) != 0
    for m in (1, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 127, 128, 256):
        k, n = 68, 128
        x, w = torch.randn(m, k), torch.randn(n, k)
        packed = sme.FP32Projections._pack(w)
        needed = ((m + 15) // 16) * 16 * k
        out = torch.full((m * n + 16,), 12345.)
        scratch = torch.full((needed + 16,), 12345.)
        args = (x.data_ptr(), packed.data_ptr(), out.data_ptr(), scratch.data_ptr(), m, n, k, 8)
        assert fn(*args, needed - 1) != 0
        assert (out == 12345.).all() and (scratch == 12345.).all()
        assert fn(*args, needed) == 0
        assert (out[m * n:] == 12345.).all() and (scratch[needed:] == 12345.).all()


@torch.no_grad()
def test_prepared_weights_refresh_and_unsupported_inputs_fall_back(native):
    w, x = torch.randn(128, 128), torch.randn(1, 16, 128)
    projections = sme.FP32Projections([(w, None)])
    first = projections.linear(x, w, None, 1).clone()
    packed = projections.weights[id(w)][1]
    w.mul_(.5)
    actual = projections.linear(x, w, None, 8)
    assert projections.weights[id(w)][1] is not packed
    torch.testing.assert_close(actual, first * .5, rtol=0, atol=0)
    for bad in (x[:, :1], torch.randn(1, 129, 128), x.double(), x.transpose(1, 2), torch.empty(1, 0, 128)):
        assert projections.linear(bad, w, None, 1) is None
    assert projections.linear(x, w, torch.zeros(128), 1) is None
    assert projections.linear(x, w.clone(), None, 1) is None
    with torch.inference_mode():
        no_version = w.clone()
    assert not sme.FP32Projections([(no_version, None)]).weights
    projections.close()
    assert not projections.weights and not projections.outputs and projections.scratch is None


def test_prepacking_option_and_unavailable_backend(monkeypatch):
    with pytest.raises(ValueError, match="prepack_weights"):
        pe.ModernBertCPU(prepack_weights=1)
    attention = CPUAttention("sdpa", 64, 64, 1)
    attention.prepare_sme([(torch.randn(128, 128), None)])
    assert attention.sme is None
    # This fallback is platform-independent and must never call SME on unsupported CPUs.
    attention.rows = object()
    monkeypatch.setattr(sme, "sme2_available", lambda: False)
    attention.prepare_sme([(torch.randn(128, 128), None)])
    assert attention.sme is None
    monkeypatch.setattr(sme, "sme2_available", lambda: True)
    def no_compiler():
        raise pe.PackedEncodersError("no compiler")

    monkeypatch.setattr(sme, "library", no_compiler)
    with pytest.warns(UserWarning, match="no compiler"):
        attention.prepare_sme([(torch.randn(128, 128), None)])
    assert attention.sme is None
    attention.close()


@torch.no_grad()
def test_model_short_long_updates_fallbacks_and_unpack(native):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(711)
        model = ModernBertModel(ModernBertConfig(vocab_size=64, hidden_size=128, intermediate_size=192,
                                               num_attention_heads=2, num_hidden_layers=3,
                                               max_position_embeddings=256, pad_token_id=0)).eval()
        original = model.forward
        storage = {k: v.data_ptr() for k, v in model.state_dict().items()}
        pe.pack(model, engine=pe.ModernBertCPU(threads=8), attention_backend="native")
        engine, saved = pe.get_engine(model), None
        assert engine.attention.sme is not None
        for length in (1, 7, 16, 17, 32, 33, 64, 65, 128, 129):
            ids = torch.arange(length)[None] % 61 + 1
            mask = torch.ones_like(ids, dtype=torch.bool)
            if length == 16:
                mask[:, 7:] = False
            ref = original(ids, attention_mask=mask).last_hidden_state
            out = model(ids, attention_mask=mask).last_hidden_state
            torch.testing.assert_close(out[mask], ref[mask], atol=2e-5, rtol=2e-5)
            if saved is None:
                saved, copy = out, out.clone()
            else:
                assert torch.equal(saved, copy)
        assert storage == {k: v.data_ptr() for k, v in model.state_dict().items()}
        ids = torch.arange(16)[None] + 1
        for weight in (model.layers[0].attn.Wqkv.weight, model.layers[0].mlp.Wo.weight):
            weight.mul_(.9)
        torch.testing.assert_close(model(ids).last_hidden_state, original(ids).last_hidden_state, atol=2e-5, rtol=2e-5)
        assert model(ids, output_hidden_states=True).hidden_states is not None
        cache = engine.attention.sme
        pe.unpack(model)
        assert not cache.weights and not cache.outputs and cache.scratch is None
        assert model.forward == original
        pe.pack(model, engine=pe.ModernBertCPU(prepack_weights=False))
        assert pe.get_engine(model).attention.sme is None
        pe.unpack(model)
        pe.pack(model, precision="int8")
        assert pe.get_engine(model).attention.sme is None
        pe.unpack(model)
    finally:
        torch.set_num_threads(old_threads)


@torch.no_grad()
def test_pack_validation_exercises_sme_and_rolls_back(native, monkeypatch):
    model = ModernBertModel(ModernBertConfig(vocab_size=64, hidden_size=128, intermediate_size=192,
                                           num_attention_heads=2, num_hidden_layers=3,
                                           max_position_embeddings=256, pad_token_id=0)).eval()
    original = model.forward
    linear = sme.FP32Projections.linear

    def broken(self, *args):
        out = linear(self, *args)
        return None if out is None else out + 1

    monkeypatch.setattr(sme.FP32Projections, "linear", broken)
    with pytest.raises(pe.ValidationError, match="validation failed"):
        pe.pack(model, attention_backend="native")
    assert model.forward == original


@torch.no_grad()
def test_thread_budget_selects_measured_dispatch_boundary(native):
    weight = torch.randn(128, 128)
    for threads, limit in ((1, 128), (2, 32), (3, 32), (4, 64), (7, 64), (8, 128)):
        projections = sme.FP32Projections([(weight, None)], threads)
        assert projections.max_rows == limit
        assert projections.linear(torch.randn(1, limit, 128), weight, None, threads) is not None
        assert projections.linear(torch.randn(1, limit + 1, 128), weight, None, threads) is None
        projections.close()
