"""The native GPTQ column loop and the per-layer FP32 calibration Gram matrices reproduce the code
they replaced, bit for bit."""
import pytest
import torch

from packed_encoders.cpu import int8
from packed_encoders.errors import PackedEncodersError


@pytest.fixture(scope='module')
def library():
    if not (int8.sme2_available() or int8.vnni_available()):
        pytest.skip('Apple SME2 or x86 AVX512-VNNI required')
    try:
        return int8.library()[0]
    except int8.Int8Unavailable as exc:
        pytest.skip(str(exc))


def reference_gptq(weights, hessians, block=128, damp=0.01):
    """The PyTorch GPTQ before the native loop, over a [B, N, K] batch, as packing ran it; also
    returns its inverse-Hessian factors."""
    w = weights.float().clone()
    h = hessians.clone()
    k = w.shape[2]
    diagonal = torch.diagonal(h, dim1=1, dim2=2)
    dead = diagonal == 0
    diagonal[dead] = 1
    w.masked_fill_(dead[:, None, :], 0)
    h += damp * diagonal.mean(1)[:, None, None] * torch.eye(k, dtype=h.dtype)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True).float()
    scale = weights.abs().amax(2).float() / 127
    safe = torch.where(scale > 0, scale, 1)[:, :, None]
    scale3 = scale[:, :, None]
    q = torch.empty_like(w)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        w1, hinv1 = w[:, :, i1:i2].clone(), hinv[:, i1:i2, i1:i2]
        errors = torch.empty_like(w1)
        for i in range(i2 - i1):
            column = w1[:, :, i:i + 1]
            quantized = torch.round(column / safe).clamp_(-127, 127)
            q[:, :, i1 + i:i1 + i + 1] = quantized
            error = (column - quantized * scale3) / hinv1[:, i, i][:, None, None]
            w1[:, :, i:] -= error * hinv1[:, i:i + 1, i:]
            errors[:, :, i:i + 1] = error
        w[:, :, i2:] -= errors @ hinv[:, i1:i2, i2:]
    return q.to(torch.int8), scale, hinv


@pytest.mark.parametrize('n, k', [(40, 200), (72, 384), (136, 768)])
def test_native_gptq_matches_batched_pytorch(library, n, k):
    # A partial last block (K = 200), row groups that do not divide N, an all-zero weight row
    # (scale 0) and a dead input column (zero Hessian diagonal).
    generator = torch.Generator().manual_seed(n + k)
    weights = torch.randn(3, n, k, generator=generator) * 0.05
    weights[1, 5] = 0
    x = torch.randn(3, 2 * k, k, generator=generator, dtype=torch.float64)
    x[2, :, 7] = 0
    hessians = x.transpose(1, 2) @ x
    q, scale, hinv = reference_gptq(weights, hessians)
    assert (q[1, 5] == 0).all() and (q != 0).any()
    for b in range(3):
        factor = int8.inverse_factor(hessians[b])
        assert torch.equal(factor[0], hinv[b]) and torch.equal(factor[1], torch.diagonal(hessians[b]) == 0)
        for workers in (1, 3):
            got_q, got_scale = int8.gptq(library, weights[b], factor, workers)
            assert torch.equal(got_scale, scale[b])
            assert torch.equal(got_q, q[b]), (b, workers, int((got_q != q[b]).sum()))


def test_native_gptq_rejects_bad_blocks(library):
    w, hinv, scale, errors = torch.zeros(4, 8), torch.eye(8), torch.ones(4), torch.empty(4, 8)
    q = torch.empty(4, 8, dtype=torch.int8)
    pointers = (w.data_ptr(), hinv.data_ptr(), scale.data_ptr(), q.data_ptr(), errors.data_ptr())
    for args in ((0, 8, 0, 8, 1), (4, 8, 0, 9, 1), (4, 8, 4, 4, 1), (4, 8, 0, 8, 0)):
        assert library.pe_q8_gptq_block(*pointers, *args) == -1
    assert library.pe_q8_gptq_block(*pointers, 4, 8, 0, 8, 2) == 0


def test_calibration_hands_over_each_layer_as_it_completes():
    # Two layers of two projections. Each layer's FP32 Gram matrices arrive as soon as its last
    # projection has run, before the next layer's first, and equal the float64 sums packing
    # recorded before. Running a projection twice, or not at all, raises.
    generator = torch.Generator().manual_seed(5)
    layers = [(torch.empty(8, 16), torch.empty(8, 24)) for _ in range(2)]
    calls = [(weight, torch.randn(1, 30, weight.shape[1], generator=generator)) for layer in layers for weight in layer]
    model = torch.nn.Module()
    model.embeddings = torch.nn.Module()
    model.embeddings.tok_embeddings = torch.nn.Embedding(10, 4)
    events, got = [], {}

    class Engine:
        max_position = 64

        def __init__(self, calls):
            self.calls = calls

        def _core(self, ids, lengths, positions, linear):
            for n, (weight, x) in enumerate(self.calls):
                events.append(('run', n))
                linear(x, weight)

    def done(index, grams):
        events.append(('done', index))
        got[index] = list(grams)

    Engine.model = model
    int8._calibrate(Engine(calls), layers, done)
    assert events == [('run', 0), ('run', 1), ('done', 0), ('run', 2), ('run', 3), ('done', 1)]
    for n, (weight, x) in enumerate(calls):
        old = torch.zeros(weight.shape[1], weight.shape[1], dtype=torch.float64)
        old += (x[0].T @ x[0]).double()
        gram = got[n // 2][n % 2]
        assert gram.dtype == torch.float32 and torch.equal(gram.double(), old)
    with pytest.raises(PackedEncodersError, match='more than once'):
        int8._calibrate(Engine(calls + calls[:1]), layers, lambda *_: None)
    with pytest.raises(PackedEncodersError, match='every projection'):
        int8._calibrate(Engine(calls[:3]), layers, lambda *_: None)
