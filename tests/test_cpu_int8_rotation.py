"""FP32 butterfly order and exact activation quantization on the x86 int8 path, and the
pack-time block rotations."""
import pytest
import torch

from packed_encoders.cpu.int8 import library, rotate, rotation, signs, vnni_available


@pytest.fixture(scope='module')
def quantize():
    if not vnni_available():
        pytest.skip('x86 AVX512-VNNI required')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    lib, _ = library()
    yield lib.pe_q8_quantize
    torch.set_num_threads(threads)


@pytest.mark.parametrize('block', [0, 64, 128])
@torch.no_grad()
def test_rotation_and_quantization_match_ordered_fp32(quantize, block):
    rng = torch.Generator().manual_seed(317)
    for k in (block or 64, 768, 1152):
        s = signs(k, block).float() if block else None
        for m in (1, 7, 16, 32, 256):
            x = torch.randn(m, k, generator=rng) * torch.logspace(-15, 15, m)[:, None]
            x[::5].zero_()
            x[1::5, k // 2] *= 1000
            rotated = x.clone()
            if block:
                rotated *= s
                # Independent scalar butterfly layout, descending h as in the
                # original kernel. Keep FP32 rounding after every add/subtract.
                h = block // 2
                while h:
                    pairs = rotated.view(m, -1, 2, h)
                    a, b = pairs[:, :, 0].clone(), pairs[:, :, 1].clone()
                    pairs[:, :, 0], pairs[:, :, 1] = a + b, a - b
                    h //= 2
            amax = rotated.abs().amax(1)
            divisor = torch.full_like(amax, 127)
            norm = torch.tensor(1. / torch.tensor(float(block)).sqrt().item()) if block else 1.
            expected_scale = (amax / divisor) * norm
            inv = divisor / torch.where(amax > 0, amax, 1)
            expected = (torch.round(rotated * inv[:, None]).to(torch.int16) + 128).to(torch.uint8)
            rows, scale, tmp = torch.empty(m, k, dtype=torch.uint8), torch.empty(m), torch.empty(k)
            quantize(x.data_ptr(), m, k, None if s is None else s.data_ptr(), block,
                     tmp.data_ptr(), rows.data_ptr(), scale.data_ptr())
            assert torch.equal(rows, expected), (m, k, block)
            assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32)), (m, k, block)
            if block:
                assert torch.equal(tmp.view(torch.int32), rotated[-1].view(torch.int32)), (m, k, block)


@pytest.mark.parametrize('k, block', [(768, 128), (768, 64), (1152, 128), (192, 64)])
def test_block_rotation_matches_dense(k, block):
    # Packing's three products: W·R, R^T·W (the Wv fold) and R^T·G·R (the Hessians). The block
    # products are the dense ones in another summation order, so they agree to float64 rounding.
    rng = torch.Generator().manual_seed(k + block)
    w = torch.randn(48, k, generator=rng, dtype=torch.float64)
    x = torch.randn(2 * k, k, generator=rng, dtype=torch.float64)
    g, r = x.T @ x, rotation(k, block)
    for got, want in ((rotate(w, k, block), w @ r),
                      (rotate(x, k, block).T, r.T @ x.T),
                      (rotate(rotate(g, k, block).T.contiguous(), k, block).T, r.T @ g @ r)):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12 * float(want.abs().max()))
