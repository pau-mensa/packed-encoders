"""Short native attention must match the general path across batching and window boundaries."""
import platform

import pytest
import torch
import torch.nn.functional as F

from packed_encoders.cpu.attention import CPUAttention


@pytest.fixture(scope='module')
def attention():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        pytest.skip('x86 attention specialization')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    engine = CPUAttention('auto', 64, 64, 1)
    try:
        if engine.library is None:
            pytest.skip('native attention compiler unavailable')
        yield engine
    finally:
        engine.close()
        torch.set_num_threads(previous)


@pytest.mark.parametrize('half', [0, 14, 15, 16, 30, 31, 32, 64])
@torch.inference_mode()
def test_short_attention_matches_general_path(attention, half):
    rng = torch.Generator().manual_seed(43)
    # Mixed batches give the short path a larger scratch stride than its own sequence;
    # lengths around each specialization also exercise the generic path and row tails.
    lengths = (16, 32, 1, 15, 17, 31, 33, 65)
    q, k, v = [torch.randn(1, 3, sum(lengths), 64, generator=rng) * 2 for _ in range(3)]
    # Uniform logits and a very sharp row cover softmax ties and exp underflow.
    q[:, :, 0].zero_()
    q[:, :, 1] *= 100
    boundaries = torch.tensor(lengths, dtype=torch.int32)
    attention.tile, attention.threads = 4, 1  # forces the general tile path
    expected = attention._native(q, k, v, lengths, boundaries, half).clone()
    offset = 0
    for length in lengths:
        section = slice(offset, offset + length)
        mask = (torch.arange(length)[:, None] - torch.arange(length)).abs() <= half
        reference = F.scaled_dot_product_attention(q[:, :, section], k[:, :, section], v[:, :, section],
                                                  attn_mask=mask)
        torch.testing.assert_close(expected[:, :, section], reference, atol=2e-5, rtol=2e-5)
        offset += length
    by_tile = {}
    for threads in (1, 8):
        attention.threads = threads
        for tile in (16, 32, 64):
            attention.tile = tile
            for token_major in (False, True):
                actual = attention._native(q, k, v, lengths, boundaries, half, token_major)
                reference = expected.transpose(1, 2).flatten(2) if token_major else expected
                # Changing tiles can change the alignment of a clipped local window and hence
                # the softmax sum order. Full-window sequences retain the exact order.
                torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
                offset = 0
                for length in lengths:
                    if half >= length - 1:
                        section = (slice(None), slice(offset, offset + length)) if token_major else (
                            slice(None), slice(None), slice(offset, offset + length))
                        assert torch.equal(actual[section], reference[section])
                    offset += length
                key = tile, token_major
                if threads == 1:
                    by_tile[key] = actual.clone()
                else:
                    assert torch.equal(actual, by_tile[key])
        # A sequence run alone must retain exactly the results it had in the mixed batch.
        for offset, length in ((0, 16), (16, 32)):
            section = slice(offset, offset + length)
            actual = attention._native(q[:, :, section], k[:, :, section], v[:, :, section], (length,),
                                       torch.tensor([length], dtype=torch.int32), half)
            assert torch.equal(actual, by_tile[64, False][:, :, section])
