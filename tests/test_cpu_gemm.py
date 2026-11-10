"""Native GEMM cache discovery, tile equivalence, and pack-time tuning contracts."""
import ctypes

import pytest
import torch
import torch.nn.functional as F

from packed_encoders.cpu import gemm
from packed_encoders.cpu.attention import CPUAttention


def test_cache_discovery_uses_affinity_and_data_cache(tmp_path, monkeypatch):
    def cache(cpu, index, level, kind, size):
        path = tmp_path / f'cpu{cpu}/cache/index{index}'
        path.mkdir(parents=True)
        for name, value in (('level', level), ('type', kind), ('size', size)):
            (path / name).write_text(str(value))

    cache(0, 0, 1, 'Instruction', '64K')
    cache(0, 1, 1, 'Data', '48K')
    cache(0, 2, 2, 'Unified', '1M')
    cache(1, 0, 1, 'Data', '32K')
    monkeypatch.setattr(gemm.os, 'sched_getaffinity', lambda _: {0, 1}, raising=False)
    assert gemm.l1_data_cache_bytes(root=tmp_path) == 32 * 1024
    monkeypatch.setattr(gemm.os, 'sched_getaffinity', lambda _: {0})
    assert gemm.l1_data_cache_bytes(root=tmp_path) == 48 * 1024
    monkeypatch.setattr(gemm.os, 'sched_getaffinity', lambda _: {0, 2})
    assert gemm.l1_data_cache_bytes(root=tmp_path) is None
    monkeypatch.setattr(gemm.os, 'sched_getaffinity', lambda _: {1})
    (tmp_path / 'cpu1/cache/index0/size').write_text('invalid')
    assert gemm.l1_data_cache_bytes(root=tmp_path) is None


def test_cache_selection_has_no_32_or_48_kib_classes():
    assert gemm.choose_tile(32, 768, 32 * 1024) == (4, 6)
    assert gemm.choose_tile(32, 1152, 32 * 1024) == (2, 4)
    assert gemm.choose_tile(32, 768, 48 * 1024) == (3, 8)
    assert gemm.choose_tile(32, 1152, 48 * 1024) == (4, 6)
    assert gemm.choose_tile(32, 1152, 64 * 1024) == (3, 8)
    for capacity in (24, 32, 40, 48, 64):
        for k in (128, 768, 1024, 1152):
            tile = gemm.choose_tile(32, k, capacity * 1024)
            if any(4 * k * sum(t) <= capacity * 1024 * 15 // 16 for t in gemm.TILES):
                assert 4 * k * sum(tile) <= capacity * 1024 * 15 // 16
    assert gemm.choose_tile(32, 100000, 32 * 1024) == (2, 4)
    assert gemm.choose_tile(32, 768, None) == (2, 4)


@pytest.mark.parametrize('switch_cost, accepted', [(1, True), (100, False)])
def test_tuner_checks_model_order_before_accepting_group_winner(monkeypatch, switch_cost, accepted):
    # Simulate a tile that wins when equal shapes run together, but can lose when
    # different projection shapes interleave. No real-time thresholds in this test.
    clock = 0
    previous_n = None

    def linear(x, w, b, y, m, n, k, threads, rows, cols):
        nonlocal clock, previous_n
        cost = 10 if (rows, cols) == (3, 8) else 20
        if (rows, cols) == (4, 6):
            cost = 1 if n == previous_n else switch_cost
        clock += cost * 1000
        previous_n = n
        ctypes.memset(y, 0, m * n * 4)
        return 0

    monkeypatch.setattr(gemm.time, 'perf_counter_ns', lambda: clock)
    projections = [(torch.zeros(4, 16), None), (torch.zeros(8, 16), None)] * 2
    choices, report = gemm.tune_tiles(linear, projections, 1, 48 * 1024, max_tokens=16)
    assert set(choices.values()) == {(4, 6) if accepted else (3, 8)}
    for entry in report.values():
        assert entry['proposed'] == (4, 6)
        assert entry['chain_accepted'] is accepted


@pytest.fixture
def attention():
    attention = CPUAttention('auto', 64, 64, 1)
    if not attention.native_linear:
        pytest.skip('native AVX-512 GEMM required')
    yield attention
    attention.close()


@pytest.mark.parametrize('m,n,k', [(1, 64, 1), (5, 128, 17), (6, 64, 128),
                                  (7, 128, 4095), (17, 64, 4097), (32, 128, 1024)])
@torch.no_grad()
def test_prepared_fp32_math_tails_and_row_independence(attention, m, n, k):
    from packed_encoders.cpu.fp32_x86 import FP32Projections
    rng = torch.Generator().manual_seed(523)
    w, x = torch.randn(n, k, generator=rng), torch.randn(m, k, generator=rng)
    prepared = FP32Projections([w])
    fn = prepared.library.pe_linear_packed
    packed = prepared.weights[id(w)][1]
    reference = None
    for workers in (1, 3):
        out = torch.empty(m, n)
        assert fn(x.data_ptr(), packed.data_ptr(), out.data_ptr(), m, n, k, workers) == 0
        if reference is None:
            reference = out.clone()
        torch.testing.assert_close(out, reference, atol=0, rtol=0)
        one = torch.empty(1, n)
        row = x[m // 2:m // 2 + 1]
        assert fn(row.data_ptr(), packed.data_ptr(), one.data_ptr(), 1, n, k, workers) == 0
        torch.testing.assert_close(one[0], reference[m // 2], atol=0, rtol=0)
    expected = F.linear(x.double(), w.double())
    assert (reference.double() - expected).norm() / expected.norm() < 2e-6
    for mm, nn, kk, workers in ((0, n, k, 1), (m, n - 1, k, 1), (m, n, 0, 1), (m, n, k, 0)):
        saved = out.clone()
        assert fn(x.data_ptr(), packed.data_ptr(), out.data_ptr(), mm, nn, kk, workers) == 1
        torch.testing.assert_close(out, saved, atol=0, rtol=0)
    prepared.close()


@torch.no_grad()
def test_prepared_fp32_refresh_ownership_and_fallbacks(attention):
    from packed_encoders.cpu.fp32_x86 import FP32Projections
    w = torch.randn(128, 192)
    before = w.clone()
    with torch.inference_mode():
        prepared = FP32Projections([w])
        first = prepared.linear(torch.randn(17, 192), w)
        saved = first.clone()
    prepared.linear(torch.randn(17, 192), w)
    # Output buffers are borrowed; an immediate same-shape call reuses storage.
    assert not torch.equal(first, saved)
    for m in (3, 17, 7):
        w.mul_(.9)
        x = torch.randn(m, 192)
        got = prepared.linear(x, w)
        expected = F.linear(x.double(), w.double())
        assert (got.double() - expected).norm() / expected.norm() < 1e-6
    assert prepared.packed_bytes == before.numel() * 4
    assert prepared.linear(torch.randn(1, 192), w) is None
    assert prepared.linear(torch.randn(3, 384)[:, ::2], w) is None
    assert prepared.linear(torch.randn(3, 192).double(), w) is None
    assert prepared.linear(torch.randn(3, 192), w.clone()) is None
    prepared.close()
    assert not prepared.weights and not prepared.outputs


@torch.no_grad()
def test_every_tile_is_bitwise_equal_across_rows_and_workers(attention):
    rng = torch.Generator().manual_seed(521)
    fn = attention.rows.pe_linear_tile
    # Every tile has partial M/N panels; both specialized and masked K paths.
    for m, n, k in ((1, 1, 1), (7, 17, 33), (16, 23, 768), (32, 13, 1152), (256, 17, 1152)):
        x = torch.randn(m, k, generator=rng)
        w = torch.randn(n, k, generator=rng) * .05
        bias = torch.randn(n, generator=rng)
        reference = None
        for workers in (1, 8):
            for tile in gemm.TILES:
                out = torch.empty(m, n)
                assert fn(x.data_ptr(), w.data_ptr(), bias.data_ptr(), out.data_ptr(), m, n, k, workers, *tile) == 0
                if reference is None:
                    reference = out.clone()
                assert torch.equal(reference, out)
                # Moving a row to another batch/tile must not change its bits.
                one = torch.empty(1, n)
                row = x[m // 2:m // 2 + 1]
                assert fn(row.data_ptr(), w.data_ptr(), bias.data_ptr(), one.data_ptr(), 1, n, k, workers, *tile) == 0
                assert torch.equal(one[0], reference[m // 2])
        expected = F.linear(x.double(), w.double(), bias.double())
        assert (reference.double() - expected).norm() / expected.norm() < 1e-6
    # Invalid tiles must be rejected before accessing any pointer (including 4x7/8).
    for tile in ((0, 4), (5, 4), (4, 7), (4, 8), (3, 9), (2, 0)):
        assert fn(None, None, None, None, 1, 1, 1, 1, *tile) != 0


@torch.no_grad()
def test_forced_cache_sizes_and_tuning_preserve_live_weights(attention):
    rng = torch.Generator().manual_seed(71)
    w = torch.randn(17, 1152, generator=rng)
    x = torch.randn(32, 1152, generator=rng)
    results = []
    for capacity in (None, 32 * 1024, 48 * 1024):
        attention.linear_l1_bytes = capacity
        attention.linear_tiles.clear()
        results.append(attention.linear(x, w).clone())
    assert all(torch.equal(results[0], out) for out in results)
    before = w.clone()
    global_rng, threads = torch.random.get_rng_state(), torch.get_num_threads()
    default_dtype = torch.get_default_dtype()
    try:
        # The native ABI requires FP32 even if the application's default differs.
        torch.set_default_dtype(torch.float64)
        attention.tune_linear([(w, None)], max_tokens=32)
    finally:
        torch.set_default_dtype(default_dtype)
    assert set(attention.linear_tuning) == {(16, 17, 1152), (32, 17, 1152)}
    assert torch.equal(global_rng, torch.random.get_rng_state())
    assert threads == torch.get_num_threads() and torch.equal(w, before)
    assert torch.equal(attention.linear(x, w), results[0])
    w.mul_(.5)
    torch.testing.assert_close(attention.linear(x, w), results[0] * .5, rtol=0, atol=0)
    # Untuned lengths choose a tile lazily without invoking the tuner.
    attention.linear(x[:7], w)
    assert (4, 17, 1152) in attention.linear_tiles and (7, 17, 1152) not in attention.linear_tuning
