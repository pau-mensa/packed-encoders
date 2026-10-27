"""Pack-time attention dispatch and affinity-aware native worker defaults."""

import pytest
import torch

from packed_encoders.cpu import attention, topology


def test_default_workers_count_physical_cores_in_affinity(tmp_path, monkeypatch):
    for cpu, package, core in ((0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 1, 0)):
        path = tmp_path / f"cpu{cpu}/topology"
        path.mkdir(parents=True)
        (path / "physical_package_id").write_text(str(package))
        (path / "core_id").write_text(str(core))
    monkeypatch.setattr(topology.torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(topology.os, "sched_getaffinity", lambda _: {0, 1, 2, 3}, raising=False)
    assert topology.default_threads(root=tmp_path) == 3
    monkeypatch.setattr(topology.os, "sched_getaffinity", lambda _: {1, 2})
    assert topology.default_threads(root=tmp_path) == 2
    monkeypatch.setattr(topology.torch, "get_num_threads", lambda: 1)
    assert topology.default_threads(root=tmp_path) == 1
    monkeypatch.setattr(topology.torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(topology.os, "sched_getaffinity", lambda _: {4})
    assert topology.default_threads(root=tmp_path) == 1  # Affinity still caps unknown topology.


@pytest.mark.parametrize("backend", ["auto", "native", "sdpa"])
def test_tuning_selects_each_shape_without_changing_process_settings(monkeypatch, backend):
    monkeypatch.setattr(attention.platform, "system", lambda: "Linux")
    monkeypatch.setattr(attention.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(attention, "_x86_library", lambda: (object(), None))
    cpu = attention.CPUAttention(backend, 64, 64, 1)
    clock, timed = 0, []

    def native(q, k, v, lengths, boundaries, half, token_major):
        nonlocal clock
        assert q.dtype == torch.float32 and q.device.type == "cpu"
        clock += 10 if len(lengths) == 1 else 100
        timed.append("native")
        return q

    def sdpa(q, k, v, lengths, *, local):
        nonlocal clock
        clock += 100 if len(lengths) == 1 else 10
        timed.append("sdpa")
        return q

    monkeypatch.setattr(cpu, "_native", native)
    monkeypatch.setattr(cpu, "_sdpa", sdpa)
    monkeypatch.setattr(attention.time, "perf_counter_ns", lambda: clock)
    rng, threads = torch.random.get_rng_state(), torch.get_num_threads()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        cpu.tune_attention(2, 1024, 4096)
    finally:
        torch.set_default_dtype(previous)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    if backend != "auto":
        assert not timed and not cpu.attention_tuning
        return
    assert len(cpu.attention_tuning) == 10
    assert cpu.rows is not None and cpu.fused_rope and cpu.fused_norm
    for local in (False, True):
        for lengths in ((7,), (32,), (129,), (1024,)):
            assert cpu._use_native(local, lengths)
        assert not cpu._use_native(local, (17, 32, 3, 21, 31, 9, 14, 6))
    # Dispatch never times kernels or grows a per-request shape cache.
    monkeypatch.setattr(attention.time, "perf_counter_ns", lambda: pytest.fail("request timed"))
    assert cpu._use_native(True, (8192,))
    assert len(cpu.attention_tuning) == 10
    cpu.close()
    assert not cpu.attention_tuning


def test_small_token_budget_bounds_calibration(monkeypatch):
    monkeypatch.setattr(attention.platform, "system", lambda: "Linux")
    monkeypatch.setattr(attention.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(attention, "_x86_library", lambda: (object(), None))
    cpu = attention.CPUAttention("auto", 64, 64, 1)

    def check(q, k, v, *args, **kwargs):
        assert q.shape == (1, 2, 3, 64)
        return q

    monkeypatch.setattr(cpu, "_native", check)
    monkeypatch.setattr(cpu, "_sdpa", check)
    cpu.tune_attention(2, 256, 3)
    assert set(cpu.attention_tuning) == {(False, 3, 1), (True, 3, 1)}


@pytest.mark.parametrize("native_ns,sdpa_ns", [(2_000_000, 100_000), (100_000, 2_000_000)])
def test_tuning_repeats_fast_calls_without_batching_slow_calls(monkeypatch, native_ns, sdpa_ns):
    monkeypatch.setattr(attention.platform, "system", lambda: "Linux")
    monkeypatch.setattr(attention.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(attention, "_x86_library", lambda: (object(), None))
    cpu = attention.CPUAttention("auto", 64, 64, 1)
    clock, calls = 0, {"native": 0, "sdpa": 0}

    def timed(name, elapsed):
        def call(q, *args, **kwargs):
            nonlocal clock
            clock += elapsed
            calls[name] += 1
            return q
        return call

    monkeypatch.setattr(cpu, "_native", timed("native", native_ns))
    monkeypatch.setattr(cpu, "_sdpa", timed("sdpa", sdpa_ns))
    monkeypatch.setattr(attention.time, "perf_counter_ns", lambda: clock)
    cpu.tune_attention(2, 256, 3)
    for name, elapsed in (("native", native_ns), ("sdpa", sdpa_ns)):
        # Two buckets, each with one warmup and three measured samples.
        assert calls[name] == 2 * (1 + 3 * (3 if elapsed < 1_000_000 else 1))
    for bucket in cpu.attention_tuning.values():
        assert bucket["selected"] == ("native" if native_ns < sdpa_ns else "sdpa")
        assert bucket["median_ms"] == {"native": native_ns / 1e6, "sdpa": sdpa_ns / 1e6}
