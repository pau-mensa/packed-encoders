"""Native workers respect affinity even when sysfs topology is hidden or invalid."""

import pytest

from packed_encoders.cpu import topology


@pytest.mark.parametrize("topology_value", [None, "-1", "invalid"])
@pytest.mark.parametrize("requested, expected", [(8, 2), (1, 1)])
def test_affinity_caps_workers_without_valid_topology(tmp_path, monkeypatch, topology_value,
                                                    requested, expected):
    monkeypatch.setattr(topology.torch, "get_num_threads", lambda: requested)
    monkeypatch.setattr(topology.os, "sched_getaffinity", lambda _: {2, 3}, raising=False)
    if topology_value is not None:
        path = tmp_path / "cpu2/topology"
        path.mkdir(parents=True)
        (path / "physical_package_id").write_text(topology_value)
        (path / "core_id").write_text("0")
    assert topology.default_threads(root=tmp_path) == expected


@pytest.mark.parametrize("unavailable", ["missing", "denied", "empty"])
def test_unavailable_affinity_preserves_torch_budget(tmp_path, monkeypatch, unavailable):
    monkeypatch.setattr(topology.torch, "get_num_threads", lambda: 8)
    if unavailable == "missing":
        monkeypatch.delattr(topology.os, "sched_getaffinity", raising=False)
    else:
        def affinity(_):
            if unavailable == "denied":
                raise OSError("affinity unavailable")
            return set()
        monkeypatch.setattr(topology.os, "sched_getaffinity", affinity, raising=False)
    assert topology.default_threads(root=tmp_path) == 8
