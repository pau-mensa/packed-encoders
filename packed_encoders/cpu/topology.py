"""Affinity-aware default worker budget, without changing process thread settings."""

import os
from pathlib import Path

import torch


def default_threads(*, root=Path("/sys/devices/system/cpu")):
    requested = torch.get_num_threads()
    try:
        cores = set()
        for cpu in os.sched_getaffinity(0):
            topology = root / f"cpu{cpu}" / "topology"
            package = int((topology / "physical_package_id").read_text())
            core = int((topology / "core_id").read_text())
            if min(package, core) < 0:
                return requested
            cores.add((package, core))
        return min(requested, len(cores)) if cores else requested
    except (AttributeError, OSError, ValueError):
        return requested
