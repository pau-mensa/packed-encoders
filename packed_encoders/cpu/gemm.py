"""Cache-aware tile selection and opt-in packing-time tuning for native FP32 GEMM."""
from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import statistics
import time

import torch

from packed_encoders.errors import PackedEncodersError

# Register tiles, not cache-size classes. All share the same FP32 reduction tree.
# Order breaks ties in favor of the measured high-throughput tiles.
TILES = ((3, 8), (4, 6), (3, 6), (3, 4), (2, 4))


def l1_data_cache_bytes(*, root=Path('/sys/devices/system/cpu')):
    """Smallest L1D across the current affinity mask; None if any CPU is unknown.

    No affinity or process-global configuration is changed. Read once per engine;
    repack after moving a tuned engine to a different set of CPUs.
    """
    try:
        cpus = os.sched_getaffinity(0)
        sizes = []
        for cpu in cpus:
            caches = []
            for index in (root / f'cpu{cpu}' / 'cache').glob('index*'):
                if index.joinpath('level').read_text().strip() != '1':
                    continue
                if index.joinpath('type').read_text().strip() not in ('Data', 'Unified'):
                    continue
                size = index.joinpath('size').read_text().strip().upper()
                value = int(size[:-1]) * {'K': 1024, 'M': 1024 * 1024}[size[-1]] if size[-1:] in ('K', 'M') else int(size)
                if value <= 0:
                    return None
                caches.append(value)
            if not caches:
                return None
            sizes.append(min(caches))
        return min(sizes) if sizes else None
    except (OSError, ValueError, AttributeError):
        return None


def choose_tile(m, k, l1_bytes):
    """Estimate residency of one weight block plus the active activation rows.

    Reserve 1/16 of L1 for other traffic. This is a heuristic, not a promise of
    cache residency: associativity, prefetching, and register reuse also matter.
    Unknown caches use the smallest candidate. Oversized K likewise uses the
    smallest working set; GEMM remains valid even when no tile fits in L1.
    """
    if l1_bytes is None:
        return TILES[-1]
    budget = l1_bytes * 15 // 16
    for rows, cols in TILES:
        if 4 * k * (min(m, rows) + cols) <= budget:
            return rows, cols
    return min(TILES, key=lambda tile: (min(m, tile[0]) + tile[1], -tile[0] * tile[1]))


def tune_tiles(linear, projections, threads, l1_bytes, max_tokens):
    """Measure full groups of actual model weights; never reorder or retain copies.

    Each timed sweep visits every projection of a shape, rather than repeatedly
    exercising one hot weight. Small models can still fit in cache, as in use.
    Tune only 16/32/256 live tokens; other M values retain cache-aware dispatch.
    Return choices and measurements; caller publishes them after success.
    """
    projections = tuple(projections)
    groups = defaultdict(list)
    for weight, bias in projections:
        groups[tuple(weight.shape)].append((weight, bias))
    choices, report = {}, {}
    rng = torch.Generator().manual_seed(173)
    with torch.inference_mode():
        for m in sorted({min(rows, max_tokens) for rows in (16, 32, 256)}):
            for (n, k), weights in groups.items():
                x = torch.randn(m, k, generator=rng, dtype=torch.float32, device='cpu')
                out = torch.empty(m, n, dtype=torch.float32, device='cpu')
                # Resolve pointers outside the timer, just as the engine resolves its plan.
                args = [(x.data_ptr(), w.data_ptr(), None if b is None else b.data_ptr(),
                         out.data_ptr(), m, n, k, threads) for w, b in weights]
                default = choose_tile(m, k, l1_bytes)
                candidates = (default, *(tile for tile in TILES if tile != default))
                times = {tile: [] for tile in candidates}
                reference = None
                for round_ in range(4):  # warm once, then three alternating measurements
                    for tile in candidates if round_ % 2 == 0 else candidates[::-1]:
                        start = time.perf_counter_ns()
                        for pointers in args:
                            if linear(*pointers, *tile):
                                raise PackedEncodersError('native CPU projection tuning failed')
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        if round_:
                            times[tile].append(elapsed)
                        else:
                            if reference is None:
                                reference = out.clone()
                            elif not torch.equal(reference, out):
                                raise PackedEncodersError('native CPU projection tiles disagree during tuning')
                medians = {tile: statistics.median(samples) for tile, samples in times.items()}
                best = min(candidates, key=medians.get)
                # Do not replace the cache heuristic for differences within timing noise.
                selected = best if medians[best] < medians[default] * .97 else default
                key = (m, n, k)
                choices[key] = selected
                report[key] = {'default': default, 'selected': selected, 'median_ms': medians,
                               'projections': len(weights)}
            # Grouped measurements can favor a tile that loses when differently shaped
            # projections interleave. Validate the proposed combination in model order.
            defaults = {(m, n, k): choose_tile(m, k, l1_bytes) for n, k in groups}
            if all(choices[key] == tile for key, tile in defaults.items()):
                continue
            buffers = {(n, k): (torch.randn(m, k, generator=rng, dtype=torch.float32, device='cpu'),
                                torch.empty(m, n, dtype=torch.float32, device='cpu')) for n, k in groups}
            chain = []
            for w, b in projections:
                n, k = w.shape
                x, out = buffers[n, k]
                chain.append(((m, n, k), (x.data_ptr(), w.data_ptr(), None if b is None else b.data_ptr(),
                                         out.data_ptr(), m, n, k, threads)))
            times = {'default': [], 'proposed': []}
            candidates = (('default', defaults), ('proposed', choices))
            for round_ in range(4):
                for name, tiles in candidates if round_ % 2 == 0 else candidates[::-1]:
                    start = time.perf_counter_ns()
                    for key, pointers in chain:
                        if linear(*pointers, *tiles[key]):
                            raise PackedEncodersError('native CPU projection tuning failed')
                    if round_:
                        times[name].append((time.perf_counter_ns() - start) / 1e6)
            medians = {name: statistics.median(samples) for name, samples in times.items()}
            accepted = medians['proposed'] < medians['default'] * .97
            for key, default in defaults.items():
                report[key].update(proposed=choices[key], chain_median_ms=medians, chain_accepted=accepted)
                if not accepted:
                    choices[key] = report[key]['selected'] = default
    return choices, report
