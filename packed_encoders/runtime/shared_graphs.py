"""Full-forward CUDA graphs of one exact packed geometry, such as a shared-prefix plan.

Unlike `runtime.graphs`, nothing is padded: a graph replays the plan's own layout, so it is
keyed on everything the host planned (segment lengths and every index map), never on token
values. Prefix discovery stays outside capture; embedding, every layer and the gather back to
the caller's layout replay as one graph. The engine owns the math: `prepare_shared_layout(plan)`
stages a plan's device metadata outside capture, and `shared_core(ids, static)` is the
capture-safe forward.
"""
from collections import OrderedDict

import torch


class CapturedForward:
    """Fixed-address inputs and output; callers clone the output before the next replay.

    `retain`, when given, runs after warmup and its result is kept for the graph's lifetime:
    engines whose kernels cache launch metadata outside the graph pool pin it there."""

    @torch.inference_mode()
    def __init__(self, engine, count, forward, *, retain=None, warmup=2, pool=None):
        with torch.cuda.device(engine.device):
            self.ids = torch.zeros(count, dtype=torch.long, device=engine.device)
            side = torch.cuda.Stream(device=engine.device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(max(2, warmup)):
                    forward(self.ids)
                self._retained = retain() if retain is not None else None
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=pool, stream=side):
                self.output = forward(self.ids)

    def replay(self, ids):
        self.ids.copy_(ids)
        self.graph.replay()
        return self.output.clone()


class SharedGraphRunner:
    """Bounded graphs for exact sharing plans; token values are never part of the key.

    The pool is shared across plans; calls must be serialized, like the padded graph runner.
    Returns None for plans beyond `config` (a `PaddedGraphConfig`) so the caller runs eagerly.
    """

    def __init__(self, engine, config):
        self.engine, self.config = engine, config
        self._cache = OrderedDict()
        self._pool = None

    @property
    def num_graphs(self):
        return len(self._cache)

    @staticmethod
    def _key(plan):
        # Include mappings, not just shapes: equal segment lengths can have
        # different parents, source row order, and output reconstruction.
        return (tuple(plan.lengths), tuple(plan.kv_lengths), plan.n_roots, plan.root_tokens,
                *(tuple(getattr(plan, name).reshape(-1).tolist()) for name in
                  ('src', 'out', 'rope_pos', 'conv_pos', 'parent', 'fix_rows', 'fix_taps', 'kv_idx')))

    def _retain(self, static):
        """Metadata the engine's kernels cache outside the graph pool, pinned for the graph's lifetime."""
        return None

    @torch.inference_mode()
    def __call__(self, ids, plan):
        cfg = self.config
        if ids.numel() > cfg.max_tokens or max(plan.kv_lengths) > cfg.max_seq or cfg.max_graphs <= 0:
            return None
        with torch.cuda.device(self.engine.device):
            key = self._key(plan)
            entry = self._cache.get(key)
            if entry is None:
                if self._pool is None:
                    self._pool = torch.cuda.graph_pool_handle()
                static = self.engine.prepare_shared_layout(plan)
                cap = CapturedForward(self.engine, ids.numel(), lambda x: self.engine.shared_core(x, static),
                                      retain=lambda: self._retain(static), warmup=cfg.warmup, pool=self._pool)
                entry = (cap, static)  # retain every metadata tensor used by the graph
                self._cache[key] = entry
                while len(self._cache) > cfg.max_graphs:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            return entry[0].replay(ids.reshape(-1))
