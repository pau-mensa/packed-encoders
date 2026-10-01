"""Architecture-agnostic CUDA graphs over right-padded `(rows, S)` buckets.

Why padded and not packed: a packed graph bakes in how tokens split into sequences, and
kernels such as fla's chunked recurrences size their launch grids from that split on the
host. A right-padded rectangle has one fixed geometry per `(rows, S)`, and it is *exact*
for every real token as long as the engine's captured core keeps one contract:

    real tokens never read pad tokens.

Causal mixers (GatedDeltaNet, causal conv, causal attention) keep it for free because
pads follow the real tokens. Bidirectional attention keeps it by treating each row as
two segments, `[real | pad]`: varlen kernels read `seg_cu32`, masked SDPA reads `lens`.
Pad rows are computed and discarded.

The engine owns the math (`prepare_static`, `padded_core`). The runner owns bucketing,
warmup, capture into one shared memory pool, staging a batch into the static buffers,
and gathering the real rows out of the shared output buffer.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor

from packed_encoders.runtime.staging import PinnedStager

GRAPH_ENV_VAR = "PACKED_ENCODERS_GRAPH"


@dataclass(frozen=True)
class PaddedGraphConfig:
    """Bucketing for `PaddedGraphRunner`.

    - `pad_to`: S is rounded up to a multiple of this. Finer steps waste fewer pad tokens and
      cost more graphs; 16 measured best for topk queries and documents.
    - `row_buckets`: a batch of `n` sequences replays the smallest bucket `>= n`. Empty rows
      cost compute, so keep the list dense near the batch sizes you actually use.
    - `max_seq` / `max_tokens`: batches beyond either run the eager packed forward instead.
      `max_tokens` also sizes the one output buffer every graph shares.
    - `max_graphs`: LRU bound on live graphs (graphs share one pool, so memory grows slowly).
    - `split_overhead_tokens`: what one extra replay costs, in padded tokens. A batch whose
      lengths vary a lot is split, longest rows first, into several buckets when the pad
      tokens saved exceed this per extra replay. Each replay streams every weight once, so
      the natural value is where a GEMM's compute time equals its weight-read time
      (FLOPS / bandwidth: ~420 tokens on an L40S, ~300 on an H100); 512 errs toward fewer
      replays. None disables splitting.
    """

    pad_to: int = 16
    row_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    max_seq: int = 2048
    max_tokens: int = 16384
    max_graphs: int = 256
    warmup: int = 2
    split_overhead_tokens: int | None = 512


@dataclass
class PaddedStatic:
    """Fixed-address inputs of one captured graph. `extras` holds engine-owned per-shape
    constants (e.g. RoPE tables) built once, before capture."""

    rows: int
    seq: int
    ids: Tensor        # (rows, seq) int64; pad ids are 0 and never read by real tokens
    seg_cu32: Tensor   # (2 * rows + 1,) int32: row i -> [i*S, i*S + len_i), [i*S + len_i, (i+1)*S)
    lens: Tensor       # (rows,) int32 real length per row (0 for unused rows)
    extras: dict[str, Any] = field(default_factory=dict)


class GraphableEngine(Protocol):
    hidden_size: int
    dtype: torch.dtype
    device: torch.device

    def prepare_static(self, static: PaddedStatic) -> None: ...

    def padded_core(self, static: PaddedStatic) -> Tensor: ...  # (rows * seq, hidden), capture-safe


@dataclass
class _Captured:
    graph: torch.cuda.CUDAGraph
    static: PaddedStatic


def graphs_globally_disabled() -> bool:
    return os.environ.get(GRAPH_ENV_VAR, "1") == "0"


class PaddedGraphRunner:
    """Replays one CUDA graph per `(rows, S)` bucket; returns None for batches it does not
    cover so the caller runs its eager forward."""

    def __init__(self, engine: GraphableEngine, config: PaddedGraphConfig | None = None):
        self.engine = engine
        self.config = config or PaddedGraphConfig()
        self._cache: OrderedDict[tuple[int, int], _Captured] = OrderedDict()
        self._pool = None
        self._out: Tensor | None = None
        self._stager = PinnedStager(engine.device)

    # ------------------------------------------------------------------ planning
    def _bucket(self, n: int, longest: int) -> tuple[int, int] | None:
        cfg = self.config
        rows = next((r for r in cfg.row_buckets if r >= n), None)
        if rows is None or longest > cfg.max_seq or longest == 0:
            return None
        seq = -(-longest // cfg.pad_to) * cfg.pad_to
        return (rows, seq) if rows * seq <= cfg.max_tokens else None

    def plan(self, lengths: Sequence[int]) -> tuple[int, int] | None:
        """The single `(rows, S)` bucket that holds the whole batch, or None if out of bounds."""
        return self._bucket(len(lengths), max(lengths)) if lengths else None

    def plan_groups(self, lengths: Sequence[int]) -> list[tuple[list[int], tuple[int, int]]] | None:
        """How to replay this batch: `[(sequence indices, (rows, S)), ...]`, or None if no plan
        fits the bounds. One group unless splitting pays (see `split_overhead_tokens`).

        Rows are sorted longest first and cut into consecutive groups, each padded to its own
        longest row. A DP over cut points, with group sizes limited to the row buckets (a
        group between buckets pays for pad rows anyway), keeps planning O(n x buckets)."""
        n = len(lengths)
        if n == 0:
            return None
        single = self._bucket(n, max(lengths))
        over = self.config.split_overhead_tokens
        # A batch too big for one bucket stays eager: packed, it has no padding at all, and at
        # that size the eager launch overhead is spread over plenty of GPU work. Splitting only
        # trims padding off batches that would be graphed anyway. (Measured: splitting 128-query
        # batches into two 64-row graphs lost to eager on topk small, 1443 vs 1761 queries/s, L40S.)
        if single is None:
            return None
        if over is None or n == 1:
            return [(list(range(n)), single)]
        order = sorted(range(n), key=lambda i: -lengths[i])
        inf = float("inf")
        best = [inf] * (n + 1)              # best[i]: cost of the i longest rows, as groups
        cut: list[tuple[int, tuple[int, int]] | None] = [None] * (n + 1)
        best[0] = 0.0
        sizes = sorted(set(self.config.row_buckets))
        for i in range(n):
            if best[i] == inf:
                continue
            longest = lengths[order[i]]
            rest = n - i
            for m in [b for b in sizes if b < rest] + [rest]:
                bucket = self._bucket(m, longest)
                if bucket is None:
                    continue
                cost = best[i] + bucket[0] * bucket[1] + (over if i else 0)
                if cost < best[i + m]:
                    best[i + m], cut[i + m] = cost, (i, bucket)
        if single[0] * single[1] <= best[n]:
            return [(list(range(n)), single)]
        groups, j = [], n
        while j:
            i, bucket = cut[j]
            groups.append((order[i:j], bucket))
            j = i
        return groups[::-1]

    @property
    def num_graphs(self) -> int:
        return len(self._cache)

    @torch.inference_mode()
    def capture(self, rows: int, seqs: Sequence[int]) -> None:
        """Pre-capture buckets (otherwise each is captured on first use)."""
        with torch.cuda.device(self.engine.device):
            for s in seqs:
                self._get(rows, -(-int(s) // self.config.pad_to) * self.config.pad_to)

    # ------------------------------------------------------------------ replay
    # Everything touching the static buffers runs under inference_mode, whatever the caller
    # uses: the buffers are then inference tensors, and in-place staging into them is legal
    # from both `no_grad` and `inference_mode` callers.
    @staticmethod
    def _layout(lens_g: Tensor, rows: int, seq: int) -> tuple[Tensor, Tensor, Tensor]:
        """Host metadata for one bucket: token slots in the (rows*S) rectangle, the [real | pad]
        segment bounds, and per-row lengths."""
        n = lens_g.numel()
        lens = torch.zeros(rows, dtype=torch.long)
        lens[:n] = lens_g
        starts = torch.arange(rows, dtype=torch.long) * seq
        cu = torch.zeros(n + 1, dtype=torch.long)
        torch.cumsum(lens_g, 0, out=cu[1:])
        dst = torch.arange(int(cu[-1])) - torch.repeat_interleave(cu[:-1] - starts[:n], lens_g)
        seg = torch.zeros(2 * rows + 1, dtype=torch.long)
        seg[1::2] = starts + lens
        seg[2::2] = starts + seq
        return dst, seg, lens

    def _replay(self, cap: _Captured, tokens: Tensor, d_dst: Tensor, d_seg: Tensor, d_lens: Tensor) -> Tensor:
        st = cap.static
        st.ids.zero_()
        st.ids.view(-1).index_copy_(0, d_dst, tokens)
        st.seg_cu32.copy_(d_seg)
        st.lens.copy_(d_lens)
        cap.graph.replay()
        return self._out[: st.rows * st.seq].index_select(0, d_dst)

    @torch.inference_mode()
    def __call__(self, ids: Tensor, lengths: Sequence[int]) -> Tensor | None:
        """ids: (T,) device token ids of the sequences back to back; lengths: host ints.
        Returns (T, hidden) for exactly those tokens, or None if the batch is unbucketed."""
        # Capture, replay and Triton launches use the current device, not the tensors': pin it.
        with torch.cuda.device(self.engine.device):
            return self._call(ids, lengths)

    def _call(self, ids: Tensor, lengths: Sequence[int]) -> Tensor | None:
        groups = self.plan_groups(lengths)
        if groups is None:
            return None
        lens_all = torch.as_tensor(list(lengths), dtype=torch.long)
        ids = ids.reshape(-1)
        if len(groups) == 1:                      # the common case: no reordering
            rows, seq = groups[0][1]
            d_dst, d_seg, d_lens = self._stager.put(list(self._layout(lens_all, rows, seq)))
            return self._replay(self._get(rows, seq), ids, d_dst, d_seg, d_lens)
        # Several buckets: `src` maps each group's tokens back to their place in `ids`. All
        # metadata goes up in one staged copy; the shared output buffer is gathered from before
        # the next replay overwrites it (same stream, so in order).
        cu = torch.zeros(len(lengths) + 1, dtype=torch.long)
        torch.cumsum(lens_all, 0, out=cu[1:])
        parts, plans = [], []
        for idx, (rows, seq) in groups:
            sel = torch.as_tensor(idx, dtype=torch.long)
            lens_g = lens_all[sel]
            dst, seg, lens = self._layout(lens_g, rows, seq)
            within = torch.arange(int(lens_g.sum())) - torch.repeat_interleave(
                torch.cumsum(lens_g, 0) - lens_g, lens_g)
            src = torch.repeat_interleave(cu[:-1][sel], lens_g) + within
            parts += [dst, seg, lens, src]
            plans.append((rows, seq))
        staged = self._stager.put(parts)
        out = torch.empty(ids.numel(), self.engine.hidden_size, device=ids.device, dtype=self.engine.dtype)
        for k, (rows, seq) in enumerate(plans):
            d_dst, d_seg, d_lens, d_src = staged[4 * k: 4 * k + 4]
            out.index_copy_(0, d_src, self._replay(self._get(rows, seq), ids.index_select(0, d_src),
                                                   d_dst, d_seg, d_lens))
        return out

    # ------------------------------------------------------------------ capture
    def _get(self, rows: int, seq: int) -> _Captured:
        key = (rows, seq)
        cap = self._cache.get(key)
        if cap is not None:
            self._cache.move_to_end(key)
            return cap
        cap = self._capture(rows, seq)
        self._cache[key] = cap
        while len(self._cache) > self.config.max_graphs:
            self._cache.popitem(last=False)
        return cap

    def _run(self, static: PaddedStatic) -> None:
        out = self.engine.padded_core(static)
        self._out[: static.rows * static.seq].copy_(out)

    def _capture(self, rows: int, seq: int) -> _Captured:
        eng, dev = self.engine, self.engine.device
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
            self._out = torch.empty(self.config.max_tokens, eng.hidden_size, device=dev, dtype=eng.dtype)
        # Capture with every row split half real / half pad so both segments are non-empty.
        half = torch.arange(2 * rows + 1, device=dev, dtype=torch.int32).mul_(seq).div_(2, rounding_mode="floor")
        static = PaddedStatic(
            rows=rows, seq=seq,
            ids=torch.zeros((rows, seq), dtype=torch.long, device=dev),
            seg_cu32=half,
            lens=torch.full((rows,), seq // 2, dtype=torch.int32, device=dev),
        )
        with torch.inference_mode():
            eng.prepare_static(static)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self.config.warmup):   # autotune + allocator warmup, off-graph
                    self._run(static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            # Capture on this runner's stream: torch.cuda.graph's default capture stream is one per
            # process, created on whichever device captured first, and entering it switches to that device.
            with torch.cuda.graph(graph, pool=self._pool, stream=side):
                self._run(static)
        return _Captured(graph=graph, static=static)
