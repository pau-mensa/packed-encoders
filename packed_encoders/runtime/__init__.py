"""Architecture-agnostic runtime shared by every engine.

- `staging`   — one pinned host buffer per consumer, one H2D copy per batch, no syncs.
- `graphs`    — CUDA graphs over right-padded `(rows, S)` buckets, a shared memory pool,
                and gathering the real tokens back out.
- `attention` — pick a varlen attention kernel for one head geometry by probing each
                candidate against fp32 SDPA, not by a lookup table.

An engine supplies the math (`prepare_static` + `padded_core` for graphs, its own eager
packed forward); nothing here knows which architecture it is running.
"""

from packed_encoders.runtime.attention import AttentionChoice, select_attention
from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner, PaddedStatic
from packed_encoders.runtime.staging import PinnedStager, packed_layout_host

__all__ = [
    "AttentionChoice",
    "select_attention",
    "PaddedGraphConfig",
    "PaddedGraphRunner",
    "PaddedStatic",
    "PinnedStager",
    "packed_layout_host",
]
