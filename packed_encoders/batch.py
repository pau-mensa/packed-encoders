"""Packed input metadata without implicit copies, synchronization or rebuilding."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class PackedBatch:
    """Real token IDs in sequence order; output has the same order, [tokens, hidden].

    Engines declare their required metadata. CUDA ModernBERT needs device cu_seqlens
    (int32), max_seqlen and device positions; host_lengths is optional. Qwen may
    instead require host lengths. CPU ModernBERT accepts host lengths or CPU
    cu_seqlens, and defaults omitted positions to zero-based per-sequence indices.
    Construction never derives or validates tensor
    values. Callers own consistency and must not mutate metadata during a call.

    Positions preserve the original model's semantics; they are not implicitly
    reset. Empty sequences, arbitrary masks and nonstandard positions are not
    universally supported. Check the selected engine's contract.
    """

    input_ids: Tensor
    cu_seqlens: Tensor | None = None
    max_seqlen: int | None = None
    position_ids: Tensor | None = None
    host_lengths: tuple[int, ...] | None = None
