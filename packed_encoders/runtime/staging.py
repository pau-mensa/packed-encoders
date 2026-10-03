"""Host-side batch metadata, staged to the GPU in one copy.

Small index tensors (cu_seqlens, positions, gather indices) are cheap to build on the CPU
and expensive to build on the GPU from device values: the latter needs a device-to-host
sync to size anything. Building them on the host and moving them with one pinned,
non-blocking copy keeps the launch stream free of syncs.

The pinned buffer is reused across batches. An event recorded after each copy guards it:
the next `put()` waits for the previous copy to leave the buffer before overwriting it
(normally already done — the copy is a few microseconds).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


class PinnedStager:
    """Pack several 1-D int64 host tensors into one pinned buffer and copy them to
    `device` in a single non-blocking transfer. Returns device views in input order."""

    def __init__(self, device: torch.device, capacity: int = 1 << 16):
        self.device = device
        self._buf = torch.empty(capacity, dtype=torch.long).pin_memory()
        self._event: torch.cuda.Event | None = None

    def put(self, parts: Sequence[Tensor]) -> list[Tensor]:
        sizes = [int(p.numel()) for p in parts]
        n = sum(sizes)
        if self._event is not None:
            self._event.synchronize()
        if n > self._buf.numel():
            self._buf = torch.empty(max(n, 2 * self._buf.numel()), dtype=torch.long).pin_memory()
        off = 0
        for p, k in zip(parts, sizes):
            self._buf[off:off + k] = p.reshape(-1)
            off += k
        with torch.cuda.device(self.device):       # record the event on the copy's own stream
            dev = self._buf[:n].to(self.device, non_blocking=True)
            self._event = torch.cuda.Event()
            self._event.record()
        return list(dev.split(sizes))


def packed_layout_host(lengths: Sequence[int]) -> tuple[Tensor, Tensor]:
    """(cu_seqlens [B+1], positions [T]) for sequences of `lengths` packed back to back,
    positions restarting at 0 in every sequence."""
    lens = torch.as_tensor(list(lengths), dtype=torch.long)
    cu = torch.zeros(lens.numel() + 1, dtype=torch.long)
    torch.cumsum(lens, 0, out=cu[1:])
    total = int(cu[-1])
    pos = torch.arange(total) - torch.repeat_interleave(cu[:-1], lens)
    return cu, pos
