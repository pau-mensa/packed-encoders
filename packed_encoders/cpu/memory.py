"""Where CPU weights live in process memory (Linux-aware, best effort elsewhere)."""

from __future__ import annotations

import ctypes
import mmap
import sys

import torch

_PAGE = mmap.PAGESIZE
_MADV_HUGEPAGE, _MADV_PAGEOUT = 14, 21


def _libc():
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int
        return libc
    except (OSError, AttributeError):
        return None


def _madvise(libc, start, end, advice):
    # Whole pages inside [start, end) only; kernels without the advice reject it, which is fine.
    start, end = -(-start // _PAGE) * _PAGE, end // _PAGE * _PAGE
    if libc is not None and end > start:
        libc.madvise(start, end - start, advice)


def _file_mappings():
    ranges = []
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split()
                if len(fields) >= 5 and fields[4] != "0":  # nonzero inode: backed by a file
                    lo, hi = fields[0].split("-")
                    ranges.append((int(lo, 16), int(hi, 16)))
    except OSError:
        pass
    return ranges


def rehome(parameters):
    """Move contiguous parameters into fresh process memory: same Parameter objects and values, new storage.

    from_pretrained usually leaves safetensors weights in a private file mapping; fresh anonymous memory
    can stream faster and is advised for huge pages. On Linux, request eviction of file-backed pages left
    behind without changing their contents: aliases of the old storage still read the same values.
    Advice is best effort, so huge pages and a reduction in resident memory are not guaranteed. Other
    platforms only copy; resident memory can grow while the original mapping or aliases remain alive.
    """
    linux = sys.platform.startswith("linux")
    libc = _libc() if linux else None
    files = _file_mappings() if linux else []
    with torch.no_grad():
        for param in parameters:
            old = param.data
            new = torch.empty_like(old, memory_format=torch.contiguous_format)
            _madvise(libc, new.data_ptr(), new.data_ptr() + new.numel() * new.element_size(), _MADV_HUGEPAGE)
            new.copy_(old)
            param.data = new
            start = old.data_ptr()
            end = start + old.numel() * old.element_size()
            for lo, hi in files:
                if lo < end and start < hi:
                    _madvise(libc, max(lo, start), min(hi, end), _MADV_PAGEOUT)
