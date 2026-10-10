"""Full encoder CUDA graphs with host-planned, fixed prefix/suffix geometry (runtime.shared_graphs),
plus the fla metadata Qwen3.5's kernels need pinned across replays."""
import weakref

import torch

from packed_encoders.errors import PackedEncodersError
from packed_encoders.runtime.shared_graphs import CapturedForward, SharedGraphRunner as _SharedGraphRunner


def retain_fla_metadata(engine, layout):
    """Pin the chunk indices consumed by FLA 0.5's conv and GDN kernels.

    Warmup populates FLA's bounded identity cache outside the graph pool. CUDA
    graphs only retain its addresses, so global cache eviction must not free it.
    These calls match FLA's argument identities and its 64-token chunk size.
    """
    if not any(layer.linear for layer in engine.layers):
        return []
    from fla.ops.utils.index import prepare_chunk_indices
    from packed_encoders.arch.qwen3_5.engine import fla_tensor_cache
    pairs = [(layout.cu, layout.cu_cpu)] if not engine.fused else []
    if layout.share is None:
        pairs.append((layout.cu, layout.cu_cpu))
    else:
        sh = layout.share
        pairs.extend([(sh.cu_roots, sh.cu_roots_cpu), (sh.cu_kids, sh.cu_kids_cpu)])
    with fla_tensor_cache():
        return [prepare_chunk_indices(cu, 64, cu_seqlens_cpu=cpu) for cu, cpu in pairs]


class SuffixGraph:
    """An explicit graph for one prepared prefix and fixed positive suffix lengths.

    Token values can change on every call. Outputs own their storage. Close the
    handle to release the graph's activation pool (separate from prefix.nbytes).
    """

    @torch.inference_mode()
    def __init__(self, prefix, lengths):
        engine = prefix._require_valid()
        self._prefix = weakref.ref(prefix)
        self.lengths = tuple(lengths)
        self._captured = None
        with torch.cuda.device(engine.device):
            engine.sync_norms()
            self._layout = engine.prepare_prefix_layout(list(lengths), prefix)
            self._captured = CapturedForward(engine, sum(lengths), lambda ids: engine.prefix_core(ids, self._layout),
                                             retain=lambda: retain_fla_metadata(engine, self._layout))

    def __call__(self, input_ids):
        if self._captured is None:
            raise PackedEncodersError("suffix graph is closed")
        prefix = self._prefix()
        if prefix is None:
            self.close()
            raise PackedEncodersError("prepared prefix is closed")
        engine = prefix._require_valid()
        prefix._check_ids(engine, input_ids, self.lengths)
        with torch.inference_mode(), torch.cuda.device(engine.device):
            return self._captured.replay(input_ids)

    def close(self):
        self._captured = None
        self._layout = None

    def __enter__(self):
        if self._captured is None:
            raise PackedEncodersError("suffix graph is closed")
        return self

    def __exit__(self, *exc):
        self.close()


class SharedGraphRunner(_SharedGraphRunner):
    """`runtime.shared_graphs.SharedGraphRunner`, pinning fla's chunk indices (retain_fla_metadata)."""

    def _retain(self, static):
        return retain_fla_metadata(self.engine, static.layout)
