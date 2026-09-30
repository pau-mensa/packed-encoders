"""The short-attention kernel compiles per MAX_SEQLEN; launches must use few values."""

from __future__ import annotations

import torch

from packed_encoders._kernels import triton_packed_attention as tpa


def test_max_seqlen_constexpr_takes_few_values(monkeypatch):
    launched = []

    class Recorder:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launched.append((grid, kwargs))

    monkeypatch.setattr(tpa, "_packed_short_attention_fwd", Recorder())
    q = torch.empty(1, 12, 64)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    for max_seqlen in range(1, 129):
        config = tpa._select_config(max_seqlen)
        tpa._launch_packed_short_attention(
            q, q, q, cu, max_seqlen, half_window=None, softmax_scale=0.125,
            config=config,
        )
        grid, kwargs = launched[-1]
        assert kwargs["MAX_SEQLEN"] >= max_seqlen
        assert kwargs["MAX_SEQLEN"] - max_seqlen < kwargs["BLOCK_N"]
        assert kwargs["MAX_SEQLEN"] % kwargs["BLOCK_N"] == 0
        assert grid[2] == -(-max_seqlen // config.block_m)
    variants = {(kwargs["MAX_SEQLEN"], kwargs["BLOCK_M"], kwargs["BLOCK_N"])
                for _, kwargs in launched}
    assert len(variants) <= 4
