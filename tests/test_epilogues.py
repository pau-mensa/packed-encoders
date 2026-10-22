"""Shared epilogue integration and CUDA preparation policy, without GPU kernels.

The policy harness bypasses only the CUDA device requirement so the real CUDA
context and all four engine wrappers can run on CPU. GPU numerics/replay have
separate coverage in test_cuda_epilogues.py.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import packed_encoders as pe
from packed_encoders.engine import ModelBinding
from packed_encoders.pieces.base import Piece
from packed_encoders.pieces.numerical import LINEAR
from packed_encoders.runtime.cuda_epilogue import CUDAPreparation
from packed_encoders.runtime.epilogue import PreparationContext


@pytest.fixture(params=["modernbert", "qwen35", "lfm2", "gemma2"])
def runtime(request, monkeypatch):
    from packed_encoders.arch.modernbert import PackedModernBert
    from packed_encoders.arch.qwen3_5 import PackedQwen35
    from packed_encoders.arch.lfm2 import PackedLfm2
    from packed_encoders.arch.embedding_gemma2 import PackedEmbeddingGemma2

    monkeypatch.setattr(CUDAPreparation, "__init__", PreparationContext.__init__)
    model = nn.Embedding(64, 16, dtype=torch.bfloat16).eval()
    binding = ModelBinding(None, model, model)
    calls = []

    def linear(x, weight):
        calls.append("linear")
        return F.linear(x, weight)

    composition = SimpleNamespace(linear=Piece("test-linear", LINEAR, linear, F.linear, lambda *args: None),
                                  all=lambda: ())
    state = SimpleNamespace(pieces=composition, graph_enabled=True,
                            engine=SimpleNamespace(composition=composition,
                                                   release=lambda **kwargs: calls.append("release")))
    if request.param == "modernbert":
        engine = PackedModernBert(binding, state)
    else:
        cls = {"qwen35": PackedQwen35, "lfm2": PackedLfm2, "gemma2": PackedEmbeddingGemma2}[request.param]
        engine = cls(None, binding, state, "sdpa")
    scratch = torch.empty(8, 16, dtype=torch.bfloat16)

    def forward(batch):
        engine._require_open()
        calls.append("graph" if state.graph_enabled else "eager")
        scratch.copy_(model(batch.input_ids))
        return scratch

    monkeypatch.setattr(engine, "forward_packed", forward)
    yield engine, calls
    engine.close()


def batch():
    return pe.PackedBatch(torch.arange(8), host_lengths=(3, 5))


@pytest.mark.parametrize("dtype,bias", [(torch.bfloat16, False), (torch.float32, False),
                                        (torch.bfloat16, True), (torch.float16, True)])
@torch.no_grad()
def test_head_dtype_piece_selection_and_live_updates(runtime, dtype, bias):
    engine, calls = runtime
    projection = nn.Linear(16, 7, dtype=dtype, bias=bias)
    encoder = engine.with_epilogue(pe.epilogues.TokenProjection(projection))
    original_dtype = projection.weight.dtype
    hidden = engine.forward_packed(batch()).clone()
    expected = F.normalize(projection(hidden.to(dtype)), dim=-1)
    calls.clear()
    actual = encoder.forward_packed(batch())
    assert actual.dtype == original_dtype
    torch.testing.assert_close(actual, expected)
    assert calls == (["graph", "linear"] if dtype == torch.bfloat16 and not bias else ["graph"])
    projection.weight.mul_(.5)
    if bias:
        projection.bias.add_(.5)
    expected = F.normalize(projection(hidden.to(dtype)), dim=-1)
    torch.testing.assert_close(encoder.forward_packed(batch()), expected)
    assert projection.weight.dtype == original_dtype
    encoder.close()
    with pytest.raises(pe.PackedEncodersError, match="closed"):
        encoder.forward_packed(batch())
    assert engine.forward_packed(batch()).shape == (8, 16)


@torch.no_grad()
def test_existing_graph_dispatch_metadata_and_output_ownership(runtime):
    engine, calls = runtime
    inputs, seen = batch(), []

    def forward(hidden, metadata):
        seen.append(metadata)
        return hidden[:, :7]  # A view of the backbone's reusable graph output.

    encoder = engine.with_epilogue(SimpleNamespace(prepare=lambda ops: forward))
    expected = engine.forward_packed(inputs)[:, :7].clone()
    calls.clear()
    actual = encoder.forward_packed(inputs)
    assert calls == ["graph"] and seen == [inputs]
    other = pe.PackedBatch(inputs.input_ids + 8, host_lengths=inputs.host_lengths)
    engine.forward_packed(other)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    engine.state.graph_enabled = False
    encoder.forward_packed(other)
    assert calls[-1] == "eager"
    engine.close()
    with pytest.raises(pe.PackedEncodersError, match="closed"):
        encoder.forward_packed(inputs)


@torch.no_grad()
def test_preparation_failure_releases_head_and_keeps_runtime_usable(runtime):
    engine, _ = runtime
    operations = []

    def prepare(ops):
        operations.append(ops.prepare_linear(nn.Linear(16, 7, dtype=torch.bfloat16)))
        raise RuntimeError("bad custom head")

    with pytest.raises(RuntimeError, match="bad custom head"):
        engine.with_epilogue(SimpleNamespace(prepare=prepare))
    assert not engine._epilogues
    assert operations[0]._weight is None
    with pytest.raises(pe.PackedEncodersError, match="closed"):
        operations[0](torch.randn(8, 16))
    assert engine.forward_packed(batch()).shape == (8, 16)


def test_precision_and_execution_guards(runtime):
    engine, _ = runtime
    for projection, precision in [(nn.Linear(16, 7), "int8"),
                                  (nn.Linear(16, 7, dtype=torch.bfloat16), "fp32"),
                                  (nn.Linear(16, 7, dtype=torch.float64), None),
                                  (nn.Linear(16, 7, device="meta"), None)]:
        with pytest.raises(pe.PackedEncodersError):
            engine.with_epilogue(pe.epilogues.TokenProjection(projection, precision=precision))
    encoder = engine.with_epilogue(pe.epilogues.TokenProjection(nn.Linear(16, 7), precision="fp32"))
    with pytest.raises(pe.PackedEncodersError, match="no_grad"):
        encoder.forward_packed(batch())
    with torch.no_grad():
        with torch.autocast("cpu"):
            with pytest.raises(pe.PackedEncodersError, match="autocast"):
                encoder.forward_packed(batch())
        engine.binding.weight_source.train()
        with pytest.raises(pe.PackedEncodersError, match="eval"):
            encoder.forward_packed(batch())


def test_cuda_context_rejects_cpu_backbone():
    runtime = SimpleNamespace(binding=SimpleNamespace(weight_source=nn.Linear(4, 4)))
    with pytest.raises(pe.PackedEncodersError, match="CUDA backbone"):
        CUDAPreparation(runtime)
