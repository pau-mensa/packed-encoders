"""Numerical CUDA heads and real graph replay; no checkpoint downloads."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import packed_encoders as pe
from packed_encoders.engine import ModelBinding
from packed_encoders.pieces.base import Piece
from packed_encoders.pieces.numerical import LINEAR
from packed_encoders.runtime.cuda_epilogue import CUDAEpilogueSupport

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class GraphBackbone(CUDAEpilogueSupport):
    """Small real graph with borrowed output, using the production composition path."""

    def __init__(self, device):
        model = nn.Embedding(32, 16, device=device, dtype=torch.bfloat16).eval()
        self.binding = ModelBinding(None, model, model)
        self.composition = SimpleNamespace(linear=Piece("torch", LINEAR, F.linear, F.linear, lambda *args: None))
        self._closed = False
        self._init_epilogues()
        self.ids = torch.arange(8, device=device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(device), torch.no_grad():
            warmup = torch.cuda.Stream(device=device)
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    model(self.ids)
            torch.cuda.current_stream().wait_stream(warmup)
            with torch.cuda.graph(self.graph):
                self.output = model(self.ids)

    def _require_open(self):
        if self._closed:
            raise pe.PackedEncodersError("closed")

    def forward_packed(self, batch):
        self._require_open()
        self.ids.copy_(batch.input_ids)
        self.graph.replay()
        return self.output

    def close(self):
        self._close_epilogues()
        self.graph = self.output = None
        self._closed = True


@pytest.mark.parametrize("dtype,bias", [(torch.bfloat16, False), (torch.float32, True), (torch.float16, False)])
@torch.no_grad()
def test_projection_after_real_replay_and_owned_outputs(dtype, bias):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    runtime = GraphBackbone(device)
    try:
        projection = nn.Linear(16, 7, bias=bias, device=device, dtype=dtype)
        seen = []

        def prepare(ops):
            project = ops.prepare_linear(projection)

            def forward(hidden, batch):
                assert not torch.cuda.is_current_stream_capturing()
                assert torch.cuda.current_device() == device.index
                seen.append(batch)
                return F.normalize(project(hidden), dim=-1)

            return forward

        encoder = runtime.with_epilogue(SimpleNamespace(prepare=prepare))
        identity = runtime.with_epilogue(SimpleNamespace(prepare=lambda ops: lambda hidden, batch: hidden[:, :7]))
        inputs = pe.PackedBatch(torch.arange(8, device=device), host_lengths=(3, 5))
        reference_hidden = runtime.binding.weight_source(inputs.input_ids)
        reference = F.normalize(projection(reference_hidden.to(dtype)), dim=-1)
        result = encoder.forward_packed(inputs)
        saved_hidden = identity.forward_packed(inputs)
        torch.testing.assert_close(result, reference)
        assert result.dtype == dtype and seen == [inputs]
        encoder.forward_packed(pe.PackedBatch(inputs.input_ids + 8, host_lengths=(3, 5)))
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        torch.testing.assert_close(saved_hidden, reference_hidden[:, :7], rtol=0, atol=0)
        projection.weight.mul_(.7)
        reference = F.normalize(projection(reference_hidden.to(dtype)), dim=-1)
        torch.testing.assert_close(encoder.forward_packed(inputs), reference)
        with torch.autocast("cuda"):
            with pytest.raises(pe.PackedEncodersError, match="autocast"):
                encoder.forward_packed(inputs)
    finally:
        runtime.close()
    with pytest.raises(pe.PackedEncodersError, match="closed"):
        encoder.forward_packed(inputs)


@torch.no_grad()
@pytest.mark.parametrize("normalize", [False, True])
def test_builtin_projection_output_survives_real_graph_replay(normalize):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    runtime = GraphBackbone(device)
    try:
        projection = nn.Linear(16, 7, device=device, dtype=torch.bfloat16)
        encoder = runtime.with_epilogue(pe.epilogues.TokenProjection(projection, normalize=normalize))
        inputs = pe.PackedBatch(torch.arange(8, device=device), host_lengths=(3, 5))
        actual = encoder.forward_packed(inputs)
        expected = actual.clone()
        encoder.forward_packed(pe.PackedBatch(inputs.input_ids + 8, host_lengths=(3, 5)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        runtime.close()


@torch.no_grad()
def test_modernbert_cuda_projection_matches_packed_hidden_states():
    from transformers import ModernBertConfig, ModernBertModel
    from packed_encoders.arch.modernbert import ModernBert

    model = ModernBertModel(ModernBertConfig(
        vocab_size=64, hidden_size=256, intermediate_size=512, num_attention_heads=4,
        num_hidden_layers=2, max_position_embeddings=256, pad_token_id=0,
    )).cuda().bfloat16().eval()
    try:
        pe.pack(model, engine=ModernBert(), attention_backend="sdpa", validate=False)
        runtime = pe.get_engine(model)
        inputs = pe.PackedBatch(torch.arange(3, 11, device="cuda"),
                                cu_seqlens=torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32),
                                max_seqlen=5, position_ids=torch.tensor([0, 1, 2, 0, 1, 2, 3, 4], device="cuda"))
        for dtype, bias in ((torch.bfloat16, False), (torch.float32, True)):
            projection = nn.Linear(256, 64, bias=bias, device="cuda", dtype=dtype)
            encoder = runtime.with_epilogue(pe.epilogues.TokenProjection(projection))
            hidden = runtime.forward_packed(inputs)
            expected = F.normalize(projection(hidden.to(dtype)), dim=-1)
            torch.testing.assert_close(encoder.forward_packed(inputs), expected, rtol=.03, atol=.01)
            assert runtime.forward_packed(inputs).shape == (8, 256)
    finally:
        pe.unpack(model)
