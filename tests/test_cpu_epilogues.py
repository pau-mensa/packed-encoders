"""CPU retriever heads: packed metadata, native projections and owned lifetimes."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import ModernBertConfig, ModernBertModel

import packed_encoders as pe


@pytest.fixture
def model():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = ModernBertModel(ModernBertConfig(
            vocab_size=64, hidden_size=128, intermediate_size=192,
            num_attention_heads=2, num_hidden_layers=2, max_position_embeddings=256,
            pad_token_id=0,
        )).eval()
    yield model
    pe.unpack(model)
    torch.set_num_threads(threads)


def batch():
    return pe.PackedBatch(torch.arange(18) % 61 + 3, host_lengths=(3, 5, 4, 6))


@pytest.mark.parametrize("backend", ["auto", "sdpa"])
@pytest.mark.parametrize("normalize,bias,width", [(True, False, 64), (False, True, 37)])
@torch.no_grad()
def test_projection_matches_hf_and_keeps_backbone_output(model, backend, normalize, bias, width):
    inputs = batch()
    projection = nn.Linear(128, width, bias=bias)
    ids = torch.zeros(4, 6, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    offset = 0
    for row, length in enumerate(inputs.host_lengths):
        ids[row, :length] = inputs.input_ids[offset:offset + length]
        mask[row, :length] = True
        offset += length
    reference_hidden = model(ids, attention_mask=mask).last_hidden_state[mask]
    reference = projection(reference_hidden)
    if normalize:
        reference = F.normalize(reference, dim=-1)
    pe.pack(model, engine=pe.ModernBertCPU(threads=2, max_tokens=8), attention_backend=backend)
    runtime = pe.get_engine(model)
    encoder = runtime.with_epilogue(pe.epilogues.TokenProjection(projection, normalize=normalize))
    result = encoder.forward_packed(inputs)
    torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(model(ids, attention_mask=mask).last_hidden_state[mask],
                               reference_hidden, atol=2e-5, rtol=2e-5)
    assert runtime.forward_packed(inputs).shape == (18, 128)
    saved = result.clone()
    encoder.forward_packed(pe.PackedBatch(inputs.input_ids.flip(0), host_lengths=inputs.host_lengths))
    torch.testing.assert_close(result, saved, rtol=0, atol=0)


@torch.no_grad()
def test_custom_head_prepared_once_and_receives_local_metadata(model):
    pe.pack(model, engine=pe.ModernBertCPU(max_tokens=8), attention_backend="sdpa")
    runtime = pe.get_engine(model)
    inputs = batch()
    positions = torch.cat([torch.arange(n) + 2 for n in inputs.host_lengths])
    inputs = pe.PackedBatch(inputs.input_ids, cu_seqlens=torch.tensor([0, 3, 8, 12, 18], dtype=torch.int32),
                            position_ids=positions)
    calls, preparations = [], []
    first, second = nn.Linear(128, 64), nn.Linear(64, 64)

    def prepare(cpu):
        preparations.append(cpu)
        a, b = cpu.prepare_linear(first), cpu.prepare_linear(second)

        def forward(hidden, chunk):
            calls.append(chunk)
            x = a(hidden)
            return b(x) + x  # Same-width projections must not overwrite x.

        return forward

    encoder = runtime.with_epilogue(SimpleNamespace(prepare=prepare))
    hidden = runtime.forward_packed(inputs)
    expected = first(hidden)
    expected = second(expected) + expected
    torch.testing.assert_close(encoder.forward_packed(inputs), expected, atol=2e-5, rtol=2e-5)
    assert len(preparations) == 1
    assert [chunk.host_lengths for chunk in calls] == [(3, 5), (4,), (6,)]
    assert [chunk.cu_seqlens.tolist() for chunk in calls] == [[0, 3, 8], [0, 4], [0, 6]]
    assert [chunk.max_seqlen for chunk in calls] == [5, 4, 6]
    assert torch.equal(torch.cat([chunk.input_ids for chunk in calls]), inputs.input_ids)
    assert torch.equal(torch.cat([chunk.position_ids for chunk in calls]), positions)
    with pytest.raises(pe.PackedEncodersError, match="only available during"):
        preparations[0].prepare_linear(first)


@torch.no_grad()
def test_custom_scratch_is_owned_across_chunks_and_requests(model):
    pe.pack(model, engine=pe.ModernBertCPU(max_tokens=8))
    scratch = torch.empty(8, 3)

    def forward(hidden, chunk):
        output = scratch[:hidden.shape[0]]
        output.copy_(chunk.input_ids[:, None])
        return output

    encoder = pe.get_engine(model).with_epilogue(SimpleNamespace(prepare=lambda cpu: forward))
    output = encoder.forward_packed(batch())
    expected = batch().input_ids[:, None].expand(-1, 3).float()
    torch.testing.assert_close(output, expected)
    scratch.zero_()
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("backend", ["auto", "sdpa"])
@torch.no_grad()
def test_prepared_linears_compose_and_read_weight_updates(model, backend):
    pe.pack(model, attention_backend=backend)
    runtime = pe.get_engine(model)
    first, second = nn.Linear(128, 64, bias=False), nn.Linear(64, 64, bias=False)
    operations = []

    def prepare(cpu):
        operations.extend([cpu.prepare_linear(first), cpu.prepare_linear(second)])
        return lambda hidden, chunk: operations[1](operations[0](hidden))

    encoder = runtime.with_epilogue(SimpleNamespace(prepare=prepare))
    x = torch.randn(7, 128)
    y = operations[0](x)
    z = operations[1](y)
    torch.testing.assert_close(y, first(x), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(z, second(first(x)), atol=2e-5, rtol=2e-5)
    # A non-contiguous input takes the PyTorch fallback.
    x = torch.randn(128, 7).T
    torch.testing.assert_close(operations[0](x), first(x), atol=2e-5, rtol=2e-5)
    first.weight.mul_(.7)
    second.weight.add_(.01)
    torch.testing.assert_close(operations[0](x.contiguous()), first(x), atol=2e-5, rtol=2e-5)
    hidden = runtime.forward_packed(batch())
    torch.testing.assert_close(encoder.forward_packed(batch()), second(first(hidden)), atol=2e-5, rtol=2e-5)


@torch.no_grad()
def test_close_and_failed_preparation_release_operations(model):
    pe.pack(model)
    runtime = pe.get_engine(model)
    projection = nn.Linear(128, 64, bias=False)
    operations = []

    def failing(cpu):
        operations.append(cpu.prepare_linear(projection))
        raise ValueError("preparation failed")

    with pytest.raises(ValueError, match="preparation failed"):
        runtime.with_epilogue(SimpleNamespace(prepare=failing))
    assert operations[0]._sme is None and operations[0]._weight is None
    with pytest.raises(pe.PackedEncodersError, match="closed"):
        operations[0](torch.randn(2, 128))
    assert not runtime._epilogues
    with pytest.raises(pe.PackedEncodersError, match="callable"):
        runtime.with_epilogue(SimpleNamespace(prepare=lambda cpu: None))
    first = runtime.with_epilogue(pe.epilogues.TokenProjection(projection))
    second = runtime.with_epilogue(pe.epilogues.TokenProjection(projection))
    first.close()
    first.close()
    with pytest.raises(pe.PackedEncodersError, match="closed"):
        first.forward_packed(batch())
    assert second.forward_packed(batch()).shape == (18, 64)
    assert runtime.forward_packed(batch()).shape == (18, 128)
    operation = second._context._operations[0]
    pe.unpack(model)
    assert operation._sme is None and operation._weight is None
    assert not runtime._epilogues
    second.close()
    with pytest.raises(pe.PackedEncodersError, match="closed"):
        second.forward_packed(batch())
    with pytest.raises(pe.PackedEncodersError, match="closed"):
        runtime.with_epilogue(pe.epilogues.TokenProjection(projection))


@pytest.mark.parametrize("kind", ["precision", "dtype", "hooks", "subclass", "bias", "noncontiguous"])
def test_invalid_projection_rejected(model, kind):
    pe.pack(model, attention_backend="sdpa")
    projection = nn.Linear(128, 64)
    if kind == "dtype":
        projection.double()
    elif kind == "hooks":
        projection.register_forward_hook(lambda *args: None)
    elif kind == "subclass":
        class CustomLinear(nn.Linear):
            pass
        projection = CustomLinear(128, 64)
    elif kind == "bias":
        projection.bias = nn.Parameter(torch.zeros(1))
    elif kind == "noncontiguous":
        projection.weight = nn.Parameter(torch.zeros(128, 64).T)
    epilogue = pe.epilogues.TokenProjection(projection, precision="int8" if kind == "precision" else "fp32")
    with pytest.raises(pe.PackedEncodersError):
        pe.get_engine(model).with_epilogue(epilogue)


@pytest.mark.parametrize("bad", [lambda x: x[:-1], lambda x: x[None], lambda x: x.long(),
                                lambda x: x[:, :0], lambda x: None])
@torch.no_grad()
def test_invalid_epilogue_output_rejected(model, bad):
    pe.pack(model, attention_backend="sdpa")
    encoder = pe.get_engine(model).with_epilogue(
        SimpleNamespace(prepare=lambda cpu: lambda hidden, chunk: bad(hidden)))
    with pytest.raises(pe.PackedEncodersError, match="preserving token count"):
        encoder.forward_packed(batch())


@pytest.mark.parametrize("change", ["width", "dtype"])
@torch.no_grad()
def test_inconsistent_chunk_outputs_rejected(model, change):
    pe.pack(model, engine=pe.ModernBertCPU(max_tokens=8), attention_backend="sdpa")

    def forward(hidden, chunk):
        if change == "width":
            return hidden[:, :hidden.shape[0]]
        return hidden.double() if hidden.shape[0] == 8 else hidden

    encoder = pe.get_engine(model).with_epilogue(SimpleNamespace(prepare=lambda cpu: forward))
    with pytest.raises(pe.PackedEncodersError, match="agree across chunks"):
        encoder.forward_packed(batch())


def test_inference_guards_and_activation_validation(model):
    pe.pack(model, attention_backend="sdpa")
    encoder = pe.get_engine(model).with_epilogue(pe.epilogues.TokenProjection(nn.Linear(128, 64)))
    operation = encoder._context._operations[0]
    with pytest.raises(pe.PackedEncodersError, match="no_grad"):
        encoder.forward_packed(batch())
    with pytest.raises(pe.PackedEncodersError, match="no_grad"):
        operation(torch.randn(2, 128))
    with torch.no_grad():
        for x in (torch.randn(2, 127), torch.randn(2, 128).double(), torch.tensor(1.),
                  torch.empty(2, 128, device="meta")):
            with pytest.raises(pe.PackedEncodersError, match="activations"):
                operation(x)
        with torch.autocast("cpu"):
            with pytest.raises(pe.PackedEncodersError, match="autocast"):
                encoder.forward_packed(batch())
            with pytest.raises(pe.PackedEncodersError, match="autocast"):
                operation(torch.randn(2, 128))
        model.train()
        with pytest.raises(pe.PackedEncodersError, match="eval"):
            encoder.forward_packed(batch())


@torch.no_grad()
def test_fp32_head_on_int8_backbone(model):
    pe.pack(model, precision="int8", engine=pe.ModernBertCPU(max_tokens=8))
    runtime = pe.get_engine(model)
    if runtime.precision != "int8":
        pytest.skip("native int8 unavailable")
    projection = nn.Linear(128, 64, bias=False)
    encoder = runtime.with_epilogue(pe.epilogues.TokenProjection(projection))
    reference = F.normalize(projection(runtime.forward_packed(batch())), dim=-1)
    torch.testing.assert_close(encoder.forward_packed(batch()), reference, atol=2e-5, rtol=2e-5)


@torch.no_grad()
def test_head_tuning_does_not_replace_backbone_tiles(model, monkeypatch):
    from packed_encoders.cpu import attention

    def tune(linear, projections, threads, l1_bytes, max_tokens):
        weight, _ = next(iter(projections))
        key = (16, *weight.shape)
        return {key: (2, 4)}, {key: {"selected": (2, 4)}}

    pe.pack(model, attention_backend="auto")
    runtime = pe.get_engine(model)
    cpu = runtime.attention
    if not cpu.native_linear:
        pytest.skip("native AVX-512 GEMM required")
    key = (16, 128, 128)
    cpu.linear_tiles[key] = (3, 8)
    cpu.linear_tuning[key] = {"selected": (3, 8)}
    runtime.tune_gemm = True
    monkeypatch.setattr(attention, "tune_tiles", tune)
    encoder = runtime.with_epilogue(pe.epilogues.TokenProjection(nn.Linear(128, 128)))
    operation = encoder._context._operations[0]
    assert operation._tiles[key] == (2, 4)
    assert cpu.linear_tiles[key] == (3, 8)
    assert cpu.linear_tuning[key]["selected"] == (3, 8)
    x = torch.randn(16, 128)
    saved = operation(x)
    expected = saved.clone()
    assert saved.data_ptr() not in {t.data_ptr() for t in cpu.linear_buffers.values()}
    operation(x + 1)
    torch.testing.assert_close(saved, expected, rtol=0, atol=0)


@pytest.mark.parametrize("normalize", [False, True])
@torch.no_grad()
def test_builtin_head_returns_its_owned_output_without_another_copy(model, monkeypatch, normalize):
    pe.pack(model, attention_backend="sdpa")
    encoder = pe.get_engine(model).with_epilogue(
        pe.epilogues.TokenProjection(nn.Linear(128, 64), normalize=normalize))
    hidden = torch.randn(18, 128)
    produced = []
    forward = encoder._forward

    def record(*args):
        output = forward(*args)
        produced.append(output)
        return output

    record._packed_encoders_owned_output = forward._packed_encoders_owned_output
    encoder._forward = record
    monkeypatch.setattr(encoder._runtime, "_forward_epilogue", lambda inputs, apply: apply(hidden, inputs))
    assert encoder.forward_packed(batch()) is produced[0]
