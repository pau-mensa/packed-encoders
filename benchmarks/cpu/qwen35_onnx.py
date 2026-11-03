"""Portable ONNX Qwen3.5 text baseline: ordinary ops, chunked delta rule, no custom kernels.

Sequence width is explicit in each artifact; batch size is dynamic. Weights live in an
external file, shared by all sequence widths. The unit-lower-triangular delta-rule solve
uses block triangular inversion, doubling block size at each stage, avoiding
an unsupported SolveTriangular op or a token-by-token ONNX Loop. Every exported width
must pass the benchmark's independent HF parity gate before being timed.
"""
from __future__ import annotations

import math
import copy
import tempfile
from pathlib import Path

import numpy as np
import torch


class Graph:
    def __init__(self, directory, seq, write_weights):
        import onnx
        self.onnx = onnx
        self.directory, self.seq = Path(directory), seq
        self.nodes, self.initializers, self.weights = [], [], {}
        self.counter = 0
        self.weight_path = self.directory / "weights.fp32.bin"
        self.file = (tempfile.NamedTemporaryFile(mode="wb", prefix=".qwen35-weights-", dir=self.directory, delete=False)
                     if write_weights else None)
        self.offset = 0

    def constant(self, value, dtype=None):
        name = f"c{len(self.initializers)}"
        self.initializers.append(self.onnx.numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(self, kind, *inputs, name=None, **attrs):
        self.counter += 1
        output = f"v{self.counter}"
        self.nodes.append(self.onnx.helper.make_node(kind, list(inputs), [output], name=name or output, **attrs))
        return output

    def weight(self, tensor, name, transpose=False, plus_one=False):
        if name in self.weights:
            return self.weights[name]
        shape = list(reversed(tensor.shape)) if transpose else list(tensor.shape)
        size = tensor.numel() * 4
        if self.file:
            values = tensor.detach().float()
            if plus_one:
                values = values + 1
            if transpose:
                values = values.T
            array = values.contiguous().numpy()
            self.file.write(array.tobytes())
        proto = self.onnx.TensorProto(name=name, data_type=self.onnx.TensorProto.FLOAT, dims=shape,
                                     data_location=self.onnx.TensorProto.EXTERNAL)
        for key, value in (("location", self.weight_path.name), ("offset", str(self.offset)), ("length", str(size))):
            entry = proto.external_data.add()
            entry.key, entry.value = key, value
        self.offset += size
        self.initializers.append(proto)
        self.weights[name] = name
        return name

    def reshape(self, x, shape):
        return self.op("Reshape", x, self.constant(shape, np.int64))

    def trans(self, x, perm):
        return self.op("Transpose", x, perm=perm)

    def unsqueeze(self, x, axes):
        return self.op("Unsqueeze", x, self.constant(axes, np.int64))

    def slice(self, x, start, end, axis):
        return self.op("Slice", x, self.constant([start], np.int64), self.constant([end], np.int64),
                       self.constant([axis], np.int64))

    def mul(self, x, y):
        return self.op("Mul", x, y)

    def silu(self, x):
        return self.mul(x, self.op("Sigmoid", x))

    def norm(self, x, weight, eps, name, zero_centered=True, l2=False):
        square = self.mul(x, x)
        if l2:
            variance = self.op("ReduceSum", square, self.constant([-1], np.int64), keepdims=1)
        else:
            variance = self.op("ReduceMean", square, self.constant([-1], np.int64), keepdims=1)
        denom = self.op("Sqrt", self.op("Add", variance, self.constant(eps, np.float32)))
        out = self.op("Div", x, denom)
        return out if weight is None else self.mul(out, self.weight(weight, name, plus_one=zero_centered))

    def linear(self, x, module, name):
        if module.bias is not None:
            raise ValueError("ONNX Qwen baseline requires bias-free projections")
        return self.op("MatMul", x, self.weight(module.weight, name + ".weight", transpose=True), name=name)

    def chunk_delta(self, q, k, v, decay, beta, nk, nv, hk, hv):
        s = self.seq
        c = min(64, 2 ** math.ceil(math.log2(s)))
        chunks = (s + c - 1) // c
        # Heads are B,H,S,D, gates B,H,S; grouped heads use a Gather rather than materialized repeats.
        q, k = (self.trans(self.reshape(x, [0, s, nk, hk]), [0, 2, 1, 3]) for x in (q, k))
        if nk != nv:
            idx = self.constant(np.arange(nv, dtype=np.int64) // (nv // nk))
            q, k = (self.op("Gather", x, idx, axis=1) for x in (q, k))
        q, k = (self.norm(x, None, 1e-6, "", l2=True) for x in (q, k))
        q = self.mul(q, self.constant(hk ** -.5, np.float32))
        v = self.trans(self.reshape(v, [0, s, nv, hv]), [0, 2, 1, 3])
        decay, beta = (self.trans(x, [0, 2, 1]) for x in (decay, beta))
        if chunks * c != s:
            pads = self.constant([0, 0, 0, 0, 0, 0, chunks * c - s, 0], np.int64)
            q, k, v = (self.op("Pad", x, pads) for x in (q, k, v))
            pads = self.constant([0, 0, 0, 0, 0, chunks * c - s], np.int64)
            decay, beta = (self.op("Pad", x, pads) for x in (decay, beta))
        q, k, v = (self.reshape(x, [0, nv, chunks, c, d]) for x, d in ((q, hk), (k, hk), (v, hv)))
        decay, beta = (self.reshape(x, [0, nv, chunks, c]) for x in (decay, beta))
        cumulative = self.op("CumSum", decay, self.constant(3, np.int64))
        pair = self.op("Sub", self.unsqueeze(cumulative, [4]), self.unsqueeze(cumulative, [3]))
        lower = self.constant(np.tril(np.ones((c, c), dtype=bool)))
        pair = self.op("Exp", self.op("Where", lower, pair, self.constant(-np.inf, np.float32)))
        kt = self.trans(k, [0, 1, 2, 4, 3])
        kb = self.mul(k, self.unsqueeze(beta, [-1]))
        vb = self.mul(v, self.unsqueeze(beta, [-1]))
        system = self.mul(self.op("MatMul", kb, kt), pair)
        system = self.op("Where", self.constant(np.tril(np.ones((c, c), dtype=bool), -1)),
                         system, self.constant(0., np.float32))
        # Invert unit-lower-triangular blocks: [A 0; B D]^-1 =
        # [A^-1 0; -D^-1 B A^-1 D^-1]. All diagonal blocks of a given
        # size run in one batched MatMul. A nilpotent power series is much
        # less stable for the near-collinear keys found in trained models.
        inverse = None
        for level in range(1, int(math.log2(c)) + 1):
            size, half = 2 ** level, 2 ** (level - 1)
            nb = c // size
            blocks = self.reshape(system, [0, nv, chunks, nb, size, nb, size])
            blocks = self.reshape(self.trans(blocks, [0, 1, 2, 3, 5, 4, 6]), [0, nv, chunks, nb * nb, size, size])
            diagonal = self.op("Gather", blocks, self.constant(np.arange(nb, dtype=np.int64) * (nb + 1)), axis=3)
            below = self.slice(self.slice(diagonal, half, size, 4), 0, half, 5)
            if inverse is None:
                one = self.op("Add", self.mul(below, self.constant(0., np.float32)), self.constant(1., np.float32))
                a_inv = d_inv = one
            else:
                previous = self.reshape(inverse, [0, nv, chunks, nb, 2, half, half])
                a_inv = self.op("Gather", previous, self.constant(0, np.int64), axis=4)
                d_inv = self.op("Gather", previous, self.constant(1, np.int64), axis=4)
            off = self.op("Neg", self.op("MatMul", self.op("MatMul", d_inv, below), a_inv))
            zero = self.mul(a_inv, self.constant(0., np.float32))
            top = self.op("Concat", a_inv, zero, axis=5)
            bottom = self.op("Concat", off, d_inv, axis=5)
            inverse = self.op("Concat", top, bottom, axis=4)
        inverse = self.reshape(inverse, [0, nv, chunks, c, c]) if inverse is not None else self.constant([[1.]], np.float32)
        cum_exp = self.unsqueeze(self.op("Exp", cumulative), [-1])
        u = self.op("MatMul", inverse, vb)
        w = self.op("MatMul", inverse, self.mul(kb, cum_exp))
        intra = self.mul(self.op("MatMul", q, kt), pair)
        q = self.mul(q, cum_exp)
        last = self.slice(cumulative, c - 1, c, 3)
        end_decay = self.unsqueeze(self.op("Exp", last), [-1])
        k = self.mul(k, self.unsqueeze(self.op("Exp", self.op("Sub", last, cumulative)), [-1]))
        output, state = [], None
        def chunk(x, i):
            return self.op("Gather", x, self.constant(i, np.int64), axis=2)
        for i in range(chunks):
            values = chunk(u, i)
            if state is not None:
                values = self.op("Sub", values, self.op("MatMul", chunk(w, i), state))
            out = self.op("MatMul", chunk(intra, i), values)
            if state is not None:
                out = self.op("Add", out, self.op("MatMul", chunk(q, i), state))
            output.append(out)
            if i + 1 < chunks:
                update = self.op("MatMul", self.trans(chunk(k, i), [0, 1, 3, 2]), values)
                state = update if state is None else self.op("Add", self.mul(state, chunk(end_decay, i)), update)
        output = self.op("Concat", *output, axis=2) if len(output) > 1 else output[0]
        if chunks * c != s:
            output = self.slice(output, 0, s, 2)
        return self.trans(output, [0, 2, 1, 3])

    def gdn(self, x, g, name, mask):
        s, nk, nv, hk, hv = self.seq, g.num_k_heads, g.num_v_heads, g.head_k_dim, g.head_v_dim
        x = self.mul(x, mask)
        qkv = self.trans(self.linear(x, g.in_proj_qkv, name + ".qkv"), [0, 2, 1])
        inputs = [qkv, self.weight(g.conv1d.weight, name + ".conv")]
        if g.conv1d.bias is not None:
            inputs.append(self.weight(g.conv1d.bias, name + ".conv_bias"))
        qkv = self.op("Conv", *inputs, group=2 * nk * hk + nv * hv,
                      pads=[g.conv_kernel_size - 1, 0], kernel_shape=[g.conv_kernel_size])
        qkv = self.trans(self.silu(qkv), [0, 2, 1])
        q = self.slice(qkv, 0, nk * hk, 2)
        k = self.slice(qkv, nk * hk, 2 * nk * hk, 2)
        v = self.slice(qkv, 2 * nk * hk, 2 * nk * hk + nv * hv, 2)
        z = self.linear(x, g.in_proj_z, name + ".z")
        beta = self.op("Sigmoid", self.linear(x, g.in_proj_b, name + ".beta"))
        a = self.op("Add", self.linear(x, g.in_proj_a, name + ".a"), self.weight(g.dt_bias, name + ".dt"))
        decay = self.mul(self.op("Neg", self.op("Exp", self.weight(g.A_log, name + ".alog"))), self.op("Softplus", a))
        out = self.chunk_delta(q, k, v, decay, beta, nk, nv, hk, hv)
        out = self.norm(out, g.norm.weight, g.norm.variance_epsilon, name + ".norm", zero_centered=False)
        out = self.mul(out, self.silu(self.reshape(z, [0, s, nv, hv])))
        return self.linear(self.reshape(out, [0, s, nv * hv]), g.out_proj, name + ".out")

    def attention(self, x, a, cfg, name, cos, sin, bias):
        s, nq, nk, hd = self.seq, cfg.num_attention_heads, cfg.num_key_value_heads, a.head_dim
        qg = self.reshape(self.linear(x, a.q_proj, name + ".q"), [0, s, nq, 2 * hd])
        q = self.slice(qg, 0, hd, 3)
        gate = self.slice(qg, hd, 2 * hd, 3)
        k = self.reshape(self.linear(x, a.k_proj, name + ".k"), [0, s, nk, hd])
        v = self.trans(self.reshape(self.linear(x, a.v_proj, name + ".v"), [0, s, nk, hd]), [0, 2, 1, 3])
        q = self.norm(q, a.q_norm.weight, a.q_norm.eps, name + ".qnorm")
        k = self.norm(k, a.k_norm.weight, a.k_norm.eps, name + ".knorm")
        r = cos.shape[-1]
        cos, sin = (self.constant(t.numpy()[:, :, None]) for t in (cos, sin))
        def rope(t):
            rot = self.slice(t, 0, r, 3)
            half = self.op("Concat", self.op("Neg", self.slice(rot, r // 2, r, 3)),
                           self.slice(rot, 0, r // 2, 3), axis=3)
            rotated = self.op("Add", self.mul(rot, cos), self.mul(half, sin))
            return self.trans(self.op("Concat", rotated, self.slice(t, r, hd, 3), axis=3), [0, 2, 1, 3])
        q, k = rope(q), rope(k)
        if nq != nk:
            idx = self.constant(np.arange(nq, dtype=np.int64) // (nq // nk))
            k, v = (self.op("Gather", t, idx, axis=1) for t in (k, v))
        scores = self.mul(self.op("MatMul", q, self.trans(k, [0, 1, 3, 2])), self.constant(a.scaling, np.float32))
        probs = self.op("Softmax", self.op("Add", scores, bias), axis=-1)
        out = self.trans(self.op("MatMul", probs, v), [0, 2, 1, 3])
        out = self.reshape(self.mul(out, self.op("Sigmoid", gate)), [0, s, nq * hd])
        return self.linear(out, a.o_proj, name + ".out")

    def build(self, model):
        cfg, s = model.config, self.seq
        mask = self.unsqueeze(self.op("Cast", "attention_mask", to=self.onnx.TensorProto.FLOAT), [-1])
        keep = self.unsqueeze(self.op("Cast", "attention_mask", to=self.onnx.TensorProto.BOOL), [1, 2])
        if getattr(cfg, "is_causal", True):
            keep = self.op("And", keep, self.constant(np.tril(np.ones((1, 1, s, s), dtype=bool))))
        bias = self.op("Where", keep, self.constant(0., np.float32), self.constant(-np.finfo(np.float32).max, np.float32))
        x = self.op("Gather", self.weight(model.embed_tokens.weight, "embed"), "input_ids", axis=0)
        with torch.no_grad():
            cos, sin = model.rotary_emb(torch.empty(1, s, cfg.hidden_size), torch.arange(s)[None, None].expand(3, 1, -1))
        for i, (layer, kind) in enumerate(zip(model.layers, cfg.layer_types)):
            name = f"layers.{i}"
            h = self.norm(x, layer.input_layernorm.weight, layer.input_layernorm.eps, name + ".inputnorm")
            if kind == "linear_attention":
                mix = self.gdn(h, layer.linear_attn, name + ".gdn", mask)
            else:
                mix = self.attention(h, layer.self_attn, cfg, name + ".attn", cos, sin, bias)
            x = self.op("Add", x, mix)
            h = self.norm(x, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps, name + ".postnorm")
            gate = self.linear(h, layer.mlp.gate_proj, name + ".mlp.gate")
            up = self.linear(h, layer.mlp.up_proj, name + ".mlp.up")
            x = self.op("Add", x, self.linear(self.mul(self.silu(gate), up), layer.mlp.down_proj, name + ".mlp.down"))
        out = self.norm(x, model.norm.weight, model.norm.eps, "finalnorm")
        self.nodes.append(self.onnx.helper.make_node("Identity", [out], ["last_hidden_state"]))
        h = self.onnx.helper
        graph = h.make_graph(self.nodes, "qwen35_cpu", [h.make_tensor_value_info(n, self.onnx.TensorProto.INT64, ["batch", s])
                                                       for n in ("input_ids", "attention_mask")],
                             [h.make_tensor_value_info("last_hidden_state", self.onnx.TensorProto.FLOAT, ["batch", s, cfg.hidden_size])],
                             self.initializers)
        result = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], producer_name="packed-encoders-qwen35-baseline")
        result.ir_version = 9
        if self.file:
            self.file.close()
            Path(self.file.name).replace(self.weight_path)
        elif self.weight_path.stat().st_size != self.offset:
            raise ValueError("ONNX external weights mismatch; export into a fresh directory")
        return result


def export(model, directory, seq):
    import onnx
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    graph = Graph(directory, seq, not (directory / "weights.fp32.bin").exists())
    try:
        model_proto = graph.build(model)
    finally:
        if graph.file and not graph.file.closed:
            graph.file.close()
        if graph.file:
            Path(graph.file.name).unlink(missing_ok=True)
    path = directory / f"fp32_s{seq}.onnx"
    onnx.save_model(model_proto, str(path))
    onnx.checker.check_model(str(path))
    return path


def quantize(src, dst, variant):
    """Only constant projection MatMuls; recurrence, softmax, embedding and norms stay FP32."""
    import onnx
    if variant == "int8_dynamic":
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(src), str(dst), per_channel=True, weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul"], use_external_data_format=True,
                         extra_options={"MatMulConstBOnly": True})
    elif variant == "int8_w8a8":
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        quantizer = MatMulNBitsQuantizer(onnx.load(str(src)), bits=8, block_size=128,
                                         is_symmetric=True, accuracy_level=4)
        quantizer.process()
        quantizer.model.save_model_to_file(str(dst), use_external_data_format=True)
    else:
        raise ValueError(f"unknown ONNX variant: {variant}")


def reuse_quantized_weights(template_fp32, template_quantized, src, dst):
    """Use exactly the same quantized projections at another sequence width.

    Only activation shapes change. Copy each quantized projection's operator subgraph,
    preserving its external weight references, into the new FP32 graph. This avoids
    duplicating gigabytes of identical quantized weights for row-trimmed baselines.
    All four paths must be in the same directory.
    """
    import onnx
    paths = tuple(map(Path, (template_fp32, template_quantized, src, dst)))
    if len({p.parent.resolve() for p in paths}) != 1:
        raise ValueError("quantized weight reuse requires graphs in the same directory")
    original, quantized, target = (onnx.load(str(p), load_external_data=False) for p in paths[:3])
    old = {n.name: n for n in original.graph.node}
    initializers = {t.name: t for t in quantized.graph.initializer}
    producers = {out: node for node in quantized.graph.node for out in node.output}
    replacements, extra = [], {}
    for node in target.graph.node:
        if node.op_type != "MatMul" or node.name not in old or not node.name.startswith("layers."):
            replacements.append(node)
            continue
        before = old[node.name]
        group, visited = [], set()
        activation, output = before.input[0], before.output[0]
        def visit(value):
            if value == activation or value in visited:
                return
            visited.add(value)
            if value in initializers:
                extra[value] = initializers[value]
                return
            producer = producers[value]
            visited.update(producer.output)
            for inp in producer.input:
                if inp:
                    visit(inp)
            group.append(producer)
        visit(output)
        # A projection must actually have been quantized in the template.
        if not any(n.op_type in ("MatMulInteger", "MatMulNBits") for n in group):
            raise ValueError(f"template did not quantize {node.name}")
        names = {activation: node.input[0], output: node.output[0]}
        def rename(value):
            if value in initializers or value == "":
                return value
            return names.setdefault(value, node.name + "/quant/" + value)
        for producer in group:
            cloned = copy.deepcopy(producer)
            cloned.name = node.name + "/quant/" + producer.name
            for i, value in enumerate(cloned.input):
                cloned.input[i] = rename(value)
            for i, value in enumerate(cloned.output):
                cloned.output[i] = rename(value)
            replacements.append(cloned)
    used = {value for node in replacements for value in node.input}
    tensors = {t.name: t for t in target.graph.initializer}
    tensors.update(extra)
    domains = {op.domain for op in target.opset_import}
    target.opset_import.extend(op for op in quantized.opset_import if op.domain not in domains)
    del target.graph.node[:]
    target.graph.node.extend(replacements)
    del target.graph.initializer[:]
    target.graph.initializer.extend(t for name, t in tensors.items() if name in used)
    staging = paths[3].with_suffix(".tmp.onnx")
    onnx.save(target, str(staging))
    onnx.checker.check_model(str(staging))
    staging.replace(paths[3])
    return paths[3]
