"""Python call preparation at baseline 822c9b8; used only by scheduling A/B."""
import ctypes
import torch
from packed_encoders.cpu.int8 import _version, TABLES
from packed_encoders.errors import PackedEncodersError


def core(self, ids, lengths, positions):
    engine = self.engine
    if list(map(_version, self.weights)) != self.versions:
        raise PackedEncodersError("CPU int8 projection weights changed in place since packing; unpack and repack")
    tables = {kind: (cos[positions].contiguous(), sin[positions].contiguous())
              for kind, (cos, sin) in engine.tables.items()}
    boundaries = torch.tensor(lengths, dtype=torch.int32)
    x = engine.model.embeddings(ids[None])
    normed = engine.layers[0].attn_norm(x).contiguous()
    tokens = x.shape[1]
    self._reserve(tokens)
    out = torch.empty_like(x)
    b = self.buffers
    for kind, index in TABLES.items():
        b.cos[index], b.sin[index] = (t.data_ptr() for t in tables[kind])
    b.residual, b.out = x.data_ptr(), out.data_ptr()
    rope = self.rope[:3 * engine.heads * tokens * 64].view(3, 1, engine.heads, tokens, 64)
    model, buffers = ctypes.addressof(self.geometry), ctypes.addressof(b)
    if self.library.pe_q8_input(model, ctypes.addressof(self.first), buffers, normed.data_ptr(), tokens, self.workers):
        raise PackedEncodersError("native CPU int8 input projection failed")
    for kind, segment in zip(engine.kinds, self.segments):
        q, k, v = rope.unbind(0)
        # x86 segments read token-major context; Apple's gather heads from head-major.
        attention = engine.attention.local if kind == "sliding_attention" else engine.attention.global_
        context = attention(q, k, v, lengths, boundaries, token_major=self.x86).contiguous()
        if self.library.pe_q8_segment(model, ctypes.addressof(segment), buffers, context.data_ptr(),
                                      tokens, self.workers):
            raise PackedEncodersError("native CPU int8 layer failed")
    return out[0]


def _native(self, q, k, v, lengths, boundaries, half, token_major=False):
    # Keys within `half` positions; at least the longest length is global.
    q, k, v = (t.contiguous() for t in (q, k, v))
    _, heads, tokens, dim = q.shape
    workers = min(self.threads, heads * len(lengths))
    # Scores [tile, columns] per worker; x86 also transposes the key window, padded to 16 keys.
    columns = -(-min(self.tile + 2 * half, max(lengths)) // 16) * 16
    needed = workers * (self.tile + 64) * columns
    if self.scratch is None or self.scratch.numel() < needed:
        self.scratch = torch.empty(needed, dtype=torch.float32)
    direct = token_major and self.token_major
    shape = (1, tokens, heads * dim) if direct else q.shape
    if self.output is None or self.output.shape != shape:
        self.output = torch.empty(shape, dtype=torch.float32)
    status = self.library.pe_local_attention(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), self.output.data_ptr(), boundaries.data_ptr(),
        tokens, heads, len(lengths), self.tile, half, direct, workers, self.scratch.data_ptr(),
        self.scratch.numel(),
    )
    if status:
        raise PackedEncodersError(f"native CPU attention failed: {status}")
    return self.output if direct or not token_major else self._layout(self.output, True)
