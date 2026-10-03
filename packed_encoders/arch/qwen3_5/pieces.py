"""Qwen's numerical composition; sequence mixing retains its probed runtime policy."""
from dataclasses import dataclass, fields
from types import SimpleNamespace

from packed_encoders.errors import UnsupportedTargetError
from packed_encoders.pieces.base import Piece, PieceValidation


@dataclass(frozen=True)
class Qwen35Pieces:
    linear: Piece
    rms_norm: Piece
    add_rms_norm: Piece
    swiglu: Piece
    conv_split: Piece
    gated_rms_norm: Piece
    qk_norm_rope: Piece
    qk_norm_rope_pair: Piece
    sigmoid_gate: Piece
    gdn_chunk: Piece
    gdn_recurrent: Piece
    attention_packed: Piece
    attention_padded: Piece

    def all(self):
        return tuple(getattr(self, f.name) for f in fields(self))

    def bind(self):
        from packed_encoders.pieces import hybrid as h
        from packed_encoders.pieces.numerical import LINEAR
        from packed_encoders.pieces.varlen import PACKED, PADDED

        contracts = (LINEAR, h.RMS, h.ADD_RMS, h.SWIGLU, h.CONV, h.GATED_RMS, h.QK_ROPE, h.QK_PAIR, h.GATE, h.GDN, h.GDN, PACKED, PADDED)
        for slot, piece, required in zip(fields(self), self.all(), contracts):
            if piece.contract != required:
                raise UnsupportedTargetError(f"incompatible {slot.name} piece {piece.name}: expected {required}")
        return SimpleNamespace(**{f.name: getattr(self, f.name).execute for f in fields(self)})

    def validate(self, engine):
        import torch
        from packed_encoders.runtime.staging import packed_layout_host

        gen = torch.Generator(device=engine.device).manual_seed(17)
        def rand(*shape):
            return torch.randn(shape, device=engine.device, dtype=engine.dtype, generator=gen)
        reports = {}
        def probe(slot, *args):
            reports[slot] = getattr(self, slot).validate(*args)
        def skip(slot, reason):
            p = getattr(self, slot)
            reports[slot] = PieceValidation(p.name, None, p.rtol, p.atol, reason)

        n, d = 17, engine.hidden_size
        first = engine.layers[0]
        x = rand(n, d)
        probe("linear", x, first.gate_up)
        probe("rms_norm", x, first.w_in, engine.eps)
        probe("add_rms_norm", x, rand(n, d), first.w_in, engine.eps)
        probe("swiglu", rand(n, first.inter), rand(n, first.inter))
        gdn = next((l for l in engine.layers if l.linear), None)
        if engine.fused and gdn is not None:
            proj = rand(n, sum(gdn.split_in))
            _, pos = packed_layout_host([5, 1, 11])
            probe("conv_split", proj, gdn.conv_w, gdn.conv_b, pos.to(engine.device), gdn.kd, gdn.vd, gdn.gate_off, gdn.nv, gdn.nv, True)
            probe("gated_rms_norm", rand(n, gdn.nv, gdn.hv), rand(n, gdn.nv, gdn.hv), gdn.gn_w, gdn.gn_eps)
        else:
            for slot in ("conv_split", "gated_rms_norm"):
                skip(slot, "no fused GatedDeltaNet layer")
        attn = next((l for l in engine.layers if not l.linear), None)
        if attn is not None:
            cos, sin = engine._rope(torch.arange(n, device=engine.device))
            if engine.fused:
                qw = attn.nq * attn.hd * (2 if attn.gated else 1)
                probe("qk_norm_rope_pair", rand(n, qw+2*attn.nkv*attn.hd), 0, attn.hd*(2 if attn.gated else 1), attn.nq, qw, attn.hd, attn.nkv, attn.wqk, cos, sin, attn.qk_eps)
                skip("qk_norm_rope", "fused Q/K pair selected")
            else:
                probe("qk_norm_rope", rand(n, attn.nq, attn.hd), attn.wq, cos, sin, attn.qk_eps)
                skip("qk_norm_rope_pair", "unfused Q/K selected")
            if engine.fused and attn.gated:
                probe("sigmoid_gate", rand(n, attn.nq, attn.hd), rand(n, attn.nq, attn.hd))
            else:
                skip("sigmoid_gate", "no fused attention output gate")
        else:
            for slot in ("qk_norm_rope", "qk_norm_rope_pair", "sigmoid_gate"):
                skip(slot, "no softmax attention layer")
        if gdn is not None:
            cu_cpu, _ = packed_layout_host([5, 1, 11])
            cu = cu_cpu.to(engine.device)
            heads = gdn.nv if engine.gdn.expand_gva else gdn.nk
            args = (rand(1, n, heads, gdn.hk), rand(1, n, heads, gdn.hk),
                    rand(1, n, gdn.nv, gdn.hv), rand(1, n, gdn.nv), rand(1, n, gdn.nv), gdn.A_log, gdn.dt_bias)
            probe("gdn_chunk", *args, cu, cu_cpu)
            # Also exercise the rectangular layout captured in CUDA graphs.
            padded_args = (rand(2, 9, heads, gdn.hk), rand(2, 9, heads, gdn.hk),
                           rand(2, 9, gdn.nv, gdn.hv), rand(2, 9, gdn.nv), rand(2, 9, gdn.nv), gdn.A_log, gdn.dt_bias)
            self.gdn_chunk.validate(*padded_args)
            if engine.gdn.recurrent_max_len:
                probe("gdn_recurrent", *args, cu, cu_cpu)
            else:
                skip("gdn_recurrent", "recurrent policy disabled by probe")
        else:
            for slot in ("gdn_chunk", "gdn_recurrent"):
                skip(slot, "no GatedDeltaNet layer")
        if engine.attention is not None:
            from packed_encoders.runtime.graphs import PaddedStatic
            cu_cpu, _ = packed_layout_host([5, 1, 11])
            q, k, v = rand(n, attn.nq, attn.hd), rand(n, attn.nkv, attn.hd), rand(n, attn.nkv, attn.hd)
            probe("attention_packed", engine.attention, q, k, v, cu_cpu.to(engine.device, torch.int32), 11, [5, 1, 11], engine.causal)
            static = PaddedStatic(2, 9, torch.zeros(2, 9, dtype=torch.long, device=engine.device),
                                  torch.tensor([0, 5, 9, 18, 18], dtype=torch.int32, device=engine.device),
                                  torch.tensor([5, 9], dtype=torch.int32, device=engine.device))
            probe("attention_padded", engine.attention, rand(18, attn.nq, attn.hd),
                  rand(18, attn.nkv, attn.hd), rand(18, attn.nkv, attn.hd), static, engine.causal)
        else:
            for slot in ("attention_packed", "attention_padded"):
                skip(slot, "no softmax attention layer")
        return reports


def default_pieces():
    from packed_encoders.pieces.hybrid import default_numerical, gdn_piece
    from packed_encoders.pieces.varlen import attention_pieces
    packed, padded = attention_pieces()
    return Qwen35Pieces(**default_numerical(), gdn_chunk=gdn_piece(), gdn_recurrent=gdn_piece(recurrent=True),
                        attention_packed=packed, attention_padded=padded)
