"""Token-level parity forensics for topk-embed-v1 under packed-encoders.

    python benchmarks/qwen35_parity_probe.py --model topk-io/topk-embed-v1-small --data sample.json

A mean cosine hides single tokens. This collects *unnormalised* per-token vectors (the head
output before the L2 norm) for the same documents, batched in data order as the speed bench
does, from four encoders:

  stock   the model as shipped, bf16
  eager   pe.pack, graphs off
  graphs  pe.pack, CUDA graphs
  fp32    the shipped model in fp32, attention per document with SDPA (flex in fp32 exceeds
          sm_89 shared memory; per-document SDPA is the same math without a mask)

and reports, for the tokens where packed and stock disagree most: which token it is, how
large its vector is before normalisation, and which bf16 encoder the fp32 reference agrees
with. A token whose pre-norm vector is tiny is ill-conditioned: any bf16 rounding turns its
direction around, in stock and packed alike.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys

import torch
import torch.nn.functional as F


def compat_shims() -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    if "block_type" in inspect.getsource(mq.Qwen3_5DecoderLayer.__init__):
        mq.Qwen3_5DecoderLayer.layer_type = property(lambda self: self.block_type)
    if not hasattr(mq.Qwen3_5GatedDeltaNet, "chunk_gated_delta_rule"):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        mq.Qwen3_5GatedDeltaNet.chunk_gated_delta_rule = staticmethod(chunk_gated_delta_rule)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="topk-io/topk-embed-v1-small")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-docs", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    compat_shims()
    import packed_encoders as pe
    from packed_encoders.state import ATTR
    from sentence_transformers.multi_vector_encoder import MultiVectorEncoder

    docs = json.load(open(a.data))["documents"][: a.n_docs]
    model = MultiVectorEncoder(a.model, model_kwargs={"torch_dtype": torch.bfloat16}, trust_remote_code=True,
                               device="cuda").eval()
    st, net = model[0], model[0].auto_model
    keys = [p for p in inspect.signature(net.forward).parameters if p not in ("pixel_values", "image_scatter_index",
                                                                            "vision_inputs", "return_dict")]
    batches = [st.preprocess(docs[s:s + a.batch], task="document") for s in range(0, len(docs), a.batch)]
    ids = torch.cat([b["packed_ids"].reshape(-1) for b in batches])
    doc_of = torch.cat([torch.repeat_interleave(torch.arange(s, s + len(b["cu_seqlens"]) - 1),
                                                b["cu_seqlens"].diff())
                        for s, b in zip(range(0, len(docs), a.batch), batches)])
    pos_of = torch.cat([b["position_ids"].reshape(-1) for b in batches])

    @torch.inference_mode()
    def run(m):
        outs = []
        for b in batches:
            f = {k: b[k].to("cuda") for k in keys}
            outs.append(m(**f)[f["attention_mask"].bool()].float().cpu())
        return torch.cat(outs)

    vec = {}
    net.config.normalize = False                         # raw head output, before the L2 norm
    vec["stock"] = run(net)
    pe.pack(model)
    state = getattr(net, ATTR)
    state.normalize = False
    with pe.no_cuda_graph(model):
        vec["eager"] = run(net)
    vec["graphs"] = run(net)
    pe.unpack(model)
    del model, net
    torch.cuda.empty_cache()

    m32 = MultiVectorEncoder(a.model, model_kwargs={"torch_dtype": torch.float32}, trust_remote_code=True,
                             device="cuda").eval()
    n32 = m32[0].auto_model
    n32.config.normalize = False
    hb = sys.modules[type(n32.model.language_model.layers[3].self_attn).__module__]
    hb.document_mask = lambda di, *, causal: (torch.unique_consecutive(di, return_counts=True)[1].tolist(), causal)
    hb.compiled_flex_attention = lambda q, k, v, *, block_mask, enable_gqa, kernel_options=None: torch.cat(
        [F.scaled_dot_product_attention(qs, ks, vs, is_causal=block_mask[1], enable_gqa=enable_gqa)
         for qs, ks, vs in zip(q.split(block_mask[0], 2), k.split(block_mask[0], 2), v.split(block_mask[0], 2))], 2)
    vec["fp32"] = run(n32)

    norm = {k: v.norm(dim=-1) for k, v in vec.items()}
    cos = lambda x, y: F.cosine_similarity(vec[x], vec[y], dim=-1)  # noqa: E731
    pairs = {p: cos(*p.split("~")) for p in ("eager~stock", "graphs~stock", "stock~fp32", "eager~fp32", "graphs~fp32")}
    res = {"model": a.model, "n_tokens": int(ids.numel()), "gpu": torch.cuda.get_device_name(0),
           "torch": str(torch.__version__), "pairs": {}, "norm_fp32_quantiles": {}, "worst": []}
    q = torch.tensor([0.0, 1e-4, 1e-3, 1e-2, 0.5])
    res["norm_fp32_quantiles"] = dict(zip(map(str, q.tolist()), torch.quantile(norm["fp32"], q).tolist()))
    print(f"{a.model} | {res['gpu']} | torch {res['torch']} | {res['n_tokens']} doc tokens")
    print("pre-norm |v| (fp32) quantiles:", {k: round(v, 4) for k, v in res["norm_fp32_quantiles"].items()})
    for p, c in pairs.items():
        res["pairs"][p] = {"cos_mean": c.mean().item(), "cos_min": c.min().item(),
                           "n_below_0.9": int((c < 0.9).sum()), "n_below_0.99": int((c < 0.99).sum())}
        print(f"  {p:13s} cos mean {c.mean():.6f} min {c.min():.5f}  <0.99: {res['pairs'][p]['n_below_0.99']:5d}"
              f"  <0.9: {res['pairs'][p]['n_below_0.9']}")
    tok = st.tokenizer if hasattr(st, "tokenizer") else getattr(st, "processor", None)
    order = torch.argsort(pairs["eager~stock"])[: a.top]
    print(f"worst {a.top} tokens by cos(eager, stock):")
    for i in order.tolist():
        t = int(ids[i])
        row = {"index": i, "doc": int(doc_of[i]), "position": int(pos_of[i]), "token_id": t,
               "token": tok.decode([t]) if tok is not None else None,
               "norm": {k: round(norm[k][i].item(), 5) for k in vec},
               "cos": {p: round(c[i].item(), 5) for p, c in pairs.items()}}
        res["worst"].append(row)
        print(f"  doc {row['doc']:4d} pos {row['position']:5d} tok {t:7d} {row['token']!r:14s} |v| "
              + " ".join(f"{k}={v:.4f}" for k, v in row["norm"].items()) + " | "
              + " ".join(f"{p}={v:.4f}" for p, v in row["cos"].items()))
    small = norm["fp32"] < torch.quantile(norm["fp32"], 1e-3)
    for p, c in pairs.items():
        res["pairs"][p]["cos_min_excluding_smallest_0.1pct_norm"] = c[~small].min().item()
    print("cos min over tokens outside the smallest 0.1% of |v|:",
          {p: round(v["cos_min_excluding_smallest_0.1pct_norm"], 5) for p, v in res["pairs"].items()})
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("PARITY PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
