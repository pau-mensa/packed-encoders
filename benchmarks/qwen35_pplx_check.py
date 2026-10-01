"""pplx-embed-v2-context-9b-preview under packed-encoders (the Qwen3.5 `hf` entry): parity, retrieval, speed.

    python benchmarks/qwen35_pplx_check.py --data sample.json --out pplx.json [--no-graphs | --graph-max-tokens N]

One process, one GPU: fp32 reference vectors (the dtype the model ships in) -> bf16 stock -> `pe.pack`
-> bf16 packed. Parity compares the model's pre-int8 query / chunk projections (its `_pool`); retrieval
uses its own int8 `encode(..., normalize_embeddings=True)`, a document scoring as its best chunk; speed
is end-to-end `encode` / `encode_queries` (tokenisation included), best of two after a warmup.

`--data` is JSON with `queries` (list of str), `documents` (list of str) and `qrels`
({query index: {document index: relevance}}, string keys). Documents are split into 3-sentence chunks
and encoded as one contextual document each. The PR's numbers used BEIR SciFact test: 100 queries
(random.Random(0)), all their gold documents, filled at random to 1,000 documents. Requires
transformers >= 5.4, accelerate, flash-linear-attention, and ~34 GB of disk for the fp32 checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL = "perplexity-ai/pplx-embed-v2-context-9b-preview"


def chunk(text: str, per: int = 3) -> list[str]:
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return [" ".join(sents[i:i + per]) for i in range(0, len(sents), per)] or [text]


def ndcg10(q_emb, d_emb, qrels) -> float:
    scores = []
    for qi, q in enumerate(q_emb):
        s = np.array([float((d @ q).max()) for d in d_emb])
        top = np.argsort(-s)[:10]
        rel = qrels.get(str(qi), {})
        dcg = sum(1 / math.log2(r + 2) for r, d in enumerate(top) if str(d) in rel)
        idcg = sum(1 / math.log2(r + 2) for r in range(min(len(rel), 10)))
        scores.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(scores))


@torch.inference_mode()
def pooled(model, rows, task, bs):
    out = []
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        feats = {k: v.to(model.device) for k, v in model.prepare_inputs(batch, task).items()}
        p = model._pool(feats, task).float().cpu()
        out += [p[j, : len(r)] for j, r in enumerate(batch)]
    return torch.cat(out)            # (total chunks, 2048)


def retrieval(model, queries, docs, qrels, bs):
    q = model.encode_queries(queries, batch_size=bs, normalize_embeddings=True)
    d = model.encode(docs, batch_size=bs, normalize_embeddings=True)
    return ndcg10([x[0] for x in q], d, qrels)


def timed(model, rows, task, bs, reps=2):
    fn = model.encode_queries if task == "query" else model.encode
    fn(rows, batch_size=bs)                         # warmup (captures graphs when packed)
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn(rows, batch_size=bs)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return len(rows) / best


def load(dtype):
    from transformers import AutoModel

    t = time.perf_counter()
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=dtype, device_map="cuda").eval()
    return m, time.perf_counter() - t


def cos_stats(a, b):
    c = F.cosine_similarity(a, b, dim=-1)
    return {"mean": c.mean().item(), "min": c.min().item(), "p01": c.quantile(0.01).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-fp32", action="store_true")
    ap.add_argument("--fp32-retrieval", action="store_true", help="also fp32 nDCG (7 min on an L40S)")
    ap.add_argument("--stock-retrieval", action="store_true", help="also stock bf16 nDCG")
    ap.add_argument("--graph-max-tokens", type=int, default=None,
                    help="pack with PaddedGraphConfig(max_tokens=N) instead of the default buckets")
    ap.add_argument("--no-graphs", action="store_true", help="pack with cuda_graph=False")
    a = ap.parse_args()

    def save():
        json.dump(res, open(a.out, "w"), indent=1, default=str)

    data = json.load(open(a.data))
    queries = [[q] for q in data["queries"]]
    docs = [chunk(d) for d in data["documents"]]
    qrels = data["qrels"]
    n_par_docs = 128
    res = {"model": MODEL, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "n_queries": len(queries), "n_docs": len(docs), "chunks_per_doc": float(np.mean([len(d) for d in docs]))}

    from huggingface_hub import snapshot_download

    t = time.perf_counter()
    snapshot_download(MODEL)
    res["download_s"] = time.perf_counter() - t
    print(res, flush=True)

    ref = {}
    if not a.skip_fp32:
        m, res["load_fp32_s"] = load(torch.float32)
        ref["q"] = pooled(m, queries, "query", 32)
        ref["d"] = pooled(m, docs[:n_par_docs], "document", 8)
        if a.fp32_retrieval:
            t = time.perf_counter()
            res["ndcg10_fp32"] = retrieval(m, queries, docs, qrels, 16)
            res["retrieval_fp32_s"] = time.perf_counter() - t
        save()
        print({k: v for k, v in res.items() if "fp32" in k}, flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    m, res["load_bf16_s"] = load(torch.bfloat16)
    tok = m.tokenizer
    res["doc_tokens_mean"] = float(np.mean([len(tok("[D] " + "<|chunk_sep|>".join(d))["input_ids"]) for d in docs]))
    stock = {"q": pooled(m, queries, "query", 32), "d": pooled(m, docs[:n_par_docs], "document", 8)}
    if a.stock_retrieval:
        res["ndcg10_stock_bf16"] = retrieval(m, queries, docs, qrels, 16)
    qs, ds = queries * 3, docs[:128]
    speed = {"stock": {}, "packed": {}}
    for bs in (8, 32):
        speed["stock"][f"queries_b{bs}"] = timed(m, qs, "query", bs)
        speed["stock"][f"docs_b{bs}"] = timed(m, ds, "document", bs)
    res["speed_items_per_s"] = speed
    save()
    print(res, speed, flush=True)

    # The stock passes leave the caching allocator holding large free blocks that CUDA-graph pools
    # can't use; a user packing after running the stock model would do the same.
    res["reserved_before_pack_gb"] = torch.cuda.memory_reserved() / 2**30
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    import packed_encoders as pe
    from packed_encoders.runtime import PaddedGraphConfig
    from packed_encoders.state import ATTR

    t = time.perf_counter()
    cap = a.graph_max_tokens
    pe.pack(m, cuda_graph=False if a.no_graphs else PaddedGraphConfig(max_tokens=cap) if cap else None)
    res["graphs"] = "off" if a.no_graphs else f"max_tokens={cap}" if cap else "default"
    res["pack_s"] = time.perf_counter() - t
    st = getattr(m, ATTR)
    rep = st.report
    res["report"] = {k: getattr(rep, k) for k in ("attention_backend", "eager_cos_mean", "eager_cos_min",
                                                  "graph_cos_mean", "graph_cos_min", "fused", "gdn_kernel_rel_err",
                                                  "gdn_rejected", "fusion_rejected", "attention_rejected")}
    res["entry"], res["causal"] = st.entry, st.engine.causal
    save()
    graphs_held = {}                 # did graphs survive each phase (they are dropped on running out of memory)?
    packed = {"q": pooled(m, queries, "query", 32), "d": pooled(m, docs[:n_par_docs], "document", 8)}
    graphs_held["parity"] = st.runner is not None
    res["ndcg10_packed_bf16"] = retrieval(m, queries, docs, qrels, 16)
    graphs_held["retrieval"] = st.runner is not None
    for bs in (8, 32):
        speed["packed"][f"queries_b{bs}"] = timed(m, qs, "query", bs)
        speed["packed"][f"docs_b{bs}"] = timed(m, ds, "document", bs)
    graphs_held["timing"] = st.runner is not None
    res["graphs_held_after"] = graphs_held
    res["num_graphs"] = st.runner.num_graphs if st.runner is not None else 0
    res["max_mem_packed_gb"] = torch.cuda.max_memory_allocated() / 2**30
    res["reserved_packed_gb"] = torch.cuda.memory_reserved() / 2**30

    par = {}
    for k, name in (("q", "queries"), ("d", "doc_chunks")):
        par[f"{name}_packed_vs_stock"] = cos_stats(packed[k], stock[k])
        if ref:
            par[f"{name}_stock_vs_fp32"] = cos_stats(stock[k], ref[k])
            par[f"{name}_packed_vs_fp32"] = cos_stats(packed[k], ref[k])
        qp = lambda x: torch.round(torch.tanh(x) * 127).clamp(-128, 127)   # noqa: E731 — the model's int8 step
        par[f"{name}_int8_equal_packed_vs_stock"] = (qp(packed[k]) == qp(stock[k])).float().mean().item()
    res["parity"] = par
    print(json.dumps(res, indent=1, default=str), flush=True)
    save()


if __name__ == "__main__":
    main()
