"""Real-text accuracy of the ONNX baselines against stock HF FP32, after the CPU engine's accuracy table.

    python benchmarks/cpu_avx2/onnx_accuracy.py --out results/onnx_acc.json [--models fp32_opt,int8]

Corpus: the repository README's paragraphs (blank-line separated) with 29-512 tokens, each run alone (no padding).
Per model: relL2 over all tokens, token cosine median / p5 / min, mean-pooled cosine min, and the
number of tokens below 0.5 cosine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bench import MODEL
from onnx_bench import session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=str(Path(__file__).parents[2] / "README.md"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="fp32_opt,int8,int8_pc")
    ap.add_argument("--dir", default=str(Path(__file__).parent / "onnx_models"))
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    paras = [p.strip() for p in open(a.readme).read().split("\n\n") if p.strip()]
    encoded = [tok(p, return_tensors="pt", truncation=True, max_length=512)["input_ids"] for p in paras]
    encoded = [e for e in encoded if 29 <= e.shape[1] <= 512]
    lengths = [e.shape[1] for e in encoded]
    print(f"{len(encoded)} paragraphs, {sum(lengths)} tokens, lengths {min(lengths)}-{max(lengths)}", flush=True)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()
    with torch.inference_mode():
        ref = [model(input_ids=e, attention_mask=torch.ones_like(e)).last_hidden_state[0] for e in encoded]
    del model
    res = {"paragraphs": len(encoded), "tokens": sum(lengths), "rows": {}}
    for name in a.models.split(","):
        sess = session(f"{a.dir}/{name}.onnx", a.threads, 1)
        got = [torch.from_numpy(sess.run(None, {"input_ids": e.numpy().astype(np.int64),
                                                "attention_mask": np.ones(e.shape, np.int64)})[0][0])
               for e in encoded]
        g, r = torch.cat(got), torch.cat(ref)
        cos = F.cosine_similarity(g, r, dim=-1)
        pooled = torch.stack([F.cosine_similarity(x.mean(0), y.mean(0), dim=0) for x, y in zip(got, ref)])
        row = {"relL2": float((g - r).norm() / r.norm()), "cos_median": float(cos.median()),
               "cos_p5": float(cos.quantile(0.05)), "cos_min": float(cos.min()),
               "pooled_cos_min": float(pooled.min()), "tokens_below_0.5": int((cos < 0.5).sum())}
        res["rows"][name] = row
        print(f"[acc] {name:9s} " + " ".join(f"{k}={v:.5g}" for k, v in row.items()), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("ACC DONE", flush=True)


if __name__ == "__main__":
    main()
