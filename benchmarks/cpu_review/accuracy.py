"""Paired real-text FP32/GPTQ/W8A8 checks using a saved tokenized corpus."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from compare import MODEL, environment, session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ort-model', required=True)
    parser.add_argument('--corpus', type=Path, default=Path(__file__).with_name('corpus.json'))
    parser.add_argument('--source', type=Path, default=Path(__file__).parents[2] / 'README.md')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if not args.corpus.exists():
        source = args.source.read_bytes()
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        paragraphs = []
        for paragraph in source.decode().split('\n\n'):
            ids = tokenizer(paragraph.strip(), truncation=True, max_length=512)['input_ids']
            if 29 <= len(ids) <= 512:
                paragraphs.append({'text': paragraph.strip(), 'input_ids': ids})
            if len(paragraphs) == 24:
                break
        args.corpus.write_text(json.dumps({'model': MODEL, 'source_sha256': hashlib.sha256(source).hexdigest(),
                                          'paragraphs': paragraphs}, indent=2) + '\n')
    corpus = json.loads(args.corpus.read_text())
    if corpus['model'] != MODEL or not corpus['paragraphs']:
        raise ValueError('corpus must contain paragraphs tokenized for the benchmark model')
    ids = [torch.tensor(p['input_ids'], dtype=torch.long)[None] for p in corpus['paragraphs']]
    results = {**environment(), 'model': MODEL, 'corpus_sha256': hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
               'lengths': [x.numel() for x in ids], 'rows': {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import packed_encoders as pe

    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation='sdpa').eval()
    with torch.inference_mode():
        reference = [model(x, attention_mask=torch.ones_like(x)).last_hidden_state[0] for x in ids]

        def record(name, outputs):
            got, want = torch.cat(outputs), torch.cat(reference)
            cos = F.cosine_similarity(got, want, dim=-1)
            pooled = torch.stack([F.cosine_similarity(x.mean(0), y.mean(0), dim=0)
                                  for x, y in zip(outputs, reference)])
            row = {'relL2': float((got - want).norm() / want.norm()), 'cos_median': float(cos.median()),
                   'cos_p5': float(cos.quantile(.05)), 'cos_min': float(cos.min()),
                   'pooled_cos_min': float(pooled.min()), 'tokens_below_0.5': int((cos < .5).sum())}
            results['rows'][name] = row
            args.out.write_text(json.dumps(results, indent=2) + '\n')
            print(name, json.dumps(row), flush=True)

        for precision in ('fp32', 'int8'):
            pe.pack(model, engine=pe.ModernBertCPU(threads=args.threads), precision=precision)
            try:
                if pe.get_engine(model).precision != precision:
                    raise RuntimeError(f'{precision} fell back; cannot label this result {precision}')
                record('engine_' + precision, [model(x, attention_mask=torch.ones_like(x)).last_hidden_state[0]
                                               for x in ids])
            finally:
                pe.unpack(model)
        del model
        ort = session(args.ort_model, args.threads, 1)
        record('ort_w8a8', [torch.from_numpy(ort.run(None, {'input_ids': x.numpy(),
                                                         'attention_mask': torch.ones_like(x).numpy()})[0][0])
                            for x in ids])


if __name__ == '__main__':
    main()
