"""Paired workloads for HF, engine FP32/int8 and trimmed ORT FP32/MatMulNBits W8A8.

One system per process; pin affinity externally. See README.md for commands.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel

from bench import MODEL, workloads
from onnx_bench import feeds, run, session


def environment():
    cpu = platform.processor() or platform.machine()
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), cpu)
    elif platform.system() == 'Darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    return {'date': datetime.now(timezone.utc).date().isoformat(), 'cpu': cpu,
            'platform': platform.platform(),
            'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'torch_threads': torch.get_num_threads(), 'omp_wait_policy': os.environ.get('OMP_WAIT_POLICY'),
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'onnxruntime')
                         if importlib.util.find_spec(name)}}


def benchmark_workloads(vocab):
    loads = workloads(vocab)
    # Separate seed preserves all previously published workloads.
    rng = torch.Generator().manual_seed(8192)
    for name, lengths in [('b32_s8192', [8192] * 32),
                          ('b32_rand_pad8192', torch.randint(1, 8193, (32,), generator=rng).tolist())]:
        ids = torch.randint(5, vocab, (32, 8192), generator=rng)
        loads[name] = ids, (torch.arange(8192)[None] < torch.tensor(lengths)[:, None]).long()
    return loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=("hf", "engine", "ort"))
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--native-threads", default="same", help="same, default (physical-core cap), or an integer")
    parser.add_argument("--precision", choices=("fp32", "int8"), default="fp32")
    parser.add_argument("--backend", choices=("auto", "native"), default="auto")
    parser.add_argument("--model", help="ORT model path (FP32 optimized or MatMulNBits W8A8)")
    parser.add_argument("--spin", type=int, choices=(0, 1), default=1)
    parser.add_argument("--only", default="b1_s16_pad32,b1_s32,b8_s32")
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1, help="complete warmup batches")
    parser.add_argument("--ort-row-warmup", action="store_true", help="warm one ORT row before timing a long batch")
    parser.add_argument("--reference-cache", type=Path, default=Path('.cache/cpu-benchmark-oracles'),
                        help="reuse HF oracle tensors for 8192-token comparisons")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.system == "ort" and not args.model:
        parser.error("ORT requires --model")
    if args.trials < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("threads/trials must be positive and warmup nonnegative")
    try:
        native_threads = (None if args.native_threads == "default" else args.threads
                          if args.native_threads == "same" else int(args.native_threads))
        if native_threads is not None and native_threads < 1:
            raise ValueError
    except ValueError:
        parser.error("native-threads must be same, default, or a positive integer")
    torch.set_num_threads(args.threads)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()
    loads = benchmark_workloads(model.config.vocab_size)
    unknown = set(args.only.split(",")) - loads.keys()
    if unknown:
        parser.error(f"unknown workloads: {sorted(unknown)}")
    loads = {k: v for k, v in loads.items() if k in args.only.split(",")}
    results = {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               **environment(), "hf_revision": getattr(model.config, '_commit_hash', None), "rows": {}}
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference = {}
        for name, (ids, mask) in loads.items():
            cache = None
            if ids.shape[1] == 8192 and results['hf_revision'] is not None:
                identity = hashlib.sha256((model.config.to_json_string()
                                           + results['hf_revision']
                                           + json.dumps(results['versions'], sort_keys=True)
                                           + str(args.threads)).encode())
                identity.update(ids.numpy().tobytes())
                identity.update(mask.numpy().tobytes())
                identity.update(b'rowwise-hf-fp32-oracle-v1')
                args.reference_cache.mkdir(parents=True, exist_ok=True)
                cache = args.reference_cache / (identity.hexdigest() + '.pt')
                results.setdefault('reference_keys', {})[name] = identity.hexdigest()
                if cache.exists():
                    reference[name] = torch.load(cache, weights_only=True)
                    continue
            # Bound oracle memory at long lengths. Independent rows have identical
            # semantics; the HF latency below still measures the complete batch.
            reference_rows = []
            for index, (row, m) in enumerate(zip(ids, mask), 1):
                if cache is not None:
                    # All workloads are right-padded. Trimming an untimed oracle
                    # preserves live-token positions and avoids quadratic padding work.
                    length = int(m.sum())
                    row, m = row[:length], m[:length]
                reference_rows.append(model(row[None], attention_mask=m[None]).last_hidden_state[0, m.bool()])
                if cache is not None and index % 8 == 0:
                    print(f'HF oracle {name}: {index}/{len(ids)} rows', flush=True)
            reference[name] = torch.cat(reference_rows)
            del reference_rows
            if cache is not None:
                temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
                torch.save(reference[name], temporary)
                os.replace(temporary, cache)
        if args.system == "engine":
            import packed_encoders as pe

            start = time.perf_counter()
            pe.pack(model, engine=pe.ModernBertCPU(threads=native_threads),
                    precision=args.precision, attention_backend=args.backend)
            runtime = pe.get_engine(model)
            if runtime.precision != args.precision:
                raise RuntimeError(f"requested {args.precision}, got fallback {runtime.precision}")
            results["pack_s"] = time.perf_counter() - start
            results["native_threads"] = runtime.attention.threads
            results["attention_tuning"] = {str(k): v for k, v in runtime.attention.attention_tuning.items()}
        if args.system == "ort":
            del model
            ort = session(args.model, args.threads, args.spin)
        for name, (ids, mask) in loads.items():
            if args.system == "ort":
                # One row at a time: trim even the padded short query. No padding
                # speedup is credited against work a baseline need not perform.
                chunks = feeds(ids, mask, "rowtrim") if len(ids) == 1 or ids.shape[1] > 128 else feeds(ids, mask, "batch")

                def call():
                    if ids.shape[1] < 8192:
                        return run(ort, chunks)
                    outputs = []
                    for index, chunk in enumerate(chunks, 1):
                        outputs.extend(run(ort, [chunk]))
                        print(f'{name} ORT: {index}/{len(chunks)} rows', flush=True)
                    return outputs
            else:
                call = lambda: model(ids, attention_mask=mask).last_hidden_state
            if args.system == "ort" and args.ort_row_warmup:
                run(ort, chunks[:1])
            for _ in range(args.warmup):
                call()
            samples = []
            for _ in range(args.trials):
                start = time.perf_counter()
                output = call()
                samples.append((time.perf_counter() - start) * 1000)
            if args.system == "ort":
                got = torch.cat([torch.from_numpy(o[0]) for o in output]) if len(chunks) == len(ids) else torch.from_numpy(output[0])[mask.bool()]
            else:
                got = output[mask.bool()]
            want = reference[name]
            cos = F.cosine_similarity(got, want, dim=-1)
            row = {"median_ms": statistics.median(samples), "samples_ms": samples,
                   "lengths": mask.sum(1).tolist(),
                   "live_tokens": int(mask.sum()), "relL2": float((got - want).norm() / want.norm()),
                   "cos_min": float(cos.min()), "cos_median": float(cos.median())}
            if args.precision == "fp32" and row["relL2"] > 2e-4:
                raise RuntimeError(f"{args.system} FP32 parity failed for {name}: {row}")
            results["rows"][name] = row
            output_path.write_text(json.dumps(results, indent=2) + "\n")
            print(name, json.dumps(row), flush=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
