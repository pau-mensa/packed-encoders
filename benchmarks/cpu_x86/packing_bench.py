"""Fresh-process pe.pack timing/RSS; excludes loading and any subsequent inference.

Example (each source is a checkout or an extracted git archive):
  .venv/bin/python benchmarks/cpu_x86/packing_bench.py \
    --sources before=/tmp/before after=/tmp/after --output /tmp/packing.jsonl
"""
import argparse
import dataclasses
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import threading
import time
import warnings


def rss_mib():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    raise RuntimeError('VmRSS unavailable')


def worker(a):
    # Set affinity before any native pools are initialized. One physical core per worker,
    # with 8 threads contained in CCD0 on this 9950X; no SMT siblings.
    os.sched_setaffinity(0, range(a.threads))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(Path(a.source).resolve()))
    import torch
    from transformers import AutoModel
    import packed_encoders as pe

    assert Path(pe.__file__).is_relative_to(Path(a.source).resolve()), pe.__file__
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = AutoModel.from_pretrained(
        'answerdotai/ModernBERT-base', local_files_only=True,
        attn_implementation='sdpa',
    ).float().cpu().eval()
    gc.collect()
    before = rss_mib()
    hwm_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    samples = [before]
    stop = threading.Event()

    def sample():
        while not stop.wait(.005):
            samples.append(rss_mib())

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start = time.perf_counter()
        pe.pack(model, engine=pe.ModernBertCPU(threads=a.threads),
                attention_backend='native', precision=a.precision)
        pack_s = time.perf_counter() - start
    after = rss_mib()
    hwm_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    samples.append(after)
    stop.set()
    monitor.join()
    engine = pe.get_engine(model)
    assert engine.state.precision == a.precision
    assert engine.state.attention_backend == 'native'
    report = dataclasses.asdict(engine.state.validation_report)
    prepared_bytes = sum(t.numel() * t.element_size()
                         for group in engine.int8.prepared for t in group if t is not None
                         ) if engine.int8 else 0
    gc.collect()
    after_gc = rss_mib()
    # HWM is an exact parent-process maximum if packing exceeded the prior maximum.
    # Compiler children are not included in this parent RSS metric.
    peak = hwm_after if hwm_after > hwm_before else max(samples)
    result = dict(label=a.label, source=str(Path(a.source).resolve()),
                  precision=a.precision, threads=a.threads,
                  affinity=sorted(os.sched_getaffinity(0)), pack_s=pack_s,
                  rss_before_mib=before, rss_after_mib=after, rss_after_gc_mib=after_gc,
                  peak_pack_rss_mib=peak, peak_extra_mib=peak - before,
                  hwm_before_mib=hwm_before, hwm_after_mib=hwm_after,
                  sampled_peak_mib=max(samples), memory_samples=len(samples),
                  prepared_mib=prepared_bytes / 2**20, validation=report,
                  warnings=[str(w.message) for w in caught],
                  torch=torch.__version__)
    if a.digest and engine.int8:
        import hashlib
        digest = hashlib.sha256()
        for group in engine.int8.prepared:
            for tensor in group:
                if tensor is not None:
                    digest.update(memoryview(tensor.numpy()).cast('B'))
        result['prepared_sha256'] = digest.hexdigest()
    pe.unpack(model)
    del engine
    gc.collect()
    result['rss_after_unpack_mib'] = rss_mib()
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', help='label=source-directory pairs')
    p.add_argument('--output', type=Path)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--thread-counts', type=int, nargs='+', default=[1, 8])
    p.add_argument('--precisions', nargs='+', default=['fp32', 'int8'])
    p.add_argument('--worker', action='store_true')
    p.add_argument('--source')
    p.add_argument('--label')
    p.add_argument('--threads', type=int)
    p.add_argument('--precision')
    p.add_argument('--digest', action='store_true', help='hash prepared tensors outside the measured region')
    a = p.parse_args()
    if a.worker:
        worker(a)
        return
    sources = [s.split('=', 1) for s in a.sources]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with a.output.open('w') as out:
        for threads in a.thread_counts:
            for precision in a.precisions:
                for round_ in range(a.rounds):
                    for label, source in sources[::1 if round_ % 2 == 0 else -1]:
                        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                                   '--source', source, '--label', label, '--threads', str(threads),
                                   '--precision', precision]
                        if a.digest:
                            command.append('--digest')
                        completed = subprocess.run(command, capture_output=True, text=True, timeout=240)
                        if completed.returncode:
                            raise RuntimeError(f'{command}\n{completed.stdout}\n{completed.stderr}')
                        result = json.loads(completed.stdout.strip().splitlines()[-1])
                        result['round'] = round_ + 1
                        out.write(json.dumps(result) + '\n')
                        out.flush()
                        results.append(result)
                        print(json.dumps({k: result[k] for k in (
                            'label', 'threads', 'precision', 'round', 'pack_s',
                            'peak_pack_rss_mib', 'peak_extra_mib', 'rss_after_gc_mib')}), flush=True)
    for threads in a.thread_counts:
        for precision in a.precisions:
            for label, _ in sources:
                rows = [r for r in results if (r['threads'], r['precision'], r['label']) ==
                        (threads, precision, label)]
                print(json.dumps(dict(label=label, threads=threads, precision=precision,
                      median_s=statistics.median(r['pack_s'] for r in rows),
                      median_peak_mib=statistics.median(r['peak_pack_rss_mib'] for r in rows),
                      median_extra_mib=statistics.median(r['peak_extra_mib'] for r in rows))), flush=True)


if __name__ == '__main__':
    main()
