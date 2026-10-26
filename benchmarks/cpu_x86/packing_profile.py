"""Attribute pe.pack cost in one fresh process; timing only packing, not model loading."""
import argparse
from collections import defaultdict
import cProfile
import functools
import json
import os
from pathlib import Path
import pstats
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--source', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    os.sched_setaffinity(0, range(a.threads))
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path.insert(0, str(a.source.resolve()))
    import torch
    from transformers import AutoModel
    import packed_encoders as pe
    from packed_encoders.arch import modernbert_cpu
    from packed_encoders.cpu import attention, int8
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    model = AutoModel.from_pretrained('answerdotai/ModernBERT-base', local_files_only=True,
                                     attn_implementation='sdpa').float().cpu().eval()
    times = defaultdict(list)
    lock = threading.Lock()

    def instrument(owner, name):
        original = getattr(owner, name)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                with lock:
                    times[name].append(time.perf_counter() - start)
        setattr(owner, name, timed)

    for name in ('gptq', 'inverse_factor', 'rotate', 'library', '_calibrate', 'pack_weights'):
        instrument(int8, name)
    instrument(attention, '_x86_library')
    instrument(modernbert_cpu.PackedModernBertCPU, 'validate')
    instrument(int8.Int8Projections, '__init__')
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    pe.pack(model, engine=pe.ModernBertCPU(threads=a.threads), precision='int8', attention_backend='native')
    profile.disable()
    elapsed = time.perf_counter() - start
    a.output.parent.mkdir(parents=True, exist_ok=True)
    profile.dump_stats(str(a.output) + '.prof')
    result = dict(threads=a.threads, pack_s=elapsed,
                  phases={k:dict(calls=len(v), sum_s=sum(v), max_s=max(v)) for k, v in times.items()})
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    pstats.Stats(profile).strip_dirs().sort_stats('tottime').print_stats(35)
    pstats.Stats(profile).strip_dirs().sort_stats('cumtime').print_stats(20)


if __name__ == '__main__':
    main()
