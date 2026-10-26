"""Compare previous dispatch, forced cache capacities, and pack-time tuning.

Run serially, pinned to CPU 2 for 1t or CPUs 0-7 for 8t on the measured 9950X:
  python benchmarks/cpu_x86/cache_gemm.py --threads 1 [--rehome] [--rounds 9]
Forced 32 KiB changes tile policy, not hardware cache capacity. Results do not
predict speed on a real 32 KiB CPU. Every result must match bitwise.
"""
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import torch
from transformers import AutoModel
import packed_encoders as pe
from packed_encoders.cpu.attention import _openmp_runtime
from bench import inputs, workloads


def baseline(directory):
    source = Path(__file__).parent / 'native_probe/fp32_cache_baseline.c'
    # C adapter gives both versions the same ctypes ABI/call overhead.
    wrapper = Path(directory) / 'baseline.c'
    wrapper.write_text(source.read_text() + '''
int pe_linear_tile(const float *x, const float *w, const float *b, float *y,
                   int M, int N, int K, int workers, int rows, int cols) {
    (void)rows; (void)cols;
    return pe_linear(x, w, b, y, M, N, K, workers);
}
''')
    output = Path(directory) / 'baseline.so'
    subprocess.run(['gcc', '-O3', '-march=native', '-fPIC', '-fopenmp', '-ffp-contract=fast',
                    '-shared', str(wrapper), _openmp_runtime(), '-o', str(output)], check=True)
    lib = ctypes.CDLL(str(output))
    lib.pe_linear_tile.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6
    lib.pe_linear_tile.restype = ctypes.c_int
    return lib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, choices=(1, 8), required=True)
    p.add_argument('--rounds', type=int, default=9)
    p.add_argument('--rehome', action='store_true')
    a = p.parse_args()
    if a.rounds < 2:
        p.error('--rounds must be at least 2')
    torch.set_num_threads(a.threads)
    model = AutoModel.from_pretrained('answerdotai/ModernBERT-base', attn_implementation='sdpa',
                                      local_files_only=True).float().eval()
    start = time.perf_counter()
    pe.pack(model, engine=pe.ModernBertCPU(threads=a.threads, rehome_weights=a.rehome, tune_gemm=True))
    pack_s = time.perf_counter() - start
    att = pe.get_engine(model).attention
    if not att.native_linear:
        raise RuntimeError('native AVX-512 GEMM required')
    native, detected = att.rows.pe_linear_tile, att.linear_l1_bytes
    tuned = att.linear_tiles.copy()
    print(json.dumps({'threads': a.threads, 'rehome': a.rehome, 'l1_bytes': detected, 'pack_s': pack_s,
                      'tuning': {str(key): {**value, 'median_ms': {str(t): ms for t, ms in value['median_ms'].items()}}
                                 for key, value in att.linear_tuning.items()}}), flush=True)
    with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
        previous = baseline(directory)
        variants = ('before', 'cache32', 'cache48', 'tuned')
        for name in ('b1_s16_pad32', 'b1_s32', 'b8_s32'):
            seq, lengths = workloads()[name]
            ids, mask = inputs(seq, lengths)
            times = {v: [] for v in variants}
            outputs, tiles = {}, {}
            reps = 3 if a.threads == 1 and name == 'b8_s32' else 10
            for round_ in range(a.rounds + 1):
                for variant in variants if round_ % 2 == 0 else variants[::-1]:
                    att.rows.pe_linear_tile = previous.pe_linear_tile if variant == 'before' else native
                    att.linear_l1_bytes = 32 * 1024 if variant == 'cache32' else (48 * 1024 if variant == 'cache48' else detected)
                    att.linear_tiles = tuned.copy() if variant == 'tuned' else {}
                    # Allocate and select once outside the timer, including after each switch.
                    model(ids, attention_mask=mask)
                    start = time.perf_counter()
                    for _ in range(reps):
                        out = model(ids, attention_mask=mask).last_hidden_state
                    elapsed = (time.perf_counter() - start) * 1000 / reps
                    if round_:
                        times[variant].append(elapsed)
                    outputs[variant] = out.clone()
                    tiles[variant] = {str(key): value for key, value in att.linear_tiles.items()}
                assert all(torch.equal(outputs['before'], out) for out in outputs.values()), name
            print(json.dumps({'threads': a.threads, 'rehome': a.rehome, 'workload': name, 'bitwise_equal': True,
                              'results': {v: {'median_ms': statistics.median(times[v]), 'round_ms': times[v],
                                              'tiles': tiles[v]} for v in variants}}), flush=True)
    att.rows.pe_linear_tile, att.linear_l1_bytes, att.linear_tiles = native, detected, tuned


if __name__ == '__main__':
    main()
