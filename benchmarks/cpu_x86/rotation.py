"""Compare old/new Hadamard kernels on identical prepared int8 weights.

Run serially: taskset -c 2 for --threads 1, taskset -c 0-7 for --threads 8.
The --micro option also measures the serial rotate+quantize entry point.
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
from packed_encoders.cpu import int8
from packed_encoders.cpu.attention import _openmp_runtime
from bench import inputs, workloads


def reference(directory, native):
    source = Path(__file__).parent / 'native_probe/int8_rotation_baseline.c'
    output = Path(directory) / 'reference.so'
    subprocess.run(['gcc', '-O3', '-march=native', '-fPIC', '-fopenmp', '-ffp-contract=off',
                    '-fno-trapping-math', '-fno-math-errno', '-Wall', '-Wextra', '-Werror',
                    '-Wno-unknown-pragmas', '-I', str(Path(int8.__file__).parent), '-shared',
                    str(source), _openmp_runtime(), '-o', str(output)], check=True, capture_output=True)
    library = ctypes.CDLL(str(output))
    for name in ('pe_q8_input', 'pe_q8_segment', 'pe_q8_quantize'):
        getattr(library, name).argtypes = getattr(native, name).argtypes
        getattr(library, name).restype = getattr(native, name).restype
    return library


def micro(libraries, rounds):
    rng = torch.Generator().manual_seed(41)
    for m in (16, 32, 256):
        for k in (768, 1152):
            x = torch.randn(m, k, generator=rng)
            x[0].zero_()
            x[-1, k // 2] = 100
            for block in (0, 64, 128):
                signs = int8.signs(k, block).float() if block else None
                out, scale, tmp = torch.empty(m, k, dtype=torch.uint8), torch.empty(m), torch.empty(k)
                args = (x.data_ptr(), m, k, None if signs is None else signs.data_ptr(), block,
                        tmp.data_ptr(), out.data_ptr(), scale.data_ptr())
                times = {v: [] for v in libraries}
                outputs = {}
                order = list(libraries)
                for round_ in range(rounds + 1):
                    for name in order if round_ % 2 == 0 else order[::-1]:
                        fn = libraries[name].pe_q8_quantize
                        start = time.perf_counter_ns()
                        for _ in range(100):
                            fn(*args)
                        elapsed = (time.perf_counter_ns() - start) / 1e6 / 100
                        if round_:
                            times[name].append(elapsed)
                        outputs[name] = (out.clone(), scale.view(torch.int32).clone(), tmp.view(torch.int32).clone())
                    for a, b in zip(outputs['before'][:3 if block else 2], outputs['after']):
                        assert torch.equal(a, b), (m, k, block)
                print(json.dumps({'kind': 'micro', 'tokens': m, 'k': k, 'block': block, 'bitwise_equal': True,
                                  'results': {v: {'median_ms': statistics.median(t), 'round_ms': t}
                                              for v, t in times.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=(1, 8), required=True)
    parser.add_argument('--rounds', type=int, default=15)
    parser.add_argument('--micro', action='store_true')
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error('--rounds must be at least 2')
    torch.set_num_threads(args.threads)
    native, _ = int8.library()
    with tempfile.TemporaryDirectory(prefix='rotation-bench-') as directory, torch.inference_mode():
        libraries = {'before': reference(directory, native), 'after': native}
        if args.micro:
            micro(libraries, args.rounds)
        model = AutoModel.from_pretrained('answerdotai/ModernBERT-base', attn_implementation='sdpa',
                                          local_files_only=True).float().eval()
        pe.pack(model, engine=pe.ModernBertCPU(threads=args.threads), precision='int8')
        q8 = pe.get_engine(model).int8
        if q8 is None or not q8.x86:
            raise RuntimeError('native AVX512-VNNI required')
        for name in ('b1_s16_pad32', 'b1_s32', 'b8_s32'):
            seq, lengths = workloads()[name]
            ids, mask = inputs(seq, lengths)
            times, outputs = {v: [] for v in libraries}, {}
            order = list(libraries)
            reps = 5 if args.threads == 1 and name == 'b8_s32' else 20
            for round_ in range(args.rounds + 1):
                for variant in order if round_ % 2 == 0 else order[::-1]:
                    q8.library = libraries[variant]
                    model(ids, attention_mask=mask)
                    start = time.perf_counter()
                    for _ in range(reps):
                        out = model(ids, attention_mask=mask).last_hidden_state
                    elapsed = (time.perf_counter() - start) * 1000 / reps
                    if round_:
                        times[variant].append(elapsed)
                    outputs[variant] = out.clone()
                assert torch.equal(outputs['before'], outputs['after']), name
            print(json.dumps({'kind': 'model', 'threads': args.threads, 'workload': name, 'bitwise_equal': True,
                              'results': {v: {'median_ms': statistics.median(t), 'round_ms': t}
                                          for v, t in times.items()}}), flush=True)
        q8.library = native


if __name__ == '__main__':
    main()
