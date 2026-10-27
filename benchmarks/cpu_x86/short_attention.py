"""Matched attention-kernel A/B, using the same weights and buffers in each process.

Pin --threads 1 to CPU 2 and --threads 8 to CPUs 0-7; run processes serially.
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
from packed_encoders.cpu.attention import CPUAttention, _openmp_runtime, _x86_library
from bench import inputs, workloads


def compile_library(source, output, native, isa='native'):
    subprocess.run(['gcc', '-O3', f'-march={isa}', '-fPIC', '-fopenmp', '-ffp-contract=fast',
                    '-fno-trapping-math', '-fno-math-errno', '-Wall', '-Wextra', '-Werror',
                    '-Wno-psabi', '-shared', str(source), _openmp_runtime(), '-o', str(output)],
                   check=True, capture_output=True)
    library = ctypes.CDLL(str(output))
    library.pe_local_attention.argtypes = native.pe_local_attention.argtypes
    library.pe_local_attention.restype = native.pe_local_attention.restype
    return library


def report(kind, times, **kwargs):
    print(json.dumps({'kind': kind, **kwargs, 'bitwise_equal': True,
                      'results': {v: {'median_ms': statistics.median(t), 'round_ms': t}
                                  for v, t in times.items()}}), flush=True)


def micro(libraries, threads, rounds):
    rng = torch.Generator().manual_seed(43)
    for lengths in ((16,), (32,), (32,) * 8):
        tokens, heads, tile, half = sum(lengths), 12, 64, 64
        q, k, v = [torch.randn(heads, tokens, 64, generator=rng) for _ in range(3)]
        out = torch.empty_like(q)
        lens = torch.tensor(lengths, dtype=torch.int32)
        scratch = torch.empty(threads * (64 + tile) * max(lengths))
        args = (q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), lens.data_ptr(),
                tokens, heads, len(lengths), tile, half, 1, threads, scratch.data_ptr(), scratch.numel())
        times, outputs = {v: [] for v in libraries}, {}
        order = list(libraries)
        for round_ in range(rounds + 1):
            for name in order if round_ % 2 == 0 else order[::-1]:
                fn = libraries[name].pe_local_attention
                assert fn(*args) == 0
                start = time.perf_counter_ns()
                for _ in range(200):
                    fn(*args)
                elapsed = (time.perf_counter_ns() - start) / 1e6 / 200
                if round_:
                    times[name].append(elapsed)
                outputs[name] = out.clone()
            for name in order[1:]:
                assert torch.equal(outputs[order[0]], outputs[name]), (lengths, name)
        report('micro', times, threads=threads, lengths=lengths)


def check(libraries, threads, isa):
    rng = torch.Generator().manual_seed(43)
    attention = CPUAttention('native', 64, 64, threads)
    count = 0
    for lengths in ((16,), (32,), (16, 32, 1, 3, 4, 7, 15, 17, 31, 33, 65, 129, 200)):
        q, k, v = [torch.randn(1, 3, sum(lengths), 64, generator=rng) * 2 for _ in range(3)]
        q[:, :, 0].zero_()
        q[:, :, 1] *= 100
        boundaries = torch.tensor(lengths, dtype=torch.int32)
        for half in (0, 7, 14, 15, 16, 30, 31, 32, 64, 200):
            for tile in (4, 16, 32, 64):
                attention.tile = tile
                for token_major in (False, True):
                    outputs = []
                    for library in libraries.values():
                        attention.library = library
                        outputs.append(attention._native(q, k, v, lengths, boundaries, half, token_major).clone())
                    for output in outputs[1:]:
                        assert torch.equal(outputs[0], output), (lengths, half, tile, token_major)
                    count += 1
    attention.close()
    print(json.dumps({'kind': 'correctness', 'threads': threads, 'isa': isa,
                      'cases': count, 'bitwise_equal': True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=(1, 8), required=True)
    parser.add_argument('--rounds', type=int, default=15)
    parser.add_argument('--micro', action='store_true')
    parser.add_argument('--micro-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--isa', choices=('native', 'x86-64-v3'), default='native')
    parser.add_argument('--precision', choices=('fp32', 'int8'), default='int8')
    parser.add_argument('--candidate', type=Path, help='Optional additional C source for microbench ablation')
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error('--rounds must be at least 2')
    torch.set_num_threads(args.threads)
    native, _ = _x86_library()
    with tempfile.TemporaryDirectory(prefix='attention-bench-') as directory, torch.inference_mode():
        baseline = Path(__file__).parent / 'native_probe/attention_baseline.c'
        libraries = {'before': compile_library(baseline, Path(directory) / 'before.so', native, args.isa),
                     'after': native}
        if args.isa != 'native':
            source = Path(pe.__file__).parent / 'cpu/local_attention_x86.c'
            libraries['after'] = compile_library(source, Path(directory) / 'after.so', native, args.isa)
        if args.candidate:
            libraries['candidate'] = compile_library(args.candidate, Path(directory) / 'candidate.so', native, args.isa)
        if args.check_only:
            check(libraries, args.threads, args.isa)
            return
        if args.micro or args.micro_only:
            micro(libraries, args.threads, args.rounds)
        if args.micro_only:
            return
        model = AutoModel.from_pretrained('answerdotai/ModernBERT-base', attn_implementation='sdpa',
                                          local_files_only=True).float().eval()
        pe.pack(model, engine=pe.ModernBertCPU(threads=args.threads), precision=args.precision)
        attention = pe.get_engine(model).attention
        if attention.library is None:
            raise RuntimeError('native x86 attention required')
        for name in ('b1_s16_pad32', 'b1_s32', 'b8_s32'):
            seq, lengths = workloads()[name]
            ids, mask = inputs(seq, lengths)
            times, outputs = {v: [] for v in libraries}, {}
            order = list(libraries)
            reps = 5 if args.threads == 1 and name == 'b8_s32' else 20
            for round_ in range(args.rounds + 1):
                for variant in order if round_ % 2 == 0 else order[::-1]:
                    attention.library = libraries[variant]
                    model(ids, attention_mask=mask)
                    start = time.perf_counter()
                    for _ in range(reps):
                        out = model(ids, attention_mask=mask).last_hidden_state
                    elapsed = (time.perf_counter() - start) * 1000 / reps
                    if round_:
                        times[variant].append(elapsed)
                    outputs[variant] = out.clone()
                for variant in order[1:]:
                    assert torch.equal(outputs[order[0]], outputs[variant]), (name, variant)
            report('model', times, threads=args.threads, precision=args.precision, workload=name)
        attention.library = native


if __name__ == '__main__':
    main()
