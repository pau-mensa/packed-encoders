"""Matched call-preparation and scheduling experiments on CPU 2 (1t) or CPUs 0-7 (8t).

Default: compare committed Python call preparation with the current implementation.
Explicit candidates: isolate native-library or Python probes on the baseline path.
Prepared weights and buffers are identical within each process.
"""
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch
from transformers import AutoModel

import packed_encoders as pe
from packed_encoders.cpu import int8
from packed_encoders.cpu.attention import _x86_library
from int8_rows import compile_library as compile_int8
from short_attention import compile_library as compile_attention, micro, check
from bench import inputs, workloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=(1,8), required=True)
    parser.add_argument('--precision', choices=('int8','fp32'), default='int8')
    parser.add_argument('--rounds', type=int, default=15)
    parser.add_argument('--attention-candidate', type=Path, action='append', default=[])
    parser.add_argument('--q8-candidate', type=Path, action='append', default=[])
    parser.add_argument('--python-probe', type=Path)
    parser.add_argument('--micro-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--isa', choices=('native','x86-64-v3'), default='native')
    args=parser.parse_args()
    if args.rounds < 2:
        parser.error('--rounds must be at least 2')
    if args.precision != 'int8' and (args.q8_candidate or args.python_probe):
        parser.error('INT8 candidates require --precision int8')
    probing=bool(args.python_probe or args.attention_candidate or args.q8_candidate)
    torch.set_num_threads(args.threads)
    att_native,_=_x86_library()
    root=Path(__file__).parent/'native_probe'
    with tempfile.TemporaryDirectory(prefix='scheduling-') as tmp, torch.inference_mode():
        tmp=Path(tmp)
        att={'before':compile_attention(root/'scheduling_attention_baseline.c',tmp/'att_before.so',att_native,args.isa)}
        for path in args.attention_candidate:
            att[path.stem]=compile_attention(path,tmp/('att_'+path.stem+'.so'),att_native,args.isa)
        if args.micro_only:
            micro(att,args.threads,args.rounds)
            return
        if args.check_only:
            check(att,args.threads,args.isa)
            return
        qlibs={}
        if args.precision=='int8':
            qnative,_=int8.library()
            qlibs['before']=compile_int8(root/'scheduling_int8_baseline.c',tmp/'q_before.so',qnative)
            for path in args.q8_candidate:
                qlibs[path.stem]=compile_int8(path,tmp/('q_'+path.stem+'.so'),qnative)
        variants={'before':(att['before'],qlibs.get('before'))}
        variants.update({name:(lib,qlibs.get('before')) for name,lib in att.items() if name!='before'})
        variants.update({name:(att['before'],lib) for name,lib in qlibs.items() if name!='before'})
        if args.attention_candidate and args.q8_candidate:
            variants['combined']=(att[args.attention_candidate[-1].stem],qlibs[args.q8_candidate[-1].stem])
        if len(variants)==1:
            variants['after']=(att_native,qnative if args.precision=='int8' else None)
        model=AutoModel.from_pretrained('answerdotai/ModernBERT-base',attn_implementation='sdpa',local_files_only=True).float().eval()
        pe.pack(model,engine=pe.ModernBertCPU(threads=args.threads),precision=args.precision)
        engine=pe.get_engine(model)
        import types
        from native_probe import scheduling_python_baseline as baseline_python
        native_method=engine.attention._native
        baseline_native=types.MethodType(baseline_python._native,engine.attention)
        baseline_core=types.MethodType(baseline_python.core,engine.int8) if engine.int8 is not None else None
        python_variants={}
        if args.python_probe:
            import importlib.util
            spec=importlib.util.spec_from_file_location('python_probe',args.python_probe)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            python_variants=module.variants(engine.int8)
            variants={'before':variants['before'],**{name:variants['before'] for name in python_variants}}
        core=engine.int8.core if engine.int8 is not None else None
        for name in ('b1_s16_pad32','b1_s32','b8_s32'):
            seq,lengths=workloads()[name];ids,mask=inputs(seq,lengths)
            times={v:[] for v in variants};outputs={};order=list(variants)
            reps=5 if args.threads==1 and name=='b8_s32' else 20
            for rnd in range(args.rounds+1):
                for variant in order if rnd%2==0 else order[::-1]:
                    engine.attention.library,qlib=variants[variant]
                    engine.attention._native=baseline_native if variant=='before' or probing else native_method
                    if engine.int8 is not None:
                        engine.int8.library=qlib
                        engine.int8.core=python_variants.get(variant, baseline_core if variant=='before' or probing else core)
                    model(ids,attention_mask=mask)
                    start=time.perf_counter()
                    for _ in range(reps):out=model(ids,attention_mask=mask).last_hidden_state
                    elapsed=(time.perf_counter()-start)*1000/reps
                    if rnd:times[variant].append(elapsed)
                    outputs[variant]=out.view(torch.int32).clone()
                for variant in order[1:]:assert torch.equal(outputs['before'],outputs[variant]),(name,variant)
            print(json.dumps({'kind':'model','threads':args.threads,'precision':args.precision,'workload':name,'bitwise_equal':True,
                              'results':{v:{'median_ms':statistics.median(t),'round_ms':t} for v,t in times.items()}}),flush=True)


if __name__=='__main__':main()
