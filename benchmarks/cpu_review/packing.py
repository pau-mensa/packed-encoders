"""Measure native compilation reuse across fresh model-packing processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


CHILD = '''
import json, sys, time, torch
from transformers import AutoModel
import packed_encoders as pe
from packed_encoders.cpu import build
torch.set_num_threads(8)
model = AutoModel.from_pretrained('answerdotai/ModernBERT-base', dtype=torch.float32, attn_implementation='sdpa').eval()
compilations = []
original = build._compile
def counted(commands, directory, parallel):
    start = time.perf_counter()
    original(commands, directory, parallel)
    compilations.append(time.perf_counter() - start)
build._compile = counted
start = time.perf_counter()
pe.pack(model, engine=pe.ModernBertCPU(threads=8), precision=sys.argv[1])
assert pe.get_engine(model).precision == sys.argv[1]
print(json.dumps({'pack_s':time.perf_counter()-start, 'build_s':compilations, 'precision':sys.argv[1]}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for precision in ('fp32', 'int8'):
        with tempfile.TemporaryDirectory(prefix='pe-build-benchmark-') as directory:
            for mode in ('disabled', 'cold', 'warm'):
                env = {**os.environ, 'PACKED_ENCODERS_CPU_CACHE_DIR': '' if mode == 'disabled' else directory}
                result = subprocess.check_output([sys.executable, '-c', CHILD, precision], env=env, text=True)
                row = json.loads(result.splitlines()[-1])
                if mode == 'warm' and row['build_s']:
                    raise RuntimeError('warm fresh process unexpectedly recompiled native code')
                rows[f'{precision}_{mode}'] = row
                print(precision, mode, row, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
