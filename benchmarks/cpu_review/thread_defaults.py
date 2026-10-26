"""Fresh-process comparison of capped defaults, explicit workers and passive OpenMP."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from packed_encoders.cpu.topology import default_threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=7)
    args = parser.parse_args()
    logical = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    torch.set_num_threads(logical)
    physical = default_threads()
    args.out.mkdir(parents=True, exist_ok=True)
    profiles = [('hf_logical', 'hf', logical, 'same', None),
                ('hf_physical', 'hf', physical, 'same', None),
                ('engine_default', 'engine', logical, 'default', None),
                ('engine_logical', 'engine', logical, 'same', None),
                ('engine_physical', 'engine', physical, 'same', None),
                ('engine_default_passive', 'engine', logical, 'default', 'PASSIVE')]
    rows = {}
    for name, system, threads, workers, policy in profiles:
        env = dict(os.environ)
        env.pop('OMP_WAIT_POLICY', None)
        if policy:
            env['OMP_WAIT_POLICY'] = policy
        path = args.out / f'{name}.json'
        subprocess.run([sys.executable, str(Path(__file__).with_name('compare.py')), system,
                        '--threads', str(threads), '--native-threads', workers, '--trials', str(args.trials),
                        '--only', 'b1_s32,b8_s32', '--out', str(path)], env=env, check=True)
        rows[name] = json.loads(path.read_text())
    lines = [f'# Thread defaults: {rows["hf_logical"]["cpu"]}', '',
             f'Affinity exposes {logical} logical CPUs and {physical} physical cores.', '',
             '| Profile | Torch threads | Native threads | b1_s32 ms | b8_s32 ms |',
             '|---|---:|---:|---:|---:|']
    for name, row in rows.items():
        lines.append(f'| {name} | {row["torch_threads"]} | {row.get("native_threads", "—")} | '
                     + ' | '.join(f'{row["rows"][workload]["median_ms"]:.2f}' for workload in ('b1_s32', 'b8_s32')) + ' |')
    (args.out / 'summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
