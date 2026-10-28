"""Require native hardware/compiler support before running the CPU test suites."""

import argparse
import json
from pathlib import Path
import platform
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect', required=True, choices=('avx2', 'vnni', 'sme2'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from compare import environment
    from packed_encoders.cpu.attention import CPUAttention
    from packed_encoders.cpu.int8 import library, vnni_available
    from packed_encoders.cpu.sme import library as sme_library, sme2_available

    attention = CPUAttention('native', 64, 64, 1)
    if args.expect == 'avx2':
        if platform.machine() != 'x86_64' or attention.native_linear or vnni_available():
            raise RuntimeError('AVX2-only host required; an AVX-512 host is not Intel AVX2 validation')
    elif args.expect == 'vnni':
        if not vnni_available() or not attention.native_linear:
            raise RuntimeError('AVX512-VNNI and native FP32 GEMM required')
        library()
    else:
        if not sme2_available():
            raise RuntimeError('Apple SME2 hardware required')
        library()
        sme_library()
    report = {'expected': args.expect, **environment(), 'torch': torch.__version__,
              'native_attention': attention.native_local, 'native_gemm': attention.native_linear,
              'vnni': vnni_available(), 'sme2': sme2_available()}
    attention.close()
    import pytest

    class Results:
        def __init__(self):
            self.passed, self.skipped = [], []

        def pytest_runtest_logreport(self, report):
            if report.when == 'call' and report.passed:
                self.passed.append(report.nodeid)
            elif report.skipped:
                self.skipped.append({'test': report.nodeid, 'reason': str(report.longrepr)})

    results = Results()
    root = Path(__file__).resolve().parents[2]
    suites = [*sorted((root / 'tests').glob('test_cpu_*.py')),
              *(root / 'tests' / name for name in ('test_modernbert_cpu.py', 'test_config.py',
                                                  'test_epilogues.py', 'test_registry.py'))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    status = pytest.main(['-q', *map(str, suites), f'--junitxml={args.out.with_suffix(".xml")}'], plugins=[results])
    report.update(exit_code=int(status), passed=results.passed, skipped=results.skipped)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(status)


if __name__ == '__main__':
    main()
