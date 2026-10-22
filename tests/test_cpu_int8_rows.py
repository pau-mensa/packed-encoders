"""Fused x86 dequantization preserves every FP32 result of the separate row bodies."""
import ctypes
from pathlib import Path
import subprocess

import pytest
import torch

from packed_encoders.cpu import int8
from packed_encoders.cpu.attention import _openmp_runtime


@pytest.fixture(scope='module')
def row_probe(tmp_path_factory):
    if not int8.vnni_available():
        pytest.skip('x86 AVX512-VNNI required')
    directory = tmp_path_factory.mktemp('int8-rows')
    source = Path(int8.__file__).with_name('int8_x86.c')
    probe = directory / 'probe.c'
    probe.write_text('#include "' + str(source) + '"\n' + r'''
void probe(int fused, const int32_t *c, const int32_t *corr, const float *s,
           const float *ws, float *residual, const float *weight, const float *bias,
           const float *signs, float *out, uint8_t *quantized, float *scales,
           float *rotated, int M, int K, int block, float eps) {
    float tmp[K], norm[K];
    for (int r=0;r<M;++r) {
        const int32_t *row = c+(size_t)r*K;
        float *res = residual+(size_t)r*K;
        float *dst = out+(size_t)r*K;
        uint8_t *q = quantized+(size_t)r*K;
        float *rot = rotated+(size_t)r*K;
        if (!fused) {
            for (int i=0;i<K;++i)
                tmp[i] = (float)(row[i]-corr[i])*(ws[i]*s[r]);
            pe_add_norm_row(res, tmp, weight, bias, block ? norm : dst, K, eps);
            if (block) emit_row(norm, K, signs, block, rot, q, scales+r);
        } else if (block) {
            dequant_norm(row, corr, s[r], ws, res, weight, bias, norm, K, eps);
            emit_row(norm, K, signs, block, rot, q, scales+r);
        } else {
            dequant_norm(row, corr, s[r], ws, res, weight, bias, dst, K, eps);
        }
    }
}
''')
    output = directory / 'probe.so'
    subprocess.run(['gcc', '-O3', '-march=native', '-fPIC', '-fopenmp', '-ffp-contract=off',
                    '-fno-trapping-math', '-fno-math-errno', '-Wall', '-Wextra', '-Werror',
                    '-Wno-unknown-pragmas', '-shared', str(probe), _openmp_runtime(), '-o', str(output)],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(str(output))
    lib.probe.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*12 + [ctypes.c_int]*3 + [ctypes.c_float]
    lib.probe.restype = None
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield lib.probe
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize('block', [0, 64, 128])
@torch.no_grad()
def test_fused_dequantization_exact(row_probe, block):
    rng = torch.Generator().manual_seed(731)
    for k in (128, 768, 1152) if block == 128 else (64, 128, 192, 768, 1152):
        signs = int8.signs(k, block).float() if block else None
        for m in (1, 7, 16, 32, 256):
            c = torch.randint(-20_000_000, 20_000_001, (m, k), generator=rng, dtype=torch.int32)
            corr = torch.randint(-100_000, 100_001, (k,), generator=rng, dtype=torch.int32)
            c[0] = corr  # exactly zero dequantized row
            scale = torch.logspace(-9, -3, m)
            ws = torch.rand(k, generator=rng)*0.01
            residual = torch.randn(m, k, generator=rng)
            residual[0].zero_()  # constant residual exercises the epsilon-only denominator
            weight = torch.randn(k, generator=rng)
            for bias in (None, torch.randn(k, generator=rng)):
                for eps in (1e-5, 1e-3):
                    outputs = []
                    for fused in (0, 1):
                        r = residual.clone()
                        out, rotated = torch.empty(m, k), torch.empty(m, k)
                        q, scales = torch.empty(m, k, dtype=torch.uint8), torch.empty(m)
                        ptrs = [c, corr, scale, ws, r, weight, bias, signs, out, q, scales, rotated]
                        row_probe(fused, *[None if t is None else t.data_ptr() for t in ptrs], m, k, block, eps)
                        result = (r, scales, rotated) if block else (r, out)
                        outputs.append(tuple(t.view(torch.int32) for t in result) + ((q,) if block else ()))
                    for before, after in zip(*outputs):
                        assert torch.equal(before, after), (block, m, k, bias is not None, eps)
