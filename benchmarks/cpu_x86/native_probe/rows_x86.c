// Probe: the packed engine's row kernels (rows.h) built for x86 with OpenMP on torch's libgomp.
// RoPE/GeGLU bodies are copied verbatim from rows.h; add+norm swaps NEON for 16-lane portable C.
#include <math.h>
#include <stddef.h>

static inline void pe_rope_row(const float *row, const float *co, const float *si, float *out,
                               int token, int tokens, int heads) {
#pragma clang fp contract(off)
    size_t plane = (size_t)heads * tokens * 64;
    for (int head = 0; head < heads; ++head) {
        const float *src = row + (size_t)head * 64;
        float *dst = out + ((size_t)head * tokens + token) * 64;
        for (int part = 0; part < 2; ++part) {
            const float *in = src + part * heads * 64;
            float *o = dst + part * plane;
            for (int d = 0; d < 32; ++d) {
                o[d] = in[d] * co[d] + (-in[d + 32]) * si[d];
                o[d + 32] = in[d + 32] * co[d + 32] + in[d] * si[d + 32];
            }
        }
        for (int d = 0; d < 64; ++d) dst[2 * plane + d] = src[2 * heads * 64 + d];
    }
}

static inline float pe_erf(float a) {
#pragma clang fp contract(on)
    float x = a > 4.f ? 4.f : (a < -4.f ? -4.f : a);
    float x2 = x * x;
    float p = x2 * -2.72614225801306e-10f + 2.77068142495902e-08f;
    p = x2 * p + -2.10102402082508e-06f;
    p = x2 * p + -5.69250639462346e-05f;
    p = x2 * p + -7.34990630326855e-04f;
    p = x2 * p + -2.95459980854025e-03f;
    p = x2 * p + -1.60960333262415e-02f;
    p = x * p;
    float q = x2 * -1.45660718464996e-05f + -2.13374055278905e-04f;
    q = x2 * q + -1.68282697438203e-03f;
    q = x2 * q + -7.37332916720468e-03f;
    q = x2 * q + -1.42647390514189e-02f;
    return a >= 4.f ? 1.f : (a <= -4.f ? -1.f : p / q);
}

static inline void pe_geglu_row(const float *in, float *out, int width) {
#pragma clang fp contract(on)
    const float *gate = in + width;
    for (int i = 0; i < width; ++i)
        out[i] = 0.5f * in[i] * (1.f + pe_erf(in[i] * 0.70710678118654752f)) * gate[i];
}

static inline float sum16(const float s[16]) {
    float t[8];
    for (int j = 0; j < 8; ++j) t[j] = s[j] + s[j + 8];
    for (int j = 0; j < 4; ++j) t[j] = t[j] + t[j + 4];
    return (t[0] + t[2]) + (t[1] + t[3]);
}

static inline void pe_add_norm_row(float *r, const float *b, const float *weight, const float *bias,
                                   float *out, int width, float eps) {
#pragma clang fp contract(off)
    float s[16] = {0};
    for (int i = 0; i < width; i += 16)
        for (int j = 0; j < 16; ++j) {
            float x = r[i + j] + b[i + j];
            r[i + j] = x;
            s[j] += x;
        }
    float mean = sum16(s) / width;
    for (int j = 0; j < 16; ++j) s[j] = 0.f;
    for (int i = 0; i < width; i += 16)
        for (int j = 0; j < 16; ++j) {
            float d = r[i + j] - mean;
            s[j] = fmaf(d, d, s[j]);
        }
    float rstd = 1.f / sqrtf(sum16(s) / width + eps);
    for (int i = 0; i < width; i += 16)
        for (int j = 0; j < 16; ++j) {
            float y = (r[i + j] - mean) * rstd * weight[i + j];
            out[i + j] = bias ? y + bias[i + j] : y;
        }
}

int pe_rope_layout(const float *qkv, const float *cos, const float *sin, float *out, int tokens, int heads,
                   int workers) {
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int token = 0; token < tokens; ++token)
        pe_rope_row(qkv + (size_t)token * 3 * heads * 64, cos + (size_t)token * 64, sin + (size_t)token * 64, out,
                    token, tokens, heads);
    return 0;
}

int pe_geglu(const float *in, float *out, int rows, int width, int workers) {
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int row = 0; row < rows; ++row) pe_geglu_row(in + (size_t)row * 2 * width, out + (size_t)row * width, width);
    return 0;
}

int pe_add_layer_norm(float *residual, const float *branch, const float *weight, const float *bias, float *out,
                      int rows, int width, float eps, int workers) {
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int row = 0; row < rows; ++row) {
        size_t o = (size_t)row * width;
        pe_add_norm_row(residual + o, branch + o, weight, bias, out + o, width, eps);
    }
    return 0;
}
