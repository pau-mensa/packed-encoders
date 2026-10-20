// Per-row FP32 bodies shared by the native CPU kernels (local_attention.c, int8.c, rows.c),
// and the int8 libraries' pack-time GPTQ row. Each function sets its own contraction mode, so
// including files see the same arithmetic. GCC ignores the clang pragmas; rows.c and
// int8_x86.c are built with -ffp-contract=off, so they never contract.
#ifndef PE_ROWS_H
#define PE_ROWS_H

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <math.h>
#include <stddef.h>
#include <stdint.h>

// One token's split-half RoPE (head_dim 64) and token-major -> head-major layout.
// `row` is the token's [q | k | v] projection (3 * heads * 64); `out` is [3, heads, tokens, 64].
// Keep multiply/add separate to match the unfused FP32 PyTorch operations.
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

// The rational erf (odd 13 / even 8 degree, as in Eigen/TensorFlow) is within ~4e-7 of
// erf, comparable to PyTorch's FP32 GELU; not the tanh approximation. Contraction is allowed.
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
    // FP32 erf(4) rounds to 1, so saturate exactly beyond the fitted range.
    return a >= 4.f ? 1.f : (a <= -4.f ? -1.f : p / q);
}

// Exact-erf GeGLU of one row: out[i] = GELU(in[i]) * in[width + i] for [activation | gate].
static inline void pe_geglu_row(const float *in, float *out, int width) {
#pragma clang fp contract(on)
    const float *gate = in + width;
#pragma clang loop interleave_count(4)
    for (int i = 0; i < width; ++i)
        out[i] = 0.5f * in[i] * (1.f + pe_erf(in[i] * 0.70710678118654752f)) * gate[i];
}

// Residual add + LayerNorm of one row, width a multiple of 32: r += b (bitwise PyTorch's
// add), then out = (r - mean) * rstd * weight (+ bias). Two-pass statistics with eight
// vector accumulators: FP32-accurate, not bitwise PyTorch's LayerNorm. A NULL bias means
// none. Only the explicit FMAs below fuse. The portable version keeps the same 32 partial
// sums and reduction order, so both produce the same bits.
#ifdef __ARM_NEON
static inline float pe_sum8(const float32x4_t s[8]) {
    return vaddvq_f32(vaddq_f32(vaddq_f32(vaddq_f32(s[0], s[1]), vaddq_f32(s[2], s[3])),
                                vaddq_f32(vaddq_f32(s[4], s[5]), vaddq_f32(s[6], s[7]))));
}

static inline void pe_add_norm_row(float *r, const float *b, const float *weight, const float *bias,
                                   float *out, int width, float eps) {
#pragma clang fp contract(off)
    float32x4_t s[8];
    for (int j = 0; j < 8; ++j) s[j] = vdupq_n_f32(0.f);
    for (int i = 0; i < width; i += 32)
        for (int j = 0; j < 8; ++j) {
            float32x4_t x = vaddq_f32(vld1q_f32(r + i + 4 * j), vld1q_f32(b + i + 4 * j));
            vst1q_f32(r + i + 4 * j, x);
            s[j] = vaddq_f32(s[j], x);
        }
    float mean = pe_sum8(s) / width;
    float32x4_t m = vdupq_n_f32(mean);
    for (int j = 0; j < 8; ++j) s[j] = vdupq_n_f32(0.f);
    for (int i = 0; i < width; i += 32)
        for (int j = 0; j < 8; ++j) {
            float32x4_t d = vsubq_f32(vld1q_f32(r + i + 4 * j), m);
            s[j] = vfmaq_f32(s[j], d, d);
        }
    float32x4_t rstd = vdupq_n_f32(1.f / sqrtf(pe_sum8(s) / width + eps));
    for (int i = 0; i < width; i += 32)
        for (int j = 0; j < 8; ++j) {
            float32x4_t y = vmulq_f32(vmulq_f32(vsubq_f32(vld1q_f32(r + i + 4 * j), m), rstd),
                                      vld1q_f32(weight + i + 4 * j));
            vst1q_f32(out + i + 4 * j, bias ? vaddq_f32(y, vld1q_f32(bias + i + 4 * j)) : y);
        }
}
#else
// Partial sum c holds lane c % 4 of NEON accumulator c / 4; pe_sum8's tree, lane by lane.
static inline float pe_sum32(const float p[32]) {
#pragma clang fp contract(off)
    float t[4];
    for (int l = 0; l < 4; ++l)
        t[l] = ((p[l] + p[4 + l]) + (p[8 + l] + p[12 + l])) + ((p[16 + l] + p[20 + l]) + (p[24 + l] + p[28 + l]));
    return (t[0] + t[1]) + (t[2] + t[3]);
}

static inline void pe_add_norm_row(float *r, const float *b, const float *weight, const float *bias,
                                   float *out, int width, float eps) {
#pragma clang fp contract(off)
    float p[32] = {0};
    for (int i = 0; i < width; i += 32)
        for (int c = 0; c < 32; ++c) {
            float x = r[i + c] + b[i + c];
            r[i + c] = x;
            p[c] += x;
        }
    float mean = pe_sum32(p) / width;
    for (int c = 0; c < 32; ++c) p[c] = 0.f;
    for (int i = 0; i < width; i += 32)
        for (int c = 0; c < 32; ++c) {
            float d = r[i + c] - mean;
            p[c] = fmaf(d, d, p[c]);
        }
    float rstd = 1.f / sqrtf(pe_sum32(p) / width + eps);
    for (int i = 0; i < width; i += 32)
        for (int c = 0; c < 32; ++c) {
            float y = (r[i + c] - mean) * rstd * weight[i + c];
            out[i + c] = bias ? y + bias[i + c] : y;
        }
}
#endif

// GPTQ's column loop over columns [i1, i2) of one weight row (int8.py's gptq): each column
// rounds to a symmetric int8 value at the row's scale, and its error, divided by the diagonal
// of the upper Cholesky factor hinv ([k][k], row-major), updates the block's later columns.
// The errors (i2 - i1 of them) feed the caller's update of the columns after the block. These
// are PyTorch's operations one by one: no contraction, true division, round half to even.
static inline void pe_gptq_row(float *w, const float *hinv, float scale, int8_t *q, float *errors,
                               int k, int i1, int i2) {
#pragma clang fp contract(off)
    float safe = scale > 0 ? scale : 1.f;
    for (int i = i1; i < i2; ++i) {
        const float *h = hinv + (size_t)i * k;
        float quantized = fminf(fmaxf(rintf(w[i] / safe), -127.f), 127.f);
        float error = (w[i] - quantized * scale) / h[i];
        q[i] = (int8_t)quantized;
        errors[i - i1] = error;
        for (int j = i + 1; j < i2; ++j) w[j] -= h[j] * error;
    }
}

#endif
