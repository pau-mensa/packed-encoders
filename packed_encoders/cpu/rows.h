// Per-row FP32 bodies shared by the native CPU kernels (local_attention.c, int8.c).
// Each function sets its own contraction mode, so including files see the same arithmetic.
#ifndef PE_ROWS_H
#define PE_ROWS_H

#include <arm_neon.h>
#include <math.h>
#include <stddef.h>

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

static inline float pe_sum8(const float32x4_t s[8]) {
    return vaddvq_f32(vaddq_f32(vaddq_f32(vaddq_f32(s[0], s[1]), vaddq_f32(s[2], s[3])),
                                vaddq_f32(vaddq_f32(s[4], s[5]), vaddq_f32(s[6], s[7]))));
}

// Residual add + LayerNorm of one row, width a multiple of 32: r += b (bitwise PyTorch's
// add), then out = (r - mean) * rstd * weight (+ bias). Two-pass statistics with eight
// vector accumulators: FP32-accurate, not bitwise PyTorch's LayerNorm. A NULL bias means
// none. Only the explicit FMAs below fuse.
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

#endif
