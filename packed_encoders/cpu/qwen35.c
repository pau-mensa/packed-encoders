// Packed Qwen3.5: causal depthwise convolution, grouped delta recurrence and gated RMSNorm.
// One task owns a (sequence, value head), including its recurrent state. Sequence boundaries
// are explicit; neither convolution taps nor recurrent state can cross them.
#include <math.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
// glibc exposes vector expf through its SIMD ABI. Annotate only this function;
// reductions and recurrence retain ordinary FP32 semantics (no fast-math).
#if defined(__GLIBC__) && defined(__x86_64__) && !defined(PE_QWEN_SCALAR_EXP)
#pragma omp declare simd notinbranch
extern float expf(float);
#define PE_QWEN_VECTOR_EXP 1
#else
#define PE_QWEN_VECTOR_EXP 0
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifdef __APPLE__
#include <vecLib/vForce.h>
#include "pool.h"
#else
#include <omp.h>
#endif

typedef struct {
    const float *proj, *conv, *bias, *alog, *dt, *norm;
    const int32_t *cu;
    float *out, *scratch;
    int rows, nk, nv, hk, hv, width, stride, workers;
    float eps;
    _Atomic int next;
} QwenJob;

static float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
static float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// Projection rows are [gate | up | optional alignment padding].
typedef struct {
    const float *in;
    float *out;
    int rows, width, stride;
    _Atomic int next;
} SwigluJob;

static void swiglu_worker(void *opaque, size_t worker) {
    (void)worker;
    SwigluJob *j = opaque;
    int row;
    while ((row = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed)) < j->rows) {
        const float *gate = j->in + (size_t)row * j->stride, *up = gate + j->width;
        float *out = j->out + (size_t)row * j->width;
#ifdef __APPLE__
        // Reuse the destination for vector exp; inputs may include row padding.
        for (int d = 0; d < j->width; ++d) out[d] = -gate[d];
        vvexpf(out, out, &j->width);
        for (int d = 0; d < j->width; ++d)
            out[d] = (gate[d] / (1.f + out[d])) * up[d];
#else
#if PE_QWEN_VECTOR_EXP
        #pragma omp simd
#endif
        for (int d = 0; d < j->width; ++d)
            out[d] = (gate[d] / (1.f + expf(-gate[d]))) * up[d];
#endif
    }
}

int pe_qwen35_swiglu(const float *in, float *out, int rows, int width, int stride,
                     int workers, void *pool) {
    if (rows <= 0 || width <= 0 || stride < 2 * width || workers <= 0) return 1;
    if (workers > rows) workers = rows;
    SwigluJob j = {.in=in, .out=out, .rows=rows, .width=width, .stride=stride, .next=0};
#ifdef __APPLE__
    pe_pool_apply(pool, workers, &j, swiglu_worker, 0);
#else
    (void)pool;
    #pragma omp parallel num_threads(workers)
    swiglu_worker(&j, (size_t)omp_get_thread_num());
#endif
    return 0;
}

// Qwen's zero-centered RMSNorm: normalize(residual + branch) * (1 + weight).
// The residual is engine-owned; out must be separate storage. These row jobs are
// small enough that one worker avoids pool wakeups and competition with BLAS.
int pe_qwen35_add_norm(float *residual, const float *branch, const float *weight,
                      float *out, int rows, int width, float eps) {
    if (rows <= 0 || width <= 0) return 1;
    for (int row = 0; row < rows; ++row) {
        float *x = residual + (size_t)row * width;
        const float *b = branch + (size_t)row * width;
        float *dst = out + (size_t)row * width;
        float sum = 0.f;
        int d = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
        float32x4_t sums = vdupq_n_f32(0.f);
        for (; d + 4 <= width; d += 4) {
            float32x4_t v = vaddq_f32(vld1q_f32(x + d), vld1q_f32(b + d));
            vst1q_f32(x + d, v);
            sums = vaddq_f32(sums, vmulq_f32(v, v));
        }
        sum = vaddvq_f32(sums);
#endif
        for (; d < width; ++d) {
            x[d] += b[d];
            sum += x[d] * x[d];
        }
        float inv = 1.f / sqrtf(sum / width + eps);
        d = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
        for (; d + 4 <= width; d += 4) {
            float32x4_t w = vaddq_f32(vdupq_n_f32(1.f), vld1q_f32(weight + d));
            vst1q_f32(dst + d, vmulq_f32(vmulq_n_f32(vld1q_f32(x + d), inv), w));
        }
#endif
        for (; d < width; ++d) dst[d] = x[d] * inv * (1.f + weight[d]);
    }
    return 0;
}

// Normalize each full head, then rotate only its configured prefix. The two
// RoPE halves use their own cos/sin entries (including scaled/custom tables).
static void qk_norm_rope(const float *x, const float *weight, const float *cos,
                         const float *sin, float *out, int hd, int rot, float eps) {
    float sum = 0.f;
    int d = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
    float32x4_t sums = vdupq_n_f32(0.f);
    for (; d + 4 <= hd; d += 4) {
        float32x4_t v = vld1q_f32(x + d);
        sums = vaddq_f32(sums, vmulq_f32(v, v));
    }
    sum = vaddvq_f32(sums);
#endif
    for (; d < hd; ++d) sum += x[d] * x[d];
    float inv = 1.f / sqrtf(sum / hd + eps);
    int half = rot / 2;
    for (d = 0; d < half; ++d) {
        float lo = x[d] * inv * (1.f + weight[d]);
        float hi = x[d + half] * inv * (1.f + weight[d + half]);
        out[d] = lo * cos[d] + (-hi) * sin[d];
        out[d + half] = hi * cos[d + half] + lo * sin[d + half];
    }
    for (d = rot; d < hd; ++d) out[d] = x[d] * inv * (1.f + weight[d]);
}

int pe_qwen35_qk_rope(const float *p, const float *wq, const float *wk,
                      const float *cos, const float *sin, float *q, float *k,
                      int rows, int nq, int nk, int hd, int rot, int stride,
                      float qeps, float keps) {
    if (rows <= 0 || nq <= 0 || nk <= 0 || hd <= 0 || rot < 0 || rot > hd || rot % 2 ||
        stride < 2 * (nq + nk) * hd) return 1;
    for (int row = 0; row < rows; ++row) {
        const float *src = p + (size_t)row * stride;
        const float *c = cos + (size_t)row * rot, *s = sin + (size_t)row * rot;
        for (int head = 0; head < nq; ++head)
            qk_norm_rope(src + head * 2 * hd, wq, c, s,
                         q + ((size_t)row * nq + head) * hd, hd, rot, qeps);
        for (int head = 0; head < nk; ++head)
            qk_norm_rope(src + (2 * nq + head) * hd, wk, c, s,
                         k + ((size_t)row * nk + head) * hd, hd, rot, keps);
    }
    return 0;
}

static void convolve_channels(const QwenJob *j, int token, int begin, int channel,
                               int count, float *out, float *temp) {
    int d = 0;
#if defined(__AVX512F__)
    if (j->width == 4) {
        const __m512i channels = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28,
                                                   32, 36, 40, 44, 48, 52, 56, 60);
        for (; d + 16 <= count; d += 16) {
            __m512 x = j->bias ? _mm512_loadu_ps(j->bias + channel + d) : _mm512_setzero_ps();
            for (int tap = 0; tap < 4; ++tap) {
                int src = token + tap - 3;
                if (src >= begin) {
                    __m512 w = _mm512_i32gather_ps(channels,
                                                  j->conv + (size_t)(channel + d) * 4 + tap, 4);
                    __m512 p = _mm512_loadu_ps(j->proj + (size_t)src * j->stride + channel + d);
                    x = _mm512_add_ps(x, _mm512_mul_ps(p, w));
                }
            }
            _mm512_storeu_ps(out + d, x);
        }
    }
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (j->width == 4) {
        for (; d + 4 <= count; d += 4) {
            float32x4_t x = j->bias ? vld1q_f32(j->bias + channel + d) : vdupq_n_f32(0.f);
            // Deinterleave the four taps to process adjacent channels together.
            float32x4x4_t w = vld4q_f32(j->conv + (size_t)(channel + d) * 4);
            for (int tap = 0; tap < 4; ++tap) {
                int src = token + tap - 3;
                if (src >= begin)
                    x = vaddq_f32(x, vmulq_f32(vld1q_f32(j->proj + (size_t)src * j->stride + channel + d), w.val[tap]));
            }
            vst1q_f32(out + d, x);
        }
    }
#endif
    for (; d < count; ++d) {
        float x = j->bias ? j->bias[channel + d] : 0.f;
        for (int tap = 0; tap < j->width; ++tap) {
            int src = token + tap - j->width + 1;
            if (src >= begin)
                x += j->proj[(size_t)src * j->stride + channel + d] * j->conv[(size_t)(channel + d) * j->width + tap];
        }
        out[d] = x;
    }
#ifdef __APPLE__
    for (d = 0; d < count; ++d) temp[d] = -out[d];
    vvexpf(temp, temp, &count);
    for (d = 0; d < count; ++d) out[d] *= 1.f / (1.f + temp[d]);
#else
    (void)temp;
#if PE_QWEN_VECTOR_EXP
    #pragma omp simd if (j->hk % 16 == 0 && j->hv % 16 == 0)
#endif
    for (d = 0; d < count; ++d) out[d] *= sigmoid(out[d]);
#endif
}

#if defined(__aarch64__) && defined(__ARM_NEON)
// Each value channel is independent. Tile across values so delta and output stay
// in registers while reducing over keys. Preserve the generic reduction order;
// the build keeps FP contraction disabled for both paths.
static void recurrence_neon(float *state, float *delta, float *output,
                            const float *q, const float *k, int hk, int hv,
                            float decay, float beta) {
    for (int base = 0; base < hv; base += 16) {
        float32x4_t dv0 = vld1q_f32(delta + base), ov0 = vdupq_n_f32(0.f);
        float32x4_t dv1 = vld1q_f32(delta + base + 4), ov1 = vdupq_n_f32(0.f);
        float32x4_t dv2 = vld1q_f32(delta + base + 8), ov2 = vdupq_n_f32(0.f);
        float32x4_t dv3 = vld1q_f32(delta + base + 12), ov3 = vdupq_n_f32(0.f);
        for (int d = 0; d < hk; ++d) {
            float *s = state + (size_t)d * hv + base;
            float32x4_t s0 = vmulq_n_f32(vld1q_f32(s), decay);
            vst1q_f32(s, s0);
            dv0 = vsubq_f32(dv0, vmulq_n_f32(s0, k[d]));
            float32x4_t s1 = vmulq_n_f32(vld1q_f32(s + 4), decay);
            vst1q_f32(s + 4, s1);
            dv1 = vsubq_f32(dv1, vmulq_n_f32(s1, k[d]));
            float32x4_t s2 = vmulq_n_f32(vld1q_f32(s + 8), decay);
            vst1q_f32(s + 8, s2);
            dv2 = vsubq_f32(dv2, vmulq_n_f32(s2, k[d]));
            float32x4_t s3 = vmulq_n_f32(vld1q_f32(s + 12), decay);
            vst1q_f32(s + 12, s3);
            dv3 = vsubq_f32(dv3, vmulq_n_f32(s3, k[d]));
        }
        dv0 = vmulq_n_f32(dv0, beta);
        dv1 = vmulq_n_f32(dv1, beta);
        dv2 = vmulq_n_f32(dv2, beta);
        dv3 = vmulq_n_f32(dv3, beta);
        for (int d = 0; d < hk; ++d) {
            float *s = state + (size_t)d * hv + base;
            float32x4_t s0 = vaddq_f32(vld1q_f32(s), vmulq_n_f32(dv0, k[d]));
            vst1q_f32(s, s0);
            ov0 = vaddq_f32(ov0, vmulq_n_f32(s0, q[d]));
            float32x4_t s1 = vaddq_f32(vld1q_f32(s + 4), vmulq_n_f32(dv1, k[d]));
            vst1q_f32(s + 4, s1);
            ov1 = vaddq_f32(ov1, vmulq_n_f32(s1, q[d]));
            float32x4_t s2 = vaddq_f32(vld1q_f32(s + 8), vmulq_n_f32(dv2, k[d]));
            vst1q_f32(s + 8, s2);
            ov2 = vaddq_f32(ov2, vmulq_n_f32(s2, q[d]));
            float32x4_t s3 = vaddq_f32(vld1q_f32(s + 12), vmulq_n_f32(dv3, k[d]));
            vst1q_f32(s + 12, s3);
            ov3 = vaddq_f32(ov3, vmulq_n_f32(s3, q[d]));
        }
        vst1q_f32(output + base, ov0);
        vst1q_f32(output + base + 4, ov1);
        vst1q_f32(output + base + 8, ov2);
        vst1q_f32(output + base + 12, ov3);
    }
}
#endif

#if defined(__AVX512F__) || defined(__AVX2__)
#if defined(__AVX512F__)
#define PE_VEC __m512
#define PE_LANES 16
#define PE_LOAD _mm512_loadu_ps
#define PE_STORE _mm512_storeu_ps
#define PE_SET _mm512_set1_ps
#define PE_ZERO _mm512_setzero_ps
#define PE_ADD _mm512_add_ps
#define PE_SUB _mm512_sub_ps
#define PE_MUL _mm512_mul_ps
#else
#define PE_VEC __m256
#define PE_LANES 8
#define PE_LOAD _mm256_loadu_ps
#define PE_STORE _mm256_storeu_ps
#define PE_SET _mm256_set1_ps
#define PE_ZERO _mm256_setzero_ps
#define PE_ADD _mm256_add_ps
#define PE_SUB _mm256_sub_ps
#define PE_MUL _mm256_mul_ps
#endif

// As in the NEON path, keep independent value channels in registers across
// the key reduction. No horizontal sums or FMA: each channel retains the
// generic recurrence's FP32 operation order. Odd head widths use generic C.
static inline __attribute__((always_inline)) void recurrence_x86(
        float *state, float *delta, float *output, const float *q,
        const float *k, int hk, int hv, float decay, float beta, const int vectors) {
    const PE_VEC decayv = PE_SET(decay), betav = PE_SET(beta);
    for (int base = 0; base < hv; base += vectors * PE_LANES) {
        PE_VEC dv0 = PE_LOAD(delta + base), dv1 = PE_LOAD(delta + base + PE_LANES);
        PE_VEC ov0 = PE_ZERO(), ov1 = PE_ZERO();
        PE_VEC dv2 = PE_ZERO(), dv3 = PE_ZERO(), ov2 = PE_ZERO(), ov3 = PE_ZERO();
        if (vectors == 4) {
            dv2 = PE_LOAD(delta + base + 2 * PE_LANES);
            dv3 = PE_LOAD(delta + base + 3 * PE_LANES);
        }
        for (int d = 0; d < hk; ++d) {
            float *s = state + (size_t)d * hv + base;
            const PE_VEC kd = PE_SET(k[d]);
            PE_VEC s0 = PE_MUL(PE_LOAD(s), decayv);
            PE_VEC s1 = PE_MUL(PE_LOAD(s + PE_LANES), decayv);
            PE_STORE(s, s0);
            PE_STORE(s + PE_LANES, s1);
            dv0 = PE_SUB(dv0, PE_MUL(s0, kd));
            dv1 = PE_SUB(dv1, PE_MUL(s1, kd));
            if (vectors == 4) {
                PE_VEC s2 = PE_MUL(PE_LOAD(s + 2 * PE_LANES), decayv);
                PE_VEC s3 = PE_MUL(PE_LOAD(s + 3 * PE_LANES), decayv);
                PE_STORE(s + 2 * PE_LANES, s2);
                PE_STORE(s + 3 * PE_LANES, s3);
                dv2 = PE_SUB(dv2, PE_MUL(s2, kd));
                dv3 = PE_SUB(dv3, PE_MUL(s3, kd));
            }
        }
        dv0 = PE_MUL(dv0, betav);
        dv1 = PE_MUL(dv1, betav);
        if (vectors == 4) { dv2 = PE_MUL(dv2, betav); dv3 = PE_MUL(dv3, betav); }
        for (int d = 0; d < hk; ++d) {
            float *s = state + (size_t)d * hv + base;
            const PE_VEC kd = PE_SET(k[d]), qd = PE_SET(q[d]);
            PE_VEC s0 = PE_ADD(PE_LOAD(s), PE_MUL(dv0, kd));
            PE_VEC s1 = PE_ADD(PE_LOAD(s + PE_LANES), PE_MUL(dv1, kd));
            PE_STORE(s, s0);
            PE_STORE(s + PE_LANES, s1);
            ov0 = PE_ADD(ov0, PE_MUL(s0, qd));
            ov1 = PE_ADD(ov1, PE_MUL(s1, qd));
            if (vectors == 4) {
                PE_VEC s2 = PE_ADD(PE_LOAD(s + 2 * PE_LANES), PE_MUL(dv2, kd));
                PE_VEC s3 = PE_ADD(PE_LOAD(s + 3 * PE_LANES), PE_MUL(dv3, kd));
                PE_STORE(s + 2 * PE_LANES, s2);
                PE_STORE(s + 3 * PE_LANES, s3);
                ov2 = PE_ADD(ov2, PE_MUL(s2, qd));
                ov3 = PE_ADD(ov3, PE_MUL(s3, qd));
            }
        }
        PE_STORE(output + base, ov0);
        PE_STORE(output + base + PE_LANES, ov1);
        if (vectors == 4) {
            PE_STORE(output + base + 2 * PE_LANES, ov2);
            PE_STORE(output + base + 3 * PE_LANES, ov3);
        }
    }
}
#endif

static void qwen_head(QwenJob *j, int task, float *scratch) {
    int row = task / j->nv, head = task % j->nv, kh = head / (j->nv / j->nk);
    int hk = j->hk, hv = j->hv, kd = j->nk * hk, vd = j->nv * hv;
    float *state = scratch, *q = state + (size_t)hk * hv, *k = q + hk;
    float *delta = k + hk, *output = delta + hv, *temp = output + hv;
    memset(state, 0, (size_t)hk * hv * sizeof(float));
    float a = expf(j->alog[head]), scale = 1.f / sqrtf((float)hk);
    int begin = j->cu[row], end = j->cu[row + 1];
    for (int token = begin; token < end; ++token) {
        float qs = 1e-6f, ks = 1e-6f;
        convolve_channels(j, token, begin, kh * hk, hk, q, temp);
        convolve_channels(j, token, begin, kd + kh * hk, hk, k, temp);
        for (int d = 0; d < hk; ++d) {
            qs += q[d] * q[d]; ks += k[d] * k[d];
        }
        qs = scale / sqrtf(qs); ks = 1.f / sqrtf(ks);
        for (int d = 0; d < hk; ++d) { q[d] *= qs; k[d] *= ks; }
        const float *p = j->proj + (size_t)token * j->stride;
        float beta = sigmoid(p[2 * kd + 2 * vd + head]);
        float decay = expf(-a * softplus(p[2 * kd + 2 * vd + j->nv + head] + j->dt[head]));
        convolve_channels(j, token, begin, 2 * kd + head * hv, hv, delta, temp);
        memset(output, 0, (size_t)hv * sizeof(float));
#if defined(__AVX512F__) || defined(__AVX2__)
        if (hv % (4 * PE_LANES) == 0) {
            recurrence_x86(state, delta, output, q, k, hk, hv, decay, beta, 4);
        } else if (hv % (2 * PE_LANES) == 0) {
            recurrence_x86(state, delta, output, q, k, hk, hv, decay, beta, 2);
        } else
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
        if (hv % 16 == 0) {
            recurrence_neon(state, delta, output, q, k, hk, hv, decay, beta);
        } else
#endif
        {
            // The value dimension is contiguous: the compiler vectorizes these inner loops.
            for (int d = 0; d < hk; ++d) {
                float *s = state + (size_t)d * hv;
                for (int v = 0; v < hv; ++v) { s[v] *= decay; delta[v] -= s[v] * k[d]; }
            }
            for (int v = 0; v < hv; ++v) delta[v] *= beta;
            for (int d = 0; d < hk; ++d) {
                float *s = state + (size_t)d * hv;
                for (int v = 0; v < hv; ++v) { s[v] += k[d] * delta[v]; output[v] += s[v] * q[d]; }
            }
        }
        float ss = 0.f;
        for (int v = 0; v < hv; ++v) ss += output[v] * output[v];
        float inv = 1.f / sqrtf(ss / hv + j->eps);
        float *dst = j->out + ((size_t)token * j->nv + head) * hv;
        const float *z = p + 2 * kd + vd + head * hv;
#ifdef __APPLE__
        for (int v = 0; v < hv; ++v) temp[v] = -z[v];
        vvexpf(temp, temp, &hv);
        for (int v = 0; v < hv; ++v) dst[v] = output[v] * inv * j->norm[v] * z[v] * (1.f / (1.f + temp[v]));
#else
#if PE_QWEN_VECTOR_EXP
        #pragma omp simd if (j->hk % 16 == 0 && j->hv % 16 == 0)
#endif
        for (int v = 0; v < hv; ++v) dst[v] = output[v] * inv * j->norm[v] * z[v] * sigmoid(z[v]);
#endif
    }
}

static void qwen_worker(void *opaque, size_t worker) {
    QwenJob *j = opaque;
    size_t stride = (size_t)j->hk * j->hv + 2 * j->hk + 2 * j->hv + (j->hk > j->hv ? j->hk : j->hv);
    int task;
    while ((task = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed)) < j->rows * j->nv)
        qwen_head(j, task, j->scratch + worker * stride);
}

int pe_qwen35_gdn(const float *proj, const float *conv, const float *bias, const float *alog,
                  const float *dt, const float *norm, const int32_t *cu, float *out, float *scratch,
                  int rows, int nk, int nv, int hk, int hv, int width, int stride, float eps,
                  int workers, void *pool) {
    if (rows <= 0 || nk <= 0 || nv <= 0 || nv % nk || hk <= 0 || hv <= 0 || width <= 0 ||
        workers <= 0 || stride < 2 * nk * hk + 2 * nv * hv + 2 * nv) return 1;
    QwenJob j = {.proj=proj, .conv=conv, .bias=bias, .alog=alog, .dt=dt, .norm=norm, .cu=cu,
                 .out=out, .scratch=scratch, .rows=rows, .nk=nk, .nv=nv, .hk=hk, .hv=hv,
                 .width=width, .stride=stride, .workers=workers, .eps=eps, .next=0};
#ifdef __APPLE__
    pe_pool_apply(pool, workers, &j, qwen_worker, 0);
#else
    (void)pool;
    #pragma omp parallel num_threads(workers)
    qwen_worker(&j, (size_t)omp_get_thread_num());
#endif
    return 0;
}
