// Opt-in int8 ModernBERT projections for Apple CPUs with SME2 (precision="int8").
// Build: clang -O3 -mcpu=apple-m4 -dynamiclib int8.c -o int8.dylib
//
// Weights are packed once by Python as [N/64][K/4][64][4] int8 with per-output-channel
// scales. Activation rows are quantized symmetrically per row, after a block Hadamard
// rotation with random signs (block 64 or 128, 1/sqrt(block) folded into the row scale) that
// the weights carry too, and packed as [ceil(M/16)][K/4][16][4]. The SME outer products
// accumulate int32 tiles that are stored straight to memory; their consumer dequantizes.
//
// Everything between two attentions is row-local, so one call per layer hands 32-row groups
// to workers, and each worker runs the whole chain on its group:
//   context -> quantize -> Wo -> add+norm -> rotate + quantize -> Wi -> GeGLU -> rotate +
//   quantize -> Wmo -> add+norm -> rotate + quantize -> next QKV -> RoPE.
// The two Apple M4 Pro P-cluster SME units are shared by their cores; workers interleave
// SME and NEON work. The attention-output rotation is folded into Wv/Wo by Python, so the
// context is quantized without rotation. At pack time, Python's GPTQ runs its column loop
// here too (pe_q8_gptq_block).
#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "rows.h"

typedef struct {
    int hidden, heads, width, block_hidden, block_width;
    const float *signs_hidden, *signs_width;  // rotation signs for hidden- and width-sized inputs
} PeQ8Model;

// Everything after attention i up to attention i + 1; the input step uses only the QKV fields.
typedef struct {
    const int8_t *wo, *wi, *wmo, *wqkv;  // wqkv: the next layer's, NULL after the last layer
    const float *so, *si, *smo, *sqkv;   // per-output-channel weight scales
    const float *mlp_weight, *mlp_bias, *after_weight, *after_bias;
    float mlp_eps, after_eps;
    int table;                           // the next layer's RoPE table
} PeQ8Segment;

typedef struct {
    const float *cos[2], *sin[2];        // gathered [tokens, 64] RoPE tables
    float *rope, *residual, *out;        // rope [3, heads, tokens, 64]; out: final FP32 rows
    int8_t *a_hidden, *a_width;          // packed activations, hidden- and width-sized inputs
    int32_t *c;                          // int32 products, tokens x max(3 hidden, 2 width)
    float *scales, *scratch;             // per-row activation scales; per-worker scratch
    long scratch_floats;
    int scratch_workers;
} PeQ8Buffers;

// ---------------------------------------------------------------- GEMM
// c[M, N] (row stride N) = a x w^T for M rows of a packed activation block range.
// Up to 16 rows: one A tile against four 16-column B tiles; otherwise 2x2 tiles.
__arm_locally_streaming __arm_new("za")
static void gemm(const int8_t *a, const int8_t *w, int32_t *c, int M, int N, int K) {
    const int ks = K / 4, mblocks = (M + 15) / 16;
    svbool_t p = svptrue_b8(), p32 = svptrue_b32();
    if (M <= 16) {
        for (int nb = 0; nb < N / 64; nb++) {
            const int8_t *bp = w + (size_t)nb * ks * 256;
            svzero_za();
            for (int k = 0; k < ks; k++) {
                const int8_t *bk = bp + 256 * k;
                svint8_t va = svld1_s8(p, a + 64 * k);
                svmopa_za32_s8_m(0, p, p, va, svld1_s8(p, bk));
                svmopa_za32_s8_m(1, p, p, va, svld1_s8(p, bk + 64));
                svmopa_za32_s8_m(2, p, p, va, svld1_s8(p, bk + 128));
                svmopa_za32_s8_m(3, p, p, va, svld1_s8(p, bk + 192));
            }
            int32_t *out = c + nb * 64;
            for (int i = 0; i < M; i++) {
                svst1_hor_za32(0, i, p32, out + (size_t)i * N);
                svst1_hor_za32(1, i, p32, out + (size_t)i * N + 16);
                svst1_hor_za32(2, i, p32, out + (size_t)i * N + 32);
                svst1_hor_za32(3, i, p32, out + (size_t)i * N + 48);
            }
        }
        return;
    }
    for (int nh = 0; nh < N / 32; nh++) {
        const int8_t *bp = w + (size_t)(nh / 2) * ks * 256 + (nh % 2) * 128;
        for (int mb = 0; mb < mblocks; mb += 2) {
            const int8_t *a0 = a + (size_t)mb * ks * 64;
            const int8_t *a1 = mb + 1 < mblocks ? a0 + (size_t)ks * 64 : a0;
            svzero_za();
            for (int k = 0; k < ks; k++) {
                const int8_t *bk = bp + 256 * k;
                svint8_t va0 = svld1_s8(p, a0 + 64 * k), va1 = svld1_s8(p, a1 + 64 * k);
                svint8_t vb0 = svld1_s8(p, bk), vb1 = svld1_s8(p, bk + 64);
                svmopa_za32_s8_m(0, p, p, va0, vb0);
                svmopa_za32_s8_m(1, p, p, va0, vb1);
                svmopa_za32_s8_m(2, p, p, va1, vb0);
                svmopa_za32_s8_m(3, p, p, va1, vb1);
            }
            int32_t *out = c + (size_t)mb * 16 * N + nh * 32;
            int left = M - mb * 16;
            for (int i = 0; i < 16 && i < left; i++) {
                svst1_hor_za32(0, i, p32, out + (size_t)i * N);
                svst1_hor_za32(1, i, p32, out + (size_t)i * N + 16);
            }
            for (int i = 0; i < 16 && i < left - 16; i++) {
                svst1_hor_za32(2, i, p32, out + (size_t)(16 + i) * N);
                svst1_hor_za32(3, i, p32, out + (size_t)(16 + i) * N + 16);
            }
        }
    }
}

// ---------------------------------------------------------------- quantization
// Unnormalized 64-point Walsh-Hadamard transform of 16 vectors in registers.
static inline void fwht64(float32x4_t r[16]) {
    const float32x4_t s2 = {1, 1, -1, -1}, s1 = {1, -1, 1, -1};
    for (int d = 8; d >= 1; d /= 2)  // h = 32, 16, 8, 4: vectors d apart
        for (int i = 0; i < 16; i++)
            if (!(i & d)) {
                float32x4_t a = r[i], b = r[i + d];
                r[i] = vaddq_f32(a, b);
                r[i + d] = vsubq_f32(a, b);
            }
    for (int i = 0; i < 16; i++) {  // h = 2, 1 inside each vector
        float32x4_t a = vfmaq_f32(vextq_f32(r[i], r[i], 2), r[i], s2);
        r[i] = vfmaq_f32(vrev64q_f32(a), a, s1);
    }
}

// dst = H_block (signs * src) for one block (64 or 128, unnormalized); returns max |dst| lanes.
static inline float32x4_t fwht_block(const float *src, const float *signs, float *dst, int block) {
    float32x4_t r[16], mx = vdupq_n_f32(0);
    for (int half = 0; half < block; half += 64) {
        for (int i = 0; i < 16; i++) {
            float32x4_t a = vmulq_f32(vld1q_f32(src + half + 4 * i), vld1q_f32(signs + half + 4 * i));
            if (block == 128) {  // stage h = 64 straight from the source
                float32x4_t b = vmulq_f32(vld1q_f32(src + (64 - half) + 4 * i), vld1q_f32(signs + (64 - half) + 4 * i));
                a = half ? vsubq_f32(b, a) : vaddq_f32(a, b);
            }
            r[i] = a;
        }
        fwht64(r);
        for (int i = 0; i < 16; i++) {
            mx = vmaxq_f32(mx, vabsq_f32(r[i]));
            vst1q_f32(dst + half + 4 * i, r[i]);
        }
    }
    return mx;
}

// Quantize one row with its amax and scatter it into packed A; scale includes `norm`.
static inline void quant_row(const float *row, int K, float amax, float norm, int8_t *packed, float *scale, int m) {
    scale[m] = amax / 127.0f * norm;
    float inv = amax > 0 ? 127.0f / amax : 0;
    int8_t *dst = packed + (size_t)(m / 16) * 16 * K + (m % 16) * 4;
    for (int k = 0; k < K; k += 8) {
        int32x4_t q0 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(row + k), inv));
        int32x4_t q1 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(row + k + 4), inv));
        int8x8_t b = vqmovn_s16(vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1)));
        vst1_lane_s32((int32_t *)(dst + (size_t)(k / 4) * 64), vreinterpret_s32_s8(b), 0);
        vst1_lane_s32((int32_t *)(dst + (size_t)(k / 4 + 1) * 64), vreinterpret_s32_s8(b), 1);
    }
}

// Rotate (signs != NULL) and quantize row m into packed A. `tmp` (K floats) must not alias src.
static inline void emit_row(const float *src, int K, const float *signs, int block, float *tmp,
                            int8_t *packed, float *scale, int m) {
    float32x4_t mx = vdupq_n_f32(0);
    if (signs) {
        for (int k = 0; k < K; k += block) mx = vmaxq_f32(mx, fwht_block(src + k, signs + k, tmp + k, block));
        quant_row(tmp, K, vmaxvq_f32(mx), 1.0f / sqrtf((float)block), packed, scale, m);
        return;
    }
    for (int k = 0; k < K; k += 4) mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(src + k)));
    quant_row(src, K, vmaxvq_f32(mx), 1.0f, packed, scale, m);
}

// y = c * (ws * s) for one int32 row, N a multiple of 16.
static inline void dequant_row(const int32_t *c, float s, const float *ws, float *y, int N) {
    for (int n = 0; n < N; n += 16) {
        float32x4_t r[4];
        for (int j = 0; j < 4; j++)
            r[j] = vmulq_f32(vcvtq_f32_s32(vld1q_s32(c + n + 4 * j)), vmulq_n_f32(vld1q_f32(ws + n + 4 * j), s));
        for (int j = 0; j < 4; j++) vst1q_f32(y + n + 4 * j, r[j]);
    }
}

static void zero_padding(int8_t *packed, int M, int K) {
    int blocks = (M + 15) / 16;
    memset(packed + (size_t)M / 16 * 16 * K, 0, (size_t)(blocks * 16 - M / 16 * 16) * K);
}

// ---------------------------------------------------------------- row segments
typedef struct {
    const PeQ8Model *m;
    const PeQ8Segment *seg;
    const PeQ8Buffers *b;
    const float *src;  // FP32 input rows (input step) or head-major context
    int input, tokens, rows_per, groups;
    _Atomic int next;
} Job;

static void run_group(const Job *j, int begin, int end, float *scratch) {
    const PeQ8Model *m = j->m;
    const PeQ8Segment *g = j->seg;
    const PeQ8Buffers *b = j->b;
    const int d = m->hidden, w = m->width, t = j->tokens, h = m->heads;
    const int stride = 3 * d > 2 * w ? 3 * d : 2 * w;
    float *s = b->scales;
    // This group's int32 region at the widest stride, so groups never overlap whatever N each
    // is at; C(N) is the base that puts global row `begin` at the region's start for width N.
    int32_t *region = b->c + (size_t)begin * stride;
#define C(N) (region - (size_t)begin * (N))
#define GEMM(a, K, wt, N) gemm((a) + (size_t)begin * (K), (wt), region, end - begin, (N), (K))
    if (j->input) {
        for (int r = begin; r < end; ++r)
            emit_row(j->src + (size_t)r * d, d, m->signs_hidden, m->block_hidden, scratch, b->a_hidden, s, r);
    } else {
        float *x = scratch, *y = scratch + d, *tmp = scratch + 2 * d;
        for (int r = begin; r < end; ++r) {  // gather the heads of row r
            for (int k = 0; k < h; ++k) memcpy(x + k * 64, j->src + ((size_t)k * t + r) * 64, 64 * sizeof(float));
            emit_row(x, d, NULL, 0, NULL, b->a_hidden, s, r);
        }
        GEMM(b->a_hidden, d, g->wo, d);
        for (int r = begin; r < end; ++r) {
            dequant_row(C(d) + (size_t)r * d, s[r], g->so, x, d);
            pe_add_norm_row(b->residual + (size_t)r * d, x, g->mlp_weight, g->mlp_bias, y, d, g->mlp_eps);
            emit_row(y, d, m->signs_hidden, m->block_hidden, tmp, b->a_hidden, s, r);
        }
        GEMM(b->a_hidden, d, g->wi, 2 * w);
        float *hid = scratch, *act = scratch + 2 * w, *rot = scratch + 3 * w;
        for (int r = begin; r < end; ++r) {
            dequant_row(C(2 * w) + (size_t)r * 2 * w, s[r], g->si, hid, 2 * w);
            pe_geglu_row(hid, act, w);
            emit_row(act, w, m->signs_width, m->block_width, rot, b->a_width, s, r);
        }
        GEMM(b->a_width, w, g->wmo, d);
        for (int r = begin; r < end; ++r) {
            dequant_row(C(d) + (size_t)r * d, s[r], g->smo, x, d);
            float *out = g->wqkv ? y : b->out + (size_t)r * d;
            pe_add_norm_row(b->residual + (size_t)r * d, x, g->after_weight, g->after_bias, out, d, g->after_eps);
            if (g->wqkv) emit_row(y, d, m->signs_hidden, m->block_hidden, tmp, b->a_hidden, s, r);
        }
        if (!g->wqkv) return;
    }
    GEMM(b->a_hidden, d, g->wqkv, 3 * d);
    const float *cos = b->cos[g->table], *sin = b->sin[g->table];
    for (int r = begin; r < end; ++r) {
        dequant_row(C(3 * d) + (size_t)r * 3 * d, s[r], g->sqkv, scratch, 3 * d);
        pe_rope_row(scratch, cos + (size_t)r * 64, sin + (size_t)r * 64, b->rope, r, t, h);
    }
#undef GEMM
#undef C
}

static void job_worker(void *opaque, size_t id) {
    Job *j = opaque;
    float *scratch = j->b->scratch + id * (size_t)j->b->scratch_floats;
    for (int task; (task = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed)) < j->groups;) {
        int begin = task * j->rows_per, end = begin + j->rows_per < j->tokens ? begin + j->rows_per : j->tokens;
        run_group(j, begin, end, scratch);
    }
}

static int run(const PeQ8Model *m, const PeQ8Segment *seg, const PeQ8Buffers *b, const float *src,
               int input, int tokens, int workers) {
    const int d = m->hidden, w = m->width;
    long need = 3L * d > 4L * w ? 3L * d : 4L * w;
    if (tokens < 1 || workers < 1 || m->heads * 64 != d || d % 64 || w % 64 || b->scratch_floats < need
        || (m->block_hidden != 64 && m->block_hidden != 128) || d % m->block_hidden
        || (m->block_width != 64 && m->block_width != 128) || w % m->block_width)
        return -1;
    if (workers > b->scratch_workers) workers = b->scratch_workers;
    // 32-row groups across workers (16 and 32 tied on an M4 Pro; 64-256 were slower);
    // one worker takes every row at once.
    Job j = {m, seg, b, src, input, tokens, workers > 1 ? 32 : (tokens + 31) / 32 * 32, 0, 0};
    j.groups = (tokens + j.rows_per - 1) / j.rows_per;
    zero_padding(b->a_hidden, tokens, d);
    zero_padding(b->a_width, tokens, w);
    if (workers > j.groups) workers = j.groups;
    if (workers == 1) job_worker(&j, 0);
    else dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &j, job_worker);
    return 0;
}

// The first layer: FP32 input rows -> rotate + quantize -> QKV -> RoPE.
int pe_q8_input(const PeQ8Model *m, const PeQ8Segment *seg, const PeQ8Buffers *b, const float *x,
                int tokens, int workers) {
    return run(m, seg, b, x, 1, tokens, workers);
}

// Everything after one attention (head-major context [heads, tokens, 64]) up to the next.
int pe_q8_segment(const PeQ8Model *m, const PeQ8Segment *seg, const PeQ8Buffers *b, const float *context,
                  int tokens, int workers) {
    return run(m, seg, b, context, 0, tokens, workers);
}

// Building blocks, exported for tests.
void pe_q8_gemm(const int8_t *a, const int8_t *w, int32_t *c, int M, int N, int K) { gemm(a, w, c, M, N, K); }

void pe_q8_quantize(const float *x, int M, int K, const float *signs, int block, float *tmp,
                    int8_t *packed, float *scale) {
    zero_padding(packed, M, K);
    for (int m = 0; m < M; ++m) emit_row(x + (size_t)m * K, K, signs, block, tmp, packed, scale, m);
}

// ---------------------------------------------------------------- pack-time GPTQ
typedef struct {
    float *w;
    const float *hinv, *scale;
    int8_t *q;
    float *errors;
    int n, k, i1, i2, groups;
    _Atomic int next;
} GptqJob;

static void gptq_worker(void *opaque, size_t id) {
    GptqJob *j = opaque;
    (void)id;
    for (int task; (task = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed)) < j->groups;)
        for (int r = task * 16; r < j->n && r < task * 16 + 16; ++r)
            pe_gptq_row(j->w + (size_t)r * j->k, j->hinv, j->scale[r], j->q + (size_t)r * j->k,
                        j->errors + (size_t)r * (j->i2 - j->i1), j->k, j->i1, j->i2);
}

// GPTQ's column loop over columns [i1, i2) of a row-major [n][k] weight copy (int8.py): rows
// are independent, so workers take groups of 16. errors is [n][i2 - i1].
int pe_q8_gptq_block(float *w, const float *hinv, const float *scale, int8_t *q, float *errors,
                     int n, int k, int i1, int i2, int workers) {
    if (!w || !hinv || !scale || !q || !errors || n < 1 || i1 < 0 || i2 <= i1 || i2 > k || workers < 1) return -1;
    GptqJob j = {w, hinv, scale, q, errors, n, k, i1, i2, (n + 15) / 16, 0};
    if (workers > j.groups) workers = j.groups;
    if (workers == 1) gptq_worker(&j, 0);
    else dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &j, gptq_worker);
    return 0;
}
