// Opt-in int8 ModernBERT projections for x86 CPUs with AVX512-VNNI (precision="int8"): the
// pipeline and contract of int8.c, with u8 x s8 VNNI dot products instead of SME outer products.
// Build: gcc -O3 -march=native -ffp-contract=off -fno-trapping-math -fno-math-errno -fopenmp -c int8_x86.c
//
// Weights are packed once by Python as [N/16][K/4][16][4] int8 with per-output-channel scales:
// one 64-byte vector holds 16 output channels x 4 consecutive k. Activation rows are rotated and
// quantized exactly as on Apple (symmetric int8 per row), then stored row-major as u8 = q + 128,
// because VNNI multiplies unsigned by signed bytes. The products then carry 128 * sum_k w[n][k],
// which Python precomputes per channel; subtracting it gives the s8 x s8 sums exactly, so the
// FP32 results equal Apple's whenever the shared FP32 row bodies do.
//
// One OpenMP team runs each call stage by stage, with a barrier after each: GEMM tasks cover
// (channel panel, row chunk) pairs, so a single short sequence still spreads over all threads;
// the row stages (dequantize, residual LayerNorm, GeGLU, rotate + quantize, RoPE) split the rows.
// Built with contraction off, as rows.c, so the shared row bodies give the same bits as there.
// At pack time, Python's GPTQ runs its column loop here too (pe_q8_gptq_block).
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

#include "rows.h"

typedef struct {
    int hidden, heads, width, block_hidden, block_width;
    const float *signs_hidden, *signs_width;  // rotation signs for hidden- and width-sized inputs
} PeQ8Model;

// int8.c's segment, then the per-channel corrections (128 * sum_k w[n][k]) of each projection.
typedef struct {
    const int8_t *wo, *wi, *wmo, *wqkv;  // wqkv: the next layer's, NULL after the last layer
    const float *so, *si, *smo, *sqkv;   // per-output-channel weight scales
    const float *mlp_weight, *mlp_bias, *after_weight, *after_bias;
    float mlp_eps, after_eps;
    int table;                           // the next layer's RoPE table
    const int32_t *co, *ci, *cmo, *cqkv;
} PeQ8Segment;

typedef struct {
    const float *cos[2], *sin[2];        // gathered [tokens, 64] RoPE tables
    float *rope, *residual, *out;        // rope [3, heads, tokens, 64]; out: final FP32 rows
    int8_t *a_hidden, *a_width;          // u8 activation rows [tokens][K], hidden- and width-sized inputs
    int32_t *c;                          // int32 products, tokens x max(3 hidden, 2 width)
    float *scales, *scratch;             // per-row activation scales; per-worker scratch
    long scratch_floats;
    int scratch_workers;
} PeQ8Buffers;

typedef float v16 __attribute__((vector_size(64)));
typedef float v16u __attribute__((vector_size(64), aligned(4), may_alias));
typedef int v16i __attribute__((vector_size(64)));
typedef int v16iu __attribute__((vector_size(64), aligned(4), may_alias));
typedef unsigned char v16b __attribute__((vector_size(16)));
typedef unsigned char v16bu __attribute__((vector_size(16), aligned(1), may_alias));

static inline v16 load(const float *p) { return *(const v16u *)p; }
static inline void store(float *p, v16 x) { *(v16u *)p = x; }
static inline v16i loadi(const int32_t *p) { return *(const v16iu *)p; }
static inline v16 vabs(v16 x) { return (v16)((v16i)x & 0x7fffffff); }
static inline v16 vmax(v16 a, v16 b) { v16i gt = a > b; return (v16)(((v16i)a & gt) | ((v16i)b & ~gt)); }
static inline float hmax(v16 x) {
    float m = x[0];
    for (int i = 1; i < 16; ++i) m = x[i] > m ? x[i] : m;
    return m;
}

// ---------------------------------------------------------------- GEMM
#define ROWS 8    // rows per register block
#define COLS 3    // 16-channel vectors per panel
#define CHUNK 64  // rows per GEMM task

static inline __m512i quad(const uint8_t *p) {  // 4 activation bytes in every lane
    int32_t v;
    memcpy(&v, p, 4);
    return _mm512_set1_epi32(v);
}

// c[r][16 x] = sum_k a[r][k] * w[n][k] for R <= ROWS rows (stride lda) and C <= COLS 16-channel
// vectors of packed weights (K/4 = ks steps of 64 bytes each). Named accumulators, not an array:
// with an array GCC kept a stack copy of every accumulator inside the loop (2.5x slower).
static inline __attribute__((always_inline)) void tile(const uint8_t *a, int lda, const int8_t *w, int ks,
                                                       int32_t *c, int ldc, const int R, const int C) {
    const __m512i z = _mm512_setzero_si512();
    const size_t panel = (size_t)ks * 64;
#define DECL(r) __m512i c##r##0 = z, c##r##1 = z, c##r##2 = z;
    DECL(0) DECL(1) DECL(2) DECL(3) DECL(4) DECL(5) DECL(6) DECL(7)
#undef DECL
    for (int k = 0; k < ks; ++k) {
        const int8_t *wk = w + (size_t)k * 64;
        __m512i b0 = _mm512_loadu_si512(wk);
        __m512i b1 = C > 1 ? _mm512_loadu_si512(wk + panel) : z;
        __m512i b2 = C > 2 ? _mm512_loadu_si512(wk + 2 * panel) : z;
#define ROW(r)                                                         \
        if (R > r) {                                                   \
            __m512i va = quad(a + (size_t)r * lda + 4 * k);            \
            c##r##0 = _mm512_dpbusd_epi32(c##r##0, va, b0);            \
            if (C > 1) c##r##1 = _mm512_dpbusd_epi32(c##r##1, va, b1); \
            if (C > 2) c##r##2 = _mm512_dpbusd_epi32(c##r##2, va, b2); \
        }
        ROW(0) ROW(1) ROW(2) ROW(3) ROW(4) ROW(5) ROW(6) ROW(7)
#undef ROW
    }
#define STORE(r)                                                            \
    if (R > r) {                                                            \
        _mm512_storeu_si512(c + (size_t)r * ldc, c##r##0);                  \
        if (C > 1) _mm512_storeu_si512(c + (size_t)r * ldc + 16, c##r##1);  \
        if (C > 2) _mm512_storeu_si512(c + (size_t)r * ldc + 32, c##r##2);  \
    }
    STORE(0) STORE(1) STORE(2) STORE(3) STORE(4) STORE(5) STORE(6) STORE(7)
#undef STORE
}

// One function per (vectors, rows) shape, dispatched by table: a chunk's last rows and a
// narrower last panel take the smaller shapes.
typedef void (*Tile)(const uint8_t *, int, const int8_t *, int, int32_t *, int);
#define SHAPE(r, cols)                                                                                   \
    static __attribute__((noinline)) void tile_##cols##_##r(const uint8_t *a, int lda, const int8_t *w,  \
                                                             int ks, int32_t *c, int ldc) {              \
        tile(a, lda, w, ks, c, ldc, r, cols);                                                            \
    }
#define SHAPES(cols) SHAPE(1, cols) SHAPE(2, cols) SHAPE(3, cols) SHAPE(4, cols) SHAPE(5, cols) SHAPE(6, cols) \
    SHAPE(7, cols) SHAPE(8, cols)
#define TABLE(cols) {tile_##cols##_1, tile_##cols##_2, tile_##cols##_3, tile_##cols##_4, tile_##cols##_5, \
                     tile_##cols##_6, tile_##cols##_7, tile_##cols##_8}
SHAPES(1)
SHAPES(2)
SHAPES(3)
static const Tile TILES[COLS][ROWS] = {TABLE(1), TABLE(2), TABLE(3)};
#undef SHAPE
#undef SHAPES
#undef TABLE

// c[M][N] = a[M][K] x w^T. Inside a parallel region: an orphaned worksharing loop with its barrier.
static void gemm(const uint8_t *a, int K, const int8_t *w, int32_t *c, int N, int M) {
    const int ks = K / 4, vectors = N / 16, panels = (vectors + COLS - 1) / COLS;
    const int chunks = (M + CHUNK - 1) / CHUNK;
#pragma omp for schedule(static)
    for (int task = 0; task < panels * chunks; ++task) {  // panel-major: a thread's tasks share weights
        int p = task / chunks, r0 = task % chunks * CHUNK, r1 = r0 + CHUNK < M ? r0 + CHUNK : M;
        int v0 = p * COLS, nv = vectors - v0 < COLS ? vectors - v0 : COLS;
        const int8_t *wp = w + (size_t)v0 * ks * 64;
        for (int r = r0; r < r1; r += ROWS) {
            int R = r1 - r < ROWS ? r1 - r : ROWS;
            TILES[nv - 1][R - 1](a + (size_t)r * K, K, wp, ks, c + (size_t)r * N + 16 * v0, N);
        }
    }
}

// ---------------------------------------------------------------- quantization
// Lanes h apart swap and combine inside one vector: lane i gets a[i] + a[i ^ h] when bit h of i
// is clear, a[i ^ h] - a[i] when set (int8.c's NEON lane stages, at 16 lanes).
static const v16i SWAP[4] = {
    {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14},
    {2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13},
    {4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11},
    {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7}};
static const v16 SIGN[4] = {
    {1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1},
    {1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1},
    {1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1},
    {1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1}};

// dst = H_block (signs * src) for one block (64 or 128, unnormalized, Sylvester order); returns
// max |dst| lanes. Same butterflies in the same order as int8.c, so the same bits.
// Named vectors and constant block sizes keep the butterflies in registers;
// dynamically indexing an array made GCC write the intermediate stages to the stack.
static inline __attribute__((always_inline)) v16 fwht_block(const float *src, const float *signs,
                                                            float *dst, const int block) {
#define LOAD(i) v16 r##i = load(src + 16 * (i)) * load(signs + 16 * (i))
    LOAD(0); LOAD(1); LOAD(2); LOAD(3);
    v16 r4 = {}, r5 = {}, r6 = {}, r7 = {}, mx = {};
#define PAIR(a, b) do { v16 x = r##a, y = r##b; r##a = x + y; r##b = x - y; } while (0)
    if (block == 128) {
        r4 = load(src + 64) * load(signs + 64);
        r5 = load(src + 80) * load(signs + 80);
        r6 = load(src + 96) * load(signs + 96);
        r7 = load(src + 112) * load(signs + 112);
        PAIR(0, 4); PAIR(1, 5); PAIR(2, 6); PAIR(3, 7);  // h = 64
        PAIR(4, 6); PAIR(5, 7);                         // h = 32, upper half
        PAIR(4, 5); PAIR(6, 7);                         // h = 16, upper half
    }
    PAIR(0, 2); PAIR(1, 3);                             // h = 32, lower half
    PAIR(0, 1); PAIR(2, 3);                             // h = 16, lower half
#define LANES(i, s) r##i = __builtin_shuffle(r##i, SWAP[s]) + r##i * SIGN[s]
#define OUT(i) do {                                                                                             \
        LANES(i, 3); LANES(i, 2); LANES(i, 1); LANES(i, 0);                                                     \
        mx = vmax(mx, vabs(r##i));                                                                              \
        store(dst + 16 * (i), r##i);                                                                            \
    } while (0)
    OUT(0); OUT(1); OUT(2); OUT(3);
    if (block == 128) { OUT(4); OUT(5); OUT(6); OUT(7); }
#undef LOAD
#undef PAIR
#undef LANES
#undef OUT
    return mx;
}

// Quantize one row with its amax into u8 = round(row * 127 / amax) + 128; scale includes `norm`.
static inline void quant_row(const float *row, int K, float amax, float norm, uint8_t *dst, float *scale) {
    *scale = amax / 127.0f * norm;
    float inv = amax > 0 ? 127.0f / amax : 0;
    for (int k = 0; k < K; k += 16) {
        v16 x = load(row + k) * inv;
        x = (x + 12582912.f) - 12582912.f;  // round to nearest even, as vcvtnq (|x| <= 127)
        *(v16bu *)(dst + k) = __builtin_convertvector(__builtin_convertvector(x, v16i) + 128, v16b);
    }
}

// Rotate (signs != NULL) and quantize one row into dst. `tmp` (K floats) must not alias src.
static inline void emit_row(const float *src, int K, const float *signs, int block, float *tmp, uint8_t *dst,
                            float *scale) {
    v16 mx = {};
    if (signs) {
        if (block == 128) {
            for (int k = 0; k < K; k += 128) mx = vmax(mx, fwht_block(src + k, signs + k, tmp + k, 128));
        } else {
            for (int k = 0; k < K; k += 64) mx = vmax(mx, fwht_block(src + k, signs + k, tmp + k, 64));
        }
        quant_row(tmp, K, hmax(mx), 1.0f / sqrtf((float)block), dst, scale);
        return;
    }
    for (int k = 0; k < K; k += 16) mx = vmax(mx, vabs(load(src + k)));
    quant_row(src, K, hmax(mx), 1.0f, dst, scale);
}

// y = (c - correction) * (ws * s) for one int32 row, N a multiple of 16 (int8.c's dequant_row).
static inline void dequant_row(const int32_t *c, const int32_t *correction, float s, const float *ws, float *y,
                               int N) {
    for (int n = 0; n < N; n += 16)
        store(y + n, __builtin_convertvector(loadi(c + n) - loadi(correction + n), v16) * (load(ws + n) * s));
}

// Fuse dequantization into the residual/statistics pass, avoiding a temporary FP32 row.
// The two vectors hold the same 32 partial sums as pe_add_norm_row; pe_sum32 retains
// its reduction tree, and only the variance accumulation uses FMA.
static inline v16 dequant_vec(const int32_t *c, const int32_t *correction, float s, const float *ws) {
    return __builtin_convertvector(loadi(c) - loadi(correction), v16) * (load(ws) * s);
}

static inline void dequant_norm(const int32_t *c, const int32_t *correction, float s, const float *ws,
                                float *r, const float *weight, const float *bias, float *out, int width, float eps) {
    v16 p0 = {}, p1 = {};
    for (int i = 0; i < width; i += 32) {
        v16 a = load(r + i) + dequant_vec(c + i, correction + i, s, ws + i);
        v16 b = load(r + i + 16) + dequant_vec(c + i + 16, correction + i + 16, s, ws + i + 16);
        store(r + i, a); store(r + i + 16, b);
        p0 += a; p1 += b;
    }
    float p[32];
    store(p, p0); store(p + 16, p1);
    float mean = pe_sum32(p) / width;
    p0 = (v16){}; p1 = (v16){};
    for (int i = 0; i < width; i += 32) {
        v16 a = load(r + i) - mean, b = load(r + i + 16) - mean;
        p0 = _mm512_fmadd_ps(a, a, p0); p1 = _mm512_fmadd_ps(b, b, p1);
    }
    store(p, p0); store(p + 16, p1);
    float rstd = 1.f / sqrtf(pe_sum32(p) / width + eps);
    for (int i = 0; i < width; i += 16) {
        v16 y = (load(r + i) - mean) * rstd * load(weight + i);
        store(out + i, bias ? y + load(bias + i) : y);
    }
}

// ---------------------------------------------------------------- segments
static int run(const PeQ8Model *m, const PeQ8Segment *g, const PeQ8Buffers *b, const float *src, int input,
               int tokens, int workers) {
    const int d = m->hidden, w = m->width, h = m->heads;
    long need = 3L * d > 4L * w ? 3L * d : 4L * w;
    if (tokens < 1 || workers < 1 || h * 64 != d || d % 64 || w % 64 || b->scratch_floats < need
        || (m->block_hidden != 64 && m->block_hidden != 128) || d % m->block_hidden
        || (m->block_width != 64 && m->block_width != 128) || w % m->block_width)
        return -1;
    if (workers > b->scratch_workers) workers = b->scratch_workers;
    uint8_t *ah = (uint8_t *)b->a_hidden, *aw = (uint8_t *)b->a_width;
    float *s = b->scales;
    int32_t *c = b->c;
#pragma omp parallel num_threads(workers) if (workers > 1)
    {
        float *scratch = b->scratch + (size_t)omp_get_thread_num() * b->scratch_floats;
        int qkv = 1;
        if (input) {
#pragma omp for schedule(static)
            for (int r = 0; r < tokens; ++r)
                emit_row(src + (size_t)r * d, d, m->signs_hidden, m->block_hidden, scratch, ah + (size_t)r * d, s + r);
        } else {
            float *y = scratch + d, *tmp = scratch + 2 * d;
            // Token-major context; its rotation is folded into Wv and Wo.
#pragma omp for schedule(static)
            for (int r = 0; r < tokens; ++r) emit_row(src + (size_t)r * d, d, NULL, 0, NULL, ah + (size_t)r * d, s + r);
            gemm(ah, d, g->wo, c, d, tokens);
#pragma omp for schedule(static)
            for (int r = 0; r < tokens; ++r) {
                dequant_norm(c + (size_t)r * d, g->co, s[r], g->so,
                             b->residual + (size_t)r * d, g->mlp_weight, g->mlp_bias, y, d, g->mlp_eps);
                emit_row(y, d, m->signs_hidden, m->block_hidden, tmp, ah + (size_t)r * d, s + r);
            }
            gemm(ah, d, g->wi, c, 2 * w, tokens);
            float *hid = scratch, *act = scratch + 2 * w, *rot = scratch + 3 * w;
#pragma omp for schedule(static)
            for (int r = 0; r < tokens; ++r) {
                dequant_row(c + (size_t)r * 2 * w, g->ci, s[r], g->si, hid, 2 * w);
                pe_geglu_row(hid, act, w);
                emit_row(act, w, m->signs_width, m->block_width, rot, aw + (size_t)r * w, s + r);
            }
            gemm(aw, w, g->wmo, c, d, tokens);
#pragma omp for schedule(static)
            for (int r = 0; r < tokens; ++r) {
                float *out = g->wqkv ? y : b->out + (size_t)r * d;
                dequant_norm(c + (size_t)r * d, g->cmo, s[r], g->smo,
                             b->residual + (size_t)r * d, g->after_weight, g->after_bias, out, d, g->after_eps);
                if (g->wqkv) emit_row(y, d, m->signs_hidden, m->block_hidden, tmp, ah + (size_t)r * d, s + r);
            }
            qkv = g->wqkv != NULL;
        }
        if (qkv) {
            gemm(ah, d, g->wqkv, c, 3 * d, tokens);
            const float *cos = b->cos[g->table], *sin = b->sin[g->table];
#pragma omp for schedule(static)
            for (int r = 0; r < tokens; ++r) {
                dequant_row(c + (size_t)r * 3 * d, g->cqkv, s[r], g->sqkv, scratch, 3 * d);
                pe_rope_row(scratch, cos + (size_t)r * 64, sin + (size_t)r * 64, b->rope, r, tokens, h);
            }
        }
    }
    return 0;
}

// The first layer: FP32 input rows -> rotate + quantize -> QKV -> RoPE.
int pe_q8_input(const PeQ8Model *m, const PeQ8Segment *seg, const PeQ8Buffers *b, const float *x, int tokens,
                int workers) {
    return run(m, seg, b, x, 1, tokens, workers);
}

// Everything after one attention (token-major context [tokens, heads * 64]) up to the next.
int pe_q8_segment(const PeQ8Model *m, const PeQ8Segment *seg, const PeQ8Buffers *b, const float *context,
                  int tokens, int workers) {
    return run(m, seg, b, context, 0, tokens, workers);
}

// Building blocks, exported for tests: u8 rows a[M][K] x packed weights -> c[M][N]; and
// rotate + quantize M rows into u8 rows with their scales.
void pe_q8_gemm(const uint8_t *a, const int8_t *w, int32_t *c, int M, int N, int K) {
#pragma omp parallel num_threads(1)
    gemm(a, K, w, c, N, M);
}

void pe_q8_quantize(const float *x, int M, int K, const float *signs, int block, float *tmp, uint8_t *rows,
                    float *scale) {
    for (int r = 0; r < M; ++r) emit_row(x + (size_t)r * K, K, signs, block, tmp, rows + (size_t)r * K, scale + r);
}

// Pack-time GPTQ: the column loop over columns [i1, i2) of a row-major [n][k] weight copy
// (int8.py). Rows are independent; errors is [n][i2 - i1].
int pe_q8_gptq_block(float *w, const float *hinv, const float *scale, int8_t *q, float *errors, int n, int k, int i1,
                     int i2, int workers) {
    if (!w || !hinv || !scale || !q || !errors || n < 1 || i1 < 0 || i2 <= i1 || i2 > k || workers < 1) return -1;
#pragma omp parallel for num_threads(workers) schedule(static, 16) if (workers > 1)
    for (int r = 0; r < n; ++r)
        pe_gptq_row(w + (size_t)r * k, hinv, scale[r], q + (size_t)r * k, errors + (size_t)r * (i2 - i1), k, i1, i2);
    return 0;
}
