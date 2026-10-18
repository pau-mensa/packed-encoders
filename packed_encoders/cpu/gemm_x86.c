// FP32 projections for the x86 ModernBERT CPU engine: y = x w^T (+ b) on the original row-major [N, K]
// weights, with no reorder and no copy. x is [M, K] row-major.
//
// Dot-product (NT) tiles have up to 24 named accumulators (arrays make GCC spill), reduced
// horizontally at the end. The caller chooses a tile using the CPU cache size or pack-time
// measurements. Threads split the weight blocks.
// The next block's weights are prefetched during the current block, spread over its tiles.
// Specializing K and the common prefetch counts removes address arithmetic and nested loops.
// Four k steps are unrolled without changing any accumulator's addition order.
//
// Every output is the same function of its activation row and weight row, whatever M, the tile it lands in
// or the thread count: k steps accumulate in order and every row is reduced by one tree. A token's result
// therefore does not depend on the batch it shares.
#include <immintrin.h>
#include <stddef.h>

// The other translation units also serve AVX2 CPUs. Omit only this entry point
// when the native compiler target lacks the intrinsics used below.
#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512VL__)

#define ROWS 4
#define COLS 8

static inline __attribute__((always_inline)) __m512 pair(__m512 a, __m512 b) {
    return _mm512_add_ps(_mm512_unpacklo_ps(a, b), _mm512_unpackhi_ps(a, b));
}

// Each 128-bit group g of the result holds a, b, c, d's partial sums over that group's 4 lanes.
static inline __attribute__((always_inline)) __m512 four(__m512 a, __m512 b, __m512 c, __m512 d) {
    __m512d ab = _mm512_castps_pd(pair(a, b)), cd = _mm512_castps_pd(pair(c, d));
    return _mm512_add_ps(_mm512_castpd_ps(_mm512_unpacklo_pd(ab, cd)), _mm512_castpd_ps(_mm512_unpackhi_pd(ab, cd)));
}

// The 8 horizontal sums, each (g0 + g1) + (g2 + g3) of its group partials.
static inline __attribute__((always_inline)) __m256 reduce8(__m512 a, __m512 b, __m512 c, __m512 d, __m512 e,
                                                            __m512 f, __m512 g, __m512 h) {
    __m512 p = four(a, b, c, d), q = four(e, f, g, h);
    __m512 s = _mm512_add_ps(_mm512_shuffle_f32x4(p, q, _MM_SHUFFLE(2, 0, 2, 0)),
                             _mm512_shuffle_f32x4(p, q, _MM_SHUFFLE(3, 1, 3, 1)));  // p01 p23 q01 q23
    s = _mm512_shuffle_f32x4(s, s, _MM_SHUFFLE(3, 1, 2, 0));                         // p01 q01 p23 q23
    return _mm256_add_ps(_mm512_castps512_ps256(s), _mm512_extractf32x8_ps(s, 1));
}

// y[r][c] = sum_k x[r][k] * w[c][k] (+ bias[c]) for R <= ROWS rows and C <= COLS weight rows (both K-strided).
// pf: npf rows of the next weight block, prefetched one line per k step.
static inline __attribute__((always_inline)) void tile(const float *x, const float *w, int K, const float *bias,
                                                       float *y, int ldy, const float *const *pf, int npf,
                                                       const int R, const int C) {
    const __m512 z = _mm512_setzero_ps();
#define DECL(r) __m512 a##r##0 = z, a##r##1 = z, a##r##2 = z, a##r##3 = z, a##r##4 = z, a##r##5 = z,            \
                       a##r##6 = z, a##r##7 = z;
    DECL(0) DECL(1) DECL(2) DECL(3)
#define COL(c, LOAD)                                                                                            \
    if (C > c) {                                                                                                \
        const __m512 v = LOAD(w + (size_t)c * K + k);                                                           \
        a0##c = _mm512_fmadd_ps(x0, v, a0##c);                                                                  \
        if (R > 1) a1##c = _mm512_fmadd_ps(x1, v, a1##c);                                                       \
        if (R > 2) a2##c = _mm512_fmadd_ps(x2, v, a2##c);                                                       \
        if (R > 3) a3##c = _mm512_fmadd_ps(x3, v, a3##c);                                                       \
    }
#define STEP(LOAD)                                                                                              \
    {                                                                                                           \
        const __m512 x0 = LOAD(x + k), x1 = R > 1 ? LOAD(x + K + k) : z,                                        \
                     x2 = R > 2 ? LOAD(x + 2 * (size_t)K + k) : z,                                              \
                     x3 = R > 3 ? LOAD(x + 3 * (size_t)K + k) : z;                                              \
        COL(0, LOAD) COL(1, LOAD) COL(2, LOAD) COL(3, LOAD) COL(4, LOAD) COL(5, LOAD)                           \
        COL(6, LOAD) COL(7, LOAD)                                                                               \
    }
#define FULL(p) _mm512_loadu_ps(p)
#define PART(p) _mm512_maskz_loadu_ps(tail, p)
    int k = 0;
#pragma GCC unroll 4
    for (; k + 16 <= K; k += 16) {
        for (int p = 0; p < npf; ++p) _mm_prefetch((const char *)(pf[p] + k), _MM_HINT_T0);
        STEP(FULL)
    }
    if (k < K) {
        const __mmask16 tail = (__mmask16)((1u << (K - k)) - 1);
        STEP(PART)
    }
    const __mmask8 cols = (__mmask8)((1u << C) - 1);
    const __m256 b = bias ? _mm256_maskz_loadu_ps(cols, bias) : _mm256_setzero_ps();
#define OUT(r)                                                                                                  \
    if (R > r) {                                                                                                \
        __m256 s = reduce8(a##r##0, C > 1 ? a##r##1 : z, C > 2 ? a##r##2 : z, C > 3 ? a##r##3 : z,              \
                           C > 4 ? a##r##4 : z, C > 5 ? a##r##5 : z, C > 6 ? a##r##6 : z, C > 7 ? a##r##7 : z); \
        if (bias) s = _mm256_add_ps(s, b);                                                                      \
        _mm256_mask_storeu_ps(y + (size_t)r * ldy, cols, s);                                                    \
    }
    OUT(0) OUT(1) OUT(2) OUT(3)
#undef DECL
#undef COL
#undef STEP
#undef FULL
#undef PART
#undef OUT
}

typedef void (*Tile)(const float *, const float *, int, const float *, float *, int, const float *const *, int);
// The two ModernBERT-base strides and 0/1/2 prefetch streams cover the hot tiles. Keep the
// generic path for other models and for very small M (which prefetches more rows per tile).
#define FIXED(k, r, c)                                                                                          \
    do {                                                                                                        \
        if (npf == 0) tile(x, w, k, bias, y, ldy, pf, 0, r, c);                                                 \
        else if (npf == 1) tile(x, w, k, bias, y, ldy, pf, 1, r, c);                                            \
        else if (npf == 2) tile(x, w, k, bias, y, ldy, pf, 2, r, c);                                            \
        else tile(x, w, k, bias, y, ldy, pf, npf, r, c);                                                        \
    } while (0)
#define SHAPE(r, c)                                                                                             \
    static __attribute__((noinline)) void tile_##r##_##c(const float *x, const float *w, int K,                 \
                                                         const float *bias, float *y, int ldy,                  \
                                                         const float *const *pf, int npf) {                     \
        if ((c == 4 || c == 6 || c == 8) && K == 768) FIXED(768, r, c);                                        \
        else if ((c == 4 || c == 6 || c == 8) && K == 1152) FIXED(1152, r, c);                                 \
        else tile(x, w, K, bias, y, ldy, pf, npf, r, c);                                                        \
    }
#define SIX(r) SHAPE(r, 1) SHAPE(r, 2) SHAPE(r, 3) SHAPE(r, 4) SHAPE(r, 5) SHAPE(r, 6)
#define SHAPES(r) SIX(r) SHAPE(r, 7) SHAPE(r, 8)
SHAPES(1) SHAPES(2) SHAPES(3) SIX(4)
#define ROW(r) {tile_##r##_1, tile_##r##_2, tile_##r##_3, tile_##r##_4, tile_##r##_5, tile_##r##_6, tile_##r##_7, \
                tile_##r##_8}
static const Tile TILES[ROWS][COLS] = {ROW(1), ROW(2), ROW(3),
    {tile_4_1, tile_4_2, tile_4_3, tile_4_4, tile_4_5, tile_4_6, NULL, NULL}};

// All M rows against a weight block, prefetching the rows from `next` on.
static inline __attribute__((always_inline)) void block(const float *x, const float *w, const float *bias,
                  float *y, int M, int N, int K, int n0, int next, int rows, int width) {
    int cols = N - n0 < width ? N - n0 : width;
    int ahead = next >= N ? 0 : (N - next < width ? N - next : width);
    int tiles = (M + rows - 1) / rows;
    for (int j = 0, m = 0; m < M; ++j, m += rows) {
        const float *pf[COLS];
        int npf = 0;
        for (int c = j; c < ahead; c += tiles) pf[npf++] = w + (size_t)(next + c) * K;
        TILES[(M - m < rows ? M - m : rows) - 1][cols - 1](x + (size_t)m * K, w + (size_t)n0 * K, K,
                                                         bias ? bias + n0 : NULL, y + (size_t)m * N + n0, N, pf, npf);
    }
}

int pe_linear_tile(const float *x, const float *w, const float *bias, float *y, int M, int N, int K,
                   int workers, int rows, int width) {
    if (M < 0 || N <= 0 || K <= 0 || workers <= 0 || rows < 1 || rows > ROWS || width < 1 ||
        width > COLS || rows * width > 24) return 1;
    int blocks = (N + width - 1) / width;
    // Dispatch once per GEMM, keeping row/column strides constant in the hot loop.
    // In particular, computing the prefetch spread must not divide by a runtime
    // tile height separately for every weight block.
#define RUN(r, c)                                                                                               \
    do {                                                                                                        \
        _Pragma("omp parallel for num_threads(workers) if (workers > 1) schedule(static)")                       \
        for (int b = 0; b < blocks; ++b) block(x, w, bias, y, M, N, K, b * (c), (b + 1) * (c), r, c);             \
    } while (0)
    if (rows == 3 && width == 8) RUN(3, 8);
    else if (rows == 4 && width == 6) RUN(4, 6);
    else if (rows == 3 && width == 6) RUN(3, 6);
    else if (rows == 3 && width == 4) RUN(3, 4);
    else if (rows == 2 && width == 4) RUN(2, 4);
    else RUN(rows, width);
#undef RUN
    return 0;
}

// The previous dispatch remains available for benchmark ablations.
int pe_linear(const float *x, const float *w, const float *bias, float *y, int M, int N, int K, int workers) {
    return pe_linear_tile(x, w, bias, y, M, N, K, workers, K == 1152 ? 4 : 3, K == 1152 ? 6 : 8);
}

#endif
