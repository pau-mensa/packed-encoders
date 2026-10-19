// FP32 ModernBERT local attention for x86 CPUs: pe_local_attention of local_attention.c without
// Accelerate. Q/K/V: contiguous [heads, total_tokens, 64]; output the same, or token-major
// [total_tokens, heads * 64] (the next projection's input layout). Lengths delimit sequences.
// Queries attend keys within `half` positions in their sequence; a half-width of at least the
// longest sequence gives exact global attention. No fast-math; built with contraction on (unlike
// rows.c), so the dot products accumulate with FMAs. OpenMP threads it; `pool` is ignored.
// GCC vector extensions: one AVX-512 register per v16, or two AVX2 halves on older CPUs.
// Build: gcc -O3 -march=native -ffp-contract=fast -fno-trapping-math -fno-math-errno -fopenmp -c local_attention_x86.c
#include <math.h>
#include <omp.h>
#include <stddef.h>

typedef float v16 __attribute__((vector_size(64)));
typedef float v16u __attribute__((vector_size(64), aligned(4), may_alias));
typedef int v16i __attribute__((vector_size(64)));

static inline v16 load(const float *p) { return *(const v16u *)p; }
static inline void store(float *p, v16 x) { *(v16u *)p = x; }
static inline v16 select(v16i mask, v16 a, v16 b) { return (v16)(((v16i)a & mask) | ((v16i)b & ~mask)); }
static const v16i iota = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

// exp(x) for x <= 0, as Cephes expf (about 1 ulp); below -87.3 (and -inf) gives 0.
static inline v16 vexp(v16 x) {
    v16i under = x < -87.33654f;
    x = select(under, (v16){} - 87.33654f, x);
    v16 n = x * 1.44269504088896341f + 12582912.f;  // round to nearest: 1.5 * 2^23
    n -= 12582912.f;
    v16 r = x - n * 0.693359375f - n * -2.12194440e-4f;
    v16 p = 1.9875691500e-4f * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.f;
    v16 scale = (v16)((__builtin_convertvector(n, v16i) + 127) << 23);
    return select(under, (v16){}, p * scale);
}

// Scores of R query rows (stride 64) against NV 16-key vectors of the transposed window
// (row stride ld), scaled by 1/sqrt(64). Constant R/NV after inlining keep acc in registers.
static inline __attribute__((always_inline)) void qk(const float *q, const float *kt, float *s, int ld,
                                                     const int R, const int NV) {
    v16 acc[4][4];
    for (int r = 0; r < R; ++r)
        for (int x = 0; x < NV; ++x) acc[r][x] = (v16){};
    for (int d = 0; d < 64; ++d) {
        v16 k[4];
        for (int x = 0; x < NV; ++x) k[x] = load(kt + (size_t)d * ld + 16 * x);
        for (int r = 0; r < R; ++r) {
            float qd = q[r * 64 + d];
            for (int x = 0; x < NV; ++x) acc[r][x] += qd * k[x];
        }
    }
    for (int r = 0; r < R; ++r)
        for (int x = 0; x < NV; ++x) store(s + (size_t)r * ld + 16 * x, acc[r][x] * 0.125f);
}

static void qk_block(const float *q, const float *kt, float *s, int ld, int R, int lo, int hi) {
    for (int c = lo; c < hi; c += 64) {
        int nv = (hi - c) / 16 < 4 ? (hi - c) / 16 : 4;
#define QK(r, n) case (r - 1) * 4 + n - 1: qk(q, kt + c, s + c, ld, r, n); break;
        switch ((R - 1) * 4 + nv - 1) {
            QK(1, 1) QK(1, 2) QK(1, 3) QK(1, 4) QK(2, 1) QK(2, 2) QK(2, 3) QK(2, 4)
            QK(3, 1) QK(3, 2) QK(3, 3) QK(3, 4) QK(4, 1) QK(4, 2) QK(4, 3) QK(4, 4)
        }
#undef QK
    }
}

// Exponentiated, unnormalized softmax of one score row over its window [first, last), zero
// elsewhere in [lo, hi) (16-aligned); returns 1 / sum.
static inline __attribute__((always_inline)) float softmax_row(float *s, int first, int last, int lo, int hi) {
    v16 m = (v16){} - INFINITY;
    for (int c = lo; c < hi; c += 16) {
        v16i valid = (iota + c >= first) & (iota + c < last);
        v16 x = select(valid, load(s + c), m);
        m = select(x > m, x, m);
    }
    float maximum = m[0];
    for (int i = 1; i < 16; ++i) maximum = m[i] > maximum ? m[i] : maximum;
    v16 sum = (v16){};
    for (int c = lo; c < hi; c += 16) {
        v16i valid = (iota + c >= first) & (iota + c < last);
        v16 e = select(valid, vexp(load(s + c) - maximum), (v16){});
        store(s + c, e);
        sum += e;
    }
    float total = 0.f;
    for (int i = 0; i < 16; ++i) total += sum[i];
    return 1.f / total;
}

// Output rows (stride ldo) = (P V) * inverse sums for R rows over window columns [lo, hi) of V (stride 64).
static inline __attribute__((always_inline)) void pv(const float *p, int ld, const float *v, int lo, int hi,
                                                     const float *inverse, float *out, int ldo, const int R) {
    // Named accumulators avoid GCC materializing the array on the stack before scaling/storing.
#define DECL(r) v16 a##r##0 = {}, a##r##1 = {}, a##r##2 = {}, a##r##3 = {};
    DECL(0) DECL(1) DECL(2) DECL(3)
#undef DECL
    for (int j = lo; j < hi; ++j) {
        const float *row = v + (size_t)j * 64;
        v16 v0 = load(row), v1 = load(row + 16), v2 = load(row + 32), v3 = load(row + 48);
#define STEP(r) if (R > r) { \
        float pj = p[(size_t)r * ld + j]; \
        a##r##0 += pj * v0; a##r##1 += pj * v1; \
        a##r##2 += pj * v2; a##r##3 += pj * v3; \
    }
        STEP(0) STEP(1) STEP(2) STEP(3)
#undef STEP
    }
#define OUT(r) if (R > r) { \
    float *o = out + (size_t)r * ldo; \
    store(o, a##r##0 * inverse[r]); store(o + 16, a##r##1 * inverse[r]); \
    store(o + 32, a##r##2 * inverse[r]); store(o + 48, a##r##3 * inverse[r]); \
}
    OUT(0) OUT(1) OUT(2) OUT(3)
#undef OUT
}

static void pv_block(const float *p, int ld, const float *v, int lo, int hi, const float *inverse, float *out,
                     int ldo, int R) {
    switch (R) {
        case 1: pv(p, ld, v, lo, hi, inverse, out, ldo, 1); break;
        case 2: pv(p, ld, v, lo, hi, inverse, out, ldo, 2); break;
        case 3: pv(p, ld, v, lo, hi, inverse, out, ldo, 3); break;
        default: pv(p, ld, v, lo, hi, inverse, out, ldo, 4); break;
    }
}

// One query tile [start, start + rows) of one head and sequence (q/k/v/out: its first token;
// output rows ldo apart).
static void tile_attention(const float *q, const float *k, const float *v, float *out, int ldo, float *kt,
                           float *scores, int length, int start, int rows, int half) {
    int lo = start > half ? start - half : 0;
    int hi = start + rows + half < length ? start + rows + half : length;
    int cols = hi - lo, ld = (cols + 15) / 16 * 16;
    for (int j = 0; j < cols; ++j)  // window keys, transposed: kt[d][j] = k[lo + j][d]
        for (int d = 0; d < 64; ++d) kt[(size_t)d * ld + j] = k[(size_t)(lo + j) * 64 + d];
    for (int j = cols; j < ld; ++j)
        for (int d = 0; d < 64; ++d) kt[(size_t)d * ld + j] = 0.f;
    for (int r0 = 0; r0 < rows; r0 += 4) {
        int R = rows - r0 < 4 ? rows - r0 : 4;
        // Windows are monotonic in the row: the block needs [first of its first row, last of its last).
        int first[4], last[4];
        for (int r = 0; r < R; ++r) {
            int position = start + r0 + r - lo;
            first[r] = position > half ? position - half : 0;
            last[r] = position + half + 1 < cols ? position + half + 1 : cols;
        }
        int c_lo = first[0] / 16 * 16, c_hi = (last[R - 1] + 15) / 16 * 16;
        float *s = scores + (size_t)r0 * ld;
        qk_block(q + (size_t)(start + r0) * 64, kt, s, ld, R, c_lo, c_hi);
        float inverse[4];
        for (int r = 0; r < R; ++r) inverse[r] = softmax_row(s + (size_t)r * ld, first[r], last[r], c_lo, c_hi);
        pv_block(s, ld, v + (size_t)lo * 64, first[0], last[R - 1], inverse, out + (size_t)(start + r0) * ldo, ldo, R);
    }
}

// Full-window 16/32-token sequences: constant strides and no row tails or masks.
// Keep the same dot products, softmax reduction order and output scaling as the general path.
static inline __attribute__((always_inline)) void short_attention(
        const float *q, const float *k, const float *v, float *out, int ldo,
        float *kt, float *scores, const int length) {
    typedef float v4 __attribute__((vector_size(16)));
    typedef float v4u __attribute__((vector_size(16), aligned(4), may_alias));
    typedef int v4i __attribute__((vector_size(16)));
    // Transpose 4x4 blocks with vector loads/stores; the general path uses scalar stores.
    for (int j = 0; j < length; j += 4) {
        for (int d = 0; d < 64; d += 4) {
            v4 a = *(const v4u *)(k + j * 64 + d), b = *(const v4u *)(k + (j + 1) * 64 + d);
            v4 c = *(const v4u *)(k + (j + 2) * 64 + d), e = *(const v4u *)(k + (j + 3) * 64 + d);
            v4 ab0 = __builtin_shuffle(a, b, (v4i){0, 4, 1, 5});
            v4 ab1 = __builtin_shuffle(a, b, (v4i){2, 6, 3, 7});
            v4 ce0 = __builtin_shuffle(c, e, (v4i){0, 4, 1, 5});
            v4 ce1 = __builtin_shuffle(c, e, (v4i){2, 6, 3, 7});
            *(v4u *)(kt + d * length + j) = __builtin_shuffle(ab0, ce0, (v4i){0, 1, 4, 5});
            *(v4u *)(kt + (d + 1) * length + j) = __builtin_shuffle(ab0, ce0, (v4i){2, 3, 6, 7});
            *(v4u *)(kt + (d + 2) * length + j) = __builtin_shuffle(ab1, ce1, (v4i){0, 1, 4, 5});
            *(v4u *)(kt + (d + 3) * length + j) = __builtin_shuffle(ab1, ce1, (v4i){2, 3, 6, 7});
        }
    }
    for (int r0 = 0; r0 < length; r0 += 4) {
        float *s = scores + r0 * length;
        qk(q + r0 * 64, kt, s, length, 4, length / 16);
        float inverse[4];
        for (int r = 0; r < 4; ++r)
            inverse[r] = softmax_row(s + r * length, 0, length, 0, length);
        pv(s, length, v, 0, length, inverse, out + (size_t)r0 * ldo, ldo, 4);
    }
}

int pe_local_attention(const float *q, const float *k, const float *v, float *out, const int *lengths, int tokens,
                       int heads, int sequences, int tile, int half, int token_major, int workers, void *pool,
                       float *scratch, size_t scratch_floats) {
    (void)pool;
    if (tokens < 1 || heads < 1 || sequences < 1 || tile < 1 || tile > 4096 || half < 0 || workers < 1)
        return -1;
    long total = 0;
    int longest = 0;
    for (int i = 0; i < sequences; ++i) {
        if (lengths[i] < 1) return -1;
        total += lengths[i];
        if (lengths[i] > longest) longest = lengths[i];
    }
    if (total != tokens) return -1;
    if (half > longest) half = longest;  // already global; avoids index overflow
    // Per worker: the transposed window [64][ld] and the scores [tile][ld], ld padded to 16 keys.
    int columns = tile + 2 * half < longest ? tile + 2 * half : longest;
    size_t ld = (size_t)(columns + 15) / 16 * 16, per_worker = (64 + (size_t)tile) * ld;
    if (workers > heads * sequences) workers = heads * sequences;
    if (!scratch || scratch_floats < (size_t)workers * per_worker) return -2;
    int offsets[sequences], ldo = token_major ? heads * 64 : 64;
    for (int i = 0, offset = 0; i < sequences; offset += lengths[i++]) offsets[i] = offset;
#pragma omp parallel for num_threads(workers) if (workers > 1) schedule(dynamic, 1)
    for (int task = 0; task < heads * sequences; ++task) {
        int head = task / sequences, sequence = task % sequences, length = lengths[sequence];
        size_t base = ((size_t)head * tokens + offsets[sequence]) * 64;
        float *o = out + (token_major ? ((size_t)offsets[sequence] * heads + head) * 64 : base);
        float *kt = scratch + (size_t)omp_get_thread_num() * per_worker, *scores = kt + 64 * ld;
        if (tile >= length && half >= length - 1) {
            if (length == 16) {
                short_attention(q + base, k + base, v + base, o, ldo, kt, scores, 16);
                continue;
            }
            if (length == 32) {
                short_attention(q + base, k + base, v + base, o, ldo, kt, scores, 32);
                continue;
            }
        }
        for (int start = 0; start < length; start += tile)
            tile_attention(q + base, k + base, v + base, o, ldo, kt, scores, length, start,
                           length - start < tile ? length - start : tile, half);
    }
    return 0;
}
