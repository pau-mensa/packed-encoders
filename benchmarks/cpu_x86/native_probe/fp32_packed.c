// Item 5.2 probe: large-M FP32 GEMM chain with per-call weight panel packing (GotoBLAS/oneDNN style) on the
// original [N, K] weights. For each 64-row weight panel: transpose it into a thread-private [K][64] buffer
// (L2-resident, transient; 16x16 in-register transposes), then 6-row activation tiles broadcast x[m][k] against
// 4 panel vectors per k step (24 named accumulators, 4 + 6 loads per 24 FMAs) and store full output lines.
// KC > 0 blocks K (partial sums accumulate in y). PFN=1 prefetches the next panel (the next K block's, or the next
// panel's first) into L2 during the current one, spread over its tiles, so packing reads L2 instead of DRAM.
// PIPE=1 (full K, panels only) software-pipelines the packing: while panel p computes, panel p+1 is packed into the
// other buffer a few 16x16 transposes after each tile, its lines prefetched into L1 one tile ahead.
// KOUTER=1 loops K blocks outside panels, so one K slice of the activations stays in L2 across all panels.
// Same chain, weights and check as fp32_gemm.c.
// Build: gcc -O3 -march=native -fopenmp -DKC=0 fp32_packed.c -o fp32_packed
// Run:   OMP_NUM_THREADS=1 taskset -c 2 ./fp32_packed 256
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MR 6
#define NR 64
#ifndef KC
#define KC 0
#endif
#ifndef PFN
#define PFN 1
#endif
#ifndef KOUTER
#define KOUTER 0
#endif
#ifndef PIPE
#define PIPE 0
#endif
#ifndef PFN_HINT
#define PFN_HINT _MM_HINT_T1
#endif

// rows[i][k0..k0+16) for 16 rows (stride ld) -> out[k][i] (stride 64), i.e. one 16x16 transpose.
static inline void transpose16(const float *src, size_t ld, float *dst) {
    __m512 r[16], t[16];
    for (int i = 0; i < 16; ++i) r[i] = _mm512_loadu_ps(src + i * ld);
    for (int i = 0; i < 16; i += 2) {
        t[i] = _mm512_unpacklo_ps(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_ps(r[i], r[i + 1]);
    }
    for (int i = 0; i < 16; i += 4) {
        r[i] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i]), _mm512_castps_pd(t[i + 2])));
        r[i + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i]), _mm512_castps_pd(t[i + 2])));
        r[i + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i + 1]), _mm512_castps_pd(t[i + 3])));
        r[i + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i + 1]), _mm512_castps_pd(t[i + 3])));
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) {
            int a = i + 8 * j, b = a + 4;
            t[a] = _mm512_shuffle_f32x4(r[a], r[b], _MM_SHUFFLE(2, 0, 2, 0));
            t[b] = _mm512_shuffle_f32x4(r[a], r[b], _MM_SHUFFLE(3, 1, 3, 1));
        }
    for (int i = 0; i < 8; ++i) {
        r[i] = _mm512_shuffle_f32x4(t[i], t[i + 8], _MM_SHUFFLE(2, 0, 2, 0));
        r[i + 8] = _mm512_shuffle_f32x4(t[i], t[i + 8], _MM_SHUFFLE(3, 1, 3, 1));
    }
    for (int q = 0; q < 16; ++q) _mm512_storeu_ps(dst + (size_t)q * NR, r[q]);  // r[q] is column q
}

#ifndef GMAJOR
#define GMAJOR 0
#endif
static void pack(const float *w, int K, int k0, int kc, float *wp) {  // w: 64 rows, stride K
    if (GMAJOR)  // 16 rows at a time: 16 concurrent row streams instead of 64
        for (int g = 0; g < 4; ++g)
            for (int k = 0; k < kc; k += 16) transpose16(w + (size_t)(16 * g) * K + k0 + k, K, wp + (size_t)k * NR + 16 * g);
    else
        for (int k = 0; k < kc; k += 16)
            for (int g = 0; g < 4; ++g) transpose16(w + (size_t)(16 * g) * K + k0 + k, K, wp + (size_t)k * NR + 16 * g);
}

static inline __attribute__((always_inline)) void kernel(const float *x, int ldx, const float *wp, int kc,
                                                         float *y, int ldy, int accumulate, const int R) {
    const __m512 z = _mm512_setzero_ps();
#define DECL(r) __m512 c##r##0 = z, c##r##1 = z, c##r##2 = z, c##r##3 = z;
    DECL(0) DECL(1) DECL(2) DECL(3) DECL(4) DECL(5)
    for (int k = 0; k < kc; ++k) {
        const float *b = wp + (size_t)k * NR;
        const __m512 b0 = _mm512_loadu_ps(b), b1 = _mm512_loadu_ps(b + 16), b2 = _mm512_loadu_ps(b + 32),
                     b3 = _mm512_loadu_ps(b + 48);
#define ROW(r)                                                     \
        if (R > r) {                                               \
            const __m512 a = _mm512_set1_ps(x[(size_t)r * ldx + k]); \
            c##r##0 = _mm512_fmadd_ps(a, b0, c##r##0);             \
            c##r##1 = _mm512_fmadd_ps(a, b1, c##r##1);             \
            c##r##2 = _mm512_fmadd_ps(a, b2, c##r##2);             \
            c##r##3 = _mm512_fmadd_ps(a, b3, c##r##3);             \
        }
        ROW(0) ROW(1) ROW(2) ROW(3) ROW(4) ROW(5)
    }
#define STORE1(r, v)                                                                                   \
    _mm512_storeu_ps(y + (size_t)r * ldy + 16 * v,                                                     \
                     accumulate ? _mm512_add_ps(_mm512_loadu_ps(y + (size_t)r * ldy + 16 * v), c##r##v) : c##r##v);
#define STORE(r) if (R > r) { STORE1(r, 0) STORE1(r, 1) STORE1(r, 2) STORE1(r, 3) }
    STORE(0) STORE(1) STORE(2) STORE(3) STORE(4) STORE(5)
}
typedef void (*Kern)(const float *, int, const float *, int, float *, int, int);
#define SHAPE(r) static __attribute__((noinline)) void kern_##r(const float *x, int ldx, const float *wp, int kc, \
                                                                float *y, int ldy, int acc) { kernel(x, ldx, wp, kc, y, ldy, acc, r); }
SHAPE(1) SHAPE(2) SHAPE(3) SHAPE(4) SHAPE(5) SHAPE(6)
static const Kern KERNS[MR + 1] = {0, kern_1, kern_2, kern_3, kern_4, kern_5, kern_6};

static float *wpack[64], *wpack2[64];

// transposes [t0, t1) of a panel, group-major (g = t / (K/16), k chunk = t % (K/16))
static void pack_slice(const float *w, int K, float *wp, int t0, int t1) {
    int chunks = K / 16;
    for (int t = t0; t < t1; ++t) {
        int g = t / chunks, k = 16 * (t % chunks);
        transpose16(w + (size_t)(16 * g) * K + k, K, wp + (size_t)k * NR + 16 * g);
    }
}
static void prefetch_slice(const float *w, int K, int t0, int t1) {
    int chunks = K / 16;
    for (int t = t0; t < t1; ++t) {
        int g = t / chunks, k = 16 * (t % chunks);
        for (int i = 0; i < 16; ++i) _mm_prefetch((const char *)(w + (size_t)(16 * g + i) * K + k), _MM_HINT_T0);
    }
}

static void gemm(const float *x, const float *w, float *y, int M, int N, int K, int threads) {
    int panels = N / NR, kc = KC > 0 && KC < K ? KC : K;
#pragma omp parallel num_threads(threads) if (threads > 1)
    {
        float *wp = wpack[omp_get_thread_num()];
        if (PIPE) {
            int nt = omp_get_num_threads(), id = omp_get_thread_num();
            int p0 = (int)((long)panels * id / nt), p1 = (int)((long)panels * (id + 1) / nt);
            float *buf[2] = {wp, wpack2[id]};
            int total = 4 * (K / 16), tiles = (M + MR - 1) / MR, per = (total + tiles - 1) / tiles;
            if (p0 < p1) pack_slice(w + (size_t)p0 * NR * K, K, buf[0], 0, total);
            for (int p = p0; p < p1; ++p) {
                const float *nw = w + (size_t)(p + 1) * NR * K;
                float *cur = buf[(p - p0) & 1], *nxt = buf[(p - p0 + 1) & 1];
                int more = p + 1 < p1;
                if (more) prefetch_slice(nw, K, 0, per < total ? per : total);
                for (int m = 0, j = 0; m < M; m += MR, ++j) {
                    int a = j * per, b = a + per < total ? a + per : total, c = b + per < total ? b + per : total;
                    if (more && b < total) prefetch_slice(nw, K, b, c);
                    KERNS[M - m < MR ? M - m : MR](x + (size_t)m * K, K, cur, K, y + (size_t)m * N + p * NR, N, 0);
                    if (more && a < total) pack_slice(nw, K, nxt, a, b);
                }
            }
        } else if (KOUTER) {
            for (int k0 = 0; k0 < K; k0 += kc) {
                int n = K - k0 < kc ? K - k0 : kc;
#pragma omp for schedule(static)
                for (int p = 0; p < panels; ++p) {
                    pack(w + (size_t)p * NR * K, K, k0, n, wp);
                    int lines = p + 1 < panels && PFN ? NR * (n / 16) : 0, tiles = (M + MR - 1) / MR, per = (lines + tiles - 1) / tiles;
                    const float *next = w + (size_t)(p + 1) * NR * K + k0;
                    for (int m = 0, j = 0; m < M; m += MR, ++j) {
                        for (int l = j * per; l < lines && l < (j + 1) * per; ++l)
                            _mm_prefetch((const char *)(next + (size_t)(l / (n / 16)) * K + 16 * (l % (n / 16))), PFN_HINT);
                        KERNS[M - m < MR ? M - m : MR](x + (size_t)m * K + k0, K, wp, n, y + (size_t)m * N + p * NR, N, k0 > 0);
                    }
                }
            }
        } else
#pragma omp for schedule(static)
        for (int p = 0; p < panels; ++p)
            for (int k0 = 0; k0 < K; k0 += kc) {
                int n = K - k0 < kc ? K - k0 : kc;
                pack(w + (size_t)p * NR * K, K, k0, n, wp);
                // the next (panel, K block) this thread packs: lines (row, k line) spread over the tiles
                int np = k0 + kc < K ? p : p + 1, nk = k0 + kc < K ? k0 + kc : 0, nn = K - nk < kc ? K - nk : kc;
                int lines = np < panels && PFN ? NR * (nn / 16) : 0, tiles = (M + MR - 1) / MR, per = (lines + tiles - 1) / tiles;
                const float *next = w + (size_t)np * NR * K + nk;
                for (int m = 0, j = 0; m < M; m += MR, ++j) {
                    for (int l = j * per; l < lines && l < (j + 1) * per; ++l)
                        _mm_prefetch((const char *)(next + (size_t)(l / (nn / 16)) * K + 16 * (l % (nn / 16))), PFN_HINT);
                    KERNS[M - m < MR ? M - m : MR](x + (size_t)m * K + k0, K, wp, n, y + (size_t)m * N + p * NR, N, k0 > 0);
                }
            }
    }
}

static const int SHAPES[4][2] = {{2304, 768}, {768, 768}, {2304, 768}, {768, 1152}};
#define LAYERS 22
static int cmp(const void *x, const void *y) { double a = *(const double *)x, b = *(const double *)y; return (a > b) - (a < b); }
static float frand(void) { return (float)rand() / RAND_MAX * 2.f - 1.f; }

int main(int argc, char **argv) {
    int threads = omp_get_max_threads();
    for (int t = 0; t < threads; ++t) wpack[t] = aligned_alloc(64, (size_t)1152 * NR * 4), wpack2[t] = aligned_alloc(64, (size_t)1152 * NR * 4);
    float *w[LAYERS][4];
    srand(1);
    for (int l = 0; l < LAYERS; ++l)
        for (int s = 0; s < 4; ++s) {
            int N = SHAPES[s][0], K = SHAPES[s][1];
            w[l][s] = aligned_alloc(64, (size_t)N * K * 4);
            for (size_t i = 0; i < (size_t)N * K; ++i) w[l][s][i] = frand() * 0.05f;
            if (l == 0) {
                int M = 19;
                float *x = malloc((size_t)M * K * 4), *y = malloc((size_t)M * N * 4);
                for (size_t i = 0; i < (size_t)M * K; ++i) x[i] = frand();
                gemm(x, w[l][s], y, M, N, K, threads);
                double err = 0, ref2 = 0;
                for (int m = 0; m < M; ++m)
                    for (int n = 0; n < N; ++n) {
                        double ref = 0;
                        for (int k = 0; k < K; ++k) ref += (double)x[(size_t)m * K + k] * w[l][s][(size_t)n * K + k];
                        err += (ref - y[(size_t)m * N + n]) * (ref - y[(size_t)m * N + n]);
                        ref2 += ref * ref;
                    }
                if (sqrt(err / ref2) > 1e-6) { printf("MISMATCH shape %d relL2 %.2e\n", s, sqrt(err / ref2)); return 1; }
                free(x); free(y);
            }
        }
    double flops_per_row = 0;
    for (int s = 0; s < 4; ++s) flops_per_row += 2.0 * SHAPES[s][0] * SHAPES[s][1] * LAYERS;
    char *list = argc > 1 ? argv[1] : "256";
    printf("packed MR=%d NR=%d KC=%d PFN=%d KOUTER=%d PIPE=%d threads=%d (88-GEMM chain, median of 21)\n", MR, NR, KC, PFN, KOUTER, PIPE, threads);
    for (char *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
        int M = atoi(tok);
        float *x = aligned_alloc(64, (size_t)M * 1152 * 4 + 64), *y = aligned_alloc(64, (size_t)M * 2304 * 4 + 64);
        for (size_t i = 0; i < (size_t)M * 1152; ++i) x[i] = frand();
        double t[21];
        for (int rep = -3; rep < 21; ++rep) {
            double t0 = omp_get_wtime();
            for (int l = 0; l < LAYERS; ++l)
                for (int s = 0; s < 4; ++s) gemm(x, w[l][s], y, M, SHAPES[s][0], SHAPES[s][1], threads);
            if (rep >= 0) t[rep] = omp_get_wtime() - t0;
        }
        qsort(t, 21, sizeof(double), cmp);
        double med = t[10];
        printf("M=%4d  %7.2f ms  %6.0f GFLOPS  %5.1f%% of %d-core FP32 peak\n", M, med * 1e3,
               M * flops_per_row / med / 1e9, 100.0 * M * flops_per_row / med / 1e9 / (363.0 * threads), threads);
        free(x); free(y);
    }
    return 0;
}
