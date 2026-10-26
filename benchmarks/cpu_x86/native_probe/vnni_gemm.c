// Item 4.0 probe: u8 x s8 AVX512-VNNI GEMM chain for ModernBERT-base's 88 projections.
// Weights packed [N/16][K/4][16][4] (one 64-byte vector = 16 output channels x 4 consecutive k), 22 distinct
// layers (110 MB, beyond L3). Activations u8 [M][K] row-major, broadcast 4 bytes at a time. Output int32.
// Threads split the output channels (panels of C 16-channel blocks); one parallel region per GEMM.
// Checks every shape against a scalar reference once. Pure GEMM: no quantize/dequantize.
// Build: gcc -O3 -march=native -fopenmp -DR_=8 -DC_=3 vnni_gemm.c -o vnni_gemm
// Run:   OMP_NUM_THREADS=1 taskset -c 2 ./vnni_gemm 16,32,256      (8t: OMP_NUM_THREADS=8 taskset -c 0-7)
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef R_
#define R_ 8
#endif
#ifndef C_
#define C_ 3
#endif

static void pack(const int8_t *w, int N, int K, int8_t *out) {
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            out[(((size_t)(n / 16) * (K / 4) + k / 4) * 16 + n % 16) * 4 + k % 4] = w[(size_t)n * K + k];
}

// c[r][16x..] = sum_k a[r][k] * w[n][k] for R rows and C 16-channel blocks starting at w (packed).
static inline __attribute__((always_inline)) void tile(const uint8_t *a, int lda, const int8_t *w, int ks,
                                                       int32_t *c, int ldc, const int R, const int C) {
    __m512i acc[R_][C_];
    for (int r = 0; r < R; ++r)
        for (int x = 0; x < C; ++x) acc[r][x] = _mm512_setzero_si512();
    for (int k = 0; k < ks; ++k) {
        __m512i b[C_];
        for (int x = 0; x < C; ++x) b[x] = _mm512_loadu_si512(w + ((size_t)x * ks + k) * 64);
        for (int r = 0; r < R; ++r) {
            int32_t v;
            memcpy(&v, a + (size_t)r * lda + 4 * k, 4);
            __m512i av = _mm512_set1_epi32(v);
            for (int x = 0; x < C; ++x) acc[r][x] = _mm512_dpbusd_epi32(acc[r][x], av, b[x]);
        }
    }
    for (int r = 0; r < R; ++r)
        for (int x = 0; x < C; ++x) _mm512_storeu_si512(c + (size_t)r * ldc + 16 * x, acc[r][x]);
}

static void rows(const uint8_t *a, int K, const int8_t *w, int32_t *c, int N, int M, int C) {
    int ks = K / 4, r = 0;
    for (; r + R_ <= M; r += R_) {
        if (C == C_) tile(a + (size_t)r * K, K, w, ks, c + (size_t)r * N, N, R_, C_);
        else for (int x = 0; x < C; ++x) tile(a + (size_t)r * K, K, w + (size_t)x * ks * 64, ks, c + (size_t)r * N + 16 * x, N, R_, 1);
    }
    for (; r < M; ++r)
        for (int x = 0; x < C; ++x) tile(a + (size_t)r * K, K, w + (size_t)x * ks * 64, ks, c + (size_t)r * N + 16 * x, N, 1, 1);
}

static void gemm(const uint8_t *a, const int8_t *w, int32_t *c, int M, int N, int K, int threads) {
    int blocks = N / 16, panels = (blocks + C_ - 1) / C_;
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(static)
    for (int p = 0; p < panels; ++p) {
        int b0 = p * C_, nb = blocks - b0 < C_ ? blocks - b0 : C_;
        rows(a, K, w + (size_t)b0 * (K / 4) * 64, c + 16 * b0, N, M, nb);
    }
}

static const int SHAPES[4][2] = {{2304, 768}, {768, 768}, {2304, 768}, {768, 1152}};
#define LAYERS 22

static int cmp(const void *x, const void *y) { double a = *(const double *)x, b = *(const double *)y; return (a > b) - (a < b); }

int main(int argc, char **argv) {
    int threads = omp_get_max_threads();
    int8_t *w[LAYERS][4];
    srand(1);
    for (int l = 0; l < LAYERS; ++l)
        for (int s = 0; s < 4; ++s) {
            int N = SHAPES[s][0], K = SHAPES[s][1];
            int8_t *raw = malloc((size_t)N * K);
            for (size_t i = 0; i < (size_t)N * K; ++i) raw[i] = (int8_t)(rand() % 255 - 127);
            w[l][s] = aligned_alloc(64, (size_t)N * K);
            pack(raw, N, K, w[l][s]);
            if (l == 0) {  // correctness against the scalar reference, M = 19 (row block + remainder)
                int M = 19;
                uint8_t *a = malloc((size_t)M * K);
                int32_t *c = malloc((size_t)M * N * 4);
                for (size_t i = 0; i < (size_t)M * K; ++i) a[i] = (uint8_t)(rand() % 256);
                gemm(a, w[l][s], c, M, N, K, threads);
                for (int m = 0; m < M; ++m)
                    for (int n = 0; n < N; ++n) {
                        int32_t ref = 0;
                        for (int k = 0; k < K; ++k) ref += (int32_t)a[(size_t)m * K + k] * raw[(size_t)n * K + k];
                        if (ref != c[(size_t)m * N + n]) { printf("MISMATCH shape %d m %d n %d\n", s, m, n); return 1; }
                    }
                free(a); free(c);
            }
            free(raw);
        }
    double ops_per_row = 0;
    for (int s = 0; s < 4; ++s) ops_per_row += 2.0 * SHAPES[s][0] * SHAPES[s][1] * LAYERS;
    char *list = argc > 1 ? argv[1] : "16,32,256";
    printf("R=%d C=%d threads=%d (88-GEMM chain, median of 21)\n", R_, C_, threads);
    for (char *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
        int M = atoi(tok);
        uint8_t *a = aligned_alloc(64, (size_t)M * 1152 + 64);
        int32_t *c = aligned_alloc(64, (size_t)M * 2304 * 4 + 64);
        memset(a, 7, (size_t)M * 1152);
        double t[21];
        for (int rep = -3; rep < 21; ++rep) {
            double t0 = omp_get_wtime();
            for (int l = 0; l < LAYERS; ++l)
                for (int s = 0; s < 4; ++s) gemm(a, w[l][s], c, M, SHAPES[s][0], SHAPES[s][1], threads);
            if (rep >= 0) t[rep] = omp_get_wtime() - t0;
        }
        qsort(t, 21, sizeof(double), cmp);
        double med = t[10];
        printf("M=%4d  %7.2f ms  %6.0f GOPS  %5.1f%% of %d-core VNNI peak\n", M, med * 1e3, M * ops_per_row / med / 1e9,
               100.0 * M * ops_per_row / med / 1e9 / (1391.0 * threads), threads);
        free(a); free(c);
    }
    return 0;
}
