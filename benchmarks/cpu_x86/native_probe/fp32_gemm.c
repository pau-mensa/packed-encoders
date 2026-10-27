// Item 5.0 probe: FP32 NT GEMM chain on the original [N, K] weights (no reorder, no copy) for ModernBERT-base's
// 88 projections, 22 distinct layers (441 MB, beyond L3). y[m][n] = sum_k x[m][k] * w[n][k], tiles of R
// activation rows x C weight rows from fp32_tiles.h (named accumulators, horizontal reduce at the end).
// Loop: M chunks of MC rows (0 = all) outside, C-row weight blocks (split across threads) inside, R-row tiles
// innermost, so a weight block stays in L1 across the chunk. PF=1 prefetches the next weight block's rows
// during the current one, spread over its tiles; PFX=1 also the next tile's activation rows,
// PFX>=16 this tile's own rows PFX floats ahead. XPACK=1 first copies each R-row tile's activations
// interleaved by k ([K/16][R][16], one sequential stream per tile; the copy is timed). Checks every shape against a double reference once.
// Build: python gen_fp32_tiles.py > fp32_tiles.h
//        gcc -O3 -march=native -fopenmp -DR_=4 -DC_=6 -DMC=0 -DPF=1 fp32_gemm.c -o fp32_gemm
// Run:   OMP_NUM_THREADS=1 taskset -c 2 ./fp32_gemm 16,32,256      (8t: OMP_NUM_THREADS=8 taskset -c 0-7)
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fp32_tiles.h"

#ifndef R_
#define R_ 4
#endif
#ifndef C_
#define C_ 6
#endif
#ifndef MC
#define MC 0
#endif
#ifndef PF
#define PF 1
#endif
#ifndef PFX
#define PFX 0
#endif
#ifndef XPACK
#define XPACK 0
#endif

static void block(const float *x, const float *w, float *y, int M, int N, int K, int n0, int nb, int m0, int m1,
                  int next, const float *xp) {
    // rows [m0, m1) of weight block n0 (nb rows); `next` is the next block's first row (or -1)
    int tiles = (m1 - m0 + R_ - 1) / R_, nn = next < 0 ? 0 : (N - next < C_ ? N - next : C_);
    for (int j = 0, m = m0; m < m1; ++j, m += R_) {
        const float *pf[C_ + R_];
        int npf = 0, r = m1 - m < R_ ? m1 - m : R_;
        if (PF)
            for (int c = j; c < nn; c += tiles) pf[npf++] = w + (size_t)(next + c) * K;
        if (PFX == 1)  // the next tile's activation rows (wrapping to the chunk start for the next block)
            for (int i = 0, mm = m + R_ < m1 ? m + R_ : m0; i < R_ && mm + i < m1; ++i) pf[npf++] = x + (size_t)(mm + i) * K;
        if (PFX >= 16)  // this tile's own activation rows, PFX floats ahead
            for (int i = 0; i < r; ++i) pf[npf++] = x + (size_t)(m + i) * K + PFX;
        if (xp)  // tile-interleaved copy: rows m.. at xp + m * K, laid out [K/16][r][16]
            TILES[r][nb](xp + (size_t)m * K, 16, r, w + (size_t)n0 * K, K, K, y + (size_t)m * N + n0, N, pf, npf);
        else
            TILES[r][nb](x + (size_t)m * K, K, 1, w + (size_t)n0 * K, K, K, y + (size_t)m * N + n0, N, pf, npf);
    }
}

static float *xpack;

static void gemm(const float *x, const float *w, float *y, int M, int N, int K, int threads) {
    int blocks = (N + C_ - 1) / C_, mc = MC > 0 && MC < M ? MC : M;
    const float *xp = XPACK ? xpack : NULL;
#pragma omp parallel num_threads(threads) if (threads > 1)
    {
    if (XPACK) {
#pragma omp for schedule(static)
        for (int m = 0; m < M; m += R_) {
            int r = M - m < R_ ? M - m : R_;
            for (int k = 0; k < K; k += 16)
                for (int i = 0; i < r; ++i)
                    _mm512_storeu_ps(xpack + (size_t)m * K + (size_t)k * r + 16 * i, _mm512_loadu_ps(x + (size_t)(m + i) * K + k));
        }
    }
    for (int m0 = 0; m0 < M; m0 += mc) {
        int m1 = m0 + mc < M ? m0 + mc : M;
#pragma omp for schedule(static) nowait
        for (int b = 0; b < blocks; ++b) {
            int n0 = b * C_, nb = N - n0 < C_ ? N - n0 : C_;
            block(x, w, y, M, N, K, n0, nb, m0, m1, b + 1 < blocks ? n0 + C_ : -1, xp);
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
    float *w[LAYERS][4];
    srand(1);
    for (int l = 0; l < LAYERS; ++l)
        for (int s = 0; s < 4; ++s) {
            int N = SHAPES[s][0], K = SHAPES[s][1];
            w[l][s] = aligned_alloc(64, (size_t)N * K * 4);
            for (size_t i = 0; i < (size_t)N * K; ++i) w[l][s][i] = frand() * 0.05f;
            if (l == 0) {  // correctness against a double reference, M = 19 (tile rows + remainder)
                int M = 19;
                float *x = malloc((size_t)M * K * 4), *y = malloc((size_t)M * N * 4);
                xpack = aligned_alloc(64, (size_t)M * K * 4);
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
    char *list = argc > 1 ? argv[1] : "16,32,256";
    printf("R=%d C=%d MC=%d PF=%d PFX=%d XPACK=%d threads=%d (88-GEMM chain, median of 21)\n", R_, C_, MC, PF, PFX, XPACK, threads);
    for (char *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
        int M = atoi(tok);
        float *x = aligned_alloc(64, (size_t)M * 1152 * 4 + 64), *y = aligned_alloc(64, (size_t)M * 2304 * 4 + 64);
        xpack = aligned_alloc(64, (size_t)M * 1152 * 4 + 64);
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
