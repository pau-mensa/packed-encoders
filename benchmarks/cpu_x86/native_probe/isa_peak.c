// Per-core peak of the candidate GEMM instructions: 12 independent zmm accumulator chains in registers.
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#define N 240000000L
#define R12(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    __m512 b = _mm512_set1_ps(1.0000001f), c = _mm512_set1_ps(1e-7f);
    __m512i u = _mm512_set1_epi8(1), s = _mm512_set1_epi8(-1);
    __m512bh hb = (__m512bh)_mm512_set1_epi16(0x3f80);
#define DECL(i) __m512 f##i = _mm512_set1_ps(i); __m512i q##i = _mm512_set1_epi32(i); __m512 g##i = _mm512_set1_ps(i);
    R12(DECL)
    double t = now();
    for (long n = 0; n < N / 12; ++n) {
#define FMA(i) f##i = _mm512_fmadd_ps(f##i, b, c);
        R12(FMA)
    }
    double fma = now() - t;
    t = now();
    for (long n = 0; n < N / 12; ++n) {
#define VNNI(i) q##i = _mm512_dpbusd_epi32(q##i, u, s);
        R12(VNNI)
    }
    double vnni = now() - t;
    t = now();
    for (long n = 0; n < N / 12; ++n) {
#define BF(i) g##i = _mm512_dpbf16_ps(g##i, hb, hb);
        R12(BF)
    }
    double bf = now() - t;
    float sink = 0; int isink = 0;
#define SINK(i) sink += _mm512_reduce_add_ps(f##i) + _mm512_reduce_add_ps(g##i); isink += _mm512_reduce_add_epi32(q##i);
    R12(SINK)
    printf("per core (sink %g %d)\n", sink, isink);
    printf("  vfmadd231ps zmm : %.2f G instr/s -> %6.0f GFLOPS fp32\n", N / fma / 1e9, N / fma * 32 / 1e9);
    printf("  vdpbf16ps   zmm : %.2f G instr/s -> %6.0f GFLOPS bf16 (fp32 accum)\n", N / bf / 1e9, N / bf * 64 / 1e9);
    printf("  vpdpbusd    zmm : %.2f G instr/s -> %6.0f GOPS int8 (u8 x s8 -> s32)\n", N / vnni / 1e9, N / vnni * 128 / 1e9);
    return 0;
}
