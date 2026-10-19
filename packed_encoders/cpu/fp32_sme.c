// Exact FP32 projections on Apple SME2 CPUs with 512-bit streaming vectors.
// Prepared weights have layout [N/64, K, 64].
// Each output accumulates k in increasing order, independently of row/worker count.
// Only this translation unit targets SME2; the ordinary Apple kernels stay portable.
#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <stddef.h>
#include <string.h>

__arm_locally_streaming int pe_sme_fp32_available(void) { return svcntw() == 16; }

static void pack_input(const float *x, int m, int k, float *packed) {
    int full = m / 16 * 16;
    if (m != full) memset(packed + (size_t)full * k, 0, sizeof(float) * 16 * k);
    for (int mb = 0; mb < full; mb += 16)
        for (int col = 0; col < k; col += 4)
            for (int row = 0; row < 16; row += 4) {
                const float *p = x + (size_t)(mb + row) * k + col;
                float32x4_t a = vld1q_f32(p), b = vld1q_f32(p + k);
                float32x4_t c = vld1q_f32(p + 2 * k), d = vld1q_f32(p + 3 * k);
                float32x4_t lo = vzip1q_f32(a, b), hi = vzip2q_f32(a, b);
                float32x4_t cl = vzip1q_f32(c, d), ch = vzip2q_f32(c, d);
                float *out = packed + (size_t)mb * k + (size_t)col * 16 + row;
                vst1q_f32(out, vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(lo), vreinterpretq_f64_f32(cl))));
                vst1q_f32(out + 16, vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(lo), vreinterpretq_f64_f32(cl))));
                vst1q_f32(out + 32, vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(hi), vreinterpretq_f64_f32(ch))));
                vst1q_f32(out + 48, vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(hi), vreinterpretq_f64_f32(ch))));
            }
    for (int row = full; row < m; ++row)
        for (int col = 0; col < k; ++col)
            packed[(size_t)full * k + (size_t)col * 16 + row % 16] = x[(size_t)row * k + col];
}

typedef struct {
    const float *a, *w;
    float *out;
    int m, n, k, workers;
} Context;

__arm_locally_streaming __arm_new("za")
static void gemm(const Context *c, int worker) {
    svbool_t p = svptrue_b32();
    int m = c->m, n = c->n, k = c->k;
    if (m <= 16) {
        for (int nb = worker; nb < n / 64; nb += c->workers) {
            const float *w = c->w + (size_t)nb * k * 64;
            svzero_za();
            for (int col = 0; col < k; ++col) {
                svfloat32_t a = svld1_f32(p, c->a + (size_t)col * 16);
                const float *b = w + (size_t)col * 64;
                svmopa_za32_f32_m(0, p, p, a, svld1_f32(p, b));
                svmopa_za32_f32_m(1, p, p, a, svld1_f32(p, b + 16));
                svmopa_za32_f32_m(2, p, p, a, svld1_f32(p, b + 32));
                svmopa_za32_f32_m(3, p, p, a, svld1_f32(p, b + 48));
            }
            for (int row = 0; row < m; ++row) {
                float *out = c->out + (size_t)row * n + nb * 64;
                svst1_hor_za32(0, row, p, out);
                svst1_hor_za32(1, row, p, out + 16);
                svst1_hor_za32(2, row, p, out + 32);
                svst1_hor_za32(3, row, p, out + 48);
            }
        }
    } else {
        for (int nh = worker; nh < n / 32; nh += c->workers) {
            const float *w = c->w + (size_t)(nh / 2) * k * 64 + (nh % 2) * 32;
            for (int mb = 0; mb < m; mb += 32) {
                const float *ap = c->a + (size_t)mb * k;
                const float *a1p = mb + 16 < m ? ap + (size_t)k * 16 : ap;
                int left = m - mb;
                svzero_za();
                for (int col = 0; col < k; ++col) {
                    const float *b = w + (size_t)col * 64;
                    svfloat32_t a = svld1_f32(p, ap + (size_t)col * 16);
                    svfloat32_t a1 = svld1_f32(p, a1p + (size_t)col * 16);
                    svfloat32_t b0 = svld1_f32(p, b), b1 = svld1_f32(p, b + 16);
                    svmopa_za32_f32_m(0, p, p, a, b0);
                    svmopa_za32_f32_m(1, p, p, a, b1);
                    svmopa_za32_f32_m(2, p, p, a1, b0);
                    svmopa_za32_f32_m(3, p, p, a1, b1);
                }
                for (int row = 0; row < 16 && row < left; ++row) {
                    float *out = c->out + (size_t)(mb + row) * n + nh * 32;
                    svst1_hor_za32(0, row, p, out);
                    svst1_hor_za32(1, row, p, out + 16);
                }
                for (int row = 0; row < 16 && row < left - 16; ++row) {
                    float *out = c->out + (size_t)(mb + row + 16) * n + nh * 32;
                    svst1_hor_za32(2, row, p, out);
                    svst1_hor_za32(3, row, p, out + 16);
                }
            }
        }
    }
}

static void worker(void *opaque, size_t index) { gemm(opaque, (int)index); }

int pe_sme_fp32_linear(const float *x, const float *w, float *out, float *scratch,
                       int m, int n, int k, int workers, size_t scratch_floats) {
    if (!x || !w || !out || !scratch || m < 1 || m > 4096 || n < 64 || n % 64 || k < 4 || k % 4 || workers < 1)
        return -1;
    if (scratch_floats < (size_t)((m + 15) / 16) * 16 * k) return -2;
    pack_input(x, m, k, scratch);
    int panels = n / (m <= 16 ? 64 : 32);
    if (workers > panels) workers = panels;
    Context context = {scratch, w, out, m, n, k, workers};
    if (workers == 1) worker(&context, 0);
    else dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context, worker);
    return 0;
}
