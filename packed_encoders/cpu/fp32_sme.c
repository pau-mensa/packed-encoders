// Exact FP32 projections on Apple SME2 CPUs with 512-bit streaming vectors.
// Prepared weights have layout [N/64, K, 64].
// Each output accumulates k in increasing order, independently of row/worker count.
// Only this translation unit targets SME2; the ordinary Apple kernels stay portable.
//
// Workers run on the engine's pool (pool.h): the caller is worker 0, and every worker takes
// input-packing tasks, then output tiles, from atomic counters. The efficiency cluster has an
// SME unit of its own. On the M4 Pro, helpers there paid off with four or fewer workers
// (skipping one left too few to keep both P-cluster units busy) or above 128 rows; otherwise
// their last tiles held back the projection.
#include <arm_neon.h>
#include <arm_sme.h>
#include <stddef.h>
#include <string.h>

#include "pool.h"

// Rows per output-tile group above this, so a group's packed rows stay in L2 across panels.
#define ROW_CHUNK 512
#define PACK_COLUMNS 256

__arm_locally_streaming int pe_sme_fp32_available(void) { return svcntw() == 16; }

// Packed input [ceil(m/16), k, 16]: rows past m are zero. One task covers 16 rows x a column range.
static void pack_block(const float *x, int m, int k, float *packed, int mb, int c0, int c1) {
    float *base = packed + (size_t)mb * k;
    if (mb + 16 > m) {
        memset(base + (size_t)c0 * 16, 0, sizeof(float) * 16 * (size_t)(c1 - c0));
        for (int row = mb; row < m; ++row)
            for (int col = c0; col < c1; ++col) base[(size_t)col * 16 + row - mb] = x[(size_t)row * k + col];
        return;
    }
    for (int col = c0; col < c1; col += 4)
        for (int row = 0; row < 16; row += 4) {
            const float *p = x + (size_t)(mb + row) * k + col;
            float32x4_t a = vld1q_f32(p), b = vld1q_f32(p + k);
            float32x4_t c = vld1q_f32(p + 2 * k), d = vld1q_f32(p + 3 * k);
            float32x4_t lo = vzip1q_f32(a, b), hi = vzip2q_f32(a, b);
            float32x4_t cl = vzip1q_f32(c, d), ch = vzip2q_f32(c, d);
            float *out = base + (size_t)col * 16 + row;
            vst1q_f32(out, vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(lo), vreinterpretq_f64_f32(cl))));
            vst1q_f32(out + 16, vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(lo), vreinterpretq_f64_f32(cl))));
            vst1q_f32(out + 32, vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(hi), vreinterpretq_f64_f32(ch))));
            vst1q_f32(out + 48, vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(hi), vreinterpretq_f64_f32(ch))));
        }
}

typedef struct {
    const float *x, *w;
    float *a, *out;
    int m, n, k, columns, column_chunks, pack_tasks, panels, rows, tiles;
    _Atomic int pack_next, packed, next;
} Context;

__arm_locally_streaming __arm_new("za")
static void gemm(Context *c) {
    svbool_t p = svptrue_b32();
    int m = c->m, n = c->n, k = c->k;
    if (m <= 16) {
        for (int nb; (nb = atomic_fetch_add_explicit(&c->next, 1, memory_order_relaxed)) < c->tiles;) {
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
        return;
    }
    // Tiles run row group by row group; within a group, one 32-column panel each.
    for (int tile; (tile = atomic_fetch_add_explicit(&c->next, 1, memory_order_relaxed)) < c->tiles;) {
        int nh = tile % c->panels, first = tile / c->panels * c->rows;
        int last = first + c->rows < m ? first + c->rows : m;
        const float *w = c->w + (size_t)(nh / 2) * k * 64 + (nh % 2) * 32;
        for (int mb = first; mb < last; mb += 32) {
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

static void worker(void *opaque, size_t index) {
    Context *c = opaque;
    (void)index;
    for (int task; (task = atomic_fetch_add_explicit(&c->pack_next, 1, memory_order_relaxed)) < c->pack_tasks;) {
        int c0 = task % c->column_chunks * c->columns;
        pack_block(c->x, c->m, c->k, c->a, task / c->column_chunks * 16, c0, c0 + c->columns < c->k ? c0 + c->columns : c->k);
        atomic_fetch_add_explicit(&c->packed, 1, memory_order_release);
    }
    // Every packing task is claimed by a running worker, so this wait ends.
    while (atomic_load_explicit(&c->packed, memory_order_acquire) < c->pack_tasks) __asm__ volatile("yield");
    gemm(c);
}

int pe_sme_fp32_linear(void *pool, const float *x, const float *w, float *out, float *scratch,
                       int m, int n, int k, int workers, size_t scratch_floats) {
    if (!x || !w || !out || !scratch || m < 1 || m > 4096 || n < 64 || n % 64 || k < 4 || k % 4 || workers < 1)
        return -1;
    if (scratch_floats < (size_t)((m + 15) / 16) * 16 * k) return -2;
    int panels = n / (m <= 16 ? 64 : 32), rows = m <= ROW_CHUNK ? m : ROW_CHUNK;
    int tiles = panels * ((m + rows - 1) / rows);
    if (workers > tiles) workers = tiles;
    int columns = workers == 1 || k <= PACK_COLUMNS ? k : PACK_COLUMNS, column_chunks = (k + columns - 1) / columns;
    Context context = {.x = x, .w = w, .a = scratch, .out = out, .m = m, .n = n, .k = k, .columns = columns,
                       .column_chunks = column_chunks, .pack_tasks = (m + 15) / 16 * column_chunks,
                       .panels = panels, .rows = rows, .tiles = tiles};
    pe_pool_apply(pool, (size_t)workers, &context, worker, workers <= 4 || m > 128);
    return 0;
}
