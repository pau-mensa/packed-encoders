// FP32 ModernBERT local attention for Apple CPUs. No fast-math.
// Q/K/V/output: contiguous [heads, total_tokens, 64]. Lengths delimit sequences.
// Build: clang -O3 -dynamiclib -framework Accelerate modernbert_local_attention.c -o attention.dylib
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdlib.h>

typedef struct {
    const float *q, *k, *v;
    float *out, *scratch;
    const int *lengths;
    int tokens, heads, sequences, tile, workers;
} Context;

static void worker(void *opaque, size_t worker_id) {
    Context *c = opaque;
    const int dim = 64, half = 64;
    float *scores = c->scratch + worker_id * c->tile * (c->tile + 2 * half);
    for (int task = (int)worker_id; task < c->heads * c->sequences; task += c->workers) {
        int head = task / c->sequences, sequence = task % c->sequences;
        int offset = 0, length = c->lengths[sequence];
        for (int i = 0; i < sequence; ++i) offset += c->lengths[i];
        size_t base = ((size_t)head * c->tokens + offset) * dim;
        for (int start = 0; start < length; start += c->tile) {
            int rows = length - start < c->tile ? length - start : c->tile;
            int lo = start > half ? start - half : 0;
            int hi = start + rows + half < length ? start + rows + half : length;
            int cols = hi - lo;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, cols, dim,
                        0.125f, c->q + base + start * dim, dim,
                        c->k + base + lo * dim, dim, 0.f, scores, cols);
            for (int row = 0; row < rows; ++row) {
                float *s = scores + row * cols;
                float maximum = -INFINITY;
                for (int col = 0; col < cols; ++col) {
                    if (abs(start + row - lo - col) > half) s[col] = -INFINITY;
                    if (s[col] > maximum) maximum = s[col];
                }
                float shift = -maximum;
                vDSP_vsadd(s, 1, &shift, s, 1, cols);
                vvexpf(s, s, &cols);
                float sum;
                vDSP_sve(s, 1, &sum, cols);
                float inverse = 1.f / sum;
                vDSP_vsmul(s, 1, &inverse, s, 1, cols);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, dim, cols,
                        1.f, scores, cols, c->v + base + lo * dim, dim, 0.f,
                        c->out + base + start * dim, dim);
        }
    }
}

int pe_local_attention(const float *q, const float *k, const float *v,
                       float *out, const int *lengths, int tokens,
                       int heads, int sequences, int tile, int workers,
                       float *scratch, size_t scratch_floats) {
    if (tokens < 1 || heads < 1 || sequences < 1 || tile < 1 || tile > 4096 || workers < 1)
        return -1;
    long total = 0;
    for (int i = 0; i < sequences; ++i) {
        if (lengths[i] < 1) return -1;
        total += lengths[i];
    }
    if (total != tokens) return -1;
    if (workers > heads * sequences) workers = heads * sequences;
    if (!scratch || scratch_floats < (size_t)workers * tile * (tile + 128)) return -2;
    Context context = {q, k, v, out, scratch, lengths, tokens, heads, sequences, tile, workers};
    if (workers == 1) worker(&context, 0);
    else dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context, worker);
    return 0;
}
