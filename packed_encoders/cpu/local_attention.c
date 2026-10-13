// FP32 ModernBERT local attention for Apple CPUs. No fast-math.
// Q/K/V/output: contiguous [heads, total_tokens, 64]. Lengths delimit sequences.
// Queries attend keys within `half` positions in their sequence; a half-width of
// at least the longest sequence gives exact global attention.
// Build: clang -O3 -dynamiclib -framework Accelerate modernbert_local_attention.c -o attention.dylib
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <math.h>

#include "rows.h"

typedef struct {
    const float *q, *k, *v;
    float *out, *scratch;
    const int *lengths;
    int tokens, heads, sequences, tile, half, columns, workers;
} Context;

static void worker(void *opaque, size_t worker_id) {
    Context *c = opaque;
    const int dim = 64, half = c->half;
    float *scores = c->scratch + worker_id * (size_t)c->tile * c->columns;
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
            // Rows are short, so per-row vDSP/vForce calls dominated: mask by each
            // row's contiguous window and exponentiate the whole tile in one call.
            // Same operations per element as per-row calls; outputs are unchanged.
            for (int row = 0; row < rows; ++row) {
                float *s = scores + row * cols;
                int position = start + row - lo;
                int first = position > half ? position - half : 0;
                int last = position + half + 1 < cols ? position + half + 1 : cols;
                float maximum = -INFINITY;
                for (int col = first; col < last; ++col) maximum = fmaxf(maximum, s[col]);
                float shift = -maximum;
                for (int col = 0; col < first; ++col) s[col] = -INFINITY;
                for (int col = first; col < last; ++col) s[col] += shift;
                for (int col = last; col < cols; ++col) s[col] = -INFINITY;
            }
            int count = rows * cols;
            vvexpf(scores, scores, &count);
            for (int row = 0; row < rows; ++row) {
                float *s = scores + row * cols;
                float sum;
                vDSP_sve(s, 1, &sum, cols);
                float inverse = 1.f / sum;
                for (int col = 0; col < cols; ++col) s[col] *= inverse;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, dim, cols,
                        1.f, scores, cols, c->v + base + lo * dim, dim, 0.f,
                        c->out + base + start * dim, dim);
        }
    }
}

int pe_local_attention(const float *q, const float *k, const float *v,
                       float *out, const int *lengths, int tokens,
                       int heads, int sequences, int tile, int half, int workers,
                       float *scratch, size_t scratch_floats) {
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
    // Score columns per tile never exceed the window or the longest sequence.
    int columns = tile + 2 * half < longest ? tile + 2 * half : longest;
    if (workers > heads * sequences) workers = heads * sequences;
    if (!scratch || scratch_floats < (size_t)workers * tile * columns) return -2;
    Context context = {q, k, v, out, scratch, lengths, tokens, heads, sequences, tile, half, columns, workers};
    if (workers == 1) worker(&context, 0);
    else dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context, worker);
    return 0;
}

// Split-half RoPE and token-major QKV -> head-major Q/K/V in one pass (pe_rope_row).
// Tasks are token blocks covering every head, so each QKV row is read once,
// sequentially; head-major tasks re-walked the whole buffer once per head.
#define ROPE_BLOCK 32
typedef struct {
    const float *qkv, *cos, *sin;
    float *out;
    int tokens, heads, workers;
} RopeContext;

static void rope_worker(void *opaque, size_t worker_id) {
    RopeContext *c = opaque;
    int blocks = (c->tokens + ROPE_BLOCK - 1) / ROPE_BLOCK;
    for (int block = (int)worker_id; block < blocks; block += c->workers) {
        int begin = block * ROPE_BLOCK;
        int end = begin + ROPE_BLOCK < c->tokens ? begin + ROPE_BLOCK : c->tokens;
        for (int token = begin; token < end; ++token)
            pe_rope_row(c->qkv + (size_t)token * 3 * c->heads * 64, c->cos + (size_t)token * 64,
                        c->sin + (size_t)token * 64, c->out, token, c->tokens, c->heads);
    }
}

int pe_rope_layout(const float *qkv, const float *cos, const float *sin,
                   float *out, int tokens, int heads, int workers) {
    if (tokens < 1 || heads < 1 || workers < 1) return -1;
    int tasks = (tokens + ROPE_BLOCK - 1) / ROPE_BLOCK;
    if (workers > tasks) workers = tasks;
    RopeContext context = {qkv, cos, sin, out, tokens, heads, workers};
    if (workers == 1) rope_worker(&context, 0);
    else dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context, rope_worker);
    return 0;
}

// Exact-erf GeGLU: out[row, i] = GELU(in[row, i]) * in[row, width + i] for
// contiguous MLP-input rows [activation | gate] (pe_geglu_row, rational erf).
#define GEGLU_BLOCK 32
typedef struct {
    const float *in;
    float *out;
    int rows, width, workers;
} GegluContext;

static void geglu_worker(void *opaque, size_t worker_id) {
    GegluContext *c = opaque;
    int blocks = (c->rows + GEGLU_BLOCK - 1) / GEGLU_BLOCK;
    for (int block = (int)worker_id; block < blocks; block += c->workers) {
        int begin = block * GEGLU_BLOCK;
        int end = begin + GEGLU_BLOCK < c->rows ? begin + GEGLU_BLOCK : c->rows;
        for (int row = begin; row < end; ++row)
            pe_geglu_row(c->in + (size_t)row * 2 * c->width, c->out + (size_t)row * c->width, c->width);
    }
}

int pe_geglu(const float *in, float *out, int rows, int width, int workers) {
    if (rows < 1 || width < 1 || workers < 1) return -1;
    int tasks = (rows + GEGLU_BLOCK - 1) / GEGLU_BLOCK;
    if (workers > tasks) workers = tasks;
    GegluContext context = {in, out, rows, width, workers};
    if (workers == 1) geglu_worker(&context, 0);
    else dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context, geglu_worker);
    return 0;
}

// Residual add + LayerNorm over contiguous [rows, width] rows, width a multiple
// of 32 (pe_add_norm_row): residual += branch, then the normalized row into out.
#define NORM_BLOCK 32
typedef struct {
    float *residual, *out;
    const float *branch, *weight, *bias;
    int rows, width, workers;
    float eps;
} NormContext;

static void add_norm_row(const NormContext *c, int row) {
    size_t offset = (size_t)row * c->width;
    pe_add_norm_row(c->residual + offset, c->branch + offset, c->weight, c->bias, c->out + offset, c->width, c->eps);
}

static void add_norm_worker(void *opaque, size_t worker_id) {
    NormContext *c = opaque;
    int blocks = (c->rows + NORM_BLOCK - 1) / NORM_BLOCK;
    for (int block = (int)worker_id; block < blocks; block += c->workers) {
        int end = (block + 1) * NORM_BLOCK < c->rows ? (block + 1) * NORM_BLOCK : c->rows;
        for (int row = block * NORM_BLOCK; row < end; ++row) add_norm_row(c, row);
    }
}

int pe_add_layer_norm(float *residual, const float *branch, const float *weight, const float *bias, float *out,
                      int rows, int width, float eps, int workers) {
    if (rows < 1 || width < 32 || width % 32 || workers < 1) return -1;
    int tasks = (rows + NORM_BLOCK - 1) / NORM_BLOCK;
    if (workers > tasks) workers = tasks;
    NormContext context = {residual, out, branch, weight, bias, rows, width, workers, eps};
    if (workers == 1) add_norm_worker(&context, 0);
    else dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context, add_norm_worker);
    return 0;
}
