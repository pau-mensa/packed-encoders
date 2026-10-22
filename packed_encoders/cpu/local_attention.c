// FP32 ModernBERT local attention for Apple CPUs. No fast-math.
// Q/K/V: contiguous [heads, total_tokens, 64]; output the same, or token-major
// [total_tokens, heads * 64] (the next projection's input layout). Lengths delimit sequences.
// Queries attend keys within `half` positions in their sequence; a half-width of
// at least the longest sequence gives exact global attention.
// Every entry point runs its workers on `pool` (pool.h), or on GCD without one. On the pool,
// workers claim tasks in order from an atomic counter (attention: one query tile each) and
// helpers on efficiency cores sit these kernels out. GCD runs keep the static stride they had
// before the pool: int8 attention measured 6% slower at 32 tokens with claiming there.
// Build: clang -O3 -dynamiclib -framework Accelerate local_attention.c pool.c -o attention.dylib
#define ACCELERATE_NEW_LAPACK
// Only the vecLib headers in use: parsing all of Accelerate.h took ~0.3 s of a ~0.5 s build.
#include <vecLib/cblas_new.h>
#include <vecLib/vDSP.h>
#include <vecLib/vForce.h>
#include <arm_neon.h>
#include <math.h>
#include <stdatomic.h>

#include "pool.h"
#include "rows.h"

// Task loops: GCD runs stride by their worker count; pool runs (stride 0) claim from `next`.
static inline int first_task(_Atomic int *next, int stride, size_t worker_id) {
    return stride ? (int)worker_id : atomic_fetch_add_explicit(next, 1, memory_order_relaxed);
}

static inline int next_task(_Atomic int *next, int stride, int task) {
    return stride ? task + stride : atomic_fetch_add_explicit(next, 1, memory_order_relaxed);
}

typedef struct {
    const float *q, *k, *v;
    float *out, *scratch;
    const int *lengths;
    int tokens, heads, sequences, tile, half, columns, workers, token_major, head_tiles;
    _Atomic int next;
} Context;

// Query rows [start, start + tile) of one head and sequence.
static void attend(const Context *c, float *scores, int head, int offset, int length, int start) {
    const int dim = 64, half = c->half;
    size_t base = ((size_t)head * c->tokens + offset) * dim;
    // Output rows ldo apart from this head and sequence's first token.
    int ldo = c->token_major ? c->heads * dim : dim;
    float *o = c->out + (c->token_major ? ((size_t)offset * c->heads + head) * dim : base);
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
                o + (size_t)start * ldo, ldo);
}

// GCD: every tile of one head and sequence per task.
static void worker(void *opaque, size_t worker_id) {
    Context *c = opaque;
    float *scores = c->scratch + worker_id * (size_t)c->tile * c->columns;
    for (int task = (int)worker_id; task < c->heads * c->sequences; task += c->workers) {
        int head = task / c->sequences, sequence = task % c->sequences;
        int offset = 0, length = c->lengths[sequence];
        for (int i = 0; i < sequence; ++i) offset += c->lengths[i];
        for (int start = 0; start < length; start += c->tile) attend(c, scores, head, offset, length, start);
    }
}

// Pool: one tile per task, so a late worker holds back at most one tile.
static void tile_worker(void *opaque, size_t worker_id) {
    Context *c = opaque;
    float *scores = c->scratch + worker_id * (size_t)c->tile * c->columns;
    for (int task; (task = atomic_fetch_add_explicit(&c->next, 1, memory_order_relaxed)) < c->heads * c->head_tiles;) {
        int head = task / c->head_tiles, index = task % c->head_tiles, sequence = 0, offset = 0;
        while (index >= (c->lengths[sequence] + c->tile - 1) / c->tile) {
            index -= (c->lengths[sequence] + c->tile - 1) / c->tile;
            offset += c->lengths[sequence++];
        }
        attend(c, scores, head, offset, c->lengths[sequence], index * c->tile);
    }
}

int pe_local_attention(const float *q, const float *k, const float *v,
                       float *out, const int *lengths, int tokens,
                       int heads, int sequences, int tile, int half, int token_major, int workers, void *pool,
                       float *scratch, size_t scratch_floats) {
    if (tokens < 1 || heads < 1 || sequences < 1 || tile < 1 || tile > 4096 || half < 0 || workers < 1)
        return -1;
    long total = 0;
    int longest = 0, head_tiles = 0;
    for (int i = 0; i < sequences; ++i) {
        if (lengths[i] < 1) return -1;
        total += lengths[i];
        if (lengths[i] > longest) longest = lengths[i];
        head_tiles += (lengths[i] + tile - 1) / tile;
    }
    if (total != tokens) return -1;
    if (half > longest) half = longest;  // already global; avoids index overflow
    // Score columns per tile never exceed the window or the longest sequence.
    int columns = tile + 2 * half < longest ? tile + 2 * half : longest;
    if (workers > heads * sequences) workers = heads * sequences;
    if (!scratch || scratch_floats < (size_t)workers * tile * columns) return -2;
    Context context = {q, k, v, out, scratch, lengths, tokens, heads, sequences, tile, half, columns, workers,
                       token_major, head_tiles, 0};
    pe_pool_apply(pool, workers, &context, pool ? tile_worker : worker, 0);
    return 0;
}

// Split-half RoPE and token-major QKV -> head-major Q/K/V in one pass (pe_rope_row).
// Tasks are token blocks covering every head, so each QKV row is read once,
// sequentially; head-major tasks re-walked the whole buffer once per head.
#define ROPE_BLOCK 32
typedef struct {
    const float *qkv, *cos, *sin;
    float *out;
    int tokens, heads, stride;
    _Atomic int next;
} RopeContext;

static void rope_worker(void *opaque, size_t worker_id) {
    RopeContext *c = opaque;
    int blocks = (c->tokens + ROPE_BLOCK - 1) / ROPE_BLOCK;
    for (int block = first_task(&c->next, c->stride, worker_id); block < blocks;
         block = next_task(&c->next, c->stride, block)) {
        int begin = block * ROPE_BLOCK;
        int end = begin + ROPE_BLOCK < c->tokens ? begin + ROPE_BLOCK : c->tokens;
        for (int token = begin; token < end; ++token)
            pe_rope_row(c->qkv + (size_t)token * 3 * c->heads * 64, c->cos + (size_t)token * 64,
                        c->sin + (size_t)token * 64, c->out, token, c->tokens, c->heads);
    }
}

int pe_rope_layout(const float *qkv, const float *cos, const float *sin,
                   float *out, int tokens, int heads, int workers, void *pool) {
    if (tokens < 1 || heads < 1 || workers < 1) return -1;
    int tasks = (tokens + ROPE_BLOCK - 1) / ROPE_BLOCK;
    if (workers > tasks) workers = tasks;
    RopeContext context = {qkv, cos, sin, out, tokens, heads, pool ? 0 : workers, 0};
    pe_pool_apply(pool, workers, &context, rope_worker, 0);
    return 0;
}

// Exact-erf GeGLU: out[row, i] = GELU(in[row, i]) * in[row, width + i] for
// contiguous MLP-input rows [activation | gate] (pe_geglu_row, rational erf).
#define GEGLU_BLOCK 32
typedef struct {
    const float *in;
    float *out;
    int rows, width, stride;
    _Atomic int next;
} GegluContext;

static void geglu_worker(void *opaque, size_t worker_id) {
    GegluContext *c = opaque;
    int blocks = (c->rows + GEGLU_BLOCK - 1) / GEGLU_BLOCK;
    for (int block = first_task(&c->next, c->stride, worker_id); block < blocks;
         block = next_task(&c->next, c->stride, block)) {
        int begin = block * GEGLU_BLOCK;
        int end = begin + GEGLU_BLOCK < c->rows ? begin + GEGLU_BLOCK : c->rows;
        for (int row = begin; row < end; ++row)
            pe_geglu_row(c->in + (size_t)row * 2 * c->width, c->out + (size_t)row * c->width, c->width);
    }
}

int pe_geglu(const float *in, float *out, int rows, int width, int workers, void *pool) {
    if (rows < 1 || width < 1 || workers < 1) return -1;
    int tasks = (rows + GEGLU_BLOCK - 1) / GEGLU_BLOCK;
    if (workers > tasks) workers = tasks;
    GegluContext context = {in, out, rows, width, pool ? 0 : workers, 0};
    pe_pool_apply(pool, workers, &context, geglu_worker, 0);
    return 0;
}

// Residual add + LayerNorm over contiguous [rows, width] rows, width a multiple
// of 32 (pe_add_norm_row): residual += branch, then the normalized row into out.
#define NORM_BLOCK 32
typedef struct {
    float *residual, *out;
    const float *branch, *weight, *bias;
    int rows, width, stride;
    float eps;
    _Atomic int next;
} NormContext;

static void add_norm_row(const NormContext *c, int row) {
    size_t offset = (size_t)row * c->width;
    pe_add_norm_row(c->residual + offset, c->branch + offset, c->weight, c->bias, c->out + offset, c->width, c->eps);
}

static void add_norm_worker(void *opaque, size_t worker_id) {
    NormContext *c = opaque;
    int blocks = (c->rows + NORM_BLOCK - 1) / NORM_BLOCK;
    for (int block = first_task(&c->next, c->stride, worker_id); block < blocks;
         block = next_task(&c->next, c->stride, block)) {
        int end = (block + 1) * NORM_BLOCK < c->rows ? (block + 1) * NORM_BLOCK : c->rows;
        for (int row = block * NORM_BLOCK; row < end; ++row) add_norm_row(c, row);
    }
}

int pe_add_layer_norm(float *residual, const float *branch, const float *weight, const float *bias, float *out,
                      int rows, int width, float eps, int workers, void *pool) {
    if (rows < 1 || width < 32 || width % 32 || workers < 1) return -1;
    int tasks = (rows + NORM_BLOCK - 1) / NORM_BLOCK;
    if (workers > tasks) workers = tasks;
    NormContext context = {residual, out, branch, weight, bias, rows, width, pool ? 0 : workers, eps, 0};
    pe_pool_apply(pool, workers, &context, add_norm_worker, 0);
    return 0;
}
