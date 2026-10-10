// FP32 ModernBERT local attention for Apple CPUs. No fast-math.
// Q/K/V/output: contiguous [heads, total_tokens, 64]. Lengths delimit sequences.
// Queries attend keys within `half` positions in their sequence; a half-width of
// at least the longest sequence gives exact global attention.
// Build: clang -O3 -dynamiclib -framework Accelerate modernbert_local_attention.c -o attention.dylib
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <math.h>

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

// Split-half RoPE and token-major QKV -> head-major Q/K/V in one pass.
// Keep multiply/add separate to match the unfused FP32 PyTorch operations.
// Tasks are token blocks covering every head, so each QKV row is read once,
// sequentially; head-major tasks re-walked the whole buffer once per head.
#pragma clang fp contract(off)
#define ROPE_BLOCK 32
typedef struct {
    const float *qkv, *cos, *sin;
    float *out;
    int tokens, heads, workers;
} RopeContext;

static void rope_worker(void *opaque, size_t worker_id) {
    RopeContext *c = opaque;
    int blocks = (c->tokens + ROPE_BLOCK - 1) / ROPE_BLOCK;
    size_t plane = (size_t)c->heads * c->tokens * 64;
    for (int block = (int)worker_id; block < blocks; block += c->workers) {
        int begin = block * ROPE_BLOCK;
        int end = begin + ROPE_BLOCK < c->tokens ? begin + ROPE_BLOCK : c->tokens;
        for (int token = begin; token < end; ++token) {
            const float *row = c->qkv + (size_t)token * 3 * c->heads * 64;
            const float *co = c->cos + (size_t)token * 64;
            const float *si = c->sin + (size_t)token * 64;
            for (int head = 0; head < c->heads; ++head) {
                const float *src = row + (size_t)head * 64;
                float *dst = c->out + ((size_t)head * c->tokens + token) * 64;
                for (int part = 0; part < 2; ++part) {
                    const float *in = src + part * c->heads * 64;
                    float *out = dst + part * plane;
                    for (int d = 0; d < 32; ++d) {
                        out[d] = in[d] * co[d] + (-in[d + 32]) * si[d];
                        out[d + 32] = in[d + 32] * co[d + 32] + in[d] * si[d + 32];
                    }
                }
                for (int d = 0; d < 64; ++d) dst[2 * plane + d] = src[2 * c->heads * 64 + d];
            }
        }
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
// contiguous MLP-input rows [activation | gate]. The rational erf (odd 13 /
// even 8 degree, as in Eigen/TensorFlow) is within ~4e-7 of erf, comparable
// to PyTorch's FP32 GELU; not the tanh approximation. Contraction is allowed.
#pragma clang fp contract(on)
#define GEGLU_BLOCK 32
typedef struct {
    const float *in;
    float *out;
    int rows, width, workers;
} GegluContext;

static inline float erf_rational(float a) {
    float x = a > 4.f ? 4.f : (a < -4.f ? -4.f : a);
    float x2 = x * x;
    float p = x2 * -2.72614225801306e-10f + 2.77068142495902e-08f;
    p = x2 * p + -2.10102402082508e-06f;
    p = x2 * p + -5.69250639462346e-05f;
    p = x2 * p + -7.34990630326855e-04f;
    p = x2 * p + -2.95459980854025e-03f;
    p = x2 * p + -1.60960333262415e-02f;
    p = x * p;
    float q = x2 * -1.45660718464996e-05f + -2.13374055278905e-04f;
    q = x2 * q + -1.68282697438203e-03f;
    q = x2 * q + -7.37332916720468e-03f;
    q = x2 * q + -1.42647390514189e-02f;
    // FP32 erf(4) rounds to 1, so saturate exactly beyond the fitted range.
    return a >= 4.f ? 1.f : (a <= -4.f ? -1.f : p / q);
}

static void geglu_worker(void *opaque, size_t worker_id) {
    GegluContext *c = opaque;
    int blocks = (c->rows + GEGLU_BLOCK - 1) / GEGLU_BLOCK;
    for (int block = (int)worker_id; block < blocks; block += c->workers) {
        int begin = block * GEGLU_BLOCK;
        int end = begin + GEGLU_BLOCK < c->rows ? begin + GEGLU_BLOCK : c->rows;
        for (int row = begin; row < end; ++row) {
            const float *a = c->in + (size_t)row * 2 * c->width, *gate = a + c->width;
            float *out = c->out + (size_t)row * c->width;
#pragma clang loop interleave_count(4)
            for (int i = 0; i < c->width; ++i)
                out[i] = 0.5f * a[i] * (1.f + erf_rational(a[i] * 0.70710678118654752f)) * gate[i];
        }
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
