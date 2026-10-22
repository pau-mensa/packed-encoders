// FP32 row kernels for non-Apple CPUs: the RoPE/layout, GeGLU and residual add + LayerNorm
// entry points of local_attention.c, with the same 32-row tasks, threaded by OpenMP instead of
// GCD or Apple's worker pool (their `pool` argument is ignored). Linked against the OpenMP runtime
// PyTorch already loaded, so they share its thread pool.
// Build: gcc -O3 -march=native -ffp-contract=off -fno-trapping-math -fno-math-errno -fopenmp -c rows.c
// (-fno-trapping-math lets GCC vectorize the saturating erf; neither flag changes results).
#include "rows.h"

#define BLOCK 32

// Split-half RoPE and token-major QKV -> head-major Q/K/V in one pass (pe_rope_row).
int pe_rope_layout(const float *qkv, const float *cos, const float *sin, float *out, int tokens, int heads,
                   int workers, void *pool) {
    (void)pool;
    if (tokens < 1 || heads < 1 || workers < 1) return -1;
    int tasks = (tokens + BLOCK - 1) / BLOCK;
    if (workers > tasks) workers = tasks;
#pragma omp parallel for num_threads(workers) if (workers > 1) schedule(static)
    for (int block = 0; block < tasks; ++block) {
        int end = (block + 1) * BLOCK < tokens ? (block + 1) * BLOCK : tokens;
        for (int token = block * BLOCK; token < end; ++token)
            pe_rope_row(qkv + (size_t)token * 3 * heads * 64, cos + (size_t)token * 64, sin + (size_t)token * 64,
                        out, token, tokens, heads);
    }
    return 0;
}

// Exact-erf GeGLU over contiguous [activation | gate] rows (pe_geglu_row).
int pe_geglu(const float *in, float *out, int rows, int width, int workers, void *pool) {
    (void)pool;
    if (rows < 1 || width < 1 || workers < 1) return -1;
    int tasks = (rows + BLOCK - 1) / BLOCK;
    if (workers > tasks) workers = tasks;
#pragma omp parallel for num_threads(workers) if (workers > 1) schedule(static)
    for (int block = 0; block < tasks; ++block) {
        int end = (block + 1) * BLOCK < rows ? (block + 1) * BLOCK : rows;
        for (int row = block * BLOCK; row < end; ++row)
            pe_geglu_row(in + (size_t)row * 2 * width, out + (size_t)row * width, width);
    }
    return 0;
}

// Residual add + LayerNorm over contiguous [rows, width] rows, width a multiple of 32.
int pe_add_layer_norm(float *residual, const float *branch, const float *weight, const float *bias, float *out,
                      int rows, int width, float eps, int workers, void *pool) {
    (void)pool;
    if (rows < 1 || width < 32 || width % 32 || workers < 1) return -1;
    int tasks = (rows + BLOCK - 1) / BLOCK;
    if (workers > tasks) workers = tasks;
#pragma omp parallel for num_threads(workers) if (workers > 1) schedule(static)
    for (int block = 0; block < tasks; ++block) {
        int end = (block + 1) * BLOCK < rows ? (block + 1) * BLOCK : rows;
        for (int row = block * BLOCK; row < end; ++row) {
            size_t offset = (size_t)row * width;
            pe_add_norm_row(residual + offset, branch + offset, weight, bias, out + offset, width, eps);
        }
    }
    return 0;
}
