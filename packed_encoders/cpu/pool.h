// Engine-owned worker pool for the Apple CPU kernels (local_attention.c and fp32_sme.c);
// pool.c creates its threads.
//
// Each M4 Pro P cluster shares one SME unit, which the SME GEMM uses. GCD wakes short
// bursts of two or four workers on a single cluster, while long-lived threads spread across
// both. FP32 engines therefore run their kernels on these threads, which spin between calls
// and then park.
//
// Spinning threads also run on the efficiency cluster part of the time. Every kernel claims
// its tasks from an atomic counter, so work(context, i) only joins that queue: index 0 (the
// caller) always runs, and each helper index runs at most once. Unless a run allows any core,
// helpers found on an efficiency core skip it, since a straggler there would hold back a whole
// small kernel. Without a pool, pe_pool_apply is dispatch_apply_f.
#ifndef PE_POOL_H
#define PE_POOL_H

#include <dispatch/dispatch.h>
#include <os/os_sync_wait_on_address.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

#define PE_POOL_MAX 16
// The generation word's low bits count the threads (the caller included) taking part in a run,
// so threads that sit one out never read its fields and the caller never waits for them.
#define PE_POOL_BITS 5

typedef struct {
    pthread_t threads[PE_POOL_MAX];
    int size;        // threads, the caller included
    int efficiency;  // CPUs numbered below this are efficiency cores
    pid_t pid;
    uint64_t spin;
    _Atomic uint32_t generation;
    _Atomic int pending, sleepers, stop;
    size_t iterations;
    void (*work)(void *, size_t);
    void *context;
    int any_core;
} PePool;

static inline uint32_t pe_pool_active(uint32_t generation) { return generation & ((1u << PE_POOL_BITS) - 1); }

// Runs work(context, i) for i below iterations, thread t taking t, t + size, ...; helpers on
// efficiency cores skip theirs unless any_core. One caller at a time per pool. A forked child
// has no pool threads and runs every index itself.
static inline void pe_pool_apply(void *opaque, size_t iterations, void *context, void (*work)(void *, size_t),
                                 int any_core) {
    PePool *pool = opaque;
    if (iterations == 1) {
        work(context, 0);
        return;
    }
    if (!pool) {
        dispatch_apply_f(iterations, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), context, work);
        return;
    }
    if (pool->size == 1 || pool->pid != getpid()) {
        for (size_t i = 0; i < iterations; ++i) work(context, i);
        return;
    }
    uint32_t active = iterations < (size_t)pool->size ? (uint32_t)iterations : (uint32_t)pool->size;
    pool->iterations = iterations;
    pool->work = work;
    pool->context = context;
    pool->any_core = any_core;
    atomic_store_explicit(&pool->pending, (int)active - 1, memory_order_relaxed);
    uint32_t generation = atomic_load_explicit(&pool->generation, memory_order_relaxed);
    // Paired with the sleepers' increment and generation check (both sequentially consistent).
    atomic_store(&pool->generation, ((generation >> PE_POOL_BITS) + 1) << PE_POOL_BITS | active);
    if (atomic_load(&pool->sleepers)) {
        if (__builtin_available(macOS 14.4, *))
            os_sync_wake_by_address_all(&pool->generation, sizeof(uint32_t), OS_SYNC_WAKE_BY_ADDRESS_NONE);
    }
    for (size_t i = 0; i < iterations; i += (size_t)pool->size) work(context, i);
    while (atomic_load_explicit(&pool->pending, memory_order_acquire) > 0) __asm__ volatile("yield");
}

#endif
