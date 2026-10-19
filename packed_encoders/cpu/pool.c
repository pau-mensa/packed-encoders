// Threads of the engine-owned worker pool (pool.h). Built into the Apple attention library.
#include <mach/mach_time.h>
#include <stdlib.h>

#include "pool.h"

typedef struct {
    PePool *pool;
    int index;
} Start;

static _Atomic int live_threads;

int pe_pool_threads(void) { return atomic_load(&live_threads); }

// Spin through the short gaps between kernels, then park.
static uint32_t pool_wait(PePool *pool, uint32_t seen) {
    uint64_t start = mach_absolute_time();
    for (;;) {
        uint32_t generation = atomic_load_explicit(&pool->generation, memory_order_acquire);
        if (generation != seen) return generation;
        if (mach_absolute_time() - start >= pool->spin) break;
        __asm__ volatile("yield");
    }
    for (;;) {
        // Paired with the caller's generation store and sleeper check (both sequentially consistent).
        atomic_fetch_add(&pool->sleepers, 1);
        if (atomic_load(&pool->generation) == seen) {
            if (__builtin_available(macOS 14.4, *))
                os_sync_wait_on_address(&pool->generation, seen, sizeof(uint32_t), OS_SYNC_WAIT_ON_ADDRESS_NONE);
        }
        atomic_fetch_sub(&pool->sleepers, 1);
        uint32_t generation = atomic_load(&pool->generation);
        if (generation != seen) return generation;
    }
}

static void *pool_thread(void *opaque) {
    Start start = *(Start *)opaque;
    free(opaque);
    PePool *pool = start.pool;
    uint32_t seen = 0;
    for (;;) {
        seen = pool_wait(pool, seen);
        if (atomic_load(&pool->stop)) break;
        if ((uint32_t)start.index >= pe_pool_active(seen)) continue;
        size_t cpu;
        if (pool->any_core || pthread_cpu_number_np(&cpu) || cpu >= (size_t)pool->efficiency)
            for (size_t i = (size_t)start.index; i < pool->iterations; i += (size_t)pool->size)
                pool->work(pool->context, i);
        atomic_fetch_sub_explicit(&pool->pending, 1, memory_order_release);
    }
    atomic_fetch_sub(&live_threads, 1);
    return NULL;
}

// `workers` counts the caller; CPUs numbered below `efficiency` are efficiency cores (Apple
// Silicon numbers them first). NULL when the OS cannot park threads (before macOS 14.4).
void *pe_pool(int workers, int spin_us, int efficiency) {
    if (workers < 1 || workers > PE_POOL_MAX || spin_us < 0 || efficiency < 0) return NULL;
    if (__builtin_available(macOS 14.4, *)) {
    } else {
        return NULL;
    }
    PePool *pool = calloc(1, sizeof(PePool));
    if (!pool) return NULL;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    pool->spin = (uint64_t)spin_us * 1000 * timebase.denom / timebase.numer;
    pool->pid = getpid();
    pool->size = 1;
    pool->efficiency = efficiency;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INITIATED, 0);
    for (int i = 1; i < workers; ++i) {
        Start *start = malloc(sizeof(Start));
        if (!start) break;
        *start = (Start){pool, i};
        atomic_fetch_add(&live_threads, 1);
        if (pthread_create(&pool->threads[i], &attr, pool_thread, start)) {
            atomic_fetch_sub(&live_threads, 1);
            free(start);
            break;
        }
        pool->size = i + 1;  // Fewer threads than requested only lowers parallelism.
    }
    pthread_attr_destroy(&attr);
    return pool;
}

void pe_pool_free(void *opaque) {
    PePool *pool = opaque;
    if (!pool) return;
    if (pool->pid == getpid()) {
        atomic_store(&pool->stop, 1);
        uint32_t generation = atomic_load(&pool->generation);
        atomic_store(&pool->generation, ((generation >> PE_POOL_BITS) + 1) << PE_POOL_BITS);
        if (__builtin_available(macOS 14.4, *))
            os_sync_wake_by_address_all(&pool->generation, sizeof(uint32_t), OS_SYNC_WAKE_BY_ADDRESS_NONE);
        for (int i = 1; i < pool->size; ++i) pthread_join(pool->threads[i], NULL);
    }
    free(pool);
}
