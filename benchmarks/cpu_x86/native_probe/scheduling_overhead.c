#include <omp.h>
#include <stddef.h>
// Nine worksharing stages model one non-final INT8 segment. Each writes disjoint
// rows; the final stage can rely on the parallel join instead of a loop barrier.
void probe(int workers, int tokens, int reps, int nowait, volatile float *out) {
    for (int rep=0;rep<reps;++rep) {
#pragma omp parallel num_threads(workers) if(workers>1)
        {
            for(int stage=0;stage<8;++stage) {
#pragma omp for schedule(static)
                for(int r=0;r<tokens;++r) out[(size_t)r*16]=(float)r+stage;
            }
            if(nowait) {
#pragma omp for schedule(static) nowait
                for(int r=0;r<tokens;++r) out[(size_t)r*16]=(float)r+8;
            } else {
#pragma omp for schedule(static)
                for(int r=0;r<tokens;++r) out[(size_t)r*16]=(float)r+8;
            }
        }
    }
}
