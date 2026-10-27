# Further x86 packing opportunities

Measured on the Ryzen 9 9950X, 2026-10-20, against `7e0c8c1`. These are isolated
source variants in `/tmp`, not production changes. Scope and environment match
[the packing baseline](PACKING.md): real ModernBERT-base weights, fresh processes,
model loading excluded, native compilation and default pack validation included.

## Two changes worth doing first

**Compile the three independent x86 translation units concurrently.**
`CPUAttention` currently compiles `rows.c`, `local_attention_x86.c`, and
`gemm_x86.c` serially. They can compile independently, followed by the existing
link step. A single eight-thread packing probe fell from 5.23 to 4.84 s. This
does not change the generated kernels or calibration arithmetic. Bound compiler
concurrency to available CPUs in a production implementation; the probe used
three jobs with affinity to eight physical cores.

**Use one factorization task at a time on this MKL/x86 setup.**
Packing currently starts up to four concurrent inverse-Hessian tasks, each of
which can also use PyTorch's configured BLAS workers. A single executor worker
still overlaps the next factor with the main thread's GPTQ work, while reducing
the number of live factorization intermediates and competing BLAS teams.

Three alternating fresh-process rounds at eight threads:

| Variant | Median pack | Median peak parent RSS |
|---|---:|---:|
| Current HEAD | 5.246 s | 1,909 MiB |
| One factor task | 5.239 s | 1,752 MiB |

That is approximately **156 MiB less peak RSS (8.2%)**, with similar time.
One-thread packing already uses one factor task, so this does not help that case.
Intel CPUs and other BLAS implementations need their own scheduling comparison;
this result is evidence for this AMD/MKL setup.

## Combined result, preserving the prepared weights

Two alternating fresh-process rounds at eight threads, with both changes:

| Variant | Median pack | Median peak parent RSS |
|---|---:|---:|
| Current HEAD | 5.347 s | 1,898 MiB |
| Concurrent compilation + one factor task | 4.726 s | 1,759 MiB |

The matched comparison saves **11.6% of packing time** and **139 MiB of peak RSS
(7.3%)**. A separate process-tree probe measured 5.244 → 4.751 s, supporting a
practical time saving of roughly 10%, rather than an exact universal percentage.

All four prepared-tensor SHA-256 hashes in the matched comparison were identical:
`30930259c0875ba7ec4a286a139e011907254b7d567ed842cb06a13364d23977`.
This covers packed quantized weights, scales, and x86 corrections. All pack
validations passed without warnings. The focused GPTQ and rotation tests also
passed against the combined source variant (12 tests).

The separate 10 ms process-tree RSS probe includes compiler subprocesses. It
observed approximately **1,905 → 1,744 MiB** peak summed RSS. Compiler-child peak
RSS increased from **235 to 362 MiB** during compilation, but remained below the
later calibration peak. Summed RSS counts shared pages more than once and is
not a PSS/physical-memory measurement; sampling can miss brief peaks.

Reviewable patches:

- [Concurrent compilation](packing_candidates/compile_and_factors_attention.diff)
- [One factor task](packing_candidates/compile_and_factors_int8.diff)

## Promising, but changes calibration results

Calibration's `record()` still returns `F.linear(x, weight, bias)` rather than
using the existing native FP32 projection GEMM. Changing that one call reduced
an eight-thread probe from 5.23 to 4.88 s. A oneDNN probe reached 4.91 s.
However, the changed FP32 accumulation affects the subsequent Gram matrices and
GPTQ decisions. The built-in validation passed, but its relative L2 errors were:

| Calibration projection backend | Validation relative L2 |
|---|---:|
| Current `F.linear` | 0.04184 |
| Existing native GEMM | 0.04796 |
| oneDNN | 0.05896 |

Adding native calibration to the compilation/factor changes gave a 4.289 s
median over three runs, versus 5.246 s for HEAD (18.2% less time). This variant
also clears the native projection buffers after calibration. The numerical
change warrants retrieval-quality evaluation before adoption. It is not part
of the recommendation that preserves the prepared weights.

## Smaller or more involved opportunities

- **Diagonal-only damping:** `inverse_factor()` constructs a dense identity and
  its scaled copy to modify only the Hessian diagonal. Updating the diagonal
  directly removes those temporary matrices. A single probe showed no decisive
  overall gain; prioritize the two changes above. Its pack validation matched.
- **Cache compiled libraries:** the cold build remains a substantial cost, but
  persistent caching needs a deliberate design for invalidation, compiler flags,
  CPU ISA, ABI, permissions, and concurrent writers. It is a larger change than
  parallel compilation.
- **Single-thread packing:** the profile measured approximately 2.39 s compiling,
  16.15 s calibrating/quantizing, and 0.47 s validating. At eight threads those
  phases were 1.84 / 3.28 / 0.08 s. Phase wrappers overlap where work is concurrent;
  sums of individual factor timings must not be read as exclusive wall time.
  With affinity restricted to one core, the two recommended scheduling changes
  have little opportunity to help; larger gains require work on calibration or
  reuse of prepared state.

## Artifacts

- [Profiling harness](packing_profile.py), [1-thread profile](results/packing_profile_1t.json),
  [8-thread profile](results/packing_profile_8t.json)
- [Five individual probes and baseline](results/packing_candidates_8t_screen.jsonl)
- [Three-round scheduling/native-combination comparison](results/packing_candidates_8t_repeat.jsonl)
- [Two-round comparison with prepared-tensor digests](results/packing_candidates_safe_8t.jsonl)
- [Process-tree probe](packing_tree_probe.py), [HEAD result](results/packing_tree_baseline.json),
  [combined result](results/packing_tree_compile_and_factors.json)

The patch files apply to an extracted copy of `7e0c8c1`. Use
`packing_bench.py --sources label=/path/to/source ... --thread-counts 8
--precisions int8 --rounds 3 --digest --output /path/to/results.jsonl`
to reproduce; digest calculation runs outside the timed and memory-sampled region.
