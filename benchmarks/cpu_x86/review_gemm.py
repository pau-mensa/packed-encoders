"""Alternating original/revised GEMM in the real model, on identical live weights.

Pin externally: taskset -c 2 (1t) or taskset -c 0-7 (8t) on the measured 9950X.
python benchmarks/cpu_x86/review_gemm.py --threads 1 [--rehome] [--rounds 15]
Add --onnx fp32_opt.onnx for a same-process, one-thread ORT comparison.
The reference source snapshots the uncommitted kernel at the start of this review.
No weight packing or different accumulation order is involved. Each result must
match bitwise. Run benchmarks serially to avoid CPU/memory-bandwidth contention.
"""
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import torch
from transformers import AutoModel

import packed_encoders as pe
from packed_encoders.cpu.attention import _openmp_runtime
from bench import inputs, workloads


def compile_reference(source, directory):
    wrapper = Path(directory) / "reference.c"
    wrapper.write_text(source.read_text() + '''
int pe_linear_tile(const float *x, const float *w, const float *b, float *y,
                   int M, int N, int K, int workers, int rows, int cols) {
    (void)rows; (void)cols;
    return pe_linear(x, w, b, y, M, N, K, workers);
}
''')
    library_path = Path(directory) / "reference.so"
    subprocess.run(["gcc", "-O3", "-march=native", "-fPIC", "-fopenmp", "-ffp-contract=fast",
                    "-Wall", "-Wextra", "-Werror", "-shared", str(wrapper), _openmp_runtime(),
                    "-o", str(library_path)], check=True, capture_output=True)
    library = ctypes.CDLL(str(library_path))
    library.pe_linear_tile.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6
    library.pe_linear_tile.restype = ctypes.c_int
    return library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, choices=(1, 8), required=True)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--rehome", action="store_true")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).parent / "native_probe/fp32_review_baseline.c")
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least 2")
    if args.onnx and args.threads != 1:
        parser.error("same-process ORT comparison requires --threads 1 (avoids competing idle thread pools)")
    torch.set_num_threads(args.threads)
    model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa",
                                      local_files_only=True).float().eval()
    pe.pack(model, engine=pe.ModernBertCPU(threads=args.threads, rehome_weights=args.rehome))
    attention = pe.get_engine(model).attention
    if not attention.native_linear:
        raise RuntimeError("this benchmark requires the native AVX-512 FP32 GEMM")
    revised = attention.rows.pe_linear_tile
    session = None
    if args.onnx:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    with tempfile.TemporaryDirectory(prefix="gemm-review-") as directory, torch.inference_mode():
        library = compile_reference(args.reference, directory)
        variants = {"original": library.pe_linear_tile, "revised": revised}
        if session:
            variants["onnx"] = None
        for name in ("b1_s16_pad32", "b1_s32", "b8_s32"):
            seq, lengths = workloads()[name]
            ids, mask = inputs(seq, lengths)
            feeds = {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
            times = {variant: [] for variant in variants}
            outputs = {}
            reps = 3 if args.threads == 1 and name == "b8_s32" else 10
            order = list(variants)
            for round_ in range(args.rounds + 1):
                for variant in order if round_ % 2 == 0 else order[::-1]:
                    if variant != "onnx":
                        attention.rows.pe_linear_tile = variants[variant]
                    start = time.perf_counter()
                    for _ in range(reps):
                        out = (session.run(None, feeds)[0] if variant == "onnx" else
                               model(ids, attention_mask=mask).last_hidden_state)
                    elapsed = (time.perf_counter() - start) * 1000 / reps
                    if round_:
                        times[variant].append(elapsed)
                    outputs[variant] = torch.as_tensor(out).clone()
                assert torch.equal(outputs["original"], outputs["revised"]), name
            result = {variant: {"median_ms": statistics.median(samples), "round_ms": samples}
                      for variant, samples in times.items()}
            if session:
                live = mask.bool()
                result["onnx"]["relative_l2_vs_original"] = float(
                    (outputs["onnx"][live] - outputs["original"][live]).norm() / outputs["original"][live].norm())
            print(json.dumps({"threads": args.threads, "rehome": args.rehome, "workload": name,
                              "bitwise_equal": True, "results": result}), flush=True)
    attention.rows.pe_linear_tile = revised


if __name__ == "__main__":
    main()
