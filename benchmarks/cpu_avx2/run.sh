#!/usr/bin/env bash
# Reproduce every file in results/ (about 2.5 hours on an i9-13900K), then regenerate summary.md.
#
#   PE_PYTHON=/path/to/python ORT_PYTHON=/path/to/python benchmarks/cpu_avx2/run.sh [section ...]
#
# PE_PYTHON runs HF and the engine. ORT_PYTHON also needs onnxruntime (measured: 1.31.0), onnx
# and onnxscript next to the same torch and transformers; one interpreter can serve both.
# Sections, in order: sweep omp ablation onnx_export accuracy onnx summary (default: all).
# Close other heavy work first: every cell is a wall-clock median.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
PE=${PE_PYTHON:-python}
ORT=${ORT_PYTHON:-$PE}
R=$here/results
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=
SHORT=b1_s16_pad32,b1_s32,b8_s32
LONG=b32_s1024,b32_rand_pad1024
SINGLE=b1_s1024,b1_s4096
sections=${*:-sweep omp ablation onnx_export accuracy onnx summary}
mkdir -p "$R"

for section in $sections; do
  echo "== $section"
  case $section in
    sweep)
      "$PE" "$here/bench.py" --threads 1 --out "$R/sweep_t1.json" --only $SHORT,b1_s1024 --profile
      for t in 4 8 16 24 32; do
        "$PE" "$here/bench.py" --threads $t --out "$R/sweep_t$t.json" --only $SHORT,$SINGLE,b32_rand_pad1024 --profile
      done ;;
    omp)
      for t in 16 32; do
        for policy in default ACTIVE PASSIVE; do
          env $([ $policy = default ] || echo OMP_WAIT_POLICY=$policy) "$PE" "$here/bench.py" --threads $t \
            --only b1_s32,b8_s32 --trials 15 --profile --out "$R/omp_t${t}_$(echo $policy | tr A-Z a-z).json"
        done
      done ;;
    ablation)
      for t in 1 8 16; do
        only=b1_s32,b8_s32,b1_s1024,b32_rand_pad1024; [ $t = 1 ] && only=b1_s32,b8_s32,b1_s1024
        for mode in native sdpa; do
          "$PE" "$here/attention_ablation.py" --threads $t --mode $([ $mode = sdpa ] && echo sdpa_attn || echo native) \
            --tag $mode --only $only --out "$R/abl_${mode}_t$t.json"
        done
      done ;;
    onnx_export)
      "$ORT" "$here/onnx_export.py" ;;
    accuracy)
      "$ORT" "$here/onnx_accuracy.py" --out "$R/acc_ort.json" --models fp32_opt,int8,int8_pc
      "$ORT" "$here/onnx_accuracy.py" --out "$R/acc_ort_dyn.json" --models int8_lin,int8_lin_u8,int8_rr
      "$ORT" "$here/onnx_accuracy.py" --out "$R/acc_ort_proj.json" --models int8_proj,int8_proj_pc
      "$ORT" "$here/onnx_accuracy.py" --out "$R/acc_ort_w8.json" --models int8_w8,int8_w8a8 ;;
    onnx)
      ort() { "$ORT" "$here/onnx_bench.py" "$@"; }
      for t in 8 16 1; do
        "$PE" "$here/bench.py" --threads $t --out "$R/hf_engine_t${t}_short.json" --only $SHORT --trials 20
        ort --threads $t --out "$R/ort_t${t}_short.json" --only $SHORT --models fp32_opt,int8$([ $t = 16 ] || echo ,int8_pc) \
          --spin 1,0 --trials 20
      done
      for t in 16 8; do
        "$PE" "$here/bench.py" --threads $t --out "$R/hf_engine_t${t}_long.json" --only $LONG,$SINGLE --trials 3
        ort --threads $t --out "$R/ort_t${t}_long.json" --only $LONG --models fp32_opt,int8 --spin 1 --trials 3
        ort --threads $t --out "$R/ort_t${t}_single.json" --only $SINGLE --models fp32_opt,int8 --spin 1,0 --trials 5
      done
      for t in 8 1; do
        ort --threads $t --out "$R/ort_dyn_t${t}_short.json" --only $SHORT --models int8_lin,int8_lin_u8,int8_rr \
          --spin 1,0 --trials 20
        ort --threads $t --out "$R/ort_proj_t${t}_short.json" --only $SHORT --models int8_proj,int8_proj_pc --spin 1 --trials 20
      done
      for t in 8 1 16; do
        ort --threads $t --out "$R/ort_w8_t${t}_short.json" --only $SHORT --models int8_w8,int8_w8a8 --spin 1 --trials 20
      done
      ort --threads 16 --out "$R/ort_dyn_t16_long.json" --only $LONG --models int8_lin,int8_lin_u8 --spin 1 --trials 3
      ort --threads 16 --out "$R/ort_proj_t16_long.json" --only $LONG --models int8_proj --spin 1 --trials 3
      ort --threads 16 --out "$R/ort_w8_t16_long.json" --only $LONG --models int8_w8,int8_w8a8 --spin 1 --trials 3 ;;
    summary)
      python3 "$here/summarize.py" > "$R/summary.md" ;;
    *)
      echo "unknown section $section" >&2; exit 2 ;;
  esac
done
echo "RUN DONE"
