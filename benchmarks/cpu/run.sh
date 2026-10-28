#!/usr/bin/env bash
set -euo pipefail

# The interpreter must have torch, transformers, onnxruntime, onnx and onnxscript.
# ORT_DIR contains fp32_opt.onnx and int8_w8a8.onnx from cpu/onnx_export.py.
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
PE_PYTHON=${PE_PYTHON:-.venv/bin/python}
ORT_DIR=${ORT_DIR:?Set ORT_DIR to the exported ONNX model directory}
RESULT_DIR=${RESULT_DIR:-benchmarks/results/cpu}
SHORT_CPUS=${SHORT_CPUS:-0-7}
LONG_CPUS=${LONG_CPUS:-0-15}
mkdir -p "$RESULT_DIR"

for profile in hf engine_fp32 engine_native engine_int8 ort_fp32 ort_w8a8 ort_w8a8_nospin; do
    options=()
    case "$profile" in
        hf) options=(hf) ;;
        engine_fp32) options=(engine) ;;
        engine_native) options=(engine --backend native) ;;
        engine_int8) options=(engine --precision int8) ;;
        ort_fp32) options=(ort --model "$ORT_DIR/fp32_opt.onnx") ;;
        ort_w8a8) options=(ort --precision int8 --model "$ORT_DIR/int8_w8a8.onnx") ;;
        ort_w8a8_nospin) options=(ort --precision int8 --model "$ORT_DIR/int8_w8a8.onnx" --spin 0) ;;
    esac
    taskset -c "$SHORT_CPUS" "$PE_PYTHON" benchmarks/cpu/compare.py "${options[@]}" \
        --threads 8 --trials 7 --out "$RESULT_DIR/${profile}_t8_short.json"
    if [[ "$profile" == engine_native || "$profile" == ort_w8a8_nospin ]]; then
        continue
    fi
    taskset -c "$LONG_CPUS" "$PE_PYTHON" benchmarks/cpu/compare.py "${options[@]}" \
        --threads 16 --trials 3 --only b32_s1024,b32_rand_pad1024 \
        --out "$RESULT_DIR/${profile}_t16_long.json"
done
"$PE_PYTHON" benchmarks/cpu/summarize.py --results "$RESULT_DIR"
