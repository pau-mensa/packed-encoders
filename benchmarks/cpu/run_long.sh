#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
PE_PYTHON=${PE_PYTHON:-.venv/bin/python}
ORT_DIR=${ORT_DIR:?Set ORT_DIR to the ONNX model directory}
W8A8_MODEL=${W8A8_MODEL:-$ORT_DIR/int8_w8a8.onnx}
RESULT_DIR=${RESULT_DIR:-benchmarks/results/cpu}
LONG_CPUS=${LONG_CPUS:-0-15}
LONG_TRIALS=${LONG_TRIALS:-1}
LONG_PROFILES=${LONG_PROFILES:-"hf engine_fp32 engine_int8 ort_fp32 ort_w8a8"}
mkdir -p "$RESULT_DIR"

# These are actual complete 32-row batches, never extrapolated single-row runs.
# Warm one ORT row rather than an additional complete long batch.
for profile in $LONG_PROFILES; do
    case "$profile" in
        hf) options=(hf) ;;
        engine_fp32) options=(engine) ;;
        engine_int8) options=(engine --precision int8) ;;
        ort_fp32) options=(ort --model "$ORT_DIR/fp32_opt.onnx") ;;
        ort_w8a8) options=(ort --precision int8 --model "$W8A8_MODEL") ;;
        *) echo "Unknown profile: $profile" >&2; exit 2 ;;
    esac
    taskset -c "$LONG_CPUS" "$PE_PYTHON" benchmarks/cpu/compare.py "${options[@]}" \
        --threads 16 --trials "$LONG_TRIALS" --warmup 0 --ort-row-warmup \
        --only b32_s8192,b32_rand_pad8192 --out "$RESULT_DIR/${profile}_t16_8192.json"
done
"$PE_PYTHON" benchmarks/cpu/summarize.py --results "$RESULT_DIR"
