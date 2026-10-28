#!/usr/bin/env bash
# Run from the repository root on an Apple SME2 Mac.
set -euo pipefail

PE_PYTHON=${PE_PYTHON:-.venv/bin/python}
RESULT_DIR=${RESULT_DIR:-hardware-results/m4}

"$PE_PYTHON" benchmarks/cpu/hardware_check.py --expect sme2 --out "$RESULT_DIR/hardware.json"
"$PE_PYTHON" benchmarks/cpu/packing.py --out "$RESULT_DIR/packing.json"

for threads in 1 8; do
  for profile in hf engine_fp32 engine_int8; do
    system=engine
    precision=fp32
    case "$profile" in
      hf) system=hf ;;
      engine_int8) precision=int8 ;;
    esac
    "$PE_PYTHON" benchmarks/cpu/compare.py "$system" \
      --threads "$threads" --precision "$precision" \
      --only b1_s16_pad32,b1_s32,b8_s32,b1_s1024 \
      --warmup 1 --trials 7 --out "$RESULT_DIR/${profile}_t${threads}.json"
  done
done
