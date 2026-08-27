#!/usr/bin/env bash
# Active unified-BASE launcher. Activate ReaCH first.
# Examples:
#   JOBS=5 DEVICES=0,0,0,0,0 ALLOW_GPU_SHARING=1 bash scripts/run_experiments.sh
#   bash scripts/run_experiments.sh --model ALFNet --dataset PEMS08 --seed 32
#   bash scripts/run_experiments.sh --model STGCN --seed 32 --seed 42 --seed 52
#   bash scripts/run_experiments.sh --status --model ALFNet
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "${PYTHON:-python}" scripts/v4_launcher.py \
  --plan configs/paper/final_plan.json \
  --jobs "${JOBS:-1}" --devices "${DEVICES:-0}" \
  ${ALLOW_GPU_SHARING:+--allow-gpu-sharing} "$@"
