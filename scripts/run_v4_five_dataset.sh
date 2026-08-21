#!/usr/bin/env bash
# The thirty pre-registered five-dataset runs.
#
#   JOBS=1 DEVICES=0 bash scripts/run_v4_five_dataset.sh
#   JOBS=3 DEVICES=0,0,0 ALLOW_GPU_SHARING=1 bash scripts/run_v4_five_dataset.sh
#   bash scripts/run_v4_five_dataset.sh --dry-run
#   bash scripts/run_v4_five_dataset.sh --status
#
# The run list comes from configs/paper/v4_five_dataset_plan.json, not from any
# constant in the launcher. The plan is hashed and the launcher refuses to start
# if it, any config it names, or any source it registered has moved.
#
# Activate the project environment first (`conda activate ReaCH`), or set PYTHON.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec "${PYTHON:-python}" scripts/v4_launcher.py \
  --plan configs/paper/v4_five_dataset_plan.json \
  --jobs "${JOBS:-1}" \
  --devices "${DEVICES:-0}" \
  ${ALLOW_GPU_SHARING:+--allow-gpu-sharing} \
  "$@"
