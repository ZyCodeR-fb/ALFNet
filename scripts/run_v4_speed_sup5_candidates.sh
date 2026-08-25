#!/usr/bin/env bash
# Six frozen seed32 validation-only speed SUP5 runs. Test is inaccessible.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec "${PYTHON:-python}" scripts/v4_launcher.py \
  --plan configs/paper/v4_speed_sup5_candidate_plan.json \
  --jobs "${JOBS:-1}" \
  --devices "${DEVICES:-0}" \
  ${ALLOW_GPU_SHARING:+--allow-gpu-sharing} \
  "$@"
