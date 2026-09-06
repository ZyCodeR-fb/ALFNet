#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec "${PYTHON:-python}" scripts/v4_launcher.py \
  --plan configs/paper/metrla_seed32_nine_models_plan.json "$@"
