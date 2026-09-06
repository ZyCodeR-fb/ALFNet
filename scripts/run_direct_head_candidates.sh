#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python}
exec "$PYTHON" scripts/v4_launcher.py \
  --plan configs/paper/direct_head_candidate_plan.json "$@"
