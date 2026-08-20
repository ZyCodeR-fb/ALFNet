#!/usr/bin/env bash
# train -> audit -> validation eval -> collect, in that order, no step skipped.
#
#   JOBS=3 DEVICES=0,1,2 bash scripts/run_v4_development_pipeline.sh
#
# `set -e` and no `|| true` anywhere: a failing stage stops the pipeline rather
# than letting the next one aggregate whatever happens to be on disk. That is
# the whole point of having four stages instead of one script.
#
# Every stage is validation-only and none of them constructs a test context.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

echo "=== 1/4 train ==="
JOBS="${JOBS:-1}" DEVICES="${DEVICES:-0}" PYTHON="$PY" \
  bash scripts/run_v4_development_base.sh

echo "=== 2/4 audit ==="
"$PY" scripts/audit_v4_development_base.py

echo "=== 3/4 validation eval ==="
"$PY" scripts/v4_development_eval.py --device "cuda:${DEVICES%%,*}"

echo "=== 4/4 collect ==="
"$PY" scripts/collect_v4_development_base.py

echo "=== pipeline complete ==="
