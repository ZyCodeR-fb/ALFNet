#!/usr/bin/env bash
# 2026-11-06 chain (user-authorized): batch-2 extension test (MCPST/MetaDG, plan 60b911dd)
# -> three-seed training of the ten comparison methods (plan 58eecfb5; superseded 839232c9, 5e70abfa; <=4 concurrent)
# -> freeze + one test per new checkpoint -> mean/std summary.
# The seed study starts after the batch-2 test whatever its outcome (independent runs);
# it is tested only if every run completed with no failure or divergence flag.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/xd/anaconda3/envs/ReaCH/bin/python
DIR=artifacts/experiments/seeds_20261106
mkdir -p "$DIR"
LOG="$DIR/chain.log"
step() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step "chain start pid $$"

step "batch-2 test: check + queue"
if ls artifacts/evaluation/extension2_test_20261105/*/COMPLETE.json >/dev/null 2>&1; then
  step "batch-2 test already COMPLETE (restarted chain); not re-run"
elif $PY scripts/test_extension2_20261105.py check >>"$LOG" 2>&1 && \
   $PY -u scripts/test_extension2_20261105.py queue >>"$LOG" 2>&1; then
  step "batch-2 test COMPLETE"
else
  step "batch-2 test FAILED (see its ATTENTION_REQUIRED.json); continuing with the independent seed study"
fi

step "seeds: check + preflight + training queue"
$PY scripts/run_seeds_20261106.py check >>"$LOG" 2>&1 || { step "seeds plan check FAILED; stop"; exit 1; }
$PY -u scripts/run_seeds_20261106.py queue >>"$LOG" 2>&1 || { step "seeds queue FAILED; stop before test"; exit 1; }
if ! $PY - <<'EOF' >>"$LOG" 2>&1
import json, sys
from pathlib import Path
p = json.load(open('configs/paper/seeds_20261106/plan.json'))
c = json.load(open(Path('artifacts/experiments/seeds_20261106') / p['plan_sha256'] / 'COMPLETE.json'))
print('seeds COMPLETE.json', c)
sys.exit(0 if c['passed'] else 1)
EOF
then
  step "seeds training not clean (missing/failed/flagged); stop before any test access"
  exit 1
fi

step "seeds test: freeze + queue"
$PY scripts/test_seeds_20261106.py freeze >>"$LOG" 2>&1 || { step "seeds test freeze FAILED; stop"; exit 1; }
$PY -u scripts/test_seeds_20261106.py queue >>"$LOG" 2>&1 || { step "seeds test queue FAILED; stop"; exit 1; }
step "seeds report"
$PY scripts/report_seeds_20261106.py >>"$LOG" 2>&1 || { step "seeds report FAILED"; exit 1; }
step "chain COMPLETE"
