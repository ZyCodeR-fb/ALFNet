#!/usr/bin/env bash
# 2026-11-10 chain (user-authorized): three-seed study of ALFNet, MCPST, MetaDG (plan configs/paper/seeds2_20261110)
# preflight (seed-32 replay, ALFNet epoch-0 reproduction, seed-42 smoke; <=5 shards) -> training queue (<=5)
# -> collect -> freeze test plan + one test per new checkpoint (<=5) -> overview workbook.
# Tests are opened only if every run completed with no failure, missing run or divergence flag.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/xd/anaconda3/envs/ReaCH/bin/python
DIR=artifacts/experiments/seeds2_20261110
mkdir -p "$DIR"
LOG="$DIR/chain.log"
step() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step "chain start pid $$"

step "seeds2: check + preflight + training queue"
$PY scripts/run_seeds2_20261110.py check >>"$LOG" 2>&1 || { step "seeds2 plan check FAILED; stop"; exit 1; }
$PY -u scripts/run_seeds2_20261110.py queue >>"$LOG" 2>&1 || { step "seeds2 queue FAILED or preflight failed; stop before test"; exit 1; }
if ! $PY - <<'EOF' >>"$LOG" 2>&1
import json, sys
from pathlib import Path
p = json.load(open('configs/paper/seeds2_20261110/plan.json'))
c = json.load(open(Path('artifacts/experiments/seeds2_20261110') / p['plan_sha256'] / 'COMPLETE.json'))
print('seeds2 COMPLETE.json', c)
sys.exit(0 if c['passed'] else 1)
EOF
then
  step "seeds2 training not clean (missing/failed/flagged); stop before any test access"
  exit 1
fi

step "seeds2 test: freeze + queue"
$PY scripts/test_seeds2_20261110.py freeze >>"$LOG" 2>&1 || { step "seeds2 test freeze FAILED; stop"; exit 1; }
$PY -u scripts/test_seeds2_20261110.py queue >>"$LOG" 2>&1 || { step "seeds2 test queue FAILED; stop"; exit 1; }
step "seeds2 overview report"
$PY scripts/report_seeds_overview2_20261110.py >>"$LOG" 2>&1 || { step "seeds2 report FAILED"; exit 1; }
step "chain COMPLETE"
