#!/usr/bin/env bash
# 2026-11-10 recovery chain (user-authorized "冻结 10% 恢复方案"): after the base queue of plan d6f66d20 has
# finished, rerun the MetaDG runs that failed by the 1% guard limit under recovery plan 694a8c24 (limit 10%),
# then freeze the merged test plan, test each new checkpoint once and write the overview.
# The base chain (alfnet_seeds2_chain) stops before any test because its COMPLETE.json misses those runs.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/xd/anaconda3/envs/ReaCH/bin/python
BASE=artifacts/experiments/seeds2_20261110/d6f66d2073190a15ed4b8e9c6dd7a07dca5c2050c381c18257de4630fd67d06b
DIR=artifacts/experiments/seeds2_recovery_metadg_20261110
mkdir -p "$DIR"
LOG="$DIR/chain.log"
step() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step "recovery chain start pid $$"
$PY scripts/recover_seeds2_metadg_20261110.py check >>"$LOG" 2>&1 || { step "recovery plan check FAILED; stop"; exit 1; }

step "waiting for the base queue ($BASE)"
while true; do
  state=$($PY - "$BASE" <<'EOF'
import json, os, sys
q = json.load(open(os.path.join(sys.argv[1], 'queue_status.json')))
pending = any(j['status'] in ('queued', 'running') for j in q['jobs'])
print('done' if not pending else ('alive' if os.path.exists(f"/proc/{q['queue_pid']}") else 'dead'))
EOF
)
  [ "$state" = done ] && break
  [ "$state" = dead ] && { step "base queue process is gone with jobs pending; stop (inspect $BASE/queue_status.json)"; exit 1; }
  sleep 300
done
for i in $(seq 1 60); do [ -f "$BASE/COMPLETE.json" ] && break; sleep 30; done
[ -f "$BASE/COMPLETE.json" ] || { step "base COMPLETE.json missing after the queue finished; stop"; exit 1; }
step "base queue finished: $(cat $BASE/COMPLETE.json | tr -d '\n' | cut -c1-300)"

step "recovery: run (targets decided now) + collect"
$PY -u scripts/recover_seeds2_metadg_20261110.py run >>"$LOG" 2>&1 || { step "recovery run FAILED; stop before test"; exit 1; }
if ! $PY - <<'EOF' >>"$LOG" 2>&1
import json, sys
p = json.load(open('configs/paper/seeds2_recovery_metadg_20261110/plan.json'))
c = json.load(open(f"artifacts/experiments/seeds2_recovery_metadg_20261110/{p['plan_sha256']}/COMPLETE.json"))
print('recovery COMPLETE.json', c)
sys.exit(0 if c['passed'] else 1)
EOF
then
  step "recovery not clean (missing/failed/flagged/not covered); stop before any test access"
  exit 1
fi

step "seeds2 test: freeze (merged) + queue"
$PY scripts/test_seeds2_20261110.py freeze >>"$LOG" 2>&1 || { step "seeds2 test freeze FAILED; stop"; exit 1; }
$PY -u scripts/test_seeds2_20261110.py queue >>"$LOG" 2>&1 || { step "seeds2 test queue FAILED; stop"; exit 1; }
step "seeds2 overview report"
$PY scripts/report_seeds_overview2_20261110.py >>"$LOG" 2>&1 || { step "seeds2 report FAILED"; exit 1; }
step "recovery chain COMPLETE"
