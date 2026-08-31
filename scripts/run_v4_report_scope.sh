#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON=${PYTHON:-python}
JOBS=${JOBS:-5}
DEVICE=${DEVICE:-cuda:0}
SCOPE=${SCOPE:-configs/paper/four_dataset_11_model_report_scope.json}
SCOPE_SHA=$($PYTHON -c 'from pathlib import Path; from scripts.v4_utils import sha256_file; import sys; print(sha256_file(Path(sys.argv[1])))' "$SCOPE")
LOG_ROOT="outputs_v4/_report_scope/${SCOPE_SHA}"
mkdir -p "$LOG_ROOT/validation" "$LOG_ROOT/test"

evaluate_one() {
    local phase=$1
    local run=$2
    local label=${run//\//_}
    "$PYTHON" scripts/eval_v4_report_scope.py \
        --scope "$SCOPE" --phase "$phase" --run "$run" --device "$DEVICE" \
        >"$LOG_ROOT/$phase/$label.log" 2>&1
    echo "[$(date '+%F %T')] DONE $phase $run"
}
export -f evaluate_one
export PYTHON DEVICE SCOPE LOG_ROOT

run_missing() {
    local phase=$1
    local list_phase="list-$phase"
    local list_path="$LOG_ROOT/${phase}_missing.txt"
    "$PYTHON" scripts/eval_v4_report_scope.py \
        --scope "$SCOPE" --phase "$list_phase" >"$list_path"
    local count
    count=$(wc -l <"$list_path")
    echo "[$(date '+%F %T')] $phase missing=$count jobs=$JOBS device=$DEVICE"
    if (( count > 0 )); then
        xargs -r -P "$JOBS" -I '{}' bash -c \
            'evaluate_one "$1" "$2"' _ "$phase" '{}' <"$list_path"
    fi
}

echo "[$(date '+%F %T')] scope=$SCOPE_SHA"
"$PYTHON" scripts/eval_v4_report_scope.py --scope "$SCOPE" --phase status
run_missing validation
"$PYTHON" scripts/eval_v4_report_scope.py --scope "$SCOPE" --phase prepare-test
run_missing test
"$PYTHON" scripts/eval_v4_report_scope.py --scope "$SCOPE" --phase complete-test
"$PYTHON" scripts/collect_v4_report_scope.py --scope "$SCOPE"
"$PYTHON" scripts/eval_v4_report_scope.py --scope "$SCOPE" --phase status
echo "[$(date '+%F %T')] REPORT_SCOPE_COMPLETE"
