#!/usr/bin/env bash
# Keep the five METR-LA validation-only candidates alive across SSH drops.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-v4-metrla-lf-sup-candidates}"
PYTHON_BIN="${PYTHON:-python}"
ACTION="${1:-status}"
shift || true

plan_hash() {
  "$PYTHON_BIN" - <<'PY'
from scripts.build_v4_metrla_lf_sup_plan import assert_plan_still_holds, load_plan
print(assert_plan_still_holds(load_plan())["plan_sha256"])
PY
}

launcher_command() {
  echo "JOBS=${JOBS:-1} DEVICES=${DEVICES:-0} ${ALLOW_GPU_SHARING:+ALLOW_GPU_SHARING=1} PYTHON=${PYTHON_BIN} bash scripts/run_v4_metrla_lf_sup_candidates.sh $*"
}

case "$ACTION" in
  start)
    command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed" >&2; exit 1; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "session ${SESSION} already exists; refusing a duplicate queue" >&2
      exit 1
    fi
    HASH="$(plan_hash)"
    CONSOLE_DIR="outputs_v4/_launcher/${HASH}"
    mkdir -p "$CONSOLE_DIR"
    CONSOLE="${CONSOLE_DIR}/console.log"
    tmux new-session -d -s "$SESSION" -n launcher \
      "cd '$ROOT' && $(launcher_command "$@") 2>&1 | tee -a '$CONSOLE'; echo; echo '--- launcher exited, exit=' \$?; exec bash"
    tmux new-window -t "$SESSION" -n monitor \
      "cd '$ROOT' && while true; do clear; date; echo; $(launcher_command --status); echo; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null; echo; tail -n 12 '$CONSOLE' 2>/dev/null; sleep 30; done"
    tmux select-window -t "$SESSION":launcher
    sleep 2
    tmux list-windows -t "$SESSION"
    echo "attach with: bash scripts/tmux_v4_metrla_lf_sup_candidates.sh attach"
    ;;
  attach)
    exec tmux attach -t "$SESSION"
    ;;
  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux session ${SESSION}: running"
      tmux list-windows -t "$SESSION"
    else
      echo "tmux session ${SESSION}: not running"
    fi
    echo
    JOBS="${JOBS:-1}" DEVICES="${DEVICES:-0}" PYTHON="$PYTHON_BIN" \
      bash scripts/run_v4_metrla_lf_sup_candidates.sh --status
    ;;
  stop)
    PIDS="$(pgrep -f 'scripts/v4_launcher.py.*v4_metrla_lf_sup_candidate_plan' || true)"
    if [[ -n "$PIDS" ]]; then
      for PID in $PIDS; do kill -INT "$PID"; done
    fi
    echo "candidate queue stop requested; last.pt remains resumable"
    ;;
  *)
    echo "usage: $0 {start|attach|status|stop}" >&2
    exit 2
    ;;
esac
