#!/usr/bin/env bash
# Run the v4 development-base queue inside tmux, so it survives a dropped SSH
# connection and can be watched without being attached to.
#
#   JOBS=3 DEVICES=0,1,2 bash scripts/tmux_v4_development_base.sh start
#   bash scripts/tmux_v4_development_base.sh attach     # Ctrl-b d to leave it running
#   bash scripts/tmux_v4_development_base.sh status
#   bash scripts/tmux_v4_development_base.sh stop       # SIGINT; last.pt is kept
#   bash scripts/tmux_v4_development_base.sh start --dry-run
#
# `stop` sends SIGINT to the launcher rather than killing the tmux server: the
# launcher forwards it to every trainer's process group, waits, and leaves each
# `last.pt` in place, so the queue can be resumed with the same `start`.
#
# Touches nothing outside tmux and the run directory -- no systemd, no proxy, no
# shell startup files.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-v4-dev-base}"
PYTHON_BIN="${PYTHON:-python}"
ACTION="${1:-status}"
shift || true

plan_hash() {
  "$PYTHON_BIN" - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.build_v4_development_plan import assert_plan_still_holds, load_plan
print(assert_plan_still_holds(load_plan())["plan_sha256"])
PY
}

launcher_command() {
  echo "JOBS=${JOBS:-1} DEVICES=${DEVICES:-0} ${ALLOW_GPU_SHARING:+ALLOW_GPU_SHARING=1} PYTHON=${PYTHON_BIN} bash scripts/run_v4_development_base.sh $*"
}

case "$ACTION" in
  start)
    if [[ "${1:-}" == "--dry-run" ]]; then
      HASH="$(plan_hash)"
      echo "session   ${SESSION}"
      echo "console   outputs_v4/_launcher/${HASH}/console.log"
      echo "launcher  $(launcher_command)"
      echo "monitor   watch -n 30 '$(launcher_command --status)'"
      echo "(dry run: nothing started; tmux not required)"
      exit 0
    fi
    command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed" >&2; exit 1; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "session ${SESSION} already exists; refusing to start a second queue." >&2
      echo "  attach:  bash scripts/tmux_v4_development_base.sh attach" >&2
      echo "  stop:    bash scripts/tmux_v4_development_base.sh stop" >&2
      exit 1
    fi
    HASH="$(plan_hash)"
    CONSOLE_DIR="outputs_v4/_launcher/${HASH}"
    mkdir -p "$CONSOLE_DIR"
    CONSOLE="${CONSOLE_DIR}/console.log"
    echo "starting ${SESSION}; console tees to ${CONSOLE}"
    # `tee -a` on the launcher window so the central console survives the
    # session and is readable without attaching.
    tmux new-session -d -s "$SESSION" -n launcher \
      "cd '$ROOT' && $(launcher_command "$@") 2>&1 | tee -a '$CONSOLE'; echo; echo '--- launcher exited, exit=' \$?; exec bash"
    tmux new-window -t "$SESSION" -n monitor \
      "cd '$ROOT' && while true; do clear; date; echo; $(launcher_command --status); echo; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null; echo; tail -n 12 '$CONSOLE' 2>/dev/null; sleep 30; done"
    tmux select-window -t "$SESSION":launcher
    sleep 2
    tmux list-windows -t "$SESSION"
    echo "attach with: bash scripts/tmux_v4_development_base.sh attach"
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
      bash scripts/run_v4_development_base.sh --status
    ;;

  stop)
    PIDS="$(pgrep -f 'scripts/v4_launcher.py' || true)"
    if [[ -z "$PIDS" ]]; then
      echo "no launcher process found"
    else
      for PID in $PIDS; do
        echo "SIGINT -> launcher ${PID}"
        kill -INT "$PID"
      done
      echo "waiting for trainers to stop; each last.pt is preserved"
      for _ in $(seq 1 60); do
        pgrep -f 'scripts/v4_launcher.py' >/dev/null 2>&1 || break
        sleep 2
      done
    fi
    echo "resume with the same start command; COMPLETE runs are skipped"
    ;;

  *)
    echo "usage: $0 {start|attach|status|stop} [--dry-run]" >&2
    exit 2
    ;;
esac
