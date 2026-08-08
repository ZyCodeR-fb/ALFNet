#!/usr/bin/env bash
#
# The frozen experiment: ALFNet against source-native MoGERNN,
# 2 models x 3 datasets x 3 seeds = 18 training runs.
#
# The matrix is NOT defined here. It comes from configs/paper/final_runs.json
# via scripts/final_runs.py, which is the same list the audit, the evaluation
# and the aggregation read -- three hand-kept copies is three chances to
# disagree about which runs the paper is made of.
#
# The model and the protocols are frozen before this runs. Nothing here selects
# anything, and `--mode test_async` is deliberately not launched: the test split
# is spent once, by scripts/run_final_eval_18.sh, and only after
# scripts/audit_final_18.py passes.
#
# Usage:
#   bash scripts/run_final_18.sh                 # run everything, JOBS at a time
#   bash scripts/run_final_18.sh --dry-run       # print the plan only
#   JOBS=3 bash scripts/run_final_18.sh          # three at a time, after a smoke
#   MAX_EPOCH=1 SUFFIX=smoke bash scripts/run_final_18.sh   # pipeline smoke
#   bash scripts/run_final_18.sh --only alfnet   # or: mogernn, metrla, bay, pems08
#
# --only is a substring match on the run_id, so it also selects a seed slice --
# `--only s32` is the six-cell (2 models x 3 datasets) seed-32 plan used for the
# smoke below. There is deliberately no separate --seeds flag.
#   MAX_EPOCH=1 SUFFIX=smoke bash scripts/run_final_18.sh --only s32
#
# Resuming: a finished run is skipped only if it is verifiably *this* run --
# same model, variant, canonical config hash (which covers seed, dataset and
# every hyper-parameter), protocol hash, structural provenance and a completed
# stop reason. A directory that fails that check is reported and the script
# refuses to continue rather than silently reusing or overwriting it.
#
# Background:
#   tmux new -s final18 'bash scripts/run_final_18.sh'

set -uo pipefail

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
# One GPU. Serial by default: concurrency is a property of the machine, not of
# the experiment, so it is set explicitly after the smoke has shown the memory
# headroom. Wall-clock from a parallel run is not comparable across jobs and
# must never be quoted as a speed result.
JOBS="${JOBS:-1}"
MAX_EPOCH="${MAX_EPOCH:-}"
SUFFIX="${SUFFIX:-}"
DRY_RUN=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

STATUS_DIR="${OUTPUT_ROOT}/_launcher/train${SUFFIX:+-${SUFFIX}}"
mkdir -p "${STATUS_DIR}"

declare -A PID_RUN=()
declare -a LAUNCHED=()
declare -a SKIPPED=()
declare -a REFUSED=()

# --- reaping -----------------------------------------------------------------
#
# Each job writes its exit code to a status file before exiting. That file, not
# the shell's job table, is the source of truth: `wait -n` reaps an arbitrary
# child and the previous version threw its status away in the semaphore, so a
# run that failed while the queue was full could vanish and the script would
# still report success. Status files also survive a `wait` that races, and they
# are structured -- nothing here greps a log.

reap_finished() {
  local reaped=1 pid rc run_id
  for pid in "${!PID_RUN[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      run_id="${PID_RUN[${pid}]}"
      rc=0
      wait "${pid}" 2>/dev/null || rc=$?
      unset 'PID_RUN[${pid}]'
      reaped=0
      if [[ "${rc}" -eq 0 ]]; then
        echo "== done   ${run_id}"
      else
        echo "!! FAILED ${run_id} (exit ${rc})" >&2
      fi
    fi
  done
  return "${reaped}"
}

wait_for_slot() {
  while (( ${#PID_RUN[@]} >= JOBS )); do
    reap_finished || sleep 3
  done
}

drain() {
  while (( ${#PID_RUN[@]} > 0 )); do
    reap_finished || sleep 3
  done
}

# --- the plan ----------------------------------------------------------------

PLAN="$("${PYTHON}" scripts/final_runs.py --emit-plan)" || {
  echo "cannot read configs/paper/final_runs.json" >&2; exit 2; }

echo "=============================================================="
echo "final experiment  |  frozen matrix from configs/paper/final_runs.json"
echo "output root : ${OUTPUT_ROOT}"
echo "parallel    : ${JOBS}"
[[ -n "${MAX_EPOCH}" ]] && echo "max_epoch   : ${MAX_EPOCH}  (SMOKE -- not a result)"
[[ -n "${SUFFIX}"    ]] && echo "suffix      : ${SUFFIX}  (separate run_id, separate directory)"
[[ -n "${ONLY}"      ]] && echo "filter      : ${ONLY}"
echo "=============================================================="

planned=0
while IFS=$'\t' read -r run_id config seed subdir model batch_size; do
  [[ -z "${run_id}" ]] && continue
  if [[ -n "${ONLY}" && "${run_id}" != *"${ONLY}"* && "${model,,}" != "${ONLY,,}" ]]; then
    continue
  fi
  planned=$((planned + 1))

  # A smoke never claims a real run's identity: its run_id, directory and
  # status files all carry the suffix, so the frozen scripts cannot mistake it
  # for a finished result.
  effective_id="${run_id}${SUFFIX:+-${SUFFIX}}"
  run_path="${OUTPUT_ROOT}/${subdir}/${effective_id}"

  # Reuse only what is verifiably this run. A truncated epoch count means the
  # smoke's own output is never reusable either, which is the point.
  if [[ -f "${run_path}/checkpoints/training_summary.json" ]]; then
    if [[ -n "${SUFFIX}" || -n "${MAX_EPOCH}" ]]; then
      # A smoke's own leftovers are never reusable -- truncated epochs are not
      # a result -- so say so plainly instead of running the identity check.
      echo "!! REFUSE ${effective_id}: smoke output already exists; delete it" >&2
      REFUSED+=("${effective_id}")
      continue
    fi
    "${PYTHON}" scripts/audit_final_18.py --only "${run_id}" --quiet \
      --output-root "${OUTPUT_ROOT}" >/dev/null 2>&1
    verdict=$?
    case "${verdict}" in
      0)
        echo "-- skip   ${effective_id}: verified complete"
        SKIPPED+=("${effective_id}")
        continue ;;
      1)
        echo "!! REFUSE ${effective_id}: a run directory exists but does not verify" >&2
        echo "   inspect it with: ${PYTHON} scripts/audit_final_18.py --only ${run_id}" >&2
        echo "   move it aside or delete it; this script will not overwrite it." >&2
        REFUSED+=("${effective_id}")
        continue ;;
      *)
        # The checker itself could not run. Refusing every run on that basis
        # would look like a corrupt matrix; stop instead and say what it is.
        echo "!! ABORT: the verifier could not run (exit ${verdict})." >&2
        echo "   ${PYTHON} scripts/audit_final_18.py --only ${run_id} --output-root ${OUTPUT_ROOT}" >&2
        echo "   Usually PYTHON is not the project environment. Set PYTHON=... or activate it." >&2
        exit 2 ;;
    esac
  fi

  cmd=("${PYTHON}" main.py -c "${config}"
       --batch_size "${batch_size}"
       --seed "${seed}"
       --output_root "${OUTPUT_ROOT}"
       --run_id "${effective_id}")
  [[ -n "${MAX_EPOCH}" ]] && cmd+=(--max_epoch "${MAX_EPOCH}")

  echo "-- run    ${effective_id}"
  printf '          %s\n' "${cmd[*]}"
  [[ "${DRY_RUN}" -eq 1 ]] && continue

  wait_for_slot
  launch_log="${STATUS_DIR}/${effective_id}.log"
  status_file="${STATUS_DIR}/${effective_id}.status"
  rm -f "${status_file}"
  (
    "${cmd[@]}" > "${launch_log}" 2>&1
    rc=$?
    echo "${rc}" > "${status_file}"
    # `exit "${rc}"` is load-bearing: without it the subshell's status is the
    # status of the `echo`, which is 0 whatever the run did, so `wait` in
    # reap_finished reported every failure as "done". The status file caught it
    # in the final summary, but the live output lied for the whole run.
    exit "${rc}"
  ) &
  PID_RUN[$!]="${effective_id}"
  LAUNCHED+=("${effective_id}")
done <<< "${PLAN}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "planned ${planned} | would skip ${#SKIPPED[@]} | would refuse ${#REFUSED[@]}"
  echo "dry run: nothing was executed"
  [[ "${#REFUSED[@]}" -gt 0 ]] && exit 1
  exit 0
fi

drain

# --- summary, from the status files ------------------------------------------

declare -a FAILED=()
declare -a MISSING=()
for run_id in "${LAUNCHED[@]}"; do
  status_file="${STATUS_DIR}/${run_id}.status"
  if [[ ! -f "${status_file}" ]]; then
    # No status file means the job died without running its trailer -- killed,
    # OOM-reaped, or the machine went down. That is a failure, not a success.
    MISSING+=("${run_id}")
    continue
  fi
  rc="$(cat "${status_file}")"
  [[ "${rc}" != "0" ]] && FAILED+=("${run_id} (exit ${rc})")
done

echo
echo "=============================================================="
echo "planned ${planned} | skipped ${#SKIPPED[@]} | launched ${#LAUNCHED[@]}"
echo "succeeded $(( ${#LAUNCHED[@]} - ${#FAILED[@]} - ${#MISSING[@]} ))"
status=0
if [[ "${#REFUSED[@]}" -gt 0 ]]; then
  status=1
  echo "REFUSED (existing directory did not verify):"
  printf '  - %s\n' "${REFUSED[@]}"
fi
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  status=1
  echo "FAILED:"
  printf '  - %s\n' "${FAILED[@]}"
fi
if [[ "${#MISSING[@]}" -gt 0 ]]; then
  status=1
  echo "NO STATUS (killed before it could report -- treat as failed):"
  printf '  - %s\n' "${MISSING[@]}"
fi

if [[ "${status}" -ne 0 ]]; then
  echo
  echo "Re-run this script: verified runs are skipped, failed ones are retried."
  echo "Per-run logs: ${STATUS_DIR}/<run_id>.log"
  echo "=============================================================="
  exit 1
fi

echo
if [[ -n "${SUFFIX}" || -n "${MAX_EPOCH}" ]]; then
  echo "Smoke finished. These directories are NOT results; delete them when done:"
  echo "  ${OUTPUT_ROOT}/*/*-${SUFFIX:-smoke}"
else
  echo "All 18 training runs finished. Next step is the audit, which reads no"
  echo "test data:"
  echo "  ${PYTHON} scripts/audit_final_18.py"
  echo "Only if it passes:"
  echo "  bash scripts/run_final_eval_18.sh"
fi
echo "=============================================================="
