#!/usr/bin/env bash
#
# Spend the test split. Once.
#
# One `--mode test_async` per checkpoint, over exactly the 18 runs in
# configs/paper/final_runs.json -- the same list training and aggregation use.
#
# It refuses to start unless scripts/audit_final_18.py passes, because the test
# split cannot be un-spent: evaluating an incomplete or mixed matrix wastes the
# only untouched data the paper has. Override with --skip-audit only if you have
# just run the audit yourself and know why it failed.
#
# Usage:
#   bash scripts/run_final_eval_18.sh --dry-run    # print the plan, run nothing
#   bash scripts/run_final_eval_18.sh              # evaluate all 18
#   bash scripts/run_final_eval_18.sh --only alfnet-metrla-s32
#
# Serial by design. `test_async` runs at batch size 1 and the point here is a
# clean, reproducible pass, not throughput; concurrent evaluation on one GPU
# buys little and makes a failure harder to attribute.
#
# Resuming: a run whose result set is already complete and protocol-consistent
# is skipped, so an interrupted pass continues where it stopped. A partial or
# doubly-evaluated directory is refused, never overwritten.
#
#   tmux new -s eval18 'bash scripts/run_final_eval_18.sh'

set -uo pipefail

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
DRY_RUN=0
ONLY=""
SKIP_AUDIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    --skip-audit) SKIP_AUDIT=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

STATUS_DIR="${OUTPUT_ROOT}/_launcher/eval"
mkdir -p "${STATUS_DIR}"

echo "=============================================================="
echo "final evaluation  |  test_async over the frozen 18"
echo "output root : ${OUTPUT_ROOT}"
echo "=============================================================="

if [[ "${SKIP_AUDIT}" -eq 0 ]]; then
  echo "-- gate: scripts/audit_final_18.py"
  if ! "${PYTHON}" scripts/audit_final_18.py --output-root "${OUTPUT_ROOT}"; then
    echo
    echo "AUDIT FAILED -- refusing to spend the test split." >&2
    echo "Fix the training matrix first, or pass --skip-audit deliberately." >&2
    exit 1
  fi
  echo
fi

PLAN="$("${PYTHON}" scripts/final_runs.py --emit-plan)" || {
  echo "cannot read configs/paper/final_runs.json" >&2; exit 2; }

declare -a EVALUATED=()
declare -a SKIPPED=()
declare -a REFUSED=()
declare -a FAILED=()
planned=0

while IFS=$'\t' read -r run_id config seed subdir model batch_size; do
  [[ -z "${run_id}" ]] && continue
  [[ -n "${ONLY}" && "${run_id}" != "${ONLY}" ]] && continue
  planned=$((planned + 1))

  run_path="${OUTPUT_ROOT}/${subdir}/${run_id}"
  # Already complete and consistent? Then this is a resume, not a re-spend.
  if "${PYTHON}" scripts/verify_eval_result.py --only "${run_id}" \
       --output-root "${OUTPUT_ROOT}" --quiet >/dev/null 2>&1; then
    echo "-- skip   ${run_id}: result already complete"
    SKIPPED+=("${run_id}")
    continue
  fi
  # A directory that exists but does not verify is a partial or double
  # evaluation. Overwriting it would hide which protocol produced what.
  if [[ -d "${run_path}/predictions" ]] && \
     compgen -G "${run_path}/predictions/lifecycle-v3-*" >/dev/null; then
    echo "!! REFUSE ${run_id}: an incomplete or duplicated result set exists" >&2
    echo "   ${PYTHON} scripts/verify_eval_result.py --only ${run_id}" >&2
    REFUSED+=("${run_id}")
    continue
  fi

  # The exact training flags, plus the mode. Anything else and the canonical
  # config hash will not resolve the checkpoint.
  cmd=("${PYTHON}" main.py -c "${config}"
       --batch_size "${batch_size}"
       --seed "${seed}"
       --mode test_async
       --output_root "${OUTPUT_ROOT}"
       --run_id "${run_id}")

  echo "-- eval   ${run_id}"
  printf '          %s\n' "${cmd[*]}"
  [[ "${DRY_RUN}" -eq 1 ]] && continue

  launch_log="${STATUS_DIR}/${run_id}.log"
  status_file="${STATUS_DIR}/${run_id}.status"
  rm -f "${status_file}"
  "${cmd[@]}" > "${launch_log}" 2>&1
  rc=$?
  echo "${rc}" > "${status_file}"

  if [[ "${rc}" -ne 0 ]]; then
    echo "!! FAILED ${run_id} (exit ${rc}) -- ${launch_log}" >&2
    FAILED+=("${run_id} (exit ${rc})")
    continue
  fi
  # Exit 0 is not proof the tables were written. Verify the artefacts.
  if ! "${PYTHON}" scripts/verify_eval_result.py --only "${run_id}" \
         --output-root "${OUTPUT_ROOT}" --quiet >/dev/null 2>&1; then
    echo "!! FAILED ${run_id}: exited 0 but the result set does not verify" >&2
    FAILED+=("${run_id} (incomplete result)")
    continue
  fi
  echo "== done   ${run_id}"
  EVALUATED+=("${run_id}")
done <<< "${PLAN}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "planned ${planned} | would skip ${#SKIPPED[@]} | would refuse ${#REFUSED[@]}"
  echo "dry run: nothing was executed, no test data was read"
  [[ "${#REFUSED[@]}" -gt 0 ]] && exit 1
  exit 0
fi

echo
echo "=============================================================="
echo "planned ${planned} | skipped ${#SKIPPED[@]} | evaluated ${#EVALUATED[@]}"
status=0
if [[ "${#REFUSED[@]}" -gt 0 ]]; then
  status=1
  echo "REFUSED (partial or duplicated result directory):"
  printf '  - %s\n' "${REFUSED[@]}"
fi
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  status=1
  echo "FAILED:"
  printf '  - %s\n' "${FAILED[@]}"
fi
if [[ "${status}" -ne 0 ]]; then
  echo
  echo "Re-run this script: complete results are skipped, failures retried."
  echo "Per-run logs: ${STATUS_DIR}/<run_id>.log"
  echo "=============================================================="
  exit 1
fi

echo
echo "All 18 evaluated. Aggregate the paper's tables with:"
echo "  ${PYTHON} scripts/collect_final_18.py --output-dir ${OUTPUT_ROOT}/reports/final"
echo "=============================================================="
