#!/usr/bin/env bash
#
# The frozen experiment: ALFNet against source-native MoGERNN,
# 2 models x 3 datasets x 3 seeds = 18 training runs.
#
# The model and the protocol are frozen before this script runs. Nothing here
# selects anything: no candidate comparison, no hyper-parameter search, and no
# test-set inspection. `--mode test_async` is deliberately NOT launched from
# here -- the test split is spent once, and it is spent only after every
# training run has finished and been checked.
#
#   model    config family                     output dir
#   ALFNet   configs/ALFNet/gs{,_bay,_p8}.yaml        outputs/ALFNet/
#   MoGERNN  configs/MoGERNN/{METRLA,PEMSBAY,PEMS08}_lcbt_v3.yaml  outputs/MoGERNN/
#
# `configs/ALFNet/alfnet.yaml` is the paper-facing name for the METR-LA mainline
# and hashes to the same experiment as gs.yaml; the generated `gs` family is used
# here because tests/test_pemsbay_configs.py pins it as a cross-dataset twin set.
#
# Usage:
#   bash scripts/run_final_18.sh                 # run everything, 3 at a time
#   bash scripts/run_final_18.sh --dry-run       # print the commands only
#   JOBS=1 bash scripts/run_final_18.sh          # serial
#   MAX_EPOCH=1 SUFFIX=smoke bash scripts/run_final_18.sh   # pipeline smoke
#   bash scripts/run_final_18.sh --only alfnet   # or: mogernn, metrla, bay, pems08
#
# Resuming: a run whose checkpoints/training_summary.json exists is skipped, so
# re-running after an interruption continues where it stopped.
#
# Background:
#   nohup bash scripts/run_final_18.sh > final18.out 2>&1 &
#   tmux new -s final18 'bash scripts/run_final_18.sh'

set -euo pipefail

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEEDS="${SEEDS:-32 42 52}"
# One GPU. Three concurrent runs is the documented ceiling; wall-clock from a
# parallel run is not comparable across jobs and must not be quoted as speed.
JOBS="${JOBS:-3}"
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

LAUNCH_LOG_DIR="${OUTPUT_ROOT}/_launcher"
mkdir -p "${LAUNCH_LOG_DIR}"

# model | dataset | output subdir | config
MATRIX=(
  "alfnet|metrla|ALFNet|configs/ALFNet/gs.yaml"
  "alfnet|bay|ALFNet|configs/ALFNet/gs_bay.yaml"
  "alfnet|pems08|ALFNet|configs/ALFNet/gs_p8.yaml"
  "mogernn|metrla|MoGERNN|configs/MoGERNN/METRLA_lcbt_v3.yaml"
  "mogernn|bay|MoGERNN|configs/MoGERNN/PEMSBAY_lcbt_v3.yaml"
  "mogernn|pems08|MoGERNN|configs/MoGERNN/PEMS08_lcbt_v3.yaml"
)

matches_filter() {
  [[ -z "${ONLY}" ]] && return 0
  [[ "$1" == "${ONLY}" || "$2" == "${ONLY}" ]]
}

planned=0
skipped=0
launched=0
declare -a FAILED=()

run_one() {
  local model="$1" dataset="$2" subdir="$3" config="$4" seed="$5"
  local run_id="${model}-${dataset}-s${seed}${SUFFIX:+-${SUFFIX}}"
  local run_dir="${OUTPUT_ROOT}/${subdir}/${run_id}"
  local summary="${run_dir}/checkpoints/training_summary.json"

  if [[ -f "${summary}" ]]; then
    echo "-- skip ${run_id}: already finished"
    skipped=$((skipped + 1))
    return 0
  fi

  local cmd=("${PYTHON}" main.py -c "${config}"
             --batch_size "${BATCH_SIZE}"
             --seed "${seed}"
             --output_root "${OUTPUT_ROOT}"
             --run_id "${run_id}")
  [[ -n "${MAX_EPOCH}" ]] && cmd+=(--max_epoch "${MAX_EPOCH}")

  echo "-- ${run_id}"
  printf '   %s\n' "${cmd[*]}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  # main.py tees into <run>/logs/train_<stamp>.log once it starts. This
  # launcher log is separate on purpose: it captures a failure that happens
  # before the tee is installed, which is exactly when a config or provenance
  # error fires, and it keeps concurrent jobs from interleaving on one stream.
  local launch_log="${LAUNCH_LOG_DIR}/${run_id}.log"
  (
    if "${cmd[@]}" > "${launch_log}" 2>&1; then
      echo "== done ${run_id}"
    else
      echo "!! FAILED ${run_id} -- see ${launch_log}" >&2
      exit 1
    fi
  ) &
  launched=$((launched + 1))

  # Semaphore: never more than JOBS training processes on the one GPU.
  while (( $(jobs -rp | wc -l) >= JOBS )); do
    wait -n || true
  done
}

echo "=============================================================="
echo "final experiment  |  2 models x 3 datasets x 3 seeds"
echo "output root : ${OUTPUT_ROOT}"
echo "batch size  : ${BATCH_SIZE}    seeds: ${SEEDS}    parallel: ${JOBS}"
[[ -n "${MAX_EPOCH}" ]] && echo "max_epoch   : ${MAX_EPOCH}  (smoke, not a result)"
[[ -n "${ONLY}" ]] && echo "filter      : ${ONLY}"
echo "=============================================================="

for entry in "${MATRIX[@]}"; do
  IFS='|' read -r model dataset subdir config <<< "${entry}"
  matches_filter "${model}" "${dataset}" || continue
  for seed in ${SEEDS}; do
    planned=$((planned + 1))
    run_one "${model}" "${dataset}" "${subdir}" "${config}" "${seed}"
  done
done

# Collect every remaining job and remember which ones failed, rather than
# letting `set -e` abort on the first one: a failed seed should not throw away
# the fifteen runs that are still going.
status=0
while (( $(jobs -rp | wc -l) > 0 )); do
  wait -n || status=1
done
wait || status=1

echo
echo "=============================================================="
echo "planned ${planned} | skipped ${skipped} | launched ${launched}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry run: nothing was executed"
  exit 0
fi
if [[ "${status}" -ne 0 ]]; then
  echo "at least one run FAILED -- grep '!! FAILED' above, then re-run this"
  echo "script; finished runs are skipped."
  exit 1
fi
echo "all runs finished. Next, and only now:"
echo "  ${PYTHON} scripts/collect_lifecycle_results.py ${OUTPUT_ROOT}/ALFNet ${OUTPUT_ROOT}/MoGERNN --output-dir ${OUTPUT_ROOT}/reports"
echo "The test split is still unspent. Run --mode test_async only when the"
echo "training table has been checked and the paper is ready to consume it."
echo "=============================================================="
