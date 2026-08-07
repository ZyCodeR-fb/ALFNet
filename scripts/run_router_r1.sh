#!/usr/bin/env bash
#
# Current-fixed-protocol router diagnostic (r1), METR-LA, seed 32.
#
# This is NOT a reproduction of the historical fg1-s32 result. Those numbers
# were produced under training protocol 59c55e6e..., before the strict-inductive
# and MoGERNN batch-adapter corrections. Everything launched here runs under
# 59c9ad67... and is comparable only with the other runs in this script.
#
# What it answers, by holding everything but the graph router fixed:
#   fg1  noisy_topk  the historical path; its router receives exactly zero
#                    forecast gradient at moge_top_k=1, so it is included as a
#                    reference point, not as a working mechanism
#   gs   soft        dense softmax over all five supports, conditioned on state
#   g1   st_top1     hard top-1 forward, straight-through gradient
#   gm   static      one learned mixture for every node, no state input --
#                    the control that isolates whether conditioning helps
#   mg               source-native MoGERNN baseline
#
# Validation only. No test_async here: the test split is spent once.
#
# Usage:
#   bash scripts/run_router_r1.sh              # run everything
#   bash scripts/run_router_r1.sh --dry-run    # print the commands only
#   MAX_EPOCH=1 bash scripts/run_router_r1.sh  # quick pipeline check
#
# Background:
#   nohup bash scripts/run_router_r1.sh > r1.out 2>&1 &
#   tmux new -s r1 'bash scripts/run_router_r1.sh'

set -euo pipefail

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SUFFIX="${SUFFIX:-r1-s32}"
MAX_EPOCH="${MAX_EPOCH:-}"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

# variant : config
JOBS=(
  "fg1:configs/ALFNet/fg1.yaml"
  "gs:configs/ALFNet/gs.yaml"
  "g1:configs/ALFNet/g1.yaml"
  "gm:configs/ALFNet/gm.yaml"
  "mg:configs/MoGERNN/METRLA_lcbt_v3.yaml"
)

echo "=============================================================="
echo "router diagnostic r1  |  METR-LA  |  seed 32  |  validation only"
echo "output root : ${OUTPUT_ROOT}"
echo "batch size  : ${BATCH_SIZE}"
echo "jobs        : ${#JOBS[@]}"
echo "=============================================================="

failed=0
for job in "${JOBS[@]}"; do
  variant="${job%%:*}"
  config="${job#*:}"
  run_id="${variant}-${SUFFIX}"

  # Serial and fail-fast: a later run must not quietly proceed on a broken
  # protocol or a half-written manifest.
  cmd=("${PYTHON}" main.py -c "${config}"
       --batch_size "${BATCH_SIZE}"
       --output_root "${OUTPUT_ROOT}"
       --run_id "${run_id}")
  if [[ -n "${MAX_EPOCH}" ]]; then
    cmd+=(--max_epoch "${MAX_EPOCH}")
  fi

  summary="${OUTPUT_ROOT}/$( [[ "${variant}" == "mg" ]] && echo MoGERNN || echo ALFNet )/${run_id}/checkpoints/training_summary.json"
  if [[ -f "${summary}" ]]; then
    echo "-- ${run_id}: already finished, skipping (${summary})"
    continue
  fi

  echo
  echo "-- ${run_id}"
  printf '   %s\n' "${cmd[*]}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    continue
  fi

  # main.py already tees stdout and stderr into <run>/logs/train_<stamp>.log,
  # so nothing needs redirecting here to keep the record.
  if ! "${cmd[@]}"; then
    echo "!! ${run_id} FAILED -- stopping" >&2
    failed=1
    break
  fi
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "dry run: nothing was executed"
  exit 0
fi

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

echo
echo "=============================================================="
echo "all runs finished. compare them with:"
echo "  ${PYTHON} scripts/validation_curve.py ${OUTPUT_ROOT}/ALFNet/gs-${SUFFIX}"
echo "per-run routing tables are written automatically to"
echo "  ${OUTPUT_ROOT}/<model>/<run_id>/reports/router_report.txt"
echo "=============================================================="
