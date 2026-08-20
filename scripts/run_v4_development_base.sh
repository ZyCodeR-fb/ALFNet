#!/usr/bin/env bash
# The twelve pre-registered v4 development-base runs, in one fixed queue.
#
# Serial by design: one GPU, and an ALFNet run must not share it with another
# ALFNet run. Serial also means the queue order cannot become a hidden variable
# -- every run gets the same machine in the same state.
#
# Resumable: a run whose directory is marked COMPLETE is skipped, so the script
# can be re-invoked after an interruption without re-running finished work and
# without ever overwriting it. `train_v4.py` refuses to clobber a COMPLETE run
# regardless, so the skip here is a convenience, not the safety property.
#
#   bash scripts/run_v4_development_base.sh --dry-run
#   nohup bash scripts/run_v4_development_base.sh > logs/dev_base.out 2>&1 &
#
# Nothing here may be edited between runs. The plan is registered and hashed;
# `train_v4.py` recomputes it and refuses to start if the plan or any config it
# names has moved.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-cuda:0}"
DATASETS=(PEMS08 PEMSBAY)
MODELS=(ALFNet MoGERNN)
SEEDS=(32 42 52)
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

PLAN_HASH="$(python - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.build_v4_development_plan import assert_plan_still_holds, load_plan
print(assert_plan_still_holds(load_plan())["plan_sha256"])
PY
)"
OUTPUT_ROOT="outputs_v4/development_base/${PLAN_HASH}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"

echo "plan       ${PLAN_HASH}"
echo "output     ${OUTPUT_ROOT}"
echo "device     ${DEVICE}"
echo "queue      ${#DATASETS[@]} datasets x ${#MODELS[@]} models x ${#SEEDS[@]} seeds"
echo

# Seed-outermost on purpose: the four seed-32 runs form the health phase, so a
# problem specific to one dataset or model surfaces after roughly one run each
# rather than after every PEMS08 seed has finished. Their numbers decide
# nothing -- the queue does not branch on them.
INDEX=0
for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      INDEX=$((INDEX + 1))
      CONFIG="configs/paper/v4/${DATASET}_${MODEL}.yaml"
      CONFIG_HASH="$(sha256sum "$CONFIG" | cut -c1-16)"
      RUN_DIR="${OUTPUT_ROOT}/${DATASET}/${MODEL}/${CONFIG_HASH}/seed${SEED}"
      LOG="${LOG_DIR}/${DATASET}_${MODEL}_seed${SEED}.log"
      COMMAND="python scripts/train_v4.py --config ${CONFIG} --device ${DEVICE} --seed ${SEED} --output_root ${OUTPUT_ROOT} --development_plan"

      if [[ -f "${RUN_DIR}/COMPLETE" ]]; then
        echo "[${INDEX}/12] SKIP  ${DATASET} ${MODEL} seed${SEED}  (COMPLETE)"
        continue
      fi
      if [[ $DRY_RUN -eq 1 ]]; then
        echo "[${INDEX}/12] PLAN  ${DATASET} ${MODEL} seed${SEED}"
        echo "            dir ${RUN_DIR}"
        echo "            cmd ${COMMAND}"
        continue
      fi
      echo "[${INDEX}/12] START ${DATASET} ${MODEL} seed${SEED}  $(date -Is)"
      echo "            log ${LOG}"
      # `|| true` so one failure does not silently drop the ten runs behind it;
      # the failure is recorded and the queue reports it at the end.
      if $COMMAND > "$LOG" 2>&1; then
        echo "[${INDEX}/12] DONE  ${DATASET} ${MODEL} seed${SEED}  $(date -Is)"
      else
        echo "[${INDEX}/12] FAIL  ${DATASET} ${MODEL} seed${SEED}  see ${LOG}"
      fi
    done
  done
done

echo
echo "queue finished $(date -Is)"
COMPLETED="$(find "$OUTPUT_ROOT" -name COMPLETE 2>/dev/null | wc -l)"
echo "runs marked COMPLETE: ${COMPLETED}/12"
