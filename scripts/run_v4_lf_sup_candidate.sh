#!/usr/bin/env bash
# Run exactly one frozen PEMS08/seed32 validation-only candidate.
# Usage: bash scripts/run_v4_lf_sup_candidate.sh BASE 0

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--status" ]]; then
  find outputs_v4/lf_sup_candidates -type f \
    \( -name COMPLETE -o -name last.pt -o -name FAILED.json \) \
    -printf '%p\n' 2>/dev/null | sort || true
  exit 0
fi

CANDIDATE="${1:-}"
DEVICE="${2:-0}"
case "$CANDIDATE" in
  BASE|LF|SUP|LF_SUP)
    CONFIG="configs/paper/v4_candidates/PEMS08/${CANDIDATE}_ALFNet.yaml"
    MODEL="ALFNet"
    ;;
  CONTROL)
    CONFIG="configs/paper/v4_candidates/PEMS08/CONTROL_MoGERNN.yaml"
    MODEL="MoGERNN"
    ;;
  *)
    echo "candidate must be one of BASE LF SUP LF_SUP CONTROL" >&2
    exit 2
    ;;
esac

PLAN="configs/paper/v4_lf_sup_candidate_plan.json"
OUTPUT="outputs_v4/lf_sup_candidates"
LOG_DIR="$OUTPUT/logs"
mkdir -p "$LOG_DIR"

python scripts/build_v4_lf_sup_plan.py --check

CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print substr($1,1,16)}')"
RUN_DIR="$OUTPUT/PEMS08/$MODEL/$CONFIG_SHA/seed32"
EXTRA=()
if [[ -f "$RUN_DIR/COMPLETE" ]]; then
  echo "$CANDIDATE already COMPLETE at $RUN_DIR; refusing to overwrite"
  exit 0
elif [[ -f "$RUN_DIR/last.pt" ]]; then
  EXTRA+=(--resume)
  echo "resuming $CANDIDATE from $RUN_DIR/last.pt"
elif [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "$CANDIDATE has an incomplete non-resumable directory: $RUN_DIR" >&2
  exit 1
fi

echo "starting $CANDIDATE on cuda:$DEVICE; log $LOG_DIR/${CANDIDATE}.log"
python scripts/train_v4.py \
  --config "$CONFIG" \
  --device "cuda:$DEVICE" \
  --seed 32 \
  --output_root "$OUTPUT" \
  --plan "$PLAN" \
  "${EXTRA[@]}" 2>&1 | tee -a "$LOG_DIR/${CANDIDATE}.log"
