#!/usr/bin/env bash
# The twelve pre-registered v4 development-base runs.
#
#   JOBS=3 DEVICES=0,1,2 bash scripts/run_v4_development_base.sh
#   JOBS=1 DEVICES=0     bash scripts/run_v4_development_base.sh --dry-run
#   bash scripts/run_v4_development_base.sh --status
#
# A thin wrapper over scripts/v4_launcher.py, which does the work. The four
# things that decide whether this queue is trustworthy -- an atomic per-run lock,
# a bounded pool with a phase barrier, forwarding a signal to the trainer's
# process *group*, and a failure that propagates rather than being swallowed --
# are all things shell does badly, so they live in Python.
#
# Environment:
#   JOBS                how many runs at once (default 1)
#   DEVICES             comma-separated GPUs, e.g. 0,1,2 (default 0)
#   ALLOW_GPU_SHARING   set to 1 to permit JOBS > number of distinct devices
#   PYTHON              interpreter to use (default `python`)
#
# Activate the project environment first (`conda activate ReaCH`), or point
# PYTHON at its interpreter; this does not guess which environment you meant.
#
# Nothing here may be edited between runs: the plan is registered and hashed,
# and train_v4.py refuses to start if the plan or any config it names has moved.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec "${PYTHON:-python}" scripts/v4_launcher.py \
  --jobs "${JOBS:-1}" \
  --devices "${DEVICES:-0}" \
  ${ALLOW_GPU_SHARING:+--allow-gpu-sharing} \
  "$@"
