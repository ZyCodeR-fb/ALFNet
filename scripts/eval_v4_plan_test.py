#!/usr/bin/env python
"""The final test evaluator. Implemented and frozen; not run in this round.

The confirmation splits are read once. Everything below exists so that the once
happens deliberately, after the evidence that it is time, and never twice under
different code.

**Eight gates, all before a single test target is read.** The plan must not have
drifted; the readiness attestation must permit *this* plan; every result slot
must be present; the twelve reused checkpoints must still pass their identity
audit; the eighteen new runs must be COMPLETE; every slot's validation must have
been independently recomputed and agree; the validation summary must exist; and
no config, model or code hash may have moved since that summary was written.

**The scaler comes from the checkpoint.** Refitting on test would make the
normalisation a function of the split being scored, which is the one thing a
held-out split must never be.

**Teacher forcing is zero and asserted**, not merely left unset.

Access is recorded once and is resumable only under identical identity:
`TEST_OPENED.json` on the first run, per-run results that are never clobbered,
`TEST_COMPLETE` when all slots are done.

    python scripts/eval_v4_plan_test.py --plan configs/paper/v4_five_dataset_plan.json
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v4_runner import build_v4_session, to_device  # noqa: E402
from experiments.v4_validation_collector import (  # noqa: E402
    ValidationCellCollector, cell_count,
)
from lib.lcbt_v4 import build_v4_teacher_forcing_mask  # noqa: E402
from lib.runtime_context import build_runtime_context  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402
from scripts.eval_v4_plan_validation import (  # noqa: E402
    EVAL_NAME as VALIDATION_EVAL_NAME, IDENTITY_FIELDS, TOLERANCE,
    assert_identity, load_plan, resolve_runs,
)
from scripts.train_v4 import COMPLETE_MARKER  # noqa: E402

TEST_SCHEMA = 1
TEST_EVAL_NAME = "plan_test_eval.json"
OPENED_MARKER = "TEST_OPENED.json"
COMPLETE_TEST_MARKER = "TEST_COMPLETE"


class TestAccessRefused(RuntimeError):
    """The confirmation splits stay closed."""


def access_identity(plan):
    """What must be identical for a resumed test run to be the same run."""
    from scripts.build_v4_readiness import load as load_attestation

    return {
        "plan_sha256": plan["plan_sha256"],
        "attestation_sha256": load_attestation()["attestation_sha256"],
        "configs": {name: entry["sha256"]
                    for name, entry in plan["configs"].items()},
        "data": {name: entry["payload_sha256"]
                 for name, entry in plan["data"].items()},
        "code_sha256": plan["code_sha256"],
        "evaluator_sha256": sha256_file(Path(__file__)),
    }


def assert_may_open(plan, runs, root):
    """The eight gates. Nothing below reads a test target until these pass."""
    from scripts.build_v4_readiness import ReadinessRefused, assert_ready
    from scripts.collect_v4_plan_validation import SUMMARY_NAME
    problems = []
    try:                                             # 1, 2
        assert_ready(plan)
    except ReadinessRefused as error:
        problems.append(f"readiness: {error}")
    if len(runs) != plan["trial_budget"]:            # 3
        problems.append(f"{len(runs)} slots, plan declares {plan['trial_budget']}")

    reused = [run for run in runs if run.execution_action == "reuse"]
    if reused:                                       # 4
        problems.append("the canonical reproduction forbids reused checkpoints")

    incomplete = [run.name for run in runs                     # 5
                  if not (run.directory / COMPLETE_MARKER).exists()]
    if incomplete:
        problems.append(f"{len(incomplete)} slot(s) not COMPLETE: "
                        f"{incomplete[:4]}")

    missing_validation = []                                                # 6
    for run in runs:
        path = run.directory / VALIDATION_EVAL_NAME
        if not path.exists():
            missing_validation.append(run.name)
            continue
        record = json.loads(path.read_text("utf-8"))
        if (not record.get("agrees")
                or record.get("partial_diagnostic_only")
                or float(record.get("max_abs_cell_difference", 1.0)) > TOLERANCE
                or int(record.get("run_seed_used", -1)) != int(run.seed)):
            missing_validation.append(run.name)
    if missing_validation:
        problems.append(f"{len(missing_validation)} slot(s) have no agreeing "
                        f"own-seed validation recomputation: {missing_validation[:4]}")

    if not (root / "plan_validation_summary.json").exists():               # 7
        problems.append("the validation summary has not been produced")

    for name, digest in plan["code_sha256"].items():                       # 8
        path = ROOT / name
        if not path.exists() or sha256_file(path) != digest:
            problems.append(f"{name} has changed since the plan was frozen")

    if problems:
        raise TestAccessRefused(
            "the confirmation splits stay closed:\n  - " + "\n  - ".join(problems))
    return True


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    os.close(handle)
    try:
        Path(temporary).write_text(text, "utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def open_test_access(plan, root):
    """Record the opening once; a resume must present the same identity."""
    marker = Path(root) / OPENED_MARKER
    identity = access_identity(plan)
    if marker.exists():
        recorded = json.loads(marker.read_text("utf-8"))
        if recorded.get("identity") != identity:
            raise TestAccessRefused(
                f"{marker} was written under a different identity; the test "
                "split may not be reopened under changed code, configs or data")
        return recorded
    record = {"schema_version": TEST_SCHEMA, "identity": identity,
              "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "note": "the confirmation splits were opened once, under this identity"}
    _atomic_write(marker, json.dumps(record, indent=1, sort_keys=True,
                                     allow_nan=False) + "\n")
    return record


@torch.no_grad()
def _test_table(session, context, checkpoint, epoch):
    """The 33 cells on the test split, with the checkpoint's scaler."""
    from data_provider.v4_lifecycle_loader import V4LifecycleDataset
    from torch.utils.data import DataLoader
    from utils.model_output import unpack_model_output

    scaler = dict(checkpoint["scaler"])
    dataset = V4LifecycleDataset(context, ROOT / session.args.data_path,
                                 input_len=session.args.input_len,
                                 pred_len=session.args.pred_len,
                                 slide_step=session.args.slide_step,
                                 freq=session.args.freq,
                                 seed=session.run_seed, epoch=0, scaler=scaler,
                                 validation_seed=session.args.lcbt_validation_seed)
    if dataset.scaler is not scaler:
        raise TestAccessRefused("the test dataset refitted a scaler")
    loader = DataLoader(dataset, batch_size=session.train_loader.batch_size,
                        shuffle=False, num_workers=0)
    # The same accumulator and the same frozen aggregation as validation. The
    # phase is named explicitly, so every place that reads test is greppable.
    collector = ValidationCellCollector(context, scaler, allow_phase="test")

    session.model.eval()
    for batch in loader:
        batch = to_device(batch, session.device)
        forced = build_v4_teacher_forcing_mask(batch, session.run_seed, epoch, False)
        if bool(forced.any()):
            raise TestAccessRefused("teacher forcing is not zero on test")
        prediction, _ = unpack_model_output(
            session.model.forward_v4(batch, context, epoch=epoch,
                                     teacher_forcing_mask=None))
        collector.update(batch, prediction)
    return collector.table()


def evaluate_run(run, plan, device="cuda:0"):
    directory = run.directory
    path = directory / TEST_EVAL_NAME
    if path.exists():
        return json.loads(path.read_text("utf-8"))          # no-clobber, skip

    checkpoint = torch.load(directory / "best.pt", weights_only=False,
                            map_location="cpu")
    session = build_v4_session(run.config, device=device, run_seed=run.seed)
    if session.run_seed != run.seed:
        raise TestAccessRefused(f"{run.name}: session seed is not {run.seed}")
    from scripts.eval_v4_plan_validation import expected_identity

    assert_identity(checkpoint, expected_identity(run, session, plan), run.name)
    context = build_runtime_context(run.dataset, "test", seed=run.seed)
    if context.phase != "test":
        raise TestAccessRefused("the test context did not resolve")
    if context.static_asset_hash() != checkpoint["static_asset_hash"]:
        raise TestAccessRefused(f"{run.name}: test static assets differ")

    session.model.load_state_dict(checkpoint["model_state"])
    session.model.to(device)
    table = _test_table(session, context, checkpoint, int(checkpoint["best_epoch"]))
    if cell_count(table) != 33:
        raise TestAccessRefused(f"{run.name}: {cell_count(table)} cells")

    record = {
        "schema_version": TEST_SCHEMA,
        "split": "test", "read_once": True,
        "plan_sha256": plan["plan_sha256"],
        "run": run.name, "dataset": run.dataset, "model": run.model,
        "seed": run.seed, "run_seed_used": session.run_seed,
        "execution_action": run.execution_action,
        "exposure_status": plan["data"][run.dataset]["test_exposure"]["status"],
        "unit": plan["data"][run.dataset]["unit"],
        "best_epoch": int(checkpoint["best_epoch"]),
        "scaler": {"mean": float(checkpoint["scaler"]["mean"]),
                   "std": float(checkpoint["scaler"]["std"]),
                   "source": "checkpoint", "refit": False},
        "teacher_forcing_positions": 0,
        "test_exposure_hash": context.exposure_hash(),
        "checkpoint_sha256": sha256_file(directory / "best.pt"),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "cells": table,
    }
    _atomic_write(path, json.dumps(record, indent=1, sort_keys=True,
                                   allow_nan=False) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--i-understand-this-reads-the-confirmation-splits",
                        dest="confirmed", action="store_true",
                        help="required; the test split is read once")
    arguments = parser.parse_args(argv)

    plan = load_plan(arguments.plan)
    runs, root = resolve_runs(plan, arguments.output_root)
    assert_may_open(plan, runs, root)
    if not arguments.confirmed:
        raise SystemExit(
            "every gate passes, but the confirmation splits are read once. "
            "Re-run with --i-understand-this-reads-the-confirmation-splits")
    open_test_access(plan, root)
    for run in runs:
        record = evaluate_run(run, plan, arguments.device)
        print(f"TEST {run.name:26s} {record['exposure_status']:12s} "
              f"seed={record['run_seed_used']}", flush=True)
    _atomic_write(Path(root) / COMPLETE_TEST_MARKER,
                  json.dumps({"slots": len(runs),
                              "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                             indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
