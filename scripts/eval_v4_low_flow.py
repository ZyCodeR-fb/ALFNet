#!/usr/bin/env python
"""Evaluate frozen PEMS flow checkpoints by low-flow bin.

Validation is the default and the only split allowed for model selection.
``--split test`` is retrospective diagnosis on an already-open supportive test
set and requires an explicit acknowledgement.  The output is no-clobber and
never changes the 33-cell headline table.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.v4_lifecycle_loader import V4LifecycleDataset
from experiments.v4_low_flow_evaluator import LowFlowCollector
from experiments.v4_runner import build_v4_session, to_device
from lib.lcbt_v4 import build_v4_teacher_forcing_mask
from lib.runtime_context import build_runtime_context
from scripts.eval_v4_plan_validation import (
    assert_identity,
    atomic_write_json,
    expected_identity,
    load_plan,
    resolve_runs,
)
from scripts.v4_utils import sha256_file
from utils.model_output import unpack_model_output


OUTPUT_NAMES = {"validation": "v4_low_flow_validation.json",
                "test": "v4_low_flow_test_retrospective.json"}


def _load_retrospective_archive(path):
    """Read a self-consistent archived result plan without pretending code froze."""
    from scripts.build_v4_five_dataset_plan import plan_hash

    path = Path(path)
    if "configs/paper/archive" not in path.resolve().as_posix():
        raise ValueError("retrospective plan must live under configs/paper/archive")
    plan = json.loads(path.read_text("utf-8"))
    if plan.get("plan_sha256") != plan_hash(plan):
        raise ValueError("archived plan hash does not verify")
    return plan


def _assert_retrospective_identity(checkpoint, run, session, plan):
    """Old schema identity, before the named seed contract existed."""
    expected = expected_identity(run, session, plan)
    legacy_fields = tuple(key for key in expected if key != "seed_contract")
    differing = {key: (expected[key], checkpoint.get(key)) for key in legacy_fields
                 if checkpoint.get(key) != expected[key]}
    if differing:
        raise ValueError(f"{run.name} legacy checkpoint identity differs: {differing}")
    declared_sha = run.entry.get("checkpoint_sha256")
    actual_sha = sha256_file(run.directory / "best.pt")
    if declared_sha and actual_sha != declared_sha:
        raise ValueError(f"{run.name} checkpoint SHA differs from archived plan")


@torch.no_grad()
def evaluate(run, plan, split="validation", device="cuda:0", retrospective=False):
    if run.dataset not in ("PEMS03", "PEMS04", "PEMS08"):
        raise ValueError("low-flow evaluation accepts flow datasets only")
    checkpoint_path = run.directory / "best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    session = build_v4_session(run.config, device=device, run_seed=run.seed)
    if retrospective:
        _assert_retrospective_identity(checkpoint, run, session, plan)
    else:
        assert_identity(checkpoint, expected_identity(run, session, plan), run.name)
    session.model.load_state_dict(checkpoint["model_state"])
    session.model.to(device).eval()

    context = (session.validation_context if split == "validation"
               else build_runtime_context(run.dataset, "test", seed=run.seed))
    scaler = dict(checkpoint["scaler"])
    dataset = V4LifecycleDataset(
        context, ROOT / session.args.data_path,
        input_len=session.args.input_len, pred_len=session.args.pred_len,
        slide_step=session.args.slide_step, freq=session.args.freq,
        seed=session.seed_contract["derived"]["train_schedule"], epoch=0,
        validation_seed=session.args.lcbt_validation_seed, scaler=scaler)
    loader = DataLoader(dataset, batch_size=session.train_loader.batch_size,
                        shuffle=False, num_workers=0)
    collector = LowFlowCollector(context, scaler, session.args.pred_len)
    epoch = int(checkpoint["best_epoch"])
    for batch in loader:
        batch = to_device(batch, device)
        forced = build_v4_teacher_forcing_mask(
            batch, session.seed_contract["derived"]["teacher_forcing"],
            epoch, False)
        if bool(forced.any()):
            raise ValueError("low-flow validation/test must be free-running")
        prediction, _ = unpack_model_output(
            session.model.forward_v4(batch, context, epoch=epoch,
                                     teacher_forcing_mask=None))
        collector.update(batch, prediction)
    report = collector.report()
    report.update({
        "run": run.name, "dataset": run.dataset, "model": run.model,
        "seed": run.seed, "run_seed_used": session.run_seed,
        "evaluation_seed_contract": session.seed_contract,
        "checkpoint_seed_contract": checkpoint.get("seed_contract"),
        "legacy_checkpoint_fixed_init_seed": 1234 if retrospective else None,
        "retrospective_archive": bool(retrospective),
        "plan_sha256": plan["plan_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": epoch,
        "teacher_forcing_positions": 0,
        "selection_use": ("validation_diagnostic_only" if split == "validation"
                          else "forbidden_retrospective_test"),
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run", required=True,
                        help="for example PEMS08/ALFNet/seed32")
    parser.add_argument("--split", choices=("validation", "test"),
                        default="validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--allow-retrospective-test", action="store_true")
    parser.add_argument("--retrospective-archive", action="store_true",
                        help="diagnose an immutable pre-seed-contract result plan")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_retrospective_test:
        raise SystemExit("--split test requires --allow-retrospective-test")
    plan = (_load_retrospective_archive(args.plan)
            if args.retrospective_archive else load_plan(args.plan))
    runs, _ = resolve_runs(plan, args.output_root)
    matches = [run for run in runs if run.name == args.run]
    if len(matches) != 1:
        raise SystemExit(f"expected one run named {args.run!r}, got {len(matches)}")
    report = evaluate(matches[0], plan, args.split, args.device,
                      retrospective=args.retrospective_archive)
    path = matches[0].directory / OUTPUT_NAMES[args.split]
    atomic_write_json(path, report)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
