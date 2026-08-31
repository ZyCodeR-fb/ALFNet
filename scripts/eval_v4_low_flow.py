#!/usr/bin/env python
"""Evaluate frozen PEMS flow checkpoints by low-flow bin.

Validation is the default and the only split allowed for model selection.
``--split test`` is a post-freeze mechanism diagnosis and requires an explicit
acknowledgement. The output is no-clobber and never changes the 33-cell table.
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
    expected_identity as expected_plan_identity,
    load_plan,
    resolve_runs,
)
from scripts.v4_utils import sha256_file
from utils.model_output import unpack_model_output


OUTPUT_NAMES = {"validation": "v4_low_flow_validation.json",
                "test": "v4_low_flow_test.json"}


def expected_low_flow_identity(run, session, plan):
    """Resolve checkpoint identity without weakening candidate-plan checks.

    The final-plan schema indexes configs by ``dataset_model``.  The ALFNet
    development plan instead indexes seven immutable configs by candidate.
    Both schemas bind the same checkpoint fields; only the authority lookup is
    different.  Missing candidate metadata is a hard failure, never a fallback
    to the BASE config.
    """
    if plan.get("kind") != "v4_alfnet_validation_candidates":
        return expected_plan_identity(run, session, plan)
    candidate = run.entry.get("candidate")
    if not candidate or candidate not in plan.get("configs", {}):
        raise ValueError(
            f"{run.name}: candidate plan has no config authority for "
            f"{candidate!r}")
    entry = plan["configs"][candidate]
    data = plan["data"][run.dataset]
    return {
        "dataset": run.dataset,
        "model": run.model,
        "model_variant": getattr(session.args, "model_variant", None),
        "seed": run.seed,
        "config_sha256": entry["sha256"],
        "payload_sha256": data["payload_sha256"],
        "unseen_sha256": data["unseen_sha256"],
        "manifest_sha256": data["manifest_sha256"],
        "static_asset_hash": session.train_context.static_asset_hash(),
        "train_exposure_hash": session.train_context.exposure_hash(),
        "validation_exposure_hash": session.validation_context.exposure_hash(),
        "seed_contract": session.seed_contract,
    }


@torch.no_grad()
def evaluate(run, plan, split="validation", device="cuda:0"):
    if run.dataset not in ("PEMS03", "PEMS04", "PEMS08"):
        raise ValueError("low-flow evaluation accepts flow datasets only")
    checkpoint_path = run.directory / "best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    session = build_v4_session(run.config, device=device, run_seed=run.seed)
    identity = expected_low_flow_identity(run, session, plan)
    assert_identity(checkpoint, identity, run.name)
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
        "plan_sha256": plan["plan_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": epoch,
        "teacher_forcing_positions": 0,
        "selection_use": ("validation_diagnostic_only" if split == "validation"
                          else "post_freeze_test_diagnostic_only"),
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
    parser.add_argument("--allow-test-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test_diagnostic:
        raise SystemExit("--split test requires --allow-test-diagnostic")
    plan = load_plan(args.plan)
    runs, _ = resolve_runs(plan, args.output_root)
    matches = [run for run in runs if run.name == args.run]
    if len(matches) != 1:
        raise SystemExit(f"expected one run named {args.run!r}, got {len(matches)}")
    report = evaluate(matches[0], plan, args.split, args.device)
    path = matches[0].directory / OUTPUT_NAMES[args.split]
    atomic_write_json(path, report)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
