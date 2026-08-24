#!/usr/bin/env python
"""Evaluate v4 k-study and PLES diagnostics from a frozen checkpoint.

The default is validation. Test requires an explicit acknowledgement and writes
a separate no-clobber file. This evaluator never participates in training,
early stopping, model selection or the 33-cell headline table.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from data_provider.v4_lifecycle_loader import V4LifecycleDataset
from experiments.v4_mechanism_evaluator import V4MechanismCollector
from experiments.v4_runner import build_v4_session, to_device
from lib.runtime_context import build_runtime_context
from scripts.v4_utils import sha256_file
from scripts.eval_v4_plan_validation import (
    assert_identity, expected_identity, load_plan, resolve_runs,
)
from utils.model_output import unpack_model_output

ROOT = Path(__file__).resolve().parents[1]


class PairedDataset(Dataset):
    def __init__(self, lifecycle_dataset):
        self.dataset = lifecycle_dataset
        self.indices = [index for index, entry in enumerate(lifecycle_dataset.schedule)
                        if entry["scenario"] in ("join", "fault")]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = self.indices[index]
        return {"treated": self.dataset[source],
                "control": self.dataset.counterfactual(source)}


def _atomic_json(path, payload):
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    if path.exists():
        if path.read_text("utf-8") == text:
            return path
        raise RuntimeError(f"{path} exists with different content")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    os.close(handle)
    try:
        Path(temporary).write_text(text, "utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


@torch.no_grad()
def evaluate(run, plan, split="validation", device="cuda:0"):
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    checkpoint_path = run.directory / "best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    session = build_v4_session(run.config, device=device, run_seed=run.seed)
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
        seed=session.seed_contract["derived"]["train_schedule"],
        validation_seed=session.args.lcbt_validation_seed,
        scaler=scaler)
    loader = DataLoader(PairedDataset(dataset),
                        batch_size=session.train_loader.batch_size,
                        shuffle=False, num_workers=0)
    collector = V4MechanismCollector(context, scaler, session.args.input_len)
    for pair in loader:
        treated = to_device(pair["treated"], device)
        control = to_device(pair["control"], device)
        treated_prediction, _ = unpack_model_output(
            session.model.forward_v4(treated, context,
                                     epoch=int(checkpoint["best_epoch"]),
                                     teacher_forcing_mask=None))
        control_prediction, _ = unpack_model_output(
            session.model.forward_v4(control, context,
                                     epoch=int(checkpoint["best_epoch"]),
                                     teacher_forcing_mask=None))
        collector.update(treated, treated_prediction, control, control_prediction)
    report = collector.report()
    report.update({
        "run": run.name, "dataset": run.dataset, "model": run.model,
        "seed": run.seed, "plan_sha256": plan["plan_sha256"],
        "seed_contract": session.seed_contract,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": int(checkpoint["best_epoch"]),
        "selection_use": "forbidden",
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run", required=True,
                        help="DATASET/MODEL/seedN, exactly as printed by --status")
    parser.add_argument("--split", choices=("validation", "test"),
                        default="validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--allow-test", action="store_true",
                        help="required when --split test")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        raise SystemExit("--split test requires --allow-test")
    plan = load_plan(args.plan)
    runs, _ = resolve_runs(plan, args.output_root)
    matches = [run for run in runs if run.name == args.run]
    if len(matches) != 1:
        raise SystemExit(f"expected one run named {args.run!r}, got {len(matches)}")
    run = matches[0]
    report = evaluate(run, plan, args.split, args.device)
    path = run.directory / f"v4_mechanisms_{args.split}.json"
    _atomic_json(path, report)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
