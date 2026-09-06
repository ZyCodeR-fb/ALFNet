#!/usr/bin/env python
"""Freeze corrected-runner GWNet/MoGERNN seed32 controls on four datasets."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_final_configs, build_final_plan  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402

DATASETS = ("PEMS03", "PEMS04", "PEMS08", "PEMSBAY")
MODELS = ("GWNET", "MoGERNN")
SEED = 32
PLAN_PATH = ROOT / "configs/paper/corrected_baseline_seed32_plan.json"
CONFIG_ROOT = ROOT / "configs/paper/corrected_baseline_seed32"


def _gwnet(dataset):
    cfg = deepcopy(build_final_configs.common(dataset, "GWNET"))
    cfg["runtime"].update({
        "model_variant": "gwnet_corrected_runner_seed32_v1",
        "model_recipe": "source_architecture_corrected_checkpoint",
    })
    cfg["family"] = deepcopy(build_final_configs.family(
        "GWNET", build_final_configs.facts(dataset)))
    cfg["training"].update({
        "teacher_forcing_mode": "free_run",
        "teacher_forcing_decay_epochs": 40,
        "early_stopping_start_epoch": 0,
    })
    return cfg


def _mogernn(dataset):
    cfg = deepcopy(build_final_configs.mogernn(dataset))
    cfg["runtime"].update({
        "model_variant": "mogernn_corrected_runner_seed32_v2",
        "model_recipe": "source_architecture_source_early_stopping",
    })
    cfg["training"].update({
        "patience": 3,
        "teacher_forcing_mode": "linear_decay",
        "teacher_forcing_decay_epochs": 40,
        # The reference loop clears wait_count while epoch < 20.  Checkpoint
        # eligibility remains ungated; only the termination counter is delayed.
        "early_stopping_start_epoch": 20,
    })
    return cfg


def render_configs():
    files = {}
    for dataset in DATASETS:
        for model, payload in (("GWNET", _gwnet(dataset)),
                               ("MoGERNN", _mogernn(dataset))):
            path = CONFIG_ROOT / f"{dataset}_{model}.yaml"
            files[path] = (
                "# GENERATED corrected-runner seed32 baseline; do not edit.\n"
                + OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True))
    return files


def write_configs():
    for path, content in render_configs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, "utf-8")


def _code_sources():
    return tuple(dict.fromkeys(build_final_plan._code_sources() + (
        "scripts/build_corrected_baseline_seed32.py",
    )))


def build_plan():
    configs, data, runs = {}, {}, []
    for dataset in DATASETS:
        manifest_path = (
            ROOT / f"configs/paper/v4_data_manifests/{dataset}_run_manifest.json")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        data[dataset] = {
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "payload_path": manifest["payload"]["path"],
            "payload_sha256": manifest["payload"]["sha256"],
            "unseen_path": manifest["unseen"]["path"],
            "unseen_sha256": manifest["unseen"]["sha256"],
            "target": manifest["target"]["name"],
            "target_channel": int(manifest["target"]["channel"]),
        }
        for model in MODELS:
            path = CONFIG_ROOT / f"{dataset}_{model}.yaml"
            key = f"{dataset}_{model}"
            configs[key] = {
                "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            runs.append({
                "dataset": dataset, "model": model, "seed": SEED,
                "config": configs[key]["path"], "phase": 0,
                "execution_action": "train",
            })
    plan = {
        "schema_version": 1,
        "kind": "v4_corrected_baseline_seed32_validation_runs",
        "status": "development_validation_only",
        "selection_split": "validation", "test_access": "forbidden",
        "datasets": list(DATASETS), "models": list(MODELS), "seeds": [SEED],
        "runs": runs, "trial_budget": len(runs), "configs": configs,
        "data": data,
        "output_root": "outputs_v4/corrected_baseline_seed32",
        "fairness_contract": {
            "common": [
                "v4 full-N scoped input", "same K/V/T node split",
                "same 60/20/20 time split", "train-known-only scaler",
                "same origins/cohorts/loss masks",
                "validation 33-cell main/LifeAvg/MAE checkpoint metric",
                "one validation forward per epoch", "earlier checkpoint on tie",
            ],
            "model_specific": {
                "GWNET": {
                    "objective": "source MAE", "decoder": "target blind",
                    "teacher_forcing": "none", "patience_start": 0,
                    "patience": 10,
                },
                "MoGERNN": {
                    "objective": "source MSE", "decoder": "autoregressive",
                    "teacher_forcing": "linear decay to zero at epoch 40",
                    "patience_start": 20, "patience": 3,
                    "source_semantics": (
                        "reference clears wait_count while epoch < 20"),
                },
            },
            "all_epochs_checkpoint_eligible": True,
            "architecture_or_loss_changed": False,
        },
        "training_contract": {
            "max_epoch": 100,
            "model_specific_patience": {
                "GWNET": {"start_epoch": 0, "patience": 10},
                "MoGERNN": {"start_epoch": 20, "patience": 3},
            },
            "checkpoint_metric": "validation 33-cell main/LifeAvg/MAE",
            "tie_rule": "keep_earlier_epoch",
            "test_metrics_may_influence_selection": False,
        },
        "code_sha256": {
            path: sha256_file(ROOT / path) for path in _code_sources()},
    }
    plan["plan_sha256"] = build_final_plan.plan_hash(plan)
    return plan


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text("utf-8"))
    if plan.get("plan_sha256") != build_final_plan.plan_hash(plan):
        raise ValueError("corrected baseline plan hash mismatch")
    return plan


def assert_plan_still_holds(plan):
    drift = []
    for key, item in plan["configs"].items():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            drift.append(f"{key}: config changed")
    for dataset, item in plan["data"].items():
        for kind in ("manifest", "payload", "unseen"):
            if sha256_file(ROOT / item[f"{kind}_path"]) != item[f"{kind}_sha256"]:
                drift.append(f"{dataset}: {kind} changed")
    for path, digest in plan["code_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            drift.append(f"{path}: code changed")
    if drift:
        raise ValueError("corrected baseline plan drift:\n  - "
                         + "\n  - ".join(drift))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        assert_plan_still_holds(load_plan())
        stale = [str(path.relative_to(ROOT))
                 for path, content in render_configs().items()
                 if not path.exists() or path.read_text("utf-8") != content]
        if stale:
            raise SystemExit("corrected baseline configs differ: "
                             + ", ".join(stale))
        print("corrected baseline plan holds: 8 seed32 runs")
        return 0
    write_configs()
    plan = build_plan()
    PLAN_PATH.write_text(json.dumps(
        plan, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")
    print(f"plan {plan['plan_sha256']}: {len(plan['runs'])} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
