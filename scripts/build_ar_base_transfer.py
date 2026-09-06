#!/usr/bin/env python
"""Freeze AR_BASE on PEMS04/PEMS08/PEMS-BAY for seed-32 validation.

PEMS03 selected the autoregressive decoder.  This plan transfers that already
selected model/training rule to the remaining three paper datasets without
opening test data or changing any dataset-specific graph/layout setting.
"""

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

DATASETS = ("PEMS04", "PEMS08", "PEMSBAY")
SEED = 32
PLAN_PATH = ROOT / "configs/paper/ar_base_transfer_plan.json"
CONFIG_ROOT = ROOT / "configs/paper/ar_base_transfer"
TRAINING = {
    "teacher_forcing_mode": "linear_decay",
    "teacher_forcing_decay_epochs": 40,
    "max_epoch": 100,
    "patience": 10,
    # Checkpoints from every epoch remain eligible.  This only prevents the
    # patience counter from terminating during the scheduled-input transition.
    "early_stopping_start_epoch": 40,
}


def render_configs():
    files = {}
    for dataset in DATASETS:
        outer = deepcopy(build_final_configs.alfnet(dataset))
        graph = deepcopy(build_final_configs.alfgraph(dataset))
        temporal = deepcopy(build_final_configs.alftemporal(dataset))
        outer["runtime"].update({
            "model_variant": "alfnet_ar_base_transfer_v1",
            "model_recipe": "selected_ar_base",
        })
        graph["runtime"]["model_variant"] = "alfgraph_ar_base_transfer_v1"
        temporal["runtime"]["model_variant"] = \
            "alftemporal_ar_base_transfer_v1"
        outer["family"]["experts"]["graph"]["config_path"] = (
            f"./configs/paper/ar_base_transfer/{dataset}_ALFGraph.yaml")
        outer["family"]["experts"]["patch"]["config_path"] = (
            f"./configs/paper/ar_base_transfer/{dataset}_ALFTemporal.yaml")
        # Make the selected decoder explicit rather than relying on a default.
        graph["family"]["recipe"]["decoder_output_mode"] = "residual_delta"
        for payload in (outer, graph, temporal):
            payload["training"].update(TRAINING)
        for suffix, payload in (("ALFNet", outer), ("ALFGraph", graph),
                                ("ALFTemporal", temporal)):
            path = CONFIG_ROOT / f"{dataset}_{suffix}.yaml"
            files[path] = (
                "# GENERATED selected AR_BASE transfer; do not edit.\n"
                + OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True))
    return files


def write_configs():
    for path, content in render_configs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, "utf-8")


def _code_sources():
    return tuple(dict.fromkeys(build_final_plan._code_sources() + (
        "scripts/build_ar_base_transfer.py",
    )))


def build_plan():
    configs, nested, data, runs = {}, {}, {}, []
    for dataset in DATASETS:
        outer = CONFIG_ROOT / f"{dataset}_ALFNet.yaml"
        configs[dataset] = {
            "path": str(outer.relative_to(ROOT)), "sha256": sha256_file(outer)}
        nested[dataset] = {}
        for expert in ("ALFGraph", "ALFTemporal"):
            path = CONFIG_ROOT / f"{dataset}_{expert}.yaml"
            nested[dataset][expert] = {
                "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
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
        runs.append({
            "dataset": dataset, "model": "ALFNet", "seed": SEED,
            "candidate": "AR_BASE", "config": configs[dataset]["path"],
            "phase": 0, "execution_action": "train",
        })
    plan = {
        "schema_version": 1,
        "kind": "v4_ar_base_transfer_validation_runs",
        "status": "development_validation_only",
        "selection_split": "validation", "test_access": "forbidden",
        "datasets": list(DATASETS), "models": ["ALFNet"], "seeds": [SEED],
        "candidates": ["AR_BASE"], "runs": runs,
        "trial_budget": len(runs), "configs": configs,
        "nested_configs": nested, "data": data,
        "output_root": "outputs_v4/ar_base_transfer_seed32",
        "transfer_contract": {
            "source_selection_dataset": "PEMS03",
            "selected_model": "AR_BASE",
            "decoder": "12-step recurrent residual decoder",
            "dataset_specific_fields_preserved": [
                "target/channel", "payload/unseen", "adjacency/positions/layout",
                "spa_patchsize", "recur_times"],
            "disabled_experiments": [
                "DIRECT_HEAD", "OBS_MASK", "POLY", "CAL", "LF", "SUP"],
        },
        "training_contract": {
            **TRAINING,
            "checkpoint_metric": "validation 33-cell main/LifeAvg/MAE",
            "all_epochs_checkpoint_eligible": True,
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
        raise ValueError("AR_BASE transfer plan hash mismatch")
    return plan


def assert_plan_still_holds(plan):
    drift = []
    for dataset, item in plan["configs"].items():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            drift.append(f"{dataset}: outer config changed")
    for dataset, experts in plan["nested_configs"].items():
        for expert, item in experts.items():
            if sha256_file(ROOT / item["path"]) != item["sha256"]:
                drift.append(f"{dataset}/{expert}: nested config changed")
    for dataset, item in plan["data"].items():
        for kind in ("manifest", "payload", "unseen"):
            if sha256_file(ROOT / item[f"{kind}_path"]) != item[f"{kind}_sha256"]:
                drift.append(f"{dataset}: {kind} changed")
    for path, digest in plan["code_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            drift.append(f"{path}: code changed")
    if drift:
        raise ValueError("AR_BASE transfer plan drift:\n  - "
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
            raise SystemExit("AR_BASE transfer configs differ: "
                             + ", ".join(stale))
        print("AR_BASE transfer plan holds: 3 datasets, seed32")
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
