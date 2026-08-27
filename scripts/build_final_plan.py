#!/usr/bin/env python
"""Build/check the sole active unified-BASE comparison plan."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_final_configs import CONFIG_DIR, DATASETS, MODELS  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402

PLAN_PATH = ROOT / "configs/paper/final_plan.json"
MANIFEST_DIR = ROOT / "configs/paper/v4_data_manifests"
SEEDS = (32, 42, 52)


def _git_commit():
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def plan_hash(plan):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _code_sources():
    fixed = [
        "args.py", "scripts/train_v4.py", "scripts/v4_launcher.py",
        "scripts/build_final_configs.py", "scripts/audit_v4_plan.py",
        "scripts/audit_v4_plan_test.py", "scripts/eval_v4_plan_validation.py",
        "scripts/eval_v4_plan_test.py", "scripts/collect_v4_plan_validation.py",
        "scripts/collect_v4_plan_test.py",
    ]
    trees = (
        "data_provider", "experiments", "lib", "layers", "utils",
        "model/ALFGraph", "model/ALFNet", "model/ALFTemporal",
        "model/comparison_model",
    )
    active = [str(path.relative_to(ROOT)) for tree in trees for path in sorted(
        (ROOT / tree).rglob("*.py"))]
    return tuple(dict.fromkeys(fixed + active))


def build_plan(include_commit=True):
    configs, data, runs = {}, {}, []
    for dataset in DATASETS:
        manifest_path = MANIFEST_DIR / f"{dataset}_run_manifest.json"
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
            "num_nodes": int(manifest["canonical_sensor_ids"]["count"]),
            "unit": "veh_per_5min" if manifest["target"]["name"] == "q" else "mph",
            "test_exposure": {
                "status": ("confirmatory" if dataset in {"PEMS03", "PEMS04"}
                           else "supportive")},
        }
        for model_index, model in enumerate(MODELS):
            path = ROOT / "configs" / CONFIG_DIR[model] / f"{dataset}.yaml"
            key = f"{dataset}_{model}"
            configs[key] = {"path": str(path.relative_to(ROOT)),
                            "sha256": sha256_file(path)}
            for seed in SEEDS:
                runs.append({
                    "dataset": dataset, "model": model, "seed": seed,
                    "config": str(path.relative_to(ROOT)),
                    # One model family per phase gives the requested priority:
                    # all ALFNet, then all MoGERNN, then the remaining models.
                    "phase": model_index, "execution_action": "train",
                })
    plan = {
        "schema_version": 1,
        "kind": "v4_unified_base_comparison_plan",
        "status": "active",
        "model_policy": "unified_BASE",
        "protocol": "lcbt_v4_topology_known_signal_inductive",
        "datasets": list(DATASETS), "models": list(MODELS),
        "seeds": list(SEEDS), "trial_budget": len(runs), "runs": runs,
        "configs": configs, "data": data,
        "output_root": "outputs_v4/final_comparisons",
        "execution": {"total_result_slots": len(runs),
                      "new_training_runs": len(runs),
                      "gpu_jobs_to_execute": len(runs)},
        "training": {
            "input_len": 12, "pred_len": 12, "slide_step": 1,
            "split": [0.6, 0.2, 0.2], "seeds": list(SEEDS),
            "checkpoint_metric": "validation LifeAvg MAE",
            "tie_rule": "keep_earlier_epoch", "max_epoch": 100,
            "patience": 10,
            "optimization_loss": "source-declared in each generated YAML",
            "reported_metrics": ["MAE", "RMSE", "MAPE"],
            "common_inputs": "same full-N masked signal, time marks and graph",
            "alfnet_specific_inputs": "lifecycle role/age/availability only",
        },
        "reporting": {
            "primary_cells": 33,
            "rows": ["Unseen", "Steady", "Join", "Fault", "LifeAvg",
                     "Cold", "Warm", "Ready", "Onset", "Decay", "Outage"],
            "metrics": ["MAE", "RMSE", "MAPE"],
            "worst_group_derived_endpoint": False,
            "flow_and_speed_absolute_errors_must_not_be_pooled": True,
        },
        "code_sha256": {path: sha256_file(ROOT / path)
                        for path in _code_sources()},
    }
    if include_commit:
        plan["git_commit"] = _git_commit()
    plan["plan_sha256"] = plan_hash(plan)
    return plan


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text("utf-8"))
    if plan.get("plan_sha256") != plan_hash(plan):
        raise ValueError("final plan content no longer matches plan_sha256")
    return plan


def assert_plan_still_holds(plan):
    drift = []
    for name, item in plan["configs"].items():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            drift.append(f"{name}: config changed")
    for dataset, item in plan["data"].items():
        for kind in ("manifest", "payload", "unseen"):
            if sha256_file(ROOT / item[f"{kind}_path"]) != item[f"{kind}_sha256"]:
                drift.append(f"{dataset}: {kind} changed")
    for path, digest in plan["code_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            drift.append(f"{path}: code changed")
    if drift:
        raise ValueError("final plan drift:\n  - " + "\n  - ".join(drift))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        disk = assert_plan_still_holds(load_plan())
        rebuilt = build_plan(include_commit=False)
        rebuilt["git_commit"] = disk.get("git_commit")
        rebuilt["plan_sha256"] = plan_hash(rebuilt)
        if rebuilt["plan_sha256"] != disk["plan_sha256"]:
            raise SystemExit("final plan differs from generator")
        print(f"plan {disk['plan_sha256'][:16]} holds: {disk['trial_budget']} runs")
        return 0
    plan = build_plan()
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(json.dumps(
        plan, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")
    print(f"plan {plan['plan_sha256']}: {plan['trial_budget']} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
