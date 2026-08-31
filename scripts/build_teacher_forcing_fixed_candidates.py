#!/usr/bin/env python
"""Build the fair fixed-50 PEMS03/seed32 decoder-policy experiment."""

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

DATASET, SEED = "PEMS03", 32
PLAN_PATH = ROOT / "configs/paper/teacher_forcing_fixed50_plan.json"
CONFIG_ROOT = ROOT / "configs/paper/teacher_forcing_fixed50/PEMS03"
CANDIDATES = ("TF40_FIXED50", "FREE_RUN_FIXED50")
SETTINGS = {
    "TF40_FIXED50": {"teacher_forcing_mode": "linear_decay",
                     "teacher_forcing_decay_epochs": 40},
    "FREE_RUN_FIXED50": {"teacher_forcing_mode": "free_run",
                         "teacher_forcing_decay_epochs": 40},
}


def render_configs():
    files = {}
    for name in CANDIDATES:
        outer = deepcopy(build_final_configs.alfnet(DATASET))
        graph = deepcopy(build_final_configs.alfgraph(DATASET))
        temporal = deepcopy(build_final_configs.alftemporal(DATASET))
        slug = name.lower()
        outer["runtime"]["model_variant"] = f"alfnet_pems03_{slug}_v1"
        outer["runtime"]["model_recipe"] = f"decoder_policy_{slug}"
        outer["family"]["experts"]["graph"]["config_path"] = (
            f"./configs/paper/teacher_forcing_fixed50/PEMS03/{name}_ALFGraph.yaml")
        outer["family"]["experts"]["patch"]["config_path"] = (
            f"./configs/paper/teacher_forcing_fixed50/PEMS03/{name}_ALFTemporal.yaml")
        graph["runtime"]["model_variant"] = f"alfgraph_pems03_{slug}_v1"
        temporal["runtime"]["model_variant"] = f"alftemporal_pems03_{slug}_v1"
        for payload in (outer, graph, temporal):
            payload["training"].update(SETTINGS[name])
            payload["training"].update({
                "max_epoch": 50,
                # Greater than the entire budget: both arms finish all epochs.
                "patience": 51,
                "early_stopping_start_epoch": 0,
            })
        for suffix, payload in (("ALFNet", outer), ("ALFGraph", graph),
                                ("ALFTemporal", temporal)):
            path = CONFIG_ROOT / f"{name}_{suffix}.yaml"
            files[path] = (
                "# GENERATED fixed-50 validation-only decoder-policy candidate.\n"
                + OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True))
    return files


def write_configs():
    for path, text in render_configs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, "utf-8")


def _code_sources():
    return tuple(dict.fromkeys(build_final_plan._code_sources() + (
        "scripts/build_teacher_forcing_fixed_candidates.py",
        "scripts/collect_teacher_forcing_fixed_candidates.py",
    )))


def build_plan():
    manifest_path = ROOT / f"configs/paper/v4_data_manifests/{DATASET}_run_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    configs, nested, runs = {}, {}, []
    for name in CANDIDATES:
        outer = CONFIG_ROOT / f"{name}_ALFNet.yaml"
        configs[name] = {"path": str(outer.relative_to(ROOT)),
                         "sha256": sha256_file(outer)}
        nested[name] = {}
        for expert in ("ALFGraph", "ALFTemporal"):
            path = CONFIG_ROOT / f"{name}_{expert}.yaml"
            nested[name][expert] = {"path": str(path.relative_to(ROOT)),
                                     "sha256": sha256_file(path)}
        runs.append({"dataset": DATASET, "model": "ALFNet", "seed": SEED,
                     "candidate": name, "config": configs[name]["path"],
                     "phase": 0, "execution_action": "train"})
    plan = {
        "schema_version": 1,
        "kind": "v4_teacher_forcing_fixed50_candidates",
        "status": "development_validation_only",
        "selection_split": "validation", "test_access": "forbidden",
        "dataset": DATASET, "datasets": [DATASET], "models": ["ALFNet"],
        "seeds": [SEED], "candidates": list(CANDIDATES), "runs": runs,
        "trial_budget": 2, "configs": configs, "nested_configs": nested,
        "output_root": "outputs_v4/teacher_forcing_fixed50_pems03",
        "data": {DATASET: {
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "payload_path": manifest["payload"]["path"],
            "payload_sha256": manifest["payload"]["sha256"],
            "unseen_path": manifest["unseen"]["path"],
            "unseen_sha256": manifest["unseen"]["sha256"],
        }},
        "candidate_contract": SETTINGS,
        "training_contract": {
            "epochs": 50, "early_termination": False,
            "all_epochs_checkpoint_eligible": True,
            "checkpoint_metric": "validation 33-cell main/LifeAvg/MAE",
            "teacher_mask_last_horizon_position": False,
        },
        "selection_rule": {
            "primary": "lower validation 33-cell main/LifeAvg/MAE",
            "secondary": "more of the same 33 validation cells lower",
            "test_metrics_may_influence_selection": False,
        },
        "code_sha256": {path: sha256_file(ROOT / path) for path in _code_sources()},
    }
    plan["plan_sha256"] = build_final_plan.plan_hash(plan)
    return plan


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text("utf-8"))
    if plan.get("plan_sha256") != build_final_plan.plan_hash(plan):
        raise ValueError("fixed-50 candidate plan hash mismatch")
    return plan


def assert_plan_still_holds(plan):
    drift = []
    for name, item in plan["configs"].items():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            drift.append(f"{name}: outer config changed")
    for name, experts in plan["nested_configs"].items():
        for expert, item in experts.items():
            if sha256_file(ROOT / item["path"]) != item["sha256"]:
                drift.append(f"{name}/{expert}: nested config changed")
    item = plan["data"][DATASET]
    for kind in ("manifest", "payload", "unseen"):
        if sha256_file(ROOT / item[f"{kind}_path"]) != item[f"{kind}_sha256"]:
            drift.append(f"{DATASET}: {kind} changed")
    for path, digest in plan["code_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            drift.append(f"{path}: code changed")
    if drift:
        raise ValueError("fixed-50 plan drift:\n  - " + "\n  - ".join(drift))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        assert_plan_still_holds(load_plan())
        expected = render_configs()
        stale = [str(path.relative_to(ROOT)) for path, text in expected.items()
                 if not path.exists() or path.read_text("utf-8") != text]
        if stale:
            raise SystemExit("candidate configs differ: " + ", ".join(stale))
        print("fixed-50 teacher-forcing plan holds: 2 runs")
        return 0
    write_configs()
    plan = build_plan()
    PLAN_PATH.write_text(json.dumps(
        plan, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")
    print(f"plan {plan['plan_sha256']}: 2 runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
