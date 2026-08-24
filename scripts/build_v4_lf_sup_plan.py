#!/usr/bin/env python
"""Freeze the five PEMS08/seed32 validation-only LF/SUP candidate runs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_v4_five_dataset_plan import CODE_SOURCES  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402

PLAN_PATH = ROOT / "configs/paper/v4_lf_sup_candidate_plan.json"
MANIFEST = ROOT / "configs/paper/v4_data_manifests/PEMS08_run_manifest.json"
CANDIDATE_DIR = ROOT / "configs/paper/v4_candidates/PEMS08"
RUNS = (
    ("BASE", "ALFNet", "BASE_ALFNet.yaml"),
    ("LF", "ALFNet", "LF_ALFNet.yaml"),
    ("SUP", "ALFNet", "SUP_ALFNet.yaml"),
    ("LF_SUP", "ALFNet", "LF_SUP_ALFNet.yaml"),
    ("CONTROL", "MoGERNN", "CONTROL_MoGERNN.yaml"),
)


def plan_hash(plan):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def build_plan():
    manifest = json.loads(MANIFEST.read_text("utf-8"))
    configs, runs = {}, []
    for candidate, model, filename in RUNS:
        path = CANDIDATE_DIR / filename
        relative = path.relative_to(ROOT).as_posix()
        entry = {"path": relative, "sha256": sha256_file(path)}
        if model == "ALFNet":
            graph_name = ("PEMS08_ALFGraph_sup.yaml"
                          if candidate in {"SUP", "LF_SUP"}
                          else "PEMS08_ALFGraph_legacy.yaml")
            nested_paths = {
                "graph": CANDIDATE_DIR / "experts" / graph_name,
                "patch": ROOT / "configs/paper/v4/experts/PEMS08_ALFTemporal.yaml",
            }
            entry["nested"] = {
                name: {"path": nested.relative_to(ROOT).as_posix(),
                       "sha256": sha256_file(nested)}
                for name, nested in nested_paths.items()
            }
        configs[candidate] = entry
        runs.append({"candidate": candidate, "dataset": "PEMS08",
                     "model": model, "seed": 32, "config": relative})
    sources = tuple(CODE_SOURCES) + (
        "scripts/build_v4_lf_sup_candidates.py",
        "scripts/build_v4_lf_sup_plan.py",
    )
    plan = {
        "schema_version": 1,
        "kind": "v4_lf_sup_candidate_plan",
        "status": "development_validation_only",
        "dataset": "PEMS08",
        "seed": 32,
        "trial_budget": 5,
        "candidates": [name for name, _, _ in RUNS],
        "configs": configs,
        "runs": runs,
        "data": {
            "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(MANIFEST),
            "payload_path": manifest["payload"]["path"],
            "payload_sha256": manifest["payload"]["sha256"],
            "unseen_path": manifest["unseen"]["path"],
            "unseen_sha256": manifest["unseen"]["sha256"],
        },
        "selection": {
            "split": "validation",
            "checkpoint_metric": "validation life_avg_mae",
            "headline_cells": 33,
            "required_non_degradation": ["Unseen MAE", "Unseen RMSE"],
            "test_forbidden": True,
            "no_candidate_may_change_after_results": True,
        },
        "output_root": "outputs_v4/lf_sup_candidates",
        "code_sha256": {name: sha256_file(ROOT / name) for name in sources},
    }
    plan["plan_sha256"] = plan_hash(plan)
    return plan


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text("utf-8"))
    if plan.get("plan_sha256") != plan_hash(plan):
        raise ValueError("candidate plan content does not match plan_sha256")
    return plan


def assert_plan_still_holds(plan):
    drift = []
    for name, entry in plan["configs"].items():
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            drift.append(f"{name}: config")
        for nested_name, nested in entry.get("nested", {}).items():
            if sha256_file(ROOT / nested["path"]) != nested["sha256"]:
                drift.append(f"{name}/{nested_name}")
    data = plan["data"]
    for field in ("manifest", "payload", "unseen"):
        if sha256_file(ROOT / data[f"{field}_path"]) != data[f"{field}_sha256"]:
            drift.append(field)
    for name, digest in plan["code_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            drift.append(name)
    if drift:
        raise ValueError("candidate plan drift: " + ", ".join(drift))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        on_disk = assert_plan_still_holds(load_plan())
        rebuilt = build_plan()
        if rebuilt["plan_sha256"] != on_disk["plan_sha256"]:
            raise SystemExit("candidate plan differs from its generator")
        print(f"plan {on_disk['plan_sha256'][:16]} holds: 5 validation-only runs")
        return
    plan = build_plan()
    PLAN_PATH.write_text(
        json.dumps(plan, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256 = {plan['plan_sha256']}")


if __name__ == "__main__":
    main()
