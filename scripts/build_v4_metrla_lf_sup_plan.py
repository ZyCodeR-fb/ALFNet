#!/usr/bin/env python
"""Freeze five METR-LA seed32 validation-only LF/SUP candidate runs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_v4_five_dataset_plan import CODE_SOURCES  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402

PLAN_PATH = ROOT / "configs/paper/v4_metrla_lf_sup_candidate_plan.json"
MANIFEST = ROOT / "configs/paper/v4_data_manifests/METRLA_run_manifest.json"
CANDIDATE_DIR = ROOT / "configs/paper/v4_candidates/METRLA"
RUNS = (
    ("BASE", "ALFNet", "BASE_ALFNet.yaml", "METRLA_ALFGraph_legacy.yaml"),
    ("LF", "ALFNet", "LF_ALFNet.yaml", "METRLA_ALFGraph_legacy.yaml"),
    ("SUP", "ALFNet", "SUP_ALFNet.yaml", "METRLA_ALFGraph_sup.yaml"),
    ("LF_SUP", "ALFNet", "LF_SUP_ALFNet.yaml", "METRLA_ALFGraph_sup.yaml"),
    ("CONTROL", "MoGERNN", "CONTROL_MoGERNN.yaml", None),
)


def plan_hash(plan):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def build_plan():
    manifest = json.loads(MANIFEST.read_text("utf-8"))
    configs, runs = {}, []
    for candidate, model, filename, graph_name in RUNS:
        path = CANDIDATE_DIR / filename
        relative = path.relative_to(ROOT).as_posix()
        entry = {"path": relative, "sha256": sha256_file(path)}
        if model == "ALFNet":
            nested_paths = {
                "graph": CANDIDATE_DIR / "experts" / graph_name,
                "patch": ROOT / "configs/paper/v4/experts/METRLA_ALFTemporal.yaml",
            }
            entry["nested"] = {
                name: {"path": nested.relative_to(ROOT).as_posix(),
                       "sha256": sha256_file(nested)}
                for name, nested in nested_paths.items()
            }
        configs[candidate] = entry
        runs.append({
            "candidate": candidate, "dataset": "METRLA", "model": model,
            "seed": 32, "phase": 0, "config": relative,
            "execution_action": "train",
        })
    sources = tuple(CODE_SOURCES) + (
        "scripts/build_v4_metrla_lf_sup_candidates.py",
        "scripts/build_v4_metrla_lf_sup_plan.py",
        "scripts/collect_v4_metrla_lf_sup_candidates.py",
        "scripts/run_v4_metrla_candidate_preflight.py",
        "scripts/run_v4_metrla_lf_sup_candidates.sh",
        "scripts/tmux_v4_metrla_lf_sup_candidates.sh",
        "scripts/train_v4.py", "scripts/v4_launcher.py",
    )
    plan = {
        "schema_version": 1,
        "kind": "v4_metrla_lf_sup_candidate_plan",
        "status": "exploratory_validation_only",
        "purpose": "decide whether LF_SUP can be one unified flow/speed recipe",
        "dataset": "METRLA",
        "datasets": ["METRLA"],
        "models": ["ALFNet", "MoGERNN"],
        "seeds": [32],
        "seed": 32,
        "trial_budget": 5,
        "candidates": [name for name, _, _, _ in RUNS],
        "configs": configs,
        "runs": runs,
        "execution": {"gpu_jobs_to_execute": 5, "reuse_forbidden": True},
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
            "test_forbidden": True,
            "checkpoint_metric": "validation life_avg_mae",
            "headline_cells": 33,
            "comparison": "LF_SUP versus SUP",
            "adopt_unified_lf_sup_only_if": {
                "lifeavg_mae": "LF_SUP strictly lower than SUP",
                "unseen_mae": "LF_SUP no more than 1% above SUP",
                "unseen_rmse": "LF_SUP no more than 1% above SUP",
                "cell_wins": "LF_SUP lower than SUP in at least 17 of 33 cells",
            },
            "otherwise": "retain flow=LF_SUP and speed=SUP",
            "no_candidate_may_change_after_results": True,
        },
        "output_root": "outputs_v4/metrla_lf_sup_candidates",
        "code_sha256": {name: sha256_file(ROOT / name) for name in dict.fromkeys(sources)},
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
    for field in ("manifest", "payload", "unseen"):
        if sha256_file(ROOT / plan["data"][f"{field}_path"]) != \
                plan["data"][f"{field}_sha256"]:
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
        print(f"plan {on_disk['plan_sha256'][:16]} holds: five validation-only runs")
        return 0
    plan = build_plan()
    PLAN_PATH.write_text(
        json.dumps(plan, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256 = {plan['plan_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
