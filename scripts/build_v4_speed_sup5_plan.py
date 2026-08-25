#!/usr/bin/env python
"""Freeze six seed32 validation-only speed SUP5 candidate runs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_v4_five_dataset_plan import CODE_SOURCES  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402

PLAN_PATH = ROOT / "configs/paper/v4_speed_sup5_candidate_plan.json"
CONFIG_ROOT = ROOT / "configs/paper/v4_speed_sup5_candidates"
OLD_METR_PLAN = ROOT / "configs/paper/v4_metrla_lf_sup_candidate_plan.json"
OLD_METR_PLAN_SHA = (
    "df329665605fc2335c94d957dce70421bd0b59494a01a6c0c5d9d9cbb314d793"
)
OLD_METR_ROOT = ROOT / "outputs_v4/metrla_lf_sup_candidates" / OLD_METR_PLAN_SHA
OLD_METR_SUMMARY = OLD_METR_ROOT / "candidate_validation_summary.json"

RUNS = (
    ("METRLA", "SUP5", "ALFNet", "SUP5_ALFNet.yaml"),
    ("METRLA", "LF_SUP5", "ALFNet", "LF_SUP5_ALFNet.yaml"),
    ("PEMSBAY", "BASE", "ALFNet", "BASE_ALFNet.yaml"),
    ("PEMSBAY", "SUP5", "ALFNet", "SUP5_ALFNet.yaml"),
    ("PEMSBAY", "LF_SUP5", "ALFNet", "LF_SUP5_ALFNet.yaml"),
    ("PEMSBAY", "CONTROL", "MoGERNN", "CONTROL_MoGERNN.yaml"),
)


def plan_hash(plan):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _old_metr_references():
    old_plan = json.loads(OLD_METR_PLAN.read_text("utf-8"))
    if old_plan.get("plan_sha256") != OLD_METR_PLAN_SHA:
        raise ValueError("the frozen METR-LA candidate plan identity changed")
    summary = json.loads(OLD_METR_SUMMARY.read_text("utf-8"))
    if summary.get("plan_sha256") != OLD_METR_PLAN_SHA:
        raise ValueError("the frozen METR-LA summary belongs to another plan")
    records = {}
    for candidate in ("BASE", "SUP", "LF_SUP", "CONTROL"):
        evidence = summary["evidence"][candidate]
        model = "MoGERNN" if candidate == "CONTROL" else "ALFNet"
        directory = (OLD_METR_ROOT / "METRLA" / model
                     / evidence["config_sha256"][:16] / "seed32")
        checkpoint = directory / "best.pt"
        table = directory / "development_validation_table.json"
        if sha256_file(checkpoint) != evidence["checkpoint_sha256"]:
            raise ValueError(f"old METR-LA {candidate} checkpoint changed")
        records[candidate] = {
            "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "table_path": table.relative_to(ROOT).as_posix(),
            "table_sha256": sha256_file(table),
        }
    return {
        "plan_path": OLD_METR_PLAN.relative_to(ROOT).as_posix(),
        "plan_file_sha256": sha256_file(OLD_METR_PLAN),
        "plan_sha256": OLD_METR_PLAN_SHA,
        "summary_path": OLD_METR_SUMMARY.relative_to(ROOT).as_posix(),
        "summary_sha256": sha256_file(OLD_METR_SUMMARY),
        "records": records,
    }


def build_plan():
    configs, runs = {}, []
    for dataset, candidate, model, filename in RUNS:
        path = CONFIG_ROOT / dataset / filename
        key = f"{dataset}_{candidate}"
        entry = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        if model == "ALFNet":
            graph_name = "ALFGraph_BASE.yaml" if candidate == "BASE" \
                else "ALFGraph_SUP5.yaml"
            graph = CONFIG_ROOT / dataset / "experts" / graph_name
            patch = ROOT / f"configs/paper/v4/experts/{dataset}_ALFTemporal.yaml"
            entry["nested"] = {
                "graph": {"path": graph.relative_to(ROOT).as_posix(),
                          "sha256": sha256_file(graph)},
                "patch": {"path": patch.relative_to(ROOT).as_posix(),
                          "sha256": sha256_file(patch)},
            }
        configs[key] = entry
        runs.append({
            "candidate": candidate, "dataset": dataset, "model": model,
            "seed": 32, "phase": 0, "config": entry["path"],
            "execution_action": "train",
        })

    data = {}
    for dataset in ("METRLA", "PEMSBAY"):
        manifest_path = ROOT / f"configs/paper/v4_data_manifests/{dataset}_run_manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        data[dataset] = {
            "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "payload_path": manifest["payload"]["path"],
            "payload_sha256": manifest["payload"]["sha256"],
            "unseen_path": manifest["unseen"]["path"],
            "unseen_sha256": manifest["unseen"]["sha256"],
        }

    sources = tuple(CODE_SOURCES) + (
        "scripts/build_v4_speed_sup5_candidates.py",
        "scripts/build_v4_speed_sup5_plan.py",
        "scripts/collect_v4_speed_sup5_candidates.py",
        "scripts/run_v4_speed_sup5_candidate_preflight.py",
        "scripts/run_v4_speed_sup5_candidates.sh",
        "scripts/tmux_v4_speed_sup5_candidates.sh",
    )
    plan = {
        "schema_version": 1,
        "kind": "v4_speed_sup5_candidate_plan",
        "status": "exploratory_validation_only",
        "purpose": "one final bounded unified support decision",
        "datasets": ["METRLA", "PEMSBAY"],
        "models": ["ALFNet", "MoGERNN"],
        "seeds": [32],
        "trial_budget": 6,
        "configs": configs,
        "runs": runs,
        "execution": {"gpu_jobs_to_execute": 6, "reuse_forbidden": True},
        "data": data,
        "reference_evidence": {"METRLA": _old_metr_references()},
        "selection": {
            "split": "validation",
            "test_forbidden": True,
            "checkpoint_metric": "validation life_avg_mae",
            "headline_cells": 33,
            "candidate_order": ["SUP5", "LF_SUP5"],
            "pass_on_each_speed_dataset": {
                "lifeavg_mae": "candidate <= BASE",
                "lifeavg_rmse": "candidate <= BASE",
                "lifeavg_mape": "candidate <= 1.01 * BASE",
                "unseen_mae": "candidate <= 1.01 * BASE",
                "unseen_rmse": "candidate <= 1.01 * BASE",
                "cell_wins_vs_base": "at least 17 of 33",
                "cell_wins_vs_control": "at least 18 of 33",
            },
            "decision": (
                "choose unified_SUP5 if SUP5 passes both datasets; otherwise "
                "unified_LF_SUP5 if LF_SUP5 passes both; otherwise unified_BASE"
            ),
            "no_candidate_or_gate_may_change_after_results": True,
        },
        "output_root": "outputs_v4/speed_sup5_candidates",
        "code_sha256": {
            name: sha256_file(ROOT / name) for name in dict.fromkeys(sources)
        },
    }
    plan["plan_sha256"] = plan_hash(plan)
    return plan


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text("utf-8"))
    if plan.get("plan_sha256") != plan_hash(plan):
        raise ValueError("speed SUP5 plan content does not match plan_sha256")
    return plan


def assert_plan_still_holds(plan):
    drift = []
    for name, entry in plan["configs"].items():
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            drift.append(f"{name}: config")
        for nested_name, nested in entry.get("nested", {}).items():
            if sha256_file(ROOT / nested["path"]) != nested["sha256"]:
                drift.append(f"{name}/{nested_name}")
    for dataset, entry in plan["data"].items():
        for field in ("manifest", "payload", "unseen"):
            if sha256_file(ROOT / entry[f"{field}_path"]) != entry[f"{field}_sha256"]:
                drift.append(f"{dataset}/{field}")
    reference = plan["reference_evidence"]["METRLA"]
    for field in ("plan", "summary"):
        if sha256_file(ROOT / reference[f"{field}_path"]) != \
                reference[f"{field}_sha256" if field == "summary" else "plan_file_sha256"]:
            drift.append(f"reference/{field}")
    for candidate, evidence in reference["records"].items():
        for field in ("checkpoint", "table"):
            if sha256_file(ROOT / evidence[f"{field}_path"]) != evidence[f"{field}_sha256"]:
                drift.append(f"reference/{candidate}/{field}")
    for name, digest in plan["code_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            drift.append(name)
    if drift:
        raise ValueError("speed SUP5 plan drift: " + ", ".join(drift))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        on_disk = assert_plan_still_holds(load_plan())
        rebuilt = build_plan()
        if rebuilt["plan_sha256"] != on_disk["plan_sha256"]:
            raise SystemExit("speed SUP5 plan differs from its generator")
        print(f"plan {on_disk['plan_sha256'][:16]} holds: six validation-only runs")
        return 0
    plan = build_plan()
    PLAN_PATH.write_text(
        json.dumps(plan, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256 = {plan['plan_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
