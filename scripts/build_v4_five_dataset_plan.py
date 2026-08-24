#!/usr/bin/env python
"""Build the canonical five-dataset, three-seed v4 reproduction plan.

Written before the runs and hashed, so the datasets, the seeds, the endpoint,
the tie rule and the reporting obligations cannot be chosen once numbers exist.
The completed 946d experiment remains byte-for-byte under
``configs/paper/archive/v4_result_946d`` and ``outputs_v4/five_dataset/946d...``.
This generator describes a clean from-scratch reproduction after configuration
authority was fixed (notably ``slide_step: 1``) and never aliases old checkpoints
into a new plan identity.

Two things this plan is explicit about, because both are easy to overstate:

* **Not every dataset here is an untouched confirmation set.** PEMS03 and PEMS04
  have never had a test split read. METR-LA and PEMS-BAY have, under v3, so they
  are supportive rather than confirmatory. PEMS08's v3 test was read on *speed*
  while v4 targets *flow*, which is a different quantity but the same calendar
  window, so the exposure is disclosed rather than waved away.
* **Flow and speed do not average.** Three datasets are veh/5min and two are
  mph. A single `avg_mae` across them would be a number with no unit, so the
  cross-dataset summary is restricted to dimensionless quantities: per-dataset
  win counts, seed agreement and relative change.

    python scripts/build_v4_five_dataset_plan.py            # write
    python scripts/build_v4_five_dataset_plan.py --check    # verify unchanged
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.lcbt_v4 import (  # noqa: E402
    PRIMARY_SLIDE_STEP,
    SHUFFLE_POLICY,
    SPLIT_FRACTIONS,
    STRATEGY,
    TEACHER_FORCING_POLICY,
)
from lib.runtime_context import build_runtime_context  # noqa: E402
from lib.v4_seeds import SEED_POLICY, seed_bundle  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402

PLAN_PATH = ROOT / "configs/paper/v4_five_dataset_plan.json"
MANIFEST_DIR = ROOT / "configs/paper/v4_data_manifests"
CONFIG_DIR = ROOT / "configs/paper/v4"
DATASETS = ("PEMS03", "PEMS04", "PEMS08", "METRLA", "PEMSBAY")
MODELS = ("ALFNet", "MoGERNN")
SEEDS = (32, 42, 52)

# Every source whose change would make a run a different experiment.
CODE_SOURCES = (
    "args.py", "lib/lcbt_v4.py", "lib/runtime_context.py", "lib/v4_seeds.py",
    "lib/v4_batch.py", "lib/v4_payload.py", "lib/teacher_forcing.py",
    "lib/context_binding.py", "lib/graph_native_layout.py",
    "data_provider/calendar_strata.py", "data_provider/lifecycle_roles.py",
    "data_provider/v4_schedule.py", "data_provider/v4_metrics.py",
    "data_provider/v4_lifecycle_loader.py", "experiments/v4_losses.py",
    "experiments/v4_runner.py", "experiments/v4_validation_collector.py",
    "model/ALFNet/ALFNet.py", "model/ALFGraph/ALFGraph.py",
    "model/ALFTemporal/ALFTemporal.py", "model/ALFTemporal/patchstg.py",
    "model/comparison_model/MoGERNN/MoGERNN.py",
    "model/comparison_model/MoGERNN/layers/DCRNNCell.py",
    "model/comparison_model/MoGERNN/layers/GRUCell.py",
    "model/comparison_model/MoGERNN/layers/agg.py",
    "model/comparison_model/MoGERNN/layers/gnn.py",
    "model/comparison_model/family_utils.py",
    "layers/DCRNNCell.py", "layers/gnn.py", "layers/evidence_router.py",
    "layers/lifecycle_moge.py", "layers/protected_residual.py",
    "layers/source_grin.py", "layers/source_timefilter.py",
    "layers/source_tokenlearner.py", "layers/source_tsmixer.py",
    "layers/structural_node_encoder.py", "scripts/train_v4.py",
    "scripts/v4_launcher.py", "scripts/build_v4_readiness.py",
)

# What has already been read on each dataset's *test* split, stated rather than
# implied. "confirmatory" is reserved for the two that have never been opened.
TEST_EXPOSURE = {
    "PEMS03": {"status": "confirmatory",
               "v3_test_read": False,
               "note": "no test split has ever been read for this dataset"},
    "PEMS04": {"status": "confirmatory",
               "v3_test_read": False,
               "note": "no test split has ever been read for this dataset"},
    "PEMS08": {"status": "supportive",
               "v3_test_read": True,
               "note": ("the v3 test split was read on this calendar window, "
                        "targeting speed; v4 targets flow, which is a different "
                        "quantity but not an unread window")},
    "METRLA": {"status": "supportive",
               "v3_test_read": True,
               "note": ("development set throughout v3; its test split has been "
                        "inspected repeatedly and it is not a confirmation set")},
    "PEMSBAY": {"status": "supportive",
               "v3_test_read": True,
               "note": ("second development set in v3; its validation informed a "
                        "design choice and its test split has been read")},
}


def plan_hash(plan):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _node_split_hashes(dataset):
    """K/V/T identity per seed. The two models must share it exactly."""
    per_seed = {}
    for seed in SEEDS:
        context = build_runtime_context(dataset, "train", seed=seed)
        validation = build_runtime_context(dataset, "validation", seed=seed)
        identity = context.exposure_identity()
        per_seed[str(seed)] = {
            "known_nodes_sha256": identity["known_nodes_sha256"],
            "validation_unseen_sha256": identity["validation_unseen_sha256"],
            "test_unseen_sha256": identity["test_unseen_sha256"],
            "static_asset_hash": context.static_asset_hash(),
            "train_exposure_hash": context.exposure_hash(),
            "validation_exposure_hash": validation.exposure_hash(),
        }
    return per_seed


def build_plan(include_commit=True):
    configs, data, runs = {}, {}, []
    for dataset in DATASETS:
        manifest_path = MANIFEST_DIR / f"{dataset}_run_manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        data[dataset] = {
            "payload_path": manifest["payload"]["path"],
            "payload_sha256": manifest["payload"]["sha256"],
            "unseen_path": manifest["unseen"]["path"],
            "unseen_sha256": manifest["unseen"]["sha256"],
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "target": manifest["target"]["name"],
            "target_channel": int(manifest["target"]["channel"]),
            "unit": "veh_per_5min" if manifest["target"]["name"] == "q" else "mph",
            "num_nodes": int(manifest["canonical_sensor_ids"]["count"]),
            "node_id_sha256": manifest["canonical_sensor_ids"]["sha256"],
            "layout_algorithm": manifest["layout"]["algorithm"],
            "spa_patchsize": int(manifest["layout"]["spa_patchsize"]),
            "node_split_per_seed": _node_split_hashes(dataset),
            "test_exposure": TEST_EXPOSURE[dataset],
        }
        for model in MODELS:
            path = CONFIG_DIR / f"{dataset}_{model}.yaml"
            configs[f"{dataset}_{model}"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "nested": {
                    expert: {
                        "path": f"configs/paper/v4/experts/{dataset}_{expert}.yaml",
                        "sha256": sha256_file(
                            CONFIG_DIR / "experts" / f"{dataset}_{expert}.yaml"),
                    } for expert in (("ALFGraph", "ALFTemporal")
                                     if model == "ALFNet" else ())
                },
            }
            for seed in SEEDS:
                runs.append({"dataset": dataset, "model": model, "seed": seed,
                             "config": str(path.relative_to(ROOT)),
                             "phase": 0 if seed == SEEDS[0] else 1,
                             "execution_action": "train"})

    plan = {
        "schema_version": 3,
        "kind": "v4_five_dataset_plan",
        "execution": {
            "total_result_slots": len(runs),
            "provenance_verified_reused_runs": 0,
            "new_training_runs": len(runs),
            "gpu_jobs_to_execute": len(runs),
            "reuse_forbidden": True,
        },
        "status": "canonical_reproduction",
        "purpose": "five-dataset three-seed v4 reproduction",
        "protocol": STRATEGY,
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "trial_budget": len(DATASETS) * len(MODELS) * len(SEEDS),
        "runs": runs,
        "configs": configs,
        "data": data,
        "training": {
            "input_len": 12, "pred_len": 12,
            "slide_step": PRIMARY_SLIDE_STEP,
            "split_fractions": list(SPLIT_FRACTIONS),
            "scenario_ratios": {name: 0.25 for name in
                                ("zero_history", "steady", "join", "fault")},
            "base_loss": "mse",
            "max_epoch": 100, "patience": 10,
            "checkpoint_metric": "validation life_avg_mae",
            "tie_rule": "keep_earlier_epoch",
            "teacher_forcing_policy": TEACHER_FORCING_POLICY,
            "shuffle_policy": SHUFFLE_POLICY,
            "seed_policy": SEED_POLICY,
            "seed_contracts": {str(seed): seed_bundle(seed) for seed in SEEDS},
            "allowed_overrides": ["seed", "device", "output_root"],
            "resume": "epoch boundary from last.pt; COMPLETE is never overwritten",
            "failure_semantics": ("a crash writes FAILED.json and blocks the "
                                  "phase; a user stop keeps last.pt and stays "
                                  "RESUMABLE"),
            "hyper_parameters_frozen": True,
            "changes_from_completed_946d": (
                "slide_step=1 is now a fail-closed protocol constant; run_seed "
                "controls named sub-streams including model initialisation, the "
                "frozen direct node split, schedule, shuffle, teacher forcing "
                "and workers; duplicate "
                "validation forward and derived WorstGroup reporting were removed"),
        },
        "output_root": "outputs_v4/reproduction",
        "endpoints": {
            "validation": {
                "cells": 33,
                "main_rows": ["Unseen", "Steady", "Join", "Fault", "LifeAvg"],
                "phase_rows": ["Cold", "Warm", "Ready", "Onset", "Decay",
                               "Outage"],
                "metrics": ["MAE", "RMSE", "MAPE"],
                "direction": "lower is better for every metric",
                "aggregation": "three seeds, mean and sample std with ddof=1",
                "std_ddof": 1,
            },
            "test": {
                "cells": 33,
                "same_definitions_as_validation": True,
                "opened_only_after": ["30/30 COMPLETE",
                                      "validation identity audit passes",
                                      "plan, config and code hashes unchanged"],
                "read_once": True,
                "scaler": "the train scaler stored in the checkpoint; never refit",
            },
        },
        "cross_dataset_reporting": {
            "units_differ": True,
            "flow_datasets": ["PEMS03", "PEMS04", "PEMS08"],
            "speed_datasets": ["METRLA", "PEMSBAY"],
            "forbidden": ["any avg_mae, avg_rmse or pooled absolute error across "
                          "flow and speed datasets"],
            "allowed": ["per-dataset cells won out of 33", "seed agreement",
                        "relative change within a dataset"],
        },
        "reporting_obligations": {
            "every_dataset_reported": True,
            "no_dataset_may_be_dropped_after_seeing_results": True,
            "33_of_33_is_an_engineering_target_not_a_stopping_rule": True,
            "not_grounds_for_changing_dataset_split_or_hyper_parameters": True,
        },
        "relationship_to_completed_result": {
            "archive_plan": "configs/paper/archive/v4_result_946d/plan.json",
            "archive_plan_sha256": "946d0b450bb10e3e0e5bc5ee066694bce13b3bdcc29c1c5e5a7e0702b2f4d0c8",
            "archive_cells": 36,
            "archive_includes_derived_worst_group": True,
            "archive_checkpoints_may_not_be_relabelled_under_this_plan": True,
        },
        "readiness_attestation": {
            # Path and schema only. Hashing the attestation here while it hashes
            # this plan would make the two a fixed point that could be satisfied
            # only by rewriting one until they agreed, which verifies nothing.
            # The reference is one-way: attestation -> plan.
            "path": "configs/paper/v4_five_dataset_readiness.json",
            "schema_version": 1,
            "generator": "scripts/build_v4_readiness.py",
            "must_reference_this_plan_sha256": True,
            "required_flags": ["data_ready", "graph_ready", "exposure_ready",
                               "training_ready"],
            "checked_before": ["creating a run directory", "spawning a child",
                               "initialising CUDA"],
        },
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_SOURCES},
    }
    if include_commit:
        plan["git_commit"] = _git_commit()
    return plan


def load_plan(path=None):
    plan = json.loads(Path(path or PLAN_PATH).read_text("utf-8"))
    recorded, actual = plan.get("plan_sha256"), plan_hash(plan)
    if recorded != actual:
        raise ValueError(
            f"the five-dataset plan's recorded hash {recorded} does not describe "
            f"its contents {actual}; it was edited after registration")
    return plan


def assert_plan_still_holds(plan):
    drifted = []
    for name, entry in plan["configs"].items():
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            drifted.append(f"{name}: config changed")
        for expert, nested in entry.get("nested", {}).items():
            if sha256_file(ROOT / nested["path"]) != nested["sha256"]:
                drifted.append(f"{name}/{expert}: nested config changed")
    for dataset, entry in plan["data"].items():
        for kind in ("payload", "unseen", "manifest"):
            if sha256_file(ROOT / entry[f"{kind}_path"]) != entry[f"{kind}_sha256"]:
                drifted.append(f"{dataset}: {kind} changed")
    for name, digest in plan["code_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            drifted.append(f"{name}: code changed since registration")
    if drifted:
        raise ValueError(
            "the five-dataset plan no longer describes what is on disk; refusing "
            "to continue rather than re-hashing:\n  - " + "\n  - ".join(drifted))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    plan = build_plan()
    if arguments.check:
        on_disk = load_plan()
        rebuilt = dict(plan, git_commit=on_disk.get("git_commit"))
        rebuilt["plan_sha256"] = plan_hash(rebuilt)
        if rebuilt["plan_sha256"] != on_disk["plan_sha256"]:
            print("the plan on disk differs from the generator")
            return 1
        assert_plan_still_holds(on_disk)
        print(f"plan {on_disk['plan_sha256'][:16]} holds: "
              f"{on_disk['trial_budget']} runs over {on_disk['datasets']}")
        return 0
    plan["plan_sha256"] = plan_hash(plan)
    PLAN_PATH.write_text(
        json.dumps(plan, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256 = {plan['plan_sha256']}")
    print(f"{plan['trial_budget']} runs: {len(DATASETS)} datasets x "
          f"{len(MODELS)} models x {len(SEEDS)} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
