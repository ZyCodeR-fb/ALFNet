#!/usr/bin/env python
"""Write the pre-registered v4 development plan, and hash it.

The plan is written *before* the runs and hashed, so the budget, the datasets,
the metric and the tie rule cannot be chosen after a number exists. Every run
records the plan hash; `train_v4.py` recomputes it and refuses to start if the
plan or any config it names has moved.

Twelve runs exactly: 2 datasets x 2 models x 3 seeds. PEMS03 and PEMS04 appear
nowhere -- not as a dataset, not as a loader, not as a validation split -- so
they cannot inform any choice made from these numbers.

    python scripts/build_v4_development_plan.py            # write
    python scripts/build_v4_development_plan.py --check    # verify unchanged
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
    SHUFFLE_POLICY,
    SPLIT_FRACTIONS,
    STRATEGY,
    TEACHER_FORCING_POLICY,
)

PLAN_PATH = ROOT / "configs/paper/v4_development_plan.json"
MANIFEST_DIR = ROOT / "configs/paper/v4_data_manifests"
CONFIG_DIR = ROOT / "configs/paper/v4"
DATASETS = ("PEMS08", "PEMSBAY")
MODELS = ("ALFNet", "MoGERNN")
SEEDS = (32, 42, 52)
# The analysis code the plan is committing to. Hashed so a table produced by a
# different collector is identifiable as such.
ANALYSIS_SOURCES = ("experiments/v4_validation_collector.py",
                    "experiments/v4_runner.py",
                    "scripts/train_v4.py",
                    "lib/lcbt_v4.py",
                    "scripts/v4_launcher.py",
                    "scripts/audit_v4_development_base.py",
                    "scripts/v4_development_eval.py",
                    "scripts/collect_v4_development_base.py")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def plan_hash(plan):
    """Over the plan without its own hash field. Stable across processes."""
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def build_plan(include_commit=True):
    configs, data = {}, {}
    for dataset in DATASETS:
        manifest = MANIFEST_DIR / f"{dataset}_run_manifest.json"
        payload = json.loads(manifest.read_text("utf-8"))
        data[dataset] = {
            "payload_path": payload["payload"]["path"],
            "payload_sha256": payload["payload"]["sha256"],
            "unseen_path": payload["unseen"]["path"],
            "unseen_sha256": payload["unseen"]["sha256"],
            "manifest_sha256": sha256_file(manifest),
            "target": payload["target"]["name"],
            "target_channel": int(payload["target"]["channel"]),
        }
        for model in MODELS:
            path = CONFIG_DIR / f"{dataset}_{model}.yaml"
            configs[f"{dataset}_{model}"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "status": "development_base",
            }
    plan = {
        "schema_version": 1,
        "status": "preregistered_development",
        "purpose": "v4_base_diagnosis_before_limited_optimization",
        "protocol": STRATEGY,
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "trial_budget": len(DATASETS) * len(MODELS) * len(SEEDS),
        "configs": configs,
        "data": data,
        "training": {
            "max_epoch": 100,
            "patience": 10,
            "base_loss": "mse",
            "split_fractions": list(SPLIT_FRACTIONS),
            "teacher_forcing_policy": TEACHER_FORCING_POLICY,
            "checkpoint_metric": "validation life_avg_mae",
            "tie_rule": "keep_earlier_epoch",
            "allowed_overrides": ["seed", "device", "output_root"],
            # The training order is a function of (run seed, epoch), so a
            # resumed run sees the permutation an uninterrupted one would have.
            "shuffle_policy": SHUFFLE_POLICY,
        },
        "infrastructure": {
            # How a run is executed, not what it is. JOBS and the device
            # assignment are scheduling and are deliberately absent: they belong
            # in the run log, never in the model's identity.
            "run_state_schema": 1,
            "run_state_file": "last.pt",
            "resume": "epoch boundary; model, optimizer, scheduler, best, "
                      "patience wait, history and RNG state are all restored",
            "launcher_version": "v2_parallel_tmux",
            "scheduling_is_not_part_of_run_identity": True,
        },
        "provenance_corrections": [
            {
                "what": "ALFNet effective batch_size 16 -> 64",
                "kind": "configuration provenance correction",
                "not_a_tuning_decision": True,
                "selected_using_v4_results": False,
                "why": ("The v3 ALFNet YAMLs declare 16, but that value never "
                        "took effect: configs/paper/final_runs.json is frozen "
                        "at batch_size 64 and passes it on every run, and the "
                        "run it produced recorded efficiency.batch_size=64 with "
                        "train_samples_per_epoch=8905 over "
                        "optimizer_steps_per_epoch=140. Generating v4 from the "
                        "YAML default gave ALFNet a batch size no v3 experiment "
                        "used."),
                "evidence_read_once_at_registration_not_a_runtime_dependency": [
                    "configs/paper/final_runs.json",
                    "outputs/ALFNet/alfnet-pems08-s32/checkpoints/"
                    "training_summary.json",
                ],
                "learning_rates_unchanged": {"ALFNet": 5e-05, "MoGERNN": 0.001},
            },
            {
                "what": "a user stop is no longer recorded as FAILED",
                "kind": "launcher semantics correction",
                "why": ("SIGINT/SIGTERM forwarded by the launcher, with a valid "
                        "last.pt, leaves the run RESUMABLE; only a crash writes "
                        "FAILED.json. Told apart by whether the launcher sent "
                        "the signal, not by the exit code."),
            },
        ],
        "phases_allowed": ["train", "validation"],
        "test_access": "forbidden",
        "candidate_configurations_this_round": 0,
        "hyper_parameter_changes_this_round": "none",
        "next_round": {
            "max_candidate_families": 2,
            "allowed_families": [
                "order_preserving_temporal_patching (tem_patchsize=4, tem_patchnum=3)",
                "metric_aligned_composite_loss (MSE + MAE + train-defined safe-relative)",
            ],
            "max_trials_per_model": 4,
            "equal_budget_for_both_models": True,
            "must_be_preregistered_before_running": True,
        },
        "prohibited": [
            "selecting a model or configuration using PEMS03 or PEMS04",
            "constructing a test context for any dataset",
            "describing these validation tables as final or confirmatory",
        ],
        "analysis_code": {
            path: sha256_file(ROOT / path) for path in ANALYSIS_SOURCES
        },
    }
    if include_commit:
        plan["git_commit"] = _git_commit()
    return plan


def load_plan():
    """Read the plan and verify its recorded hash still describes it."""
    plan = json.loads(PLAN_PATH.read_text("utf-8"))
    recorded = plan.get("plan_sha256")
    actual = plan_hash(plan)
    if recorded != actual:
        raise ValueError(
            f"the development plan's recorded hash {recorded} does not describe "
            f"its contents {actual}; the plan was edited after registration")
    return plan


def assert_plan_still_holds(plan):
    """Every config and data asset the plan named must be unchanged."""
    drifted = []
    for name, entry in plan["configs"].items():
        actual = sha256_file(ROOT / entry["path"])
        if actual != entry["sha256"]:
            drifted.append(f"{name}: config {actual} != registered {entry['sha256']}")
    for dataset, entry in plan["data"].items():
        for kind in ("payload", "unseen"):
            actual = sha256_file(ROOT / entry[f"{kind}_path"])
            if actual != entry[f"{kind}_sha256"]:
                drifted.append(f"{dataset}: {kind} {actual} != registered")
        manifest = sha256_file(MANIFEST_DIR / f"{dataset}_run_manifest.json")
        if manifest != entry["manifest_sha256"]:
            drifted.append(f"{dataset}: manifest {manifest} != registered")
    if drifted:
        raise ValueError(
            "the pre-registered development plan no longer describes what is on "
            "disk; refusing to continue rather than re-hashing and carrying on:\n"
            + "\n".join(drifted))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    plan = build_plan()
    if arguments.check:
        if not PLAN_PATH.exists():
            print(f"{PLAN_PATH.relative_to(ROOT)} is missing")
            return 1
        on_disk = load_plan()
        # The git commit moves with every commit and is provenance, not policy.
        rebuilt = dict(plan, git_commit=on_disk.get("git_commit"))
        rebuilt["plan_sha256"] = plan_hash(rebuilt)
        if rebuilt["plan_sha256"] != on_disk["plan_sha256"]:
            print("the plan on disk differs from the generator")
            return 1
        assert_plan_still_holds(on_disk)
        print(f"plan {on_disk['plan_sha256'][:16]} holds: "
              f"{on_disk['trial_budget']} runs, {on_disk['datasets']}")
        return 0
    plan["plan_sha256"] = plan_hash(plan)
    PLAN_PATH.write_text(
        json.dumps(plan, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}  plan_sha256={plan['plan_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
