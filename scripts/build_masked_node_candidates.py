#!/usr/bin/env python
"""Build the final PEMS03/seed32 observed-node masking ablation."""

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
CANDIDATES = ("OBS_MASK", "OBS_MASK_POLY", "OBS_MASK_POLY_CAL")
PLAN_PATH = ROOT / "configs/paper/masked_node_candidate_plan.json"
CONFIG_ROOT = ROOT / "configs/paper/masked_node_candidates/PEMS03"
TRAINING = {
    "teacher_forcing_mode": "linear_decay",
    "teacher_forcing_decay_epochs": 40,
    "max_epoch": 100,
    "patience": 10,
    "early_stopping_start_epoch": 40,
}
REFERENCE_ROOT = (
    "outputs_v4/direct_head_candidates_pems03/"
    "99d853e45eb6a62edaa7bb72f4e7182fe2a8662672d3ebe3407fd24086485447/"
    "PEMS03/ALFNet/abba04ec37e280f5/seed32")
REFERENCE_FILES = {
    "development_validation_table.json":
        "2397e14e6003bce56cddc0f6959d50529fe5d2e00d485f96a1f5bcfaaf4edc90",
    "run_report.json":
        "b91157c4e0351451dc8be19d179411b1245c97f70ad73296a650cfdb1651bc00",
    "best.pt":
        "601f22aa36f73ce179f4fbaf7c0493c4fefd3adfcc1a28672c500352a73054a0",
}
MASK_RECIPE = {
    # Match the frozen master-unseen proportion, while the 0.15 convex weight
    # keeps the original lifecycle objective dominant.
    "synthetic_unseen_training_mode": "observed_node_mask_reconstruction",
    "synthetic_unseen_fraction": 0.25,
    "synthetic_unseen_loss_weight": 0.15,
}
POLY_RECIPE = {
    "reliability_polynomial_mode": "sender_gated_i_a_a2_residual",
    "reliability_polynomial_hidden_dim": 8,
    "reliability_polynomial_scale": 0.1,
}
CAL_RECIPE = {
    "low_flow_calibration_mode": "role_horizon_residual_train_scale_bins",
    "low_flow_calibration_hidden_dim": 16,
    "low_flow_calibration_scale": 0.1,
    "low_flow_bin_loss_weight": 0.10,
}


def render_configs():
    files = {}
    for name in CANDIDATES:
        outer = deepcopy(build_final_configs.alfnet(DATASET))
        graph = deepcopy(build_final_configs.alfgraph(DATASET))
        temporal = deepcopy(build_final_configs.alftemporal(DATASET))
        slug = name.lower()
        outer["runtime"]["model_variant"] = f"alfnet_pems03_{slug}_v1"
        outer["runtime"]["model_recipe"] = f"controlled_masking_{slug}"
        graph["runtime"]["model_variant"] = f"alfgraph_pems03_{slug}_v1"
        temporal["runtime"]["model_variant"] = f"alftemporal_pems03_{slug}_v1"
        outer["family"]["experts"]["graph"]["config_path"] = (
            f"./configs/paper/masked_node_candidates/PEMS03/{name}_ALFGraph.yaml")
        outer["family"]["experts"]["patch"]["config_path"] = (
            f"./configs/paper/masked_node_candidates/PEMS03/{name}_ALFTemporal.yaml")
        graph["family"]["recipe"]["decoder_output_mode"] = "residual_delta"
        outer["family"]["recipe"].update(MASK_RECIPE)
        if name in {"OBS_MASK_POLY", "OBS_MASK_POLY_CAL"}:
            outer["family"]["recipe"].update(POLY_RECIPE)
        if name == "OBS_MASK_POLY_CAL":
            outer["family"]["recipe"].update(CAL_RECIPE)
        for payload in (outer, graph, temporal):
            payload["training"].update(TRAINING)
        for suffix, payload in (("ALFNet", outer), ("ALFGraph", graph),
                                ("ALFTemporal", temporal)):
            path = CONFIG_ROOT / f"{name}_{suffix}.yaml"
            files[path] = (
                "# GENERATED final validation-only cumulative structure ablation; do not edit.\n"
                + OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True))
    return files


def write_configs():
    for path, content in render_configs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, "utf-8")


def _code_sources():
    return tuple(dict.fromkeys(build_final_plan._code_sources() + (
        "scripts/build_masked_node_candidates.py",
        "scripts/collect_masked_node_candidates.py",
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
            nested[name][expert] = {
                "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
        runs.append({"dataset": DATASET, "model": "ALFNet", "seed": SEED,
                     "candidate": name, "config": configs[name]["path"],
                     "phase": 0, "execution_action": "train"})
    plan = {
        "schema_version": 1,
        "kind": "v4_final_structure_validation_candidates",
        "status": "development_validation_only",
        "selection_split": "validation", "test_access": "forbidden",
        "dataset": DATASET, "datasets": [DATASET], "models": ["ALFNet"],
        "seeds": [SEED], "candidates": list(CANDIDATES), "runs": runs,
        "trial_budget": len(runs), "configs": configs, "nested_configs": nested,
        "output_root": "outputs_v4/final_structure_candidates_pems03",
        "reference_control": {
            "name": "AR_BASE", "path": REFERENCE_ROOT,
            "files_sha256": REFERENCE_FILES,
            "best_epoch": 58, "epochs_run": 70,
            "normal_early_stop_counterfactual": {
                "early_stopping_start_epoch": 40, "patience": 10,
                "would_stop_after_epoch": 68,
                "selected_checkpoint_unchanged": True,
            },
        },
        "data": {DATASET: {
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "payload_path": manifest["payload"]["path"],
            "payload_sha256": manifest["payload"]["sha256"],
            "unseen_path": manifest["unseen"]["path"],
            "unseen_sha256": manifest["unseen"]["sha256"],
        }},
        "candidate_contract": {
            "controlled_difference": "cumulative OBS_MASK -> POLY -> CAL",
            "AR_BASE": "frozen prior-round autoregressive control at epoch 58",
            "OBS_MASK": "AR_BASE plus train-only synthetic virtual sensors",
            "OBS_MASK_POLY": (
                "OBS_MASK plus sender-reliability-gated I/A/A2 input residual"),
            "OBS_MASK_POLY_CAL": (
                "OBS_MASK_POLY plus target-blind calibration head and fixed "
                "train-scale equal-bin Huber term"),
            "unchanged": [
                "decoder/teacher forcing", "data/split/scaler/schedule",
                "shared GRIN", "ALFTemporal/patches", "ALFGraph/supports/router",
                "FiLM/graph refiner", "base objective", "optimizer/lr", "seed",
                "epoch budget/checkpoint metric",
            ],
        },
        "augmentation_contract": {
            **MASK_RECIPE,
            "candidate_nodes": "train signal_scope K minus lifecycle cohort",
            "selection_inputs": "run_seed, epoch, sample_index, input availability",
            "future_target_used_for_selection": False,
            "validation_augmentation": False,
            "test_access": False,
            "synthetic_role": "VS",
            "teacher_forcing_on_synthetic_nodes": False,
        },
        "polynomial_contract": {
            **POLY_RECIPE,
            "supports": ["identity", "sender_gated_A", "sender_gated_A2"],
            "unavailable_sender_contribution": 0,
            "isolated_receiver_fallback": "identity",
            "zero_initialised_residual": True,
        },
        "calibration_contract": {
            **CAL_RECIPE,
            "inference_inputs": ["prediction", "lifecycle_role", "horizon"],
            "future_target_at_inference": False,
            "bin_boundaries": [0, ".05*train_std", ".10*train_std",
                               ".20*train_std", ".50*train_std", "inf"],
            "target_dependent_division": False,
            "zero_initialised_residual": True,
        },
        "training_contract": {
            **TRAINING, "early_termination": True,
            "all_epochs_checkpoint_eligible": True,
            "checkpoint_metric": "validation 33-cell main/LifeAvg/MAE",
            "equal_epoch_budget": True,
        },
        "selection_rule": {
            "primary": (
                "lowest validation main/LifeAvg/MAE among candidates that pass "
                "all mandatory gates"),
            "mandatory_unseen_gate": (
                "candidate main/Unseen MAE and RMSE must both be lower than AR_BASE"),
            "mandatory_lifeavg_rmse_gate": (
                "candidate main/LifeAvg/RMSE must be lower than AR_BASE"),
            "secondary": "candidate wins at least 17 of the same 33 cells",
            "mape": "diagnostic; cannot override MAE/RMSE gates",
            "test_metrics_may_influence_selection": False,
        },
        "code_sha256": {path: sha256_file(ROOT / path) for path in _code_sources()},
    }
    plan["plan_sha256"] = build_final_plan.plan_hash(plan)
    return plan


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text("utf-8"))
    if plan.get("plan_sha256") != build_final_plan.plan_hash(plan):
        raise ValueError("MASK_REC plan hash mismatch")
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
    reference = plan["reference_control"]
    for name, digest in reference["files_sha256"].items():
        path = ROOT / reference["path"] / name
        if not path.exists() or sha256_file(path) != digest:
            drift.append(f"AR_BASE reference changed: {name}")
    if drift:
        raise ValueError("MASK_REC plan drift:\n  - " + "\n  - ".join(drift))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        assert_plan_still_holds(load_plan())
        expected = render_configs()
        stale = [str(path.relative_to(ROOT)) for path, content in expected.items()
                 if not path.exists() or path.read_text("utf-8") != content]
        if stale:
            raise SystemExit("MASK_REC configs differ: " + ", ".join(stale))
        print("final structure plan holds: 3 new runs + frozen AR_BASE reference")
        return 0
    write_configs()
    plan = build_plan()
    PLAN_PATH.write_text(json.dumps(
        plan, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")
    print(f"plan {plan['plan_sha256']}: 3 new runs + frozen AR_BASE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
