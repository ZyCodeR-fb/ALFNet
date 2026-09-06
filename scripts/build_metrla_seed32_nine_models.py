#!/usr/bin/env python
"""Freeze the METR-LA seed-32 nine-model validation-only supplement.

This is an exploratory development experiment.  ALFNet uses the already
selected AR_BASE training recipe and the historically selected METR-LA
ALFTemporal padding capacity of 8.  Comparison models share the v4 data and
evaluation protocol but retain their own architectures and objectives; they
never receive ALFNet-only lifecycle features or losses.
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    build_ar_base_transfer,
    build_corrected_baseline_seed32,
    build_final_configs,
    build_final_plan,
)
from scripts.v4_utils import sha256_file  # noqa: E402

DATASET = "METRLA"
SEED = 32
MODELS = (
    "ALFNet", "MoGERNN", "STID", "STBP", "HA",
    "FCLSTM", "STGCN", "STGODE", "SLPF",
)
PLAN_PATH = ROOT / "configs/paper/metrla_seed32_nine_models_plan.json"
CONFIG_ROOT = ROOT / "configs/paper/metrla_seed32_nine_models"
ALF_PATCH_CAPACITY = 8


def _alf_configs():
    outer = deepcopy(build_final_configs.alfnet(DATASET))
    graph = deepcopy(build_final_configs.alfgraph(DATASET))
    temporal = deepcopy(build_final_configs.alftemporal(DATASET))
    outer["runtime"].update({
        "model_variant": "alfnet_ar_base_metrla_patch8_v1",
        "model_recipe": "selected_ar_base_metrla_patch8",
    })
    graph["runtime"]["model_variant"] = "alfgraph_ar_base_metrla_patch8_v1"
    temporal["runtime"]["model_variant"] = \
        "alftemporal_ar_base_metrla_patch8_v1"
    outer["family"]["experts"]["graph"]["config_path"] = (
        "./configs/paper/metrla_seed32_nine_models/METRLA_ALFGraph.yaml")
    outer["family"]["experts"]["patch"]["config_path"] = (
        "./configs/paper/metrla_seed32_nine_models/METRLA_ALFTemporal.yaml")
    graph["family"]["recipe"]["decoder_output_mode"] = "residual_delta"
    temporal["family"]["model"]["spa_patchsize"] = ALF_PATCH_CAPACITY
    for payload in (outer, graph, temporal):
        payload["training"].update(build_ar_base_transfer.TRAINING)
    return outer, graph, temporal


def _baseline(model):
    if model == "MoGERNN":
        return deepcopy(build_corrected_baseline_seed32._mogernn(DATASET))
    cfg = deepcopy(build_final_configs.common(
        DATASET, model, t_mark=model in {"STID", "SLPF"}))
    cfg["runtime"].update({
        "model_variant": f"{model.lower()}_metrla_seed32_v1",
        "model_recipe": "source_architecture_shared_v4_protocol",
    })
    cfg["family"] = deepcopy(
        build_final_configs.family(model, build_final_configs.facts(DATASET)))
    cfg["training"].update({
        "teacher_forcing_mode": "free_run",
        "teacher_forcing_decay_epochs": 40,
        "early_stopping_start_epoch": 0,
        "patience": 10,
    })
    return cfg


def rendered_configs():
    outer, graph, temporal = _alf_configs()
    payloads = {
        "METRLA_ALFNet.yaml": outer,
        "METRLA_ALFGraph.yaml": graph,
        "METRLA_ALFTemporal.yaml": temporal,
    }
    for model in MODELS[1:]:
        payloads[f"METRLA_{model}.yaml"] = _baseline(model)
    header = (
        "# GENERATED METR-LA seed32 validation supplement; do not edit.\n"
        "# Test access is forbidden; comparison models receive no ALF-only inputs.\n"
    )
    return {
        CONFIG_ROOT / name:
            header + OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True)
        for name, payload in payloads.items()
    }


def write_configs():
    for path, content in rendered_configs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, "utf-8")


def _code_sources():
    return tuple(dict.fromkeys(build_final_plan._code_sources() + (
        "scripts/build_ar_base_transfer.py",
        "scripts/build_corrected_baseline_seed32.py",
        "scripts/build_metrla_seed32_nine_models.py",
    )))


def build_plan():
    manifest_path = (
        ROOT / "configs/paper/v4_data_manifests/METRLA_run_manifest.json")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    data = {
        DATASET: {
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "payload_path": manifest["payload"]["path"],
            "payload_sha256": manifest["payload"]["sha256"],
            "unseen_path": manifest["unseen"]["path"],
            "unseen_sha256": manifest["unseen"]["sha256"],
            "target": manifest["target"]["name"],
            "target_channel": int(manifest["target"]["channel"]),
            "num_nodes": int(manifest["canonical_sensor_ids"]["count"]),
            "layout_algorithm": manifest["layout"]["algorithm"],
            "resolved_max_partition_size": int(
                manifest["layout"]["spa_patchsize"]),
        }
    }
    configs = {}
    for model in MODELS:
        path = CONFIG_ROOT / f"METRLA_{model}.yaml"
        configs[f"{DATASET}_{model}"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
    nested = {DATASET: {}}
    for expert in ("ALFGraph", "ALFTemporal"):
        path = CONFIG_ROOT / f"METRLA_{expert}.yaml"
        nested[DATASET][expert] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
    # The preferred four plus HA enter phase 0, so JOBS=5 starts all four
    # requested priority models immediately.  The remaining four start only
    # after the health phase succeeds.
    runs = []
    for index, model in enumerate(MODELS):
        runs.append({
            "dataset": DATASET,
            "model": model,
            "seed": SEED,
            "config": configs[f"{DATASET}_{model}"]["path"],
            "phase": 0 if index < 5 else 1,
            "execution_action": "train",
        })
    plan = {
        "schema_version": 1,
        "kind": "v4_metrla_seed32_nine_models_validation_runs",
        "status": "exploratory_development_validation_only",
        "selection_split": "validation",
        "test_access": "forbidden",
        "datasets": [DATASET],
        "models": list(MODELS),
        "seeds": [SEED],
        "trial_budget": len(runs),
        "runs": runs,
        "configs": configs,
        "nested_configs": nested,
        "data": data,
        "output_root": "outputs_v4/metrla_seed32_nine_models",
        "selection_disclosure": {
            "dataset_inclusion_after_this_result_is_exploratory": True,
            "patch_capacity_selected_from_prior_metrla_validation": True,
            "confirmation_claim_permitted": False,
        },
        "alfnet_contract": {
            "candidate": "AR_BASE",
            "decoder": "12-step recurrent residual decoder",
            "teacher_forcing": "linear decay to zero at epoch 40",
            "early_stopping_start_epoch": 40,
            "patience": 10,
            "resolved_max_partition_size": 7,
            "alftemporal_spa_patchsize_capacity": ALF_PATCH_CAPACITY,
            "padding_only_no_layout_membership_change": True,
        },
        "fairness_contract": {
            "common": [
                "same METR-LA v4 payload and target speed channel 2",
                "same full-N phase-scoped visible signal",
                "same K/V/T nodes and 60/20/20 time split",
                "same train-known-only scaler, graph, origins, cohorts and masks",
                "same run seed 32 and validation 33-cell LifeAvg-MAE selection",
                "same tie rule: keep earlier checkpoint",
            ],
            "alfnet_only": [
                "lifecycle role/age/availability context",
                "ALFGraph/ALFTemporal fusion and lifecycle router",
                "role-balanced ALF objective",
                "AR_BASE lifecycle-safe teacher forcing",
            ],
            "comparison_models": {
                "input": "common base signal/time marks/graph only",
                "alfnet_lifecycle_features": False,
                "alfnet_role_balanced_loss": False,
                "alfnet_router_or_refiner": False,
                "architecture": "source-native v4 adapter",
                "objective": "source-declared",
                "non_autoregressive_decoders": "target blind / no teacher forcing",
                "MoGERNN": (
                    "source autoregressive teacher forcing decay40, patience "
                    "starts epoch20 with patience3"),
            },
        },
        "training_contract": {
            "max_epoch": 100,
            "checkpoint_metric": "validation 33-cell main/LifeAvg/MAE",
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
        raise ValueError("METR-LA seed32 nine-model plan hash mismatch")
    return plan


def assert_plan_still_holds(plan):
    drift = []
    for key, item in plan["configs"].items():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            drift.append(f"{key}: config changed")
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
        raise ValueError("METR-LA seed32 nine-model plan drift:\n  - "
                         + "\n  - ".join(drift))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        plan = assert_plan_still_holds(load_plan())
        stale = [
            str(path.relative_to(ROOT))
            for path, content in rendered_configs().items()
            if not path.exists() or path.read_text("utf-8") != content
        ]
        if stale:
            raise SystemExit("METR-LA seed32 configs differ: " + ", ".join(stale))
        print(f"plan {plan['plan_sha256'][:16]} holds: 9 METR-LA seed32 runs")
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
