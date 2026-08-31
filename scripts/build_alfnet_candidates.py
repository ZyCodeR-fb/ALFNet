#!/usr/bin/env python
"""Build the bounded PEMS03/seed32 ALFNet validation-only candidate plan."""

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

DATASET = "PEMS03"
SEED = 32
PLAN_PATH = ROOT / "configs/paper/alfnet_candidate_plan.json"
CONFIG_ROOT = ROOT / "configs/paper/alfnet_candidates/PEMS03"
CANDIDATES = ("BASE", "UNSEEN", "LOW", "SUP", "PATCH", "LOW_SUP", "ALL")


def _recipe_flags(name):
    return {
        "unseen": name in {"UNSEEN", "ALL"},
        "low": name in {"LOW", "LOW_SUP", "ALL"},
        "support": name in {"SUP", "LOW_SUP", "ALL"},
        "patch": name in {"PATCH", "ALL"},
    }


def render_configs():
    files = {}
    for name in CANDIDATES:
        flags = _recipe_flags(name)
        outer = deepcopy(build_final_configs.alfnet(DATASET))
        graph = deepcopy(build_final_configs.alfgraph(DATASET))
        temporal = deepcopy(build_final_configs.alftemporal(DATASET))
        slug = name.lower()
        outer["runtime"]["model_variant"] = f"alfnet_pems03_{slug}_candidate_v1"
        outer["runtime"]["model_recipe"] = f"bounded_validation_candidate_{slug}"
        graph["runtime"]["model_variant"] = f"alfgraph_pems03_{slug}_candidate_v1"
        temporal["runtime"]["model_variant"] = f"alftemporal_pems03_{slug}_candidate_v1"
        outer["family"]["experts"]["graph"]["config_path"] = \
            f"./configs/paper/alfnet_candidates/PEMS03/{name}_ALFGraph.yaml"
        outer["family"]["experts"]["patch"]["config_path"] = \
            f"./configs/paper/alfnet_candidates/PEMS03/{name}_ALFTemporal.yaml"
        if flags["unseen"]:
            outer["family"]["recipe"]["graph_diffusion_role_scale"]["aas"] = 1.0
        if flags["low"]:
            outer["family"]["recipe"].update({
                "forecast_scale_loss_mode": "signed_log_huber",
                "forecast_scale_loss_weight": 0.05,
                "forecast_scale_loss_scale_fraction": 0.05,
            })
        if flags["support"]:
            graph["family"]["recipe"]["moge_support_set"] = \
                "distinct_multiscale5_v2"
        if flags["patch"]:
            temporal["family"]["model"]["spa_patchsize"] = 13
        for suffix, payload in (("ALFNet", outer), ("ALFGraph", graph),
                                ("ALFTemporal", temporal)):
            path = CONFIG_ROOT / f"{name}_{suffix}.yaml"
            files[path] = ("# GENERATED validation-only ALFNet candidate; do not edit.\n"
                           + OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True))
    return files


def write_configs():
    for path, text in render_configs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, "utf-8")


def _code_sources():
    return tuple(dict.fromkeys(build_final_plan._code_sources() + (
        "scripts/build_alfnet_candidates.py",
        "scripts/collect_alfnet_candidates.py",
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
        "kind": "v4_alfnet_validation_candidates",
        "status": "development_validation_only",
        "selection_split": "validation",
        "test_access": "forbidden",
        "dataset": DATASET, "datasets": [DATASET],
        "models": ["ALFNet"], "seeds": [SEED],
        "candidates": list(CANDIDATES), "runs": runs,
        "trial_budget": len(runs), "configs": configs,
        "nested_configs": nested,
        "output_root": "outputs_v4/alfnet_candidates_pems03",
        "data": {DATASET: {
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "payload_path": manifest["payload"]["path"],
            "payload_sha256": manifest["payload"]["sha256"],
            "unseen_path": manifest["unseen"]["path"],
            "unseen_sha256": manifest["unseen"]["sha256"],
        }},
        "candidate_contract": {
            "BASE": [], "UNSEEN": ["AAS graph output refinement"],
            "LOW": ["signed-log Huber, weight 0.05"],
            "SUP": ["five distinct multiscale supports"],
            "PATCH": ["one graph-boundary context node per patch"],
            "LOW_SUP": ["LOW", "SUP"],
            "ALL": ["UNSEEN", "LOW", "SUP", "PATCH"],
        },
        "selection_rule": {
            "primary": "most of 33 validation cells lower than rerun BASE",
            "secondary": "most PEMS03 seed32 cells lower than fixed 11-model envelope",
            "tertiary": "lowest validation LifeAvg MAE",
            "test_metrics_may_influence_selection": False,
            "BASE_meaning": "current ALFNet rerun, not MoGERNN",
        },
        "code_sha256": {path: sha256_file(ROOT / path) for path in _code_sources()},
    }
    plan["plan_sha256"] = build_final_plan.plan_hash(plan)
    return plan


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text("utf-8"))
    if plan.get("plan_sha256") != build_final_plan.plan_hash(plan):
        raise ValueError("candidate plan content no longer matches plan_sha256")
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
        raise ValueError("candidate plan drift:\n  - " + "\n  - ".join(drift))
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
        print(f"candidate plan holds: {len(CANDIDATES)} PEMS03 seed32 runs")
        return 0
    write_configs()
    plan = build_plan()
    PLAN_PATH.write_text(json.dumps(
        plan, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")
    print(f"plan {plan['plan_sha256']}: {len(CANDIDATES)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
