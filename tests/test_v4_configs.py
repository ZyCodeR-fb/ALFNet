"""Contracts for the generated unified-BASE configuration matrix."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from args import load_args_from_config
from scripts.build_final_configs import (
    CONFIG_DIR, DATASETS, MODELS, SOURCE_LOSS, render,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "configs/paper/v4_data_manifests"


def path_for(dataset, model):
    return ROOT / "configs" / CONFIG_DIR[model] / f"{dataset}.yaml"


def test_generator_is_byte_stable():
    result = subprocess.run(
        [sys.executable, "scripts/build_final_configs.py", "--check"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(render()) == 75


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("model", MODELS)
def test_every_outer_config_matches_the_data_contract(dataset, model):
    path = path_for(dataset, model)
    assert path.exists()
    args = load_args_from_config(str(path), {})
    manifest = json.loads(
        (MANIFESTS / f"{dataset}_run_manifest.json").read_text("utf-8"))
    assert args.model == model
    assert args.input_features == args.target_features == manifest["target"]["name"]
    assert Path(args.data_path).resolve() == (ROOT / manifest["payload"]["path"]).resolve()
    assert Path(args.unknown_nodes_path).resolve() == (
        ROOT / manifest["unseen"]["path"]).resolve()
    assert (args.input_len, args.pred_len, args.slide_step) == (12, 12, 1)
    assert (args.lcbt_train_end_ratio, args.lcbt_val_end_ratio,
            args.lcbt_test_start_ratio) == (0.6, 0.8, 0.8)
    assert args.loss == SOURCE_LOSS[model]
    assert args.stop_based == "val_lifecycle"
    if model == "ALFNet":
        assert args.model_variant == "alfnet_unified_base_v4"
        assert args.enc_in == 7
    else:
        assert args.enc_in == args.raw_enc_in == 1


@pytest.mark.parametrize("dataset", DATASETS)
def test_alfnet_nested_configs_are_base_and_manifest_driven(dataset):
    outer = load_args_from_config(str(path_for(dataset, "ALFNet")), {})
    graph = load_args_from_config(str(ROOT / "configs/ALFGraph" / f"{dataset}.yaml"), {})
    patch = load_args_from_config(
        str(ROOT / "configs/ALFTemporal" / f"{dataset}.yaml"), {})
    manifest = json.loads(
        (MANIFESTS / f"{dataset}_run_manifest.json").read_text("utf-8"))
    assert graph.model_family_cfg["recipe"]["moge_support_set"] == \
           "unified_base_legacy5_v1"
    assert patch.model_family_cfg["model"]["spa_patchsize"] == \
           manifest["layout"]["spa_patchsize"]
    assert outer.model_family_cfg["experts"]["graph"]["config_path"] == \
           f"./configs/ALFGraph/{dataset}.yaml"
    assert outer.model_family_cfg["experts"]["patch"]["config_path"] == \
           f"./configs/ALFTemporal/{dataset}.yaml"
    assert "safe_relative" not in json.dumps(outer.model_family_cfg)
    assert "candidate" not in json.dumps(outer.model_family_cfg)
