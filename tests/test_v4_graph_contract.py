import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lib.runtime_context import build_runtime_context
from lib.v4_payload import load_v4_payload

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("PEMS03", "PEMS04", "PEMS08", "METRLA", "PEMSBAY")


def _manifest(dataset):
    return json.loads((ROOT / "configs/paper/v4_data_manifests"
                       / f"{dataset}_run_manifest.json").read_text("utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize("dataset", DATASETS)
def test_manifest_binds_the_two_runtime_files(dataset):
    manifest = _manifest(dataset)
    assert manifest["files"] == ["raw_data.pkl", "unknown_nodes.npy"]
    assert _sha(ROOT / manifest["payload"]["path"]) == manifest["payload"]["sha256"]
    assert _sha(ROOT / manifest["unseen"]["path"]) == manifest["unseen"]["sha256"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_target_channel_and_shape_are_real(dataset):
    manifest = _manifest(dataset)
    payload = load_v4_payload(ROOT / manifest["payload"]["path"])
    assert list(payload["data"].shape) == manifest["payload"]["shape"]
    channel = int(manifest["target"]["channel"])
    assert np.isfinite(payload["data"][:, :, channel]).any()
    expected = ("q", 0) if dataset in ("PEMS03", "PEMS04", "PEMS08") else ("v", 2)
    assert (manifest["target"]["name"], channel) == expected


@pytest.mark.parametrize("dataset", DATASETS)
def test_resolved_layout_fits_the_frozen_patch_capacity(dataset):
    context = build_runtime_context(dataset, "train", seed=32)
    manifest = _manifest(dataset)
    largest = max(len(part) for part in context.layout_parts)
    assert int(manifest["layout"]["spa_patchsize"]) >= largest
    expected = ("kdtree_on_verified_wgs84" if dataset in ("METRLA", "PEMSBAY")
                else "graph_native_component_bfs")
    assert manifest["layout"]["algorithm"] == expected


@pytest.mark.parametrize("dataset", DATASETS)
def test_original_source_evidence_is_preserved(dataset):
    directory = ROOT / "data" / ({"METRLA": "METR-LA", "PEMSBAY": "PEMS-BAY"}
                                 .get(dataset, dataset)) / "original"
    assert directory.is_dir() and any(directory.iterdir())


def test_no_candidate_data_directories_remain():
    assert not list(ROOT.glob("data*/**/*candidate*"))
