"""The v4 run format: one reader, one schema, two files per dataset.

These tests are about the container and the guard rails, not about the numbers.
The defect they exist to prevent is the v3 one: four ``.pkl`` files that were
actually NumPy object containers, read by three separate hand-rolled
``np.load(..., allow_pickle=True).item()`` spellings, so there was no single
place that could say what a payload must contain.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from lib.v4_payload import (
    InvalidPayload,
    load_legacy_payload,
    load_raw_payload,
    load_v4_payload,
    payload_schema,
    save_v4_payload,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "configs/paper/v4_data_manifests"
DATASETS = ("PEMS03", "PEMS04", "PEMS08", "PEMSBAY")


def _payload(tmp_path, name="raw_data.pkl", nodes=4, steps=6, channels=2, positions=None):
    return save_v4_payload(
        tmp_path / name,
        data=np.arange(steps * nodes * channels, dtype=np.float32).reshape(steps, nodes, channels),
        adj=np.eye(nodes, dtype=np.float32),
        time_stamp=np.array([f"2018-09-01 00:{index:02d}:00" for index in range(steps)]),
        pos_stamp=positions)


# --- container format ---------------------------------------------------------


def test_v4_payload_is_a_real_pickle_not_a_numpy_container(tmp_path):
    path = _payload(tmp_path)
    with path.open("rb") as handle:
        assert handle.read(6) != b"\x93NUMPY", "v4 must not be a .npy in .pkl clothing"
        handle.seek(0)
        assert isinstance(pickle.load(handle), dict)
    assert payload_schema(path) == "v4"


def test_round_trip_preserves_values_dtypes_and_absent_positions(tmp_path):
    positions = np.array([[1.5, 2.5]] * 4, dtype=np.float32)
    with_pos = load_v4_payload(_payload(tmp_path, "a.pkl", positions=positions))
    assert np.array_equal(with_pos["pos_stamp"], positions)
    without = load_v4_payload(_payload(tmp_path, "b.pkl"))
    # Absent coordinates are None, never zeros or node indices, which would look
    # like an answer to StructuralNodeEncoder and to the layout.
    assert without["pos_stamp"] is None
    assert without["data"].dtype == np.float32 and without["adj"].dtype == np.float32


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p.pop("adj"), "missing"),
    (lambda p: p.pop("schema_version"), "schema_version"),
    (lambda p: p.update(schema_version=3), "schema_version"),
    (lambda p: p.update(extra=1), "unexpected"),
    (lambda p: p.update(data=p["data"].astype(np.float64)), "float32"),
    (lambda p: p.update(adj=np.eye(9, dtype=np.float32)), "nodes"),
    (lambda p: p.update(time_stamp=np.array(["x"])), "time_stamp length"),
    (lambda p: p.update(pos_stamp=np.zeros((9, 2), dtype=np.float32)), "pos_stamp rows"),
])
def test_every_schema_defect_raises_rather_than_being_repaired(tmp_path, mutate, message):
    path = _payload(tmp_path)
    payload = load_v4_payload(path)
    mutate(payload)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)
    with pytest.raises(InvalidPayload, match=message):
        load_v4_payload(path)


# --- the one shared reader ----------------------------------------------------


def test_legacy_container_is_read_only_down_the_explicit_legacy_path(tmp_path):
    path = tmp_path / "legacy.pkl"
    np.save(path.with_suffix(".pkl.npy"), {"data": np.zeros((2, 2, 1), np.float32)})
    path.with_suffix(".pkl.npy").rename(path)
    assert payload_schema(path) == "v3_legacy"
    assert set(load_legacy_payload(path)) == {"data"}
    # auto-detection routes it, but asserting v4 must refuse rather than fall
    # back -- a silent fallback is how a malformed v4 file would go unnoticed.
    assert set(load_raw_payload(path)) == {"data"}
    with pytest.raises(InvalidPayload, match="expected v4 payload, found v3_legacy"):
        load_raw_payload(path, schema="v4")


def test_a_plain_v3_pickle_is_classified_by_its_declaration_not_its_container(tmp_path):
    """data/METR-LA/raw_data.pkl is a genuine pickle holding float64 arrays.

    Classifying by container magic called it v4 and then rejected it for a
    float32 requirement it never claimed to meet, which broke every model that
    reads node counts off a v3 payload. The declaration is what decides.
    """
    path = tmp_path / "raw_data.pkl"
    with path.open("wb") as handle:
        pickle.dump({"data": np.zeros((3, 2, 1)), "adj": np.eye(2)}, handle)
    assert payload_schema(path) == "v3_legacy"
    assert set(load_raw_payload(path)) == {"data", "adj"}
    with pytest.raises(InvalidPayload, match="expected v4 payload"):
        load_raw_payload(path, schema="v4")


def test_real_v3_payloads_on_disk_still_load(tmp_path):
    for name in ("METR-LA", "PEMS03", "PEMS-BAY"):
        path = ROOT / "data" / name / "raw_data.pkl"
        if not path.exists():
            pytest.skip("v3 payloads not present in this checkout")
        assert payload_schema(path) == "v3_legacy"
        assert "adj" in load_raw_payload(path)


def test_a_corrupt_v4_file_is_never_retried_as_legacy(tmp_path):
    path = tmp_path / "raw_data.pkl"
    path.write_bytes(b"not a pickle and not a npy")
    with pytest.raises(InvalidPayload, match="not a canonical v4 pickle"):
        load_raw_payload(path)


def test_production_read_sites_share_the_one_entry_point():
    """Guards the consolidation itself: no module may reintroduce its own read."""
    sources = [ROOT / "lib/utils.py", ROOT / "data_provider/graph_data_loader.py",
               ROOT / "model/comparison_model/stgode/generate_matrices.py"]
    for source in sources:
        text = source.read_text("utf-8")
        assert "load_raw_payload" in text, f"{source.name} does not use the shared reader"
        assert "np.load(self.data_path" not in text
        assert "np.load(data_path, allow_pickle" not in text


# --- the built run directories ------------------------------------------------


@pytest.mark.parametrize("dataset", DATASETS)
def test_run_manifest_is_tracked_and_declares_exactly_two_files(dataset):
    manifest = json.loads((MANIFEST_DIR / f"{dataset}_run_manifest.json").read_text("utf-8"))
    assert manifest["files"] == ["raw_data.pkl", "unknown_nodes.npy"]
    assert manifest["dataset"] == dataset
    readiness = manifest["readiness"]
    assert readiness["data_ready"] and readiness["graph_ready"]
    # Part B has not landed, so nothing may claim to be trainable yet.
    assert not readiness["exposure_ready"] and not readiness["training_ready"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_manifest_agrees_with_the_frozen_registry(dataset):
    manifest = json.loads((MANIFEST_DIR / f"{dataset}_run_manifest.json").read_text("utf-8"))
    entry = json.loads((ROOT / "configs/paper/v4_dataset_registry.json").read_text("utf-8"))
    entry = entry["datasets"][dataset]
    assert manifest["payload"]["shape"] == list(entry["expected_shape"])
    assert manifest["target"]["name"] == entry["target"]
    assert manifest["target"]["channel"] < entry["expected_shape"][2]


@pytest.mark.parametrize("dataset", DATASETS)
def test_unseen_pool_is_a_quarter_of_the_nodes_and_well_formed(dataset):
    manifest = json.loads((MANIFEST_DIR / f"{dataset}_run_manifest.json").read_text("utf-8"))
    nodes = manifest["payload"]["shape"][1]
    unseen = manifest["unseen"]
    assert unseen["count"] < nodes
    assert 0.24 <= unseen["ratio"] <= 0.26, "the master pool is ~25% by design"
    path = ROOT / unseen["path"]
    if not path.exists():
        pytest.skip("run directory not built on this machine")
    ids = np.load(path)
    assert ids.dtype == np.int64 and ids.ndim == 1
    assert np.array_equal(ids, np.unique(ids))
    assert ids.min() >= 0 and ids.max() < nodes


@pytest.mark.parametrize("dataset", DATASETS)
def test_v4_adjacency_is_directed(dataset):
    """The v3 converter symmetrised the graph, which collapsed forward diffusion
    onto backward diffusion and cost the operator set a distinct member."""
    manifest = json.loads((MANIFEST_DIR / f"{dataset}_run_manifest.json").read_text("utf-8"))
    assert manifest["payload"]["adj_directed"] is True


# --- preflight actually recomputes -------------------------------------------


def _run_fixture(tmp_path):
    from scripts import v4_dataset_preflight as pre

    directory = tmp_path / "PEMS08"
    directory.mkdir()
    entry = json.loads((ROOT / "configs/paper/v4_dataset_registry.json").read_text("utf-8"))
    entry = entry["datasets"]["PEMS08"]
    steps, nodes, channels = entry["expected_shape"]
    payload_path = save_v4_payload(
        directory / "raw_data.pkl",
        data=np.zeros((steps, nodes, channels), np.float32),
        adj=np.eye(nodes, dtype=np.float32),
        time_stamp=np.array([
            str(np.datetime64(entry["start_time"]) + np.timedelta64(5 * index, "m"))
            for index in range(steps)]),
        pos_stamp=None)
    unseen_path = directory / "unknown_nodes.npy"
    np.save(unseen_path, np.arange(42, dtype=np.int64))
    manifest = {
        "dataset": "PEMS08", "run_directory": str(directory.relative_to(ROOT))
        if directory.is_relative_to(ROOT) else str(directory),
        "payload": {"path": str(payload_path), "sha256": pre.file_sha256(payload_path)},
        "unseen": {"path": str(unseen_path), "sha256": pre.file_sha256(unseen_path),
                   "count": 42},
        "readiness": {"data_ready": True, "graph_ready": True,
                      "exposure_ready": False, "training_ready": False},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    return pre, manifest, manifest_path


def test_preflight_accepts_a_consistent_run_directory(tmp_path, monkeypatch):
    pre, _, manifest_path = _run_fixture(tmp_path)
    monkeypatch.setattr(pre, "ROOT", Path("/"))
    assert pre.run_preflight(manifest_path)["dataset"] == "PEMS08"


def test_preflight_recomputes_the_payload_hash_rather_than_trusting_it(tmp_path, monkeypatch):
    pre, manifest, manifest_path = _run_fixture(tmp_path)
    monkeypatch.setattr(pre, "ROOT", Path("/"))
    payload_path = Path(manifest["payload"]["path"])
    payload = load_v4_payload(payload_path)
    payload["data"][0, 0, 0] = 1.0                       # one byte, hash unchanged in manifest
    with payload_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)
    with pytest.raises(SystemExit, match="payload hash mismatch"):
        pre.run_preflight(manifest_path)


def test_preflight_rejects_a_third_file_in_the_run_directory(tmp_path, monkeypatch):
    pre, manifest, manifest_path = _run_fixture(tmp_path)
    monkeypatch.setattr(pre, "ROOT", Path("/"))
    (Path(manifest["payload"]["path"]).parent / "extra.npy").write_bytes(b"")
    with pytest.raises(SystemExit, match="expected exactly the two v4 files"):
        pre.run_preflight(manifest_path)


def test_preflight_rejects_training_ready_without_exposure_ready(tmp_path, monkeypatch):
    pre, manifest, manifest_path = _run_fixture(tmp_path)
    monkeypatch.setattr(pre, "ROOT", Path("/"))
    manifest["readiness"]["training_ready"] = True
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(SystemExit, match="training_ready is set while exposure_ready is not"):
        pre.run_preflight(manifest_path)


def test_preflight_rejects_an_unsorted_or_out_of_range_unseen_pool(tmp_path, monkeypatch):
    pre, manifest, manifest_path = _run_fixture(tmp_path)
    monkeypatch.setattr(pre, "ROOT", Path("/"))
    unseen_path = Path(manifest["unseen"]["path"])
    np.save(unseen_path, np.array([5, 3, 999] + list(range(39)), dtype=np.int64))
    manifest["unseen"]["sha256"] = pre.file_sha256(unseen_path)
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(SystemExit) as refusal:
        pre.run_preflight(manifest_path)
    assert "outside [0, 170)" in str(refusal.value)
    assert "not sorted and unique" in str(refusal.value)
