import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from lib.v4_payload import InvalidPayload, load_v4_payload, save_v4_payload

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("PEMS03", "PEMS04", "PEMS08", "METRLA", "PEMSBAY")


def test_v4_payload_round_trip_and_closed_schema(tmp_path):
    path = save_v4_payload(
        tmp_path / "raw_data.pkl", data=np.ones((6, 4, 1), np.float32),
        adj=np.eye(4, dtype=np.float32),
        time_stamp=np.arange(6).astype("datetime64[m]"), pos_stamp=None)
    payload = load_v4_payload(path)
    assert payload["data"].dtype == np.float32
    with path.open("wb") as handle:
        pickle.dump({"data": np.ones((2, 2, 1))}, handle)
    with pytest.raises(InvalidPayload):
        load_v4_payload(path)


@pytest.mark.parametrize("dataset", DATASETS)
def test_runtime_directory_has_exactly_payload_and_unknown_nodes(dataset):
    path_name = {"PEMSBAY": "PEMS-BAY", "METRLA": "METR-LA"}.get(dataset, dataset)
    directory = ROOT / "data_v4" / path_name
    assert sorted(path.name for path in directory.iterdir()) == [
        "raw_data.pkl", "unknown_nodes.npy"]
    manifest = json.loads((ROOT / "configs/paper/v4_data_manifests"
                           / f"{dataset}_run_manifest.json").read_text("utf-8"))
    assert manifest["files"] == ["raw_data.pkl", "unknown_nodes.npy"]
