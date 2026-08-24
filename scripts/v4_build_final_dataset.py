"""Assemble the final v4 run directory for one dataset.

``data_v4/<dataset>/`` holds exactly two files -- the payload and the unseen node
pool. Manifests, graph contracts and candidates stay out of it, so a run
directory cannot be mistaken for carrying a third model input.

    python scripts/v4_build_final_dataset.py --dataset PEMS03
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.graph_native_layout import (  # noqa: E402
    connected_components, resolve_layout, structural_view,
)
from lib.v4_payload import load_legacy_payload, load_v4_payload, save_v4_payload  # noqa: E402

REGISTRY = ROOT / "configs/paper/v4_dataset_registry.json"
# Only the number of recursive splits is a policy input. Patch capacity is
# derived from the resolved membership as its exact largest part; keeping a
# hand-written +1 for four datasets but not the fifth created an undocumented
# dataset-specific architecture difference.
LAYOUT_SETTINGS = {
    dataset: {"recur_times": 5}
    for dataset in ("PEMS03", "PEMS04", "PEMS08", "METRLA", "PEMSBAY")
}
DEST_ROOT = ROOT / "data_v4"
# Directory name under data/ differs from the registry key for PEMS-BAY.
SOURCE_DIR = {"PEMS03": "PEMS03", "PEMS04": "PEMS04", "PEMS08": "PEMS08",
              "METRLA": "METR-LA", "PEMSBAY": "PEMS-BAY"}
DEST_DIR = dict(SOURCE_DIR)
ORIGINAL_SOURCE_DATASETS = {"PEMS03", "PEMS04", "PEMS08"}


def array_sha256(array):
    """Digest of exact bytes plus shape and dtype, so a reshape cannot collide."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def graph_fields(dataset, adjacency, positions):
    """The propagation graph, the structural projection and the layout.

    Merged into the run manifest rather than kept in a second file: a dataset
    with two manifests has two authorities, and the one that gets read is
    whichever the caller happened to open.
    """
    settings = LAYOUT_SETTINGS[dataset]
    view = structural_view(adjacency)
    # Resolved, not assumed: PEMS-BAY has coordinates and therefore a KD-tree
    # layout. Recording graph-native here while ALFTemporal built a KD-tree gave
    # the dataset two layouts and one hash that described neither.
    layout = resolve_layout(adjacency, positions, settings["recur_times"])
    spa_patchsize = max(len(part) for part in layout["parts"])
    return {
        "propagation_graph": {
            "role": "what the models diffuse over",
            "sha256": array_sha256(adjacency),
            "shape": list(adjacency.shape),
            "dtype": str(adjacency.dtype),
            "directed": not bool(np.allclose(adjacency, adjacency.T)),
            "weighted": bool(np.any((adjacency != 0) & (adjacency != 1))),
            "self_loops": int(np.count_nonzero(np.diag(adjacency))),
            "must_not_be_symmetrised": True,
        },
        "structural_view": {
            "role": "layout only; never diffused over",
            "derivation": "S = 1[(A + A.T) > 0] with the diagonal cleared",
            "sha256": array_sha256(view),
            "component_count": len(connected_components(view)),
            "component_sizes": [len(component) for component in connected_components(view)],
            "never_substituted_into_propagation": True,
        },
        "layout": {
            "algorithm": layout["algorithm"],
            "version": layout["version"],
            "recur_times": settings["recur_times"],
            "spa_patchsize": spa_patchsize,
            "padding_rule": (
                "spa_patchsize equals the largest resolved part; smaller parts "
                "are padded and no part is split"
            ),
            "membership_sha256": layout["membership_sha256"],
        },
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_payload(dataset, entry, source, destination):
    """Build canonical signal/time/graph arrays directly from immutable sources."""
    if dataset not in ORIGINAL_SOURCE_DATASETS:
        legacy = source / "raw_data.pkl"
        if legacy.exists():
            return load_legacy_payload(legacy)
        # Refinement removed duplicate active v3 payloads. METR-LA/PEMS-BAY's
        # canonical v4 bytes were already audited against original H5/CSV; use
        # that canonical container when only layout metadata is being rebuilt.
        canonical = destination / "raw_data.pkl"
        if not canonical.exists():
            raise SystemExit(f"{dataset}: missing canonical payload {canonical}")
        return load_v4_payload(canonical)
    signal_path = source / entry["signal_source"]
    edge_path = source / entry["edge_source"]
    with np.load(signal_path) as archive:
        keys = list(archive.keys())
        if len(keys) != 1:
            raise SystemExit(f"{signal_path}: expected one signal array, got {keys}")
        data = np.asarray(archive[keys[0]], dtype=np.float32)
    if tuple(data.shape) != tuple(entry["expected_shape"]):
        raise SystemExit(
            f"{dataset}: signal shape {data.shape} != {entry['expected_shape']}")
    stamps = pd.date_range(
        entry["start_time"], periods=data.shape[0],
        freq=f"{int(entry['freq_minutes'])}min",
    )
    if str(stamps[-1]) != entry["end_time"]:
        raise SystemExit(
            f"{dataset}: generated end {stamps[-1]} != {entry['end_time']}")
    edges = pd.read_csv(edge_path)
    columns = {str(column).lower(): column for column in edges.columns}
    if "from" not in columns or "to" not in columns:
        raise SystemExit(f"{edge_path}: missing from/to columns")
    pairs = edges[[columns["from"], columns["to"]]].to_numpy(dtype=np.int64)
    num_nodes = int(entry["num_nodes"])
    if pairs.min() < 0 or pairs.max() >= num_nodes:
        raise SystemExit(f"{edge_path}: node ids outside [0, {num_nodes})")
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adjacency[pairs[:, 0], pairs[:, 1]] = 1.0
    return {"data": data, "adj": adjacency,
            "time_stamp": stamps.to_numpy(), "pos_stamp": None}


def build(dataset):
    entry = json.loads(REGISTRY.read_text("utf-8"))["datasets"][dataset]
    source = ROOT / "data" / SOURCE_DIR[dataset]
    destination = DEST_ROOT / DEST_DIR[dataset]

    payload = _source_payload(dataset, entry, source, destination)
    positions = payload.get("pos_stamp")
    if positions is not None:
        positions = np.asarray(positions, dtype=np.float32)

    payload_path = save_v4_payload(
        destination / "raw_data.pkl",
        data=payload["data"], adj=payload["adj"],
        time_stamp=payload["time_stamp"], pos_stamp=positions)

    # Node ordering is unchanged from v3, so the frozen master unseen pool is
    # reused rather than redrawn. Redrawing it here would silently change the
    # experiment.
    source_unseen = source / "unknown_nodes.npy"
    if not source_unseen.exists():
        # Code refinement removed duplicate v3 inputs. The canonical v4 pool is
        # already beside the payload and remains immutable; rebuilding data
        # metadata must never redraw it.
        source_unseen = destination / "unknown_nodes.npy"
    if not source_unseen.exists():
        raise SystemExit(f"{dataset}: no frozen unknown_nodes.npy to reuse")
    unseen = np.asarray(np.load(source_unseen), dtype=np.int64)
    unseen = np.unique(unseen)
    num_nodes = payload["data"].shape[1]
    if unseen.min() < 0 or unseen.max() >= num_nodes:
        raise SystemExit(f"{dataset}: unseen ids outside [0, {num_nodes})")
    unseen_path = destination / "unknown_nodes.npy"
    np.save(unseen_path, unseen)

    files = sorted(p.name for p in destination.iterdir())
    if files != ["raw_data.pkl", "unknown_nodes.npy"]:
        raise SystemExit(f"{dataset}: run directory must hold exactly two files, found {files}")

    verified = load_v4_payload(payload_path)
    manifest = {
        "schema_version": 1,
        "kind": "v4_run_manifest",
        "dataset": dataset,
        "payload": {"path": payload_path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(payload_path),
                    "shape": list(verified["data"].shape),
                    "adj_directed": not bool(np.allclose(verified["adj"], verified["adj"].T)),
                    "pos_stamp": None if verified["pos_stamp"] is None else "wgs84_float32"},
        "unseen": {"path": unseen_path.relative_to(ROOT).as_posix(),
                   "sha256": sha256(unseen_path), "count": int(unseen.size),
                   "ratio": round(float(unseen.size / num_nodes), 4)},
        "target": {"name": entry["target"], "channel": entry["target_channel"]},
        "time": {"start": str(verified["time_stamp"][0]),
                 "end": str(verified["time_stamp"][-1])},
        "canonical_sensor_ids": {
            "count": num_nodes,
            "order": "payload node index order, 0-based",
            "sha256": array_sha256(np.arange(num_nodes, dtype=np.int64)),
        },
        "positions": {
            "policy": ("verified_wgs84" if verified["pos_stamp"] is not None
                       else "graph_native_no_coordinates"),
            "sha256": (None if verified["pos_stamp"] is None
                       else array_sha256(verified["pos_stamp"])),
            "note": ("real coordinates ship with this dataset"
                     if verified["pos_stamp"] is not None else
                     "no coordinates exist for this dataset; the layout is derived "
                     "from the graph. Zeros or node indices would look like an answer."),
        },
        "run_directory": destination.relative_to(ROOT).as_posix(),
        "files": files,
        **graph_fields(dataset, verified["adj"], verified["pos_stamp"]),
        # Where the runtime scopes live. They are built in memory by the
        # RuntimeDataContext and deliberately not written here: a node mask on
        # disk beside the payload is a third model input waiting to be read.
        "reserved_scopes": {
            "propagation_graph_scope": "full N, all splits",
            "available_signal_node_mask": "RuntimeDataContext.input_available_mask",
            "loss_target_node_mask": "RuntimeDataContext.loss_mask",
        },
        "readiness": {
            # data_ready/graph_ready are asserted below from the payload itself.
            # training_ready stays false until Part B's exposure wiring lands and
            # is verified; a manifest that claims readiness it cannot demonstrate
            # is worse than no manifest.
            "data_ready": True,
            "graph_ready": True,
            "exposure_ready": False,
            "training_ready": False,
        },
    }
    manifest_path = ROOT / "configs/paper/v4_data_manifests" / f"{dataset}_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    manifest["manifest"] = manifest_path.relative_to(ROOT).as_posix()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(SOURCE_DIR))
    args = parser.parse_args()
    report = build(args.dataset)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
