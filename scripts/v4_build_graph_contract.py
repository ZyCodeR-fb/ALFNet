"""Build the versioned GraphContract for a v4 dataset.

Three graph-shaped objects were previously conflated into one array. They are
separated here because they have different jobs and different correctness rules:

* **propagation graph** -- what the models diffuse over. Source-native and
  directed. Symmetrising it collapses forward and backward diffusion into the
  same operator, which is how v3 ended up with fewer distinct operators than
  labels on every PeMS dataset.
* **structural view** -- an undirected binary projection, used only for layout
  and for topology-derived node features. Never substituted back.
* **layout** -- the node ordering ALFTemporal partitions into patches. On
  datasets without GPS this comes from the graph, not from invented coordinates.

Both models must read the same propagation hash for a dataset; the contract is
what makes that checkable rather than assumed.

    python scripts/v4_build_graph_contract.py --dataset PEMS03
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.graph_native_layout import (  # noqa: E402
    LAYOUT_ALGORITHM,
    LAYOUT_VERSION,
    connected_components,
    layout_signature,
    structural_view,
)

REGISTRY_PATH = ROOT / "configs/paper/v4_dataset_registry.json"
TRACKED_MANIFESTS = ROOT / "configs/paper/v4_data_manifests"
CONTRACT_SCHEMA = 1

# Datasets with coordinates we trust. Everything else uses the graph-native
# layout and reports positions as absent rather than fabricating them.
VERIFIED_WGS84 = {"PEMSBAY"}
# recur_times / spa_patchsize come from the model configs; recorded here so the
# layout hash covers the partition actually used.
LAYOUT_SETTINGS = {
    "PEMS03": {"recur_times": 5, "spa_patchsize": 12},
    "PEMS04": {"recur_times": 5, "spa_patchsize": 11},
    "PEMS08": {"recur_times": 5, "spa_patchsize": 7},
    "PEMSBAY": {"recur_times": 5, "spa_patchsize": 12},
}


class ContractRefused(SystemExit):
    """The graph assets cannot be certified."""


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _unwrap(value):
    if isinstance(value, dict):
        return value
    return value.item() if getattr(value, "shape", None) == () else value


def _code_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:                                        # noqa: BLE001
        return None


def _load_payload(dataset):
    """The v4 payload for a dataset, candidate first, active only for PEMS-BAY."""
    candidate = ROOT / "data" / dataset / "v4_candidate" / "raw_data_v4_candidate.pkl.npy"
    if candidate.exists():
        return _unwrap(np.load(candidate, allow_pickle=True)), candidate
    if dataset == "PEMSBAY":
        active = ROOT / "data/PEMS-BAY/raw_data.pkl"
        return _unwrap(np.load(active, allow_pickle=True)), active
    raise ContractRefused(
        f"{dataset}: no v4 candidate payload. Build it first with "
        "scripts/v4_build_candidate_payload.py.")


def build(dataset, output_dir=None):
    registry = json.loads(REGISTRY_PATH.read_text("utf-8"))["datasets"]
    if dataset not in registry:
        raise ContractRefused(f"{dataset} is not in the v4 registry")
    entry = registry[dataset]
    settings = LAYOUT_SETTINGS[dataset]

    payload, payload_path = _load_payload(dataset)
    adjacency = np.asarray(payload["adj"])
    num_nodes = adjacency.shape[0]
    if num_nodes != entry["num_nodes"]:
        raise ContractRefused(
            f"{dataset}: adjacency has {num_nodes} nodes, registry says "
            f"{entry['num_nodes']}")

    symmetric = bool(np.allclose(adjacency, adjacency.T))
    if dataset != "PEMSBAY" and symmetric:
        raise ContractRefused(
            f"{dataset}: the propagation graph is symmetric. A symmetrised PeMS "
            "graph makes forward and backward diffusion the same operator; v4 "
            "requires the source-native directed graph.")

    view = structural_view(adjacency)
    components = connected_components(view)
    signature = layout_signature(adjacency, settings["recur_times"])

    # Positions: trusted coordinates, or explicitly none. No zeros, no indices.
    if dataset in VERIFIED_WGS84:
        positions = np.asarray(payload["pos_stamp"], dtype=np.float64)
        position_policy = "verified_wgs84"
        position_hash = _array_sha256(positions)
    else:
        position_policy = "graph_native_no_coordinates"
        position_hash = None

    sensor_ids = list(range(num_nodes))
    contract = {
        "graph_contract_schema": CONTRACT_SCHEMA,
        "dataset": dataset,
        "payload_version": "v4_candidate" if "v4_candidate" in payload_path.as_posix()
                           else "v3_active_unchanged",
        "target": {"name": entry["target"], "channel": entry["target_channel"]},
        "canonical_sensor_ids": {
            "count": num_nodes,
            "order": "payload node index order, 0-based",
            "sha256": _sha256_bytes(",".join(map(str, sensor_ids)).encode("utf-8")),
        },
        "propagation_graph": {
            "role": "what the models diffuse over",
            "directed": not symmetric,
            "weighted": bool(dataset == "PEMSBAY"),
            "shape": list(adjacency.shape),
            "dtype": str(adjacency.dtype),
            "sha256": _array_sha256(adjacency),
            "self_loops": int(np.count_nonzero(np.diag(adjacency))),
            "edge_rules": (
                "source-native; duplicate edges collapse to one; reciprocal "
                "edges are kept as two directed edges; self-loops are preserved "
                "as they appear in the source; ids outside [0, N) are refused at "
                "conversion time"),
            "must_not_be_symmetrised": True,
        },
        "structural_view": {
            "role": "layout and topology-derived node features only",
            "derivation": "S = 1[(A + A.T) > 0], diagonal cleared",
            "never_substituted_into_propagation": True,
            "shape": list(view.shape),
            "sha256": _array_sha256(view),
            "component_count": len(components),
            "component_sizes": [len(c) for c in components],
        },
        "positions": {
            "policy": position_policy,
            "sha256": position_hash,
            "note": ("real WGS84 coordinates" if position_policy == "verified_wgs84"
                     else "no coordinates exist for this dataset; the layout is "
                          "derived from the graph and no coordinate field is "
                          "written. Zeros or node indices would look like an answer."),
        },
        "layout": {
            "algorithm": (LAYOUT_ALGORITHM if position_policy != "verified_wgs84"
                          else "kdtree_on_verified_wgs84"),
            "version": LAYOUT_VERSION,
            "recur_times": settings["recur_times"],
            "spa_patchsize": settings["spa_patchsize"],
            "padding_rule": "parts are padded up to spa_patchsize and never split",
            "membership_sha256": _sha256_bytes(signature.encode("utf-8")),
        },
        # Reserved for Stage 2-A3. Three distinct scopes, so no model can infer
        # them from a data path.
        "reserved_scopes": {
            "propagation_graph_scope": None,
            "available_signal_node_mask": None,
            "loss_target_node_mask": None,
            "note": "populated in Stage 2-A3; deliberately distinct fields",
        },
        "status": {
            "data_ready": True,
            "graph_ready": True,
            "training_ready": False,
            "training_ready_blocked_by": (
                "GraphContract is not yet bound to ALFNet/MoGERNN and the "
                "signal/topology exposure protocol is not implemented "
                "(Stage 2-A3)"),
        },
        "provenance": {
            "payload_path": _repo_relative(payload_path),
            "payload_sha256": _sha256_file(payload_path),
            "builder": "scripts/v4_build_graph_contract.py",
            "builder_sha256": _sha256_file(Path(__file__)),
            "layout_module_sha256": _sha256_file(ROOT / "lib/graph_native_layout.py"),
            "code_commit": _code_commit(),
        },
    }

    destination = Path(output_dir) if output_dir else TRACKED_MANIFESTS
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{dataset}_graph_contract.json"
    path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False),
                    encoding="utf-8")
    return contract, path


def _repo_relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=sorted(LAYOUT_SETTINGS))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    contract, path = build(args.dataset, args.output_dir)
    print(f"{args.dataset}: {path.name}")
    print(f"  propagation directed={contract['propagation_graph']['directed']} "
          f"sha256={contract['propagation_graph']['sha256'][:16]}")
    print(f"  structural components={contract['structural_view']['component_count']} "
          f"sha256={contract['structural_view']['sha256'][:16]}")
    print(f"  positions={contract['positions']['policy']}")
    print(f"  layout={contract['layout']['algorithm']}/v{contract['layout']['version']} "
          f"membership={contract['layout']['membership_sha256'][:16]}")
    print(f"  status data={contract['status']['data_ready']} "
          f"graph={contract['status']['graph_ready']} "
          f"training={contract['status']['training_ready']}")


if __name__ == "__main__":
    main()
