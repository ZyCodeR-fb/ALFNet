"""Refuse to train on a v4 payload that does not match its manifest.

A dataset defect is cheapest to catch before a GPU-week, and most expensive to
catch after results exist. This checks the payload against its own manifest and
the frozen registry, and it is deliberately fail-closed on
``training_ready: false`` -- PEMS03, PEMS04 and PEMS08 are all not ready this
round because the graph/layout contract is unresolved, and marking them ready to
make a check pass would defeat the check.

    python scripts/v4_dataset_preflight.py --manifest data/PEMS03/v4_candidate/v4_candidate_manifest.json
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

REGISTRY_PATH = ROOT / "configs/paper/v4_dataset_registry.json"


class PreflightRefused(SystemExit):
    """The payload may not be used for training."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(manifest_path, require_training_ready=True):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise PreflightRefused(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    dataset = manifest["dataset"]
    registry = json.loads(REGISTRY_PATH.read_text("utf-8"))["datasets"]
    if dataset not in registry:
        raise PreflightRefused(f"{dataset} is not in the frozen registry")
    entry = registry[dataset]
    problems = []

    recorded = manifest["output"]["path"]
    # An <external> marker means the payload was built outside the repo (a test
    # or a scratch build); resolve it beside its own manifest instead.
    payload_path = (manifest_path.parent / Path(recorded).name
                    if recorded.startswith("<external>") else ROOT / recorded)
    if not payload_path.exists():
        raise PreflightRefused(f"payload missing: {payload_path}")
    actual = file_sha256(payload_path)
    if actual != manifest["output"]["sha256"]:
        # Raise immediately rather than collecting this one. The payload is
        # parsed further down, and a corrupted file fails there with an opaque
        # UnpicklingError before the collected problems are ever reported.
        raise PreflightRefused(
            f"{dataset} preflight REFUSED:\n  - payload sha256 {actual} != "
            f"manifest {manifest['output']['sha256']}")

    for role in ("signal", "edges"):
        source = manifest["sources"][role]
        source_path = ROOT / source["path"]
        if not source_path.exists():
            problems.append(f"{role} source missing: {source['path']}")
        elif file_sha256(source_path) != source["sha256"]:
            problems.append(f"{role} source bytes changed since conversion: {source['path']}")

    if manifest["target"]["name"] != entry["target"]:
        problems.append(f"target {manifest['target']['name']!r} != frozen {entry['target']!r}")
    if manifest["target"]["channel"] != entry["target_channel"]:
        problems.append("target channel disagrees with the registry")
    if manifest["signal"]["shape"] != list(entry["expected_shape"]):
        problems.append(f"shape {manifest['signal']['shape']} != {entry['expected_shape']}")
    channels = manifest["signal"]["shape"][2]
    if not 0 <= manifest["target"]["channel"] < channels:
        problems.append(
            f"target channel {manifest['target']['channel']} out of range for "
            f"{channels} channels")

    if manifest["time"]["start"][:19] != entry["start_time"].replace(" ", "T")[:19]:
        problems.append(f"start {manifest['time']['start']} != frozen {entry['start_time']}")
    if manifest["time"]["end"][:19] != entry["end_time"].replace(" ", "T")[:19]:
        problems.append(f"end {manifest['time']['end']} != frozen {entry['end_time']}")
    if manifest["time"]["freq_minutes"] != entry["freq_minutes"]:
        problems.append("cadence disagrees with the registry")

    payload = np.load(payload_path, allow_pickle=True)
    payload = payload.item() if payload.shape == () else payload
    stamps = pd.to_datetime(pd.Series(np.asarray(payload["time_stamp"]).ravel()))
    if stamps.duplicated().any():
        problems.append("timestamps contain duplicates")
    if not stamps.is_monotonic_increasing:
        problems.append("timestamps are not strictly increasing")
    deltas = stamps.diff().dropna().unique()
    if len(deltas) != 1 or deltas[0] != pd.Timedelta(minutes=entry["freq_minutes"]):
        problems.append(f"cadence is not a uniform {entry['freq_minutes']}min grid")

    if manifest.get("positions_status") != "resolved":
        problems.append(
            f"positions are {manifest.get('positions_status')!r}; a deterministic "
            "graph-native layout is required before training (Stage 2-A2)")
    if require_training_ready and not manifest.get("training_ready", False):
        problems.append(
            "training_ready is false: " + str(manifest.get("training_ready_blocked_by")))

    if problems:
        raise PreflightRefused(
            f"{dataset} preflight REFUSED:\n" + "\n".join(f"  - {p}" for p in problems))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="a candidate build manifest")
    source.add_argument("--run-manifest", help="a final data_v4 run manifest")
    parser.add_argument("--allow-not-ready", action="store_true",
                        help="inspect a payload that is not training-ready; this "
                             "never authorises training")
    args = parser.parse_args()
    try:
        if args.run_manifest:
            report = run_preflight(args.run_manifest)
            print(json.dumps(report, sort_keys=True))
        else:
            preflight(args.manifest, require_training_ready=not args.allow_not_ready)
    except PreflightRefused as refusal:
        print(refusal)
        raise SystemExit(1) from None
    print("preflight passed")


# --- final run directory ------------------------------------------------------


def run_preflight(manifest_path):
    """Re-verify a v4 run directory against its manifest and the registry.

    Deliberately recomputes rather than trusting the manifest it is checking: a
    manifest and the payload it describes are written by the same script, so a
    check that only reads the manifest verifies nothing about the payload.
    """
    from lib.v4_payload import load_v4_payload

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise PreflightRefused(f"run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    dataset = manifest["dataset"]
    registry = json.loads(REGISTRY_PATH.read_text("utf-8"))["datasets"]
    if dataset not in registry:
        raise PreflightRefused(f"{dataset} is not in the frozen registry")
    entry = registry[dataset]
    problems = []

    run_dir = ROOT / manifest["run_directory"]
    if not run_dir.is_dir():
        raise PreflightRefused(f"run directory missing: {run_dir}")
    # A run directory holds the payload and the unseen pool and nothing else, so
    # a stray third file cannot become an undeclared model input.
    present = sorted(p.name for p in run_dir.iterdir())
    if present != ["raw_data.pkl", "unknown_nodes.npy"]:
        problems.append(f"run directory holds {present}, expected exactly the two v4 files")

    payload_path = ROOT / manifest["payload"]["path"]
    unseen_path = ROOT / manifest["unseen"]["path"]
    for label, path, recorded in (("payload", payload_path, manifest["payload"]["sha256"]),
                                  ("unseen", unseen_path, manifest["unseen"]["sha256"])):
        if not path.exists():
            raise PreflightRefused(f"{label} missing: {path}")
        # Symlinks are allowed, but the resolved target is what gets hashed.
        actual = file_sha256(path.resolve())
        if actual != recorded:
            raise PreflightRefused(
                f"{dataset} {label} hash mismatch: manifest {recorded}, found {actual}")

    payload = load_v4_payload(payload_path)          # raises on any schema defect
    num_nodes = int(payload["data"].shape[1])
    if num_nodes != int(entry["num_nodes"]):
        problems.append(f"node count {num_nodes} != registry {entry['num_nodes']}")
    expected_shape = [int(v) for v in entry["expected_shape"]]
    if list(payload["data"].shape) != expected_shape:
        problems.append(f"data shape {list(payload['data'].shape)} != registry {expected_shape}")
    channel = int(entry["target_channel"])
    if channel >= int(payload["data"].shape[2]):
        problems.append(f"target channel {channel} outside data with "
                        f"{payload['data'].shape[2]} channels")

    stamps = pd.to_datetime(pd.Series(payload["time_stamp"]))
    for edge, key in (("start", "start_time"), ("end", "end_time")):
        expected = pd.Timestamp(entry[key])
        found = stamps.iloc[0] if edge == "start" else stamps.iloc[-1]
        if found.normalize() != expected.normalize():
            problems.append(f"{edge} {found} != registry {expected}")

    unseen = np.load(unseen_path)
    if unseen.dtype != np.int64 or unseen.ndim != 1:
        problems.append(f"unseen pool must be 1-D int64, got {unseen.ndim}-D {unseen.dtype}")
    if unseen.size and (unseen.min() < 0 or unseen.max() >= num_nodes):
        problems.append(f"unseen ids outside [0, {num_nodes})")
    if not np.array_equal(unseen, np.unique(unseen)):
        problems.append("unseen pool is not sorted and unique")
    if int(unseen.size) != int(manifest["unseen"]["count"]):
        problems.append(f"unseen count {unseen.size} != manifest {manifest['unseen']['count']}")

    readiness = manifest["readiness"]
    if readiness.get("training_ready") and not readiness.get("exposure_ready"):
        problems.append("training_ready is set while exposure_ready is not")

    if problems:
        raise PreflightRefused(f"{dataset} run preflight refused:\n  - "
                               + "\n  - ".join(problems))
    return {"dataset": dataset, "nodes": num_nodes,
            "steps": int(payload["data"].shape[0]),
            "target": entry["target"], "unseen": int(unseen.size),
            "readiness": readiness}


if __name__ == "__main__":
    main()
