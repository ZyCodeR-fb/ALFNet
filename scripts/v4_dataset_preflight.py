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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allow-not-ready", action="store_true",
                        help="inspect a payload that is not training-ready; this "
                             "never authorises training")
    args = parser.parse_args()
    try:
        preflight(args.manifest, require_training_ready=not args.allow_not_ready)
    except PreflightRefused as refusal:
        print(refusal)
        raise SystemExit(1) from None
    print("preflight passed")


if __name__ == "__main__":
    main()
