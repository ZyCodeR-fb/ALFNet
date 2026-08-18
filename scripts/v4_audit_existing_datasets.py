"""Read-only admission audit for the datasets v4 keeps as-is.

PEMS08 and PEMS-BAY need no payload rewrite: their signals and calendars are
sound. What they do need is a written record of *what the active payload
actually is*, because two of its properties are easy to assume wrongly --
PEMS08's positions are converter-derived spectral vectors rather than GPS, and
its adjacency was symmetrised by the converter rather than being the canonical
directed graph.

This script writes an audit JSON and touches no dataset file.

    python scripts/v4_audit_existing_datasets.py --output-dir docs/v4_audit
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY = ROOT / "configs/paper/v4_dataset_registry.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap(value):
    return value.item() if isinstance(value, np.ndarray) and value.shape == () else value


def audit(dataset, directory):
    registry = json.loads(REGISTRY.read_text("utf-8"))["datasets"][dataset]
    payload_path = Path(directory) / "raw_data.pkl"
    payload = _unwrap(np.load(payload_path, allow_pickle=True))

    data = np.asarray(payload["data"])
    adjacency = np.asarray(payload["adj"])
    positions = np.asarray(payload["pos_stamp"])
    stamps = np.asarray(payload["time_stamp"])

    symmetric = bool(np.allclose(adjacency, adjacency.T))
    # Real WGS84 latitudes/longitudes sit in a plausible geographic band;
    # converter-derived spectral vectors cluster near zero. This is a recorded
    # observation, not a certification.
    looks_geographic = bool(
        positions.shape[1] == 2
        and 20.0 < float(np.abs(positions[:, 0]).max()) < 80.0
        and 100.0 < float(np.abs(positions[:, 1]).max()) < 180.0
    )
    return {
        "dataset": dataset,
        "audit_only": True,
        "payload_rewritten": False,
        "payload_path": payload_path.relative_to(ROOT).as_posix(),
        "payload_sha256": file_sha256(payload_path),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "time_start": str(stamps[0]),
        "time_end": str(stamps[-1]),
        "time_semantics": registry["time_semantics"],
        "v4_target": registry["target"],
        "v4_target_channel": registry["target_channel"],
        "channel_order": registry["channel_order"],
        "adjacency": {
            "symmetric": symmetric,
            "status": ("converter-derived symmetric RBF graph, not an unmodified "
                       "canonical directed graph" if symmetric
                       else "directed; must not be symmetrised"),
        },
        "positions": {
            "unique_rows": int(len(np.unique(np.round(positions, 6), axis=0))),
            "node_count": int(positions.shape[0]),
            "looks_geographic": looks_geographic,
            "status": ("real WGS84 coordinates" if looks_geographic else
                       "converter-derived spectral vectors, NOT GPS. Distinct "
                       "rows do not imply cross-machine stability: the "
                       "eigenvector basis is not sign- or basis-fixed."),
        },
        "notes": registry.get("notes"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/v4_audit")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for dataset, directory in (("PEMS08", ROOT / "data/PEMS08"),
                               ("PEMSBAY", ROOT / "data/PEMS-BAY")):
        report = audit(dataset, directory)
        # v4 retrains both regardless: PEMS08's target changes, and the shared
        # graph/layout contract is not settled until Stage 2-A2.
        report["v4_requires_retrain"] = True
        report["training_ready"] = False
        report["training_ready_blocked_by"] = (
            "graph/layout policy unresolved; see Stage 2-A2")
        destination = output / f"{dataset}_v4_admission_audit.json"
        destination.write_text(json.dumps(report, indent=2, allow_nan=False),
                               encoding="utf-8")
        shown = destination.resolve()
        try:
            shown = shown.relative_to(ROOT)
        except ValueError:
            pass
        print(f"{dataset}: {shown}  sha256={file_sha256(destination)[:16]}")


if __name__ == "__main__":
    main()
