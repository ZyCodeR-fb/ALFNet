"""Build a versioned v4 candidate payload for PEMS03 / PEMS04.

The signal arrays in the original NPZ files are sound -- measured, they are
bit-identical to the active payload after a float32 cast. What is wrong is the
metadata the conversion invented: both datasets were stamped from 2016-07-01,
which is PEMS08's start, not theirs. That is not cosmetic. LCBT's calendar
stratification and LCE's 5/7-2/7 weekday/weekend weighting both key on
day-of-week, and the stamped starts are off by one weekday for PEMS03 and three
for PEMS04.

So dates come from ``configs/paper/v4_dataset_registry.json`` and never from a
free-form argument: a caller who passes the wrong start for a known dataset is
refused rather than handed another plausible-looking calendar.

Nothing under ``data/<dataset>/`` is overwritten. Output goes to a separate
``v4_candidate/`` directory, and positions are deliberately left unresolved --
the spectral vectors the old converter produced are degenerate on disconnected
graphs and are not basis-stable, so carrying them into v4 would launder a known
defect. Zeros or node indices would be worse: they would look like an answer.

    python scripts/v4_build_candidate_payload.py --dataset PEMS03
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY_PATH = ROOT / "configs/paper/v4_dataset_registry.json"
TRACKED_MANIFESTS = ROOT / "configs/paper/v4_data_manifests"
# PEMS08 is here despite already having correct dates: v4 changes its target
# from speed to flow and replaces the converter-symmetrised RBF graph with the
# source-native directed one, so it needs its own candidate rather than reusing
# the v3 payload.
BUILDABLE = ("PEMS03", "PEMS04", "PEMS08")


class ConversionRefused(SystemExit):
    """The inputs or arguments do not match the registry."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(dataset):
    registry = json.loads(REGISTRY_PATH.read_text("utf-8"))
    if dataset not in registry["datasets"]:
        raise ConversionRefused(f"{dataset} is not in the v4 registry")
    return registry["datasets"][dataset], registry


def build_timestamps(entry, periods):
    """Timestamps from the registry, then verified against it.

    Generated and *checked*, not merely generated: the check is what catches a
    registry whose declared span disagrees with the array length.
    """
    stamps = pd.date_range(entry["start_time"], periods=periods,
                           freq=f"{entry['freq_minutes']}min")
    if str(stamps[-1]) != entry["end_time"]:
        raise ConversionRefused(
            f"generated end {stamps[-1]} != registry end {entry['end_time']}; "
            f"the declared span does not match {periods} steps at "
            f"{entry['freq_minutes']}min")
    deltas = stamps.to_series().diff().dropna().unique()
    expected = pd.Timedelta(minutes=entry["freq_minutes"])
    if len(deltas) != 1 or deltas[0] != expected:
        raise ConversionRefused(f"timestamps are not a uniform {expected} grid")
    if stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ConversionRefused("timestamps must be unique and strictly increasing")
    return stamps.to_numpy()


def build_adjacency(edge_csv, num_nodes):
    """Directed binary adjacency from the source edge list, traceably.

    Source-native and directed: no kernel, no symmetrisation. The old converter
    symmetrised, which collapsed forward and backward diffusion into the same
    operator on every PeMS graph.
    """
    frame = pd.read_csv(edge_csv)
    columns = {c.lower(): c for c in frame.columns}
    src = columns.get("from"); dst = columns.get("to")
    if src is None or dst is None:
        raise ConversionRefused(
            f"{edge_csv}: expected 'from' and 'to' columns, found {list(frame.columns)}")
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    pairs = frame[[src, dst]].to_numpy(dtype=np.int64)
    if pairs.min() < 0 or pairs.max() >= num_nodes:
        raise ConversionRefused(
            f"{edge_csv}: node ids outside [0, {num_nodes}); the file may use "
            "1-based or raw sensor ids and needs an explicit mapping")
    adjacency[pairs[:, 0], pairs[:, 1]] = 1.0
    return adjacency, len(pairs)


def _repo_relative(path):
    """Repo-relative when inside the tree, otherwise a stable marker.

    A candidate can legitimately be built into a temporary directory, so this
    must not raise -- and an absolute machine path must never be recorded as if
    it were a repository location.
    """
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def code_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:                                        # noqa: BLE001
        return None


def build(dataset, output_dir=None, start_time=None):
    if dataset not in BUILDABLE:
        raise ConversionRefused(
            f"{dataset} has no candidate build; only {BUILDABLE} are rebuilt for v4")
    entry, registry = load_registry(dataset)

    if start_time is not None and start_time != entry["start_time"]:
        raise ConversionRefused(
            f"--start-time {start_time!r} contradicts the registry "
            f"({entry['start_time']!r}) for {dataset}. Dates are not a free "
            "parameter: the wrong start silently shifts day-of-week and "
            "corrupts calendar stratification.")

    source_dir = ROOT / "data" / dataset
    signal_path = source_dir / entry["signal_source"]
    edge_path = source_dir / entry["edge_source"]
    for path in (signal_path, edge_path):
        if not path.exists():
            raise ConversionRefused(f"missing source file: {path}")

    with np.load(signal_path) as archive:
        key = list(archive.keys())[0]
        signal = np.asarray(archive[key])
    expected_shape = tuple(entry["expected_shape"])
    if signal.shape != expected_shape:
        raise ConversionRefused(
            f"{dataset}: signal shape {signal.shape} != registry {expected_shape}")

    data = signal.astype(np.float32)
    stamps = build_timestamps(entry, periods=data.shape[0])
    adjacency, edge_count = build_adjacency(edge_path, entry["num_nodes"])

    destination = Path(output_dir) if output_dir else (source_dir / "v4_candidate")
    destination.mkdir(parents=True, exist_ok=True)
    payload_path = destination / "raw_data_v4_candidate.pkl"
    np.save(payload_path, {
        "data": data,
        "adj": adjacency,
        "time_stamp": stamps,
        # positions are deliberately absent; see positions_status below
    }, allow_pickle=True)
    actual_path = payload_path.with_suffix(".pkl.npy")

    manifest = {
        "schema": 1,
        "dataset": dataset,
        "payload_version": "v4_candidate",
        "training_ready": False,
        "training_ready_blocked_by": (
            "positions/layout unresolved: the previous spectral pos_stamp is "
            "degenerate on disconnected graphs and not basis-stable. A "
            "deterministic graph-native layout lands in Stage 2-A2."),
        "positions_status": "unresolved",
        "positions_note": (
            "No positions are written. Zeros or node indices would masquerade "
            "as a solution; absence is the honest state."),
        "source_provenance_caveat": registry["provenance_caveat"],
        "sources": {
            "signal": {"path": _repo_relative(signal_path),
                       "bytes": signal_path.stat().st_size,
                       "sha256": file_sha256(signal_path),
                       "array_key": key},
            "edges": {"path": _repo_relative(edge_path),
                      "bytes": edge_path.stat().st_size,
                      "sha256": file_sha256(edge_path),
                      "edge_count": int(edge_count)},
        },
        "signal": {"shape": list(data.shape), "dtype": str(data.dtype),
                   "channel_order": entry["channel_order"],
                   "cast_from": str(signal.dtype),
                   "cast_is_lossless_for_float32_range": True},
        "target": {"name": entry["target"], "channel": entry["target_channel"]},
        "time": {"start": str(stamps[0]), "end": str(stamps[-1]),
                 "freq_minutes": entry["freq_minutes"],
                 "steps": int(len(stamps)),
                 "semantics": entry["time_semantics"],
                 "authority": "configs/paper/v4_dataset_registry.json; the "
                              "source NPZ carries no timestamps"},
        "adjacency": {"rule": "source-native directed binary from the edge list",
                      "symmetrised": False, "kernelised": False,
                      "shape": list(adjacency.shape)},
        "conversion": {
            "command": f"python scripts/v4_build_candidate_payload.py --dataset {dataset}",
            "converter": "scripts/v4_build_candidate_payload.py",
            "converter_sha256": file_sha256(Path(__file__)),
            "code_commit": code_commit(),
        },
        "output": {"path": _repo_relative(actual_path),
                   "sha256": file_sha256(actual_path)},
    }
    body = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
    manifest_path = destination / "v4_candidate_manifest.json"
    manifest_path.write_text(body, encoding="utf-8")

    # The canonical copy is version-controlled. The runtime copy lives beside the
    # payload under data/, which .gitignore excludes -- so on its own it could not
    # tell another machine what should have been produced. The two are written
    # byte-for-byte identical and a test asserts it.
    if output_dir is None:
        TRACKED_MANIFESTS.mkdir(parents=True, exist_ok=True)
        (TRACKED_MANIFESTS / f"{dataset}_data_manifest.json").write_text(
            body, encoding="utf-8")
    return manifest, actual_path, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=BUILDABLE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-time", default=None,
                        help="optional; must equal the registry value or the "
                             "build is refused")
    args = parser.parse_args()
    manifest, payload, manifest_path = build(args.dataset, args.output_dir,
                                             args.start_time)
    print(f"{args.dataset}: {manifest['time']['start']} .. {manifest['time']['end']}")
    print(f"  payload  {payload}  sha256={manifest['output']['sha256'][:16]}")
    print(f"  manifest {manifest_path}")
    print(f"  training_ready={manifest['training_ready']} "
          f"positions={manifest['positions_status']}")


if __name__ == "__main__":
    main()
