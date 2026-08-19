"""The v4 candidate payloads must fix the metadata without touching v3.

The defect being closed is narrow and was measured before any code changed: the
PEMS03/PEMS04 signal arrays are bit-identical to the originals after a float32
cast, and only the *derived* metadata was wrong -- both were stamped from
2016-07-01, which is PEMS08's start. That shifts day-of-week by one weekday for
PEMS03 and three for PEMS04, and LCBT calendar stratification and LCE's 5/7-2/7
weighting both key on day-of-week.

These tests read the real source files, which is the point: a fixture cannot
show that the conversion preserves the actual data.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v4_build_candidate_payload import (  # noqa: E402
    ConversionRefused,
    build,
    load_registry,
)

CANDIDATES = ("PEMS03", "PEMS04")
EXPECTED = {
    "PEMS03": {"shape": (26208, 358, 1), "start": "2018-09-01", "end": "2018-11-30"},
    "PEMS04": {"shape": (16992, 307, 3), "start": "2018-01-01", "end": "2018-02-28"},
}


def _sources_present(dataset):
    entry, _ = load_registry(dataset)
    return (ROOT / "data" / dataset / entry["signal_source"]).exists()


needs_sources = pytest.mark.skipif(
    not all(_sources_present(d) for d in CANDIDATES),
    reason="original NPZ sources are not present in this checkout")


def _unwrap(blob):
    if isinstance(blob, dict):
        return blob
    return blob.item() if getattr(blob, "shape", None) == () else blob


def _payload(dataset, tmp_path):
    manifest, payload_path, _ = build(dataset, output_dir=tmp_path / dataset)
    return manifest, _unwrap(np.load(payload_path, allow_pickle=True))


# --- the signal survives conversion exactly ----------------------------------


@needs_sources
@pytest.mark.parametrize("dataset", CANDIDATES)
def test_the_candidate_signal_equals_the_original_after_a_float32_cast(dataset, tmp_path):
    entry, _ = load_registry(dataset)
    with np.load(ROOT / "data" / dataset / entry["signal_source"]) as archive:
        original = np.asarray(archive[list(archive.keys())[0]])
    _, payload = _payload(dataset, tmp_path)
    assert np.array_equal(original.astype(np.float32), payload["data"])
    assert payload["data"].dtype == np.float32


@needs_sources
@pytest.mark.parametrize("dataset", CANDIDATES)
def test_the_candidate_shape_matches_the_registry(dataset, tmp_path):
    _, payload = _payload(dataset, tmp_path)
    assert payload["data"].shape == EXPECTED[dataset]["shape"]


# --- the calendar is the corrected one ---------------------------------------


@needs_sources
@pytest.mark.parametrize("dataset", CANDIDATES)
def test_the_candidate_calendar_is_corrected_and_uniform(dataset, tmp_path):
    _, payload = _payload(dataset, tmp_path)
    stamps = pd.to_datetime(pd.Series(np.asarray(payload["time_stamp"]).ravel()))
    assert str(stamps.iloc[0].date()) == EXPECTED[dataset]["start"]
    assert str(stamps.iloc[-1].date()) == EXPECTED[dataset]["end"]
    assert stamps.is_monotonic_increasing
    assert not stamps.duplicated().any()
    deltas = stamps.diff().dropna().unique()
    assert len(deltas) == 1 and deltas[0] == pd.Timedelta(minutes=5)


@needs_sources
@pytest.mark.parametrize("dataset", CANDIDATES)
def test_the_corrected_start_is_not_the_old_wrong_one(dataset, tmp_path):
    """Both were stamped 2016-07-01, which is PEMS08's start, not theirs."""
    _, payload = _payload(dataset, tmp_path)
    stamps = pd.to_datetime(pd.Series(np.asarray(payload["time_stamp"]).ravel()))
    assert str(stamps.iloc[0].date()) != "2016-07-01"


@needs_sources
def test_the_corrected_calendar_changes_day_of_week():
    """Why the wrong start is a protocol defect, not a cosmetic one."""
    assert pd.Timestamp("2016-07-01").dayofweek != pd.Timestamp("2018-09-01").dayofweek
    assert pd.Timestamp("2016-07-01").dayofweek != pd.Timestamp("2018-01-01").dayofweek


# --- dates are not a free parameter ------------------------------------------


@needs_sources
@pytest.mark.parametrize("dataset", CANDIDATES)
def test_a_contradicting_start_time_is_refused(dataset, tmp_path):
    with pytest.raises(ConversionRefused, match="contradicts the registry"):
        build(dataset, output_dir=tmp_path / dataset, start_time="2016-07-01 00:00:00")


@needs_sources
@pytest.mark.parametrize("dataset", CANDIDATES)
def test_the_registry_start_time_is_accepted(dataset, tmp_path):
    entry, _ = load_registry(dataset)
    manifest, _, _ = build(dataset, output_dir=tmp_path / dataset,
                           start_time=entry["start_time"])
    assert manifest["dataset"] == dataset


def test_a_dataset_without_a_candidate_build_is_refused(tmp_path):
    """PEMS-BAY keeps its canonical payload; it is not rebuilt."""
    with pytest.raises(ConversionRefused, match="no candidate build"):
        build("PEMSBAY", output_dir=tmp_path)


# --- PEMS03 has one channel --------------------------------------------------


def test_pems03_targets_channel_zero_and_has_no_speed_channel():
    entry, _ = load_registry("PEMS03")
    assert entry["target"] == "q"
    assert entry["target_channel"] == 0
    assert entry["expected_shape"][2] == 1
    # v -> channel 2 would be out of range; the registry says so explicitly
    assert "out of range" in entry["notes"]


@needs_sources
def test_pems03_channel_zero_is_addressable(tmp_path):
    _, payload = _payload("PEMS03", tmp_path)
    assert payload["data"][:, :, 0].shape == (26208, 358)
    with pytest.raises(IndexError):
        _ = payload["data"][:, :, 2]


# --- adjacency is source-native and directed ---------------------------------


@needs_sources
@pytest.mark.parametrize("dataset", CANDIDATES)
def test_the_candidate_adjacency_is_not_symmetrised(dataset, tmp_path):
    manifest, payload = _payload(dataset, tmp_path)
    assert manifest["adjacency"]["symmetrised"] is False
    assert manifest["adjacency"]["kernelised"] is False
    adjacency = payload["adj"]
    assert not np.allclose(adjacency, adjacency.T), (
        "a symmetric graph collapses forward and backward diffusion")


# --- preflight ------------------------------------------------------------------
#
# The candidate-manifest preflight was retired in Stage 2 Part 0. Candidates are
# now intermediates built into a TemporaryDirectory, so there is no candidate
# manifest to preflight. The equivalent guarantees -- recomputed payload and
# unseen hashes, registry agreement, run-directory contents and the readiness
# ordering -- are asserted against the canonical run manifest in
# tests/test_v4_run_format.py.


# --- v3 is untouched -----------------------------------------------------------


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


FROZEN_V3 = {
    "data/PEMS03/raw_data.pkl":
        "2f5c6182682687de9c9e9dcffb6334bd103b652a5311f591cffee7e7df448ecb",
    "data/PEMS04/raw_data.pkl":
        "b8741a79dface8a80b6db5ffb753944407deeea645a7de3a8a64ae5607228130",
}


@pytest.mark.parametrize("path,digest", sorted(FROZEN_V3.items()))
def test_the_v3_payloads_are_byte_identical(path, digest):
    target = ROOT / path
    if not target.exists():
        pytest.skip(f"{path} is not present in this checkout")
    assert _sha256(target) == digest, "the v3 payload must never be rewritten"


@needs_sources
@pytest.mark.parametrize("dataset", CANDIDATES)
def test_building_a_candidate_does_not_touch_the_v3_payload(dataset, tmp_path):
    active = ROOT / "data" / dataset / "raw_data.pkl"
    before = _sha256(active)
    build(dataset, output_dir=tmp_path / dataset)
    assert _sha256(active) == before


# --- PEMS-BAY's DST window is left alone ---------------------------------------


def test_pems_bay_dst_transition_is_not_padded_or_deleted():
    """01:55 -> 03:00 on 2017-03-12 is spring-forward; UTC is continuous.

    Inserting 12 NaN rows or deleting the window would both invent a defect.
    """
    payload = ROOT / "data/PEMS-BAY/raw_data.pkl"
    if not payload.exists():
        pytest.skip("PEMS-BAY payload not present")
    blob = _unwrap(np.load(payload, allow_pickle=True))
    stamps = pd.to_datetime(pd.Series(np.asarray(blob["time_stamp"]).ravel()))
    gaps = stamps.diff().dropna()
    jumps = gaps[gaps != pd.Timedelta(minutes=5)]
    assert len(jumps) == 1, f"expected exactly the DST jump, found {len(jumps)}"
    assert jumps.iloc[0] == pd.Timedelta(minutes=65)
    boundary = stamps.iloc[jumps.index[0] - 1]
    assert str(boundary) == "2017-03-12 01:55:00"
