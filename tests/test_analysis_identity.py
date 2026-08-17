"""Identity must come from the frozen manifest, and output must not clobber.

Comparing the two bundles to each other is not identity. The earlier guard
accepted all of these, each verified against the real artefacts before this was
written:

  * ALFNet and MoGERNN swapped between ``--ours`` and ``--baseline``, while the
    default labels still read ALFNet/MoGERNN -- so the report stated the
    opposite of what it measured;
  * arbitrary model strings, including an ALFNet ablation standing in for the
    published mainline;
  * two bundles that both omitted the same field, because ``None == None``.

So each side is matched against its own entry in
``configs/paper/final_runs.json``, and cross-side equality is required only for
what the shared experimental design determines.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis_identity import (  # noqa: E402
    IdentityRefused,
    OutputRefused,
    analysis_hash,
    assert_shared_design,
    load_manifest,
    mark_complete,
    prepare_output_dir,
    verify_role,
)

MANIFEST = load_manifest()


def _summary(model="ALFNet", variant="gs", seed=32, dataset="METRLA", **overrides):
    summary = {
        "model": model, "model_variant": variant, "seed": seed,
        "formal_protocol_match": True,
        "protocol_hash": "p", "training_protocol_hash": "t",
        "lce_protocol_hash": "l", "ples_protocol_hash": "e",
        "dataset": {"name": dataset, "fingerprint": "f", "adjacency_hash": "a",
                    "timestamps_hash": "s", "test_values_hash": "v"},
    }
    summary.update(overrides)
    return summary


# --- role is anchored to the manifest, not to the other side ----------------


def test_the_mainline_pair_is_accepted_in_the_right_slots():
    verify_role(_summary("ALFNet", "gs"), "ours", MANIFEST)
    verify_role(_summary("MoGERNN", "mg_l3"), "baseline", MANIFEST)


def test_swapped_roles_are_refused():
    """The defect that inverted the reported sign under default labels."""
    with pytest.raises(IdentityRefused, match="must hold ALFNet"):
        verify_role(_summary("MoGERNN", "mg_l3"), "ours", MANIFEST)
    with pytest.raises(IdentityRefused, match="must hold MoGERNN"):
        verify_role(_summary("ALFNet", "gs"), "baseline", MANIFEST)


def test_an_ablation_may_not_stand_in_for_the_mainline():
    """`ad` is an ALFNet ablation; it is not the published model."""
    with pytest.raises(IdentityRefused, match="not a mainline"):
        verify_role(_summary("ALFNet", "ad"), "ours", MANIFEST)


def test_an_arbitrary_model_string_is_refused():
    with pytest.raises(IdentityRefused, match="must hold ALFNet"):
        verify_role(_summary("SomeOtherNet", "gs"), "ours", MANIFEST)


@pytest.mark.parametrize("field", ["model", "model_variant", "seed"])
def test_a_missing_identity_field_is_refused(field):
    """`None == None` used to pass when both sides omitted the same field."""
    summary = _summary()
    summary[field] = None
    with pytest.raises(IdentityRefused, match="missing or empty"):
        verify_role(summary, "ours", MANIFEST)


def test_a_missing_dataset_name_is_refused():
    summary = _summary()
    summary["dataset"] = {}
    with pytest.raises(IdentityRefused, match="dataset.name"):
        verify_role(summary, "ours", MANIFEST)


def test_a_variant_that_contradicts_the_manifest_is_refused():
    """gs_bay is a mainline variant, but not METR-LA's."""
    with pytest.raises(IdentityRefused, match="manifest says variant"):
        verify_role(_summary("ALFNet", "gs_bay", dataset="METRLA"), "ours", MANIFEST)


def test_a_run_absent_from_the_manifest_is_refused():
    with pytest.raises(IdentityRefused, match="0 entries"):
        verify_role(_summary(seed=99), "ours", MANIFEST)


def test_formal_protocol_match_must_be_true():
    with pytest.raises(IdentityRefused, match="formal_protocol_match"):
        verify_role(_summary(formal_protocol_match=False), "ours", MANIFEST)


# --- shared design ------------------------------------------------------------


def test_a_matched_pair_shares_its_design():
    assert_shared_design(_summary("ALFNet", "gs"), _summary("MoGERNN", "mg_l3"))


def test_a_cross_seed_pair_is_refused():
    with pytest.raises(IdentityRefused, match="cross-seed"):
        assert_shared_design(_summary("ALFNet", "gs", seed=32),
                             _summary("MoGERNN", "mg_l3", seed=52))


def test_a_missing_shared_field_is_refused_on_both_sides():
    left = _summary("ALFNet", "gs")
    right = _summary("MoGERNN", "mg_l3")
    left["lce_protocol_hash"] = None
    right["lce_protocol_hash"] = None
    with pytest.raises(IdentityRefused, match="missing or empty"):
        assert_shared_design(left, right)


def test_a_different_dataset_fingerprint_is_refused():
    right = _summary("MoGERNN", "mg_l3")
    right["dataset"]["fingerprint"] = "different"
    with pytest.raises(IdentityRefused, match="fingerprint"):
        assert_shared_design(_summary("ALFNet", "gs"), right)


# --- content-addressed, no-clobber output ------------------------------------


def test_the_same_inputs_give_the_same_hash():
    payload = {"b": 2, "a": 1}
    assert analysis_hash(payload) == analysis_hash({"a": 1, "b": 2})
    assert analysis_hash(payload) != analysis_hash({"a": 1, "b": 3})


def test_a_fresh_directory_is_created_exclusively(tmp_path):
    destination = prepare_output_dir(tmp_path, "abc123")
    assert destination.is_dir()
    assert destination.name == "abc123"


def test_an_existing_directory_is_never_overwritten(tmp_path):
    prepare_output_dir(tmp_path, "abc123")
    with pytest.raises(OutputRefused, match="already exists"):
        prepare_output_dir(tmp_path, "abc123")


def test_an_incomplete_directory_is_reported_as_such(tmp_path):
    prepare_output_dir(tmp_path, "abc123")
    with pytest.raises(OutputRefused, match="incomplete"):
        prepare_output_dir(tmp_path, "abc123")

    other = prepare_output_dir(tmp_path, "def456")
    mark_complete(other, {"ok": True})
    with pytest.raises(OutputRefused, match="is complete"):
        prepare_output_dir(tmp_path, "def456")


@pytest.mark.parametrize("frozen", ["predictions", "checkpoints", "reports/final"])
def test_writing_into_a_frozen_location_is_refused(tmp_path, frozen):
    root = tmp_path / frozen
    root.mkdir(parents=True)
    with pytest.raises(OutputRefused, match="frozen location"):
        prepare_output_dir(root, "abc123")


def test_the_complete_marker_is_written_last(tmp_path):
    destination = prepare_output_dir(tmp_path, "abc123")
    assert not (destination / "COMPLETE").exists()
    mark_complete(destination, {"schema_version": 1})
    assert json.loads((destination / "COMPLETE").read_text())["schema_version"] == 1
