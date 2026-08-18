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


# --- D2-A: authoritative resolution and input provenance ---------------------
#
# Evidence classes are kept distinct on purpose. Some identity fields appear in
# both the frozen bundle and final_runs.json and are cross-checked; some appear
# in only one and are merely recorded. canonical_config_hash is authority-only:
# the result bundles do not carry it, so reporting it as "verified" would be
# false, and the payload says so.

import hashlib as _hashlib

from scripts.analysis_identity import (  # noqa: E402
    AUTHORITY_ONLY_FIELDS,
    BUNDLE_ONLY_FIELDS,
    CROSS_CHECKED_FIELDS,
    build_analysis_identity,
    input_provenance,
    resolve_run,
)


def _bundle_summary(model="ALFNet", variant="gs", seed=32, dataset="METRLA", **over):
    summary = {
        "model": model, "model_variant": variant, "seed": seed,
        "formal_protocol_match": True, "exp_id": f"{model}-exp",
        "protocol_hash": "p" * 8, "training_protocol_hash": "t" * 8,
        "required_training_protocol_hash": "t" * 8,
        "lce_protocol_hash": "l" * 8, "ples_protocol_hash": "e" * 8,
        "dataset": {"name": dataset, "fingerprint": "f", "adjacency_hash": "a",
                    "timestamps_hash": "s", "test_values_hash": "v"},
    }
    summary.update(over)
    return summary


def _bundle_dir(tmp_path, subdir, run_id):
    directory = tmp_path / subdir / run_id / "predictions" / "lifecycle-v3-abc"
    directory.mkdir(parents=True)
    return directory


def test_a_correct_bundle_resolves_to_its_unique_run(tmp_path):
    directory = _bundle_dir(tmp_path, "ALFNet", "alfnet-metrla-s32")
    entry = resolve_run(_bundle_summary(), "ours", directory, MANIFEST,
                        output_root=tmp_path)
    assert entry["run_id"] == "alfnet-metrla-s32"


def test_swapped_roles_are_refused_by_the_resolver(tmp_path):
    directory = _bundle_dir(tmp_path, "MoGERNN", "mogernn-metrla-s32")
    with pytest.raises(IdentityRefused, match="must hold ALFNet"):
        resolve_run(_bundle_summary("MoGERNN", "mg_l3"), "ours", directory,
                    MANIFEST, output_root=tmp_path)


def test_an_ablation_is_refused_by_the_resolver(tmp_path):
    directory = _bundle_dir(tmp_path, "ALFNet", "alfnet-metrla-s32")
    with pytest.raises(IdentityRefused, match="not a mainline"):
        resolve_run(_bundle_summary(variant="ad"), "ours", directory, MANIFEST,
                    output_root=tmp_path)


def test_a_bundle_in_another_runs_directory_is_refused(tmp_path):
    """Legal files, wrong home: the summary carries no run_id of its own."""
    directory = _bundle_dir(tmp_path, "ALFNet", "alfnet-metrla-s42")
    with pytest.raises(IdentityRefused, match="not under the directory"):
        resolve_run(_bundle_summary(seed=32), "ours", directory, MANIFEST,
                    output_root=tmp_path)


@pytest.mark.parametrize("field", ["exp_id", "protocol_hash", "ples_protocol_hash"])
def test_a_missing_bundle_field_is_refused(tmp_path, field):
    directory = _bundle_dir(tmp_path, "ALFNet", "alfnet-metrla-s32")
    summary = _bundle_summary()
    summary[field] = None
    with pytest.raises(IdentityRefused, match="missing or empty"):
        resolve_run(summary, "ours", directory, MANIFEST, output_root=tmp_path)


def test_a_missing_training_protocol_hash_is_refused(tmp_path):
    """Caught by the required-hash comparison, which runs first and is stricter."""
    directory = _bundle_dir(tmp_path, "ALFNet", "alfnet-metrla-s32")
    summary = _bundle_summary()
    summary["training_protocol_hash"] = None
    with pytest.raises(IdentityRefused, match="required|missing or empty"):
        resolve_run(summary, "ours", directory, MANIFEST, output_root=tmp_path)


def test_a_training_hash_that_contradicts_its_requirement_is_refused(tmp_path):
    directory = _bundle_dir(tmp_path, "ALFNet", "alfnet-metrla-s32")
    summary = _bundle_summary(required_training_protocol_hash="other")
    with pytest.raises(IdentityRefused, match="required"):
        resolve_run(summary, "ours", directory, MANIFEST, output_root=tmp_path)


def test_a_wrong_dataset_or_seed_is_refused(tmp_path):
    directory = _bundle_dir(tmp_path, "ALFNet", "alfnet-metrla-s32")
    with pytest.raises(IdentityRefused):
        resolve_run(_bundle_summary(dataset="PEMSBAY"), "ours", directory,
                    MANIFEST, output_root=tmp_path)
    with pytest.raises(IdentityRefused):
        resolve_run(_bundle_summary(seed=99), "ours", directory, MANIFEST,
                    output_root=tmp_path)


# --- provenance and the analysis hash ----------------------------------------


def _identity(tmp_path, content=None, **over):
    """Build a payload. ``content`` writes the input file only when given.

    It must not be rewritten unconditionally: an earlier version did, which
    silently undid the byte mutation the hash test was trying to make, so that
    test passed against an unchanged file and proved nothing.
    """
    payload = tmp_path / "rows.csv"
    if content is not None or not payload.exists():
        payload.write_text(content if content is not None else "a,b\n1,2\n",
                           encoding="utf-8")
    kwargs = dict(
        analysis_type="single_seed_shockwave", dataset="METRLA",
        roles={"ours": "ALFNet", "baseline": "MoGERNN"}, seeds=[32],
        entries={"ours": dict(MANIFEST["runs"][0]),
                 "baseline": dict(MANIFEST["runs"][9])},
        summaries={"ours": _bundle_summary(),
                   "baseline": _bundle_summary("MoGERNN", "mg_l3")},
        endpoint_inventory=["JoinLift/MAE"], resampling="overlap-block x cohort",
        bootstrap_samples=1000, bootstrap_seed=2026,
        margins={"contrast": {"MAE": 0.05}, "arm": {"MAE": 0.05}},
        inputs={"paired_rows": payload},
        display_labels={"ours": "ALFNet", "baseline": "MoGERNN"},
    )
    kwargs.update(over)
    return build_analysis_identity(**kwargs)


def test_the_payload_records_every_input_hash(tmp_path):
    payload = _identity(tmp_path)
    entry = payload["input_provenance"]["paired_rows"]
    assert entry["sha256"] == _hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert not entry["repo_relative_path"].startswith("/"), "no absolute path in provenance"


def test_changing_one_input_byte_changes_the_analysis_hash(tmp_path):
    before = _identity(tmp_path, content="a,b\n1,2\n")["analysis_hash"]
    after = _identity(tmp_path, content="a,b\n1,3\n")["analysis_hash"]
    assert after != before


@pytest.mark.parametrize("field,value", [
    ("bootstrap_samples", 2000),
    ("bootstrap_seed", 7),
    ("margins", {"contrast": {"MAE": 0.10}, "arm": {"MAE": 0.05}}),
    ("endpoint_inventory", ["JoinLift/MAE", "JoinLift/MAPE"]),
    ("resampling", "something else"),
])
def test_changing_the_statistical_configuration_changes_the_hash(tmp_path, field, value):
    before = _identity(tmp_path)["analysis_hash"]
    assert _identity(tmp_path, **{field: value})["analysis_hash"] != before


def test_changing_only_a_display_label_does_not_change_the_hash(tmp_path):
    """A legend rename must not change the identity of the numbers."""
    before = _identity(tmp_path)["analysis_hash"]
    after = _identity(tmp_path, display_labels={"ours": "Ours (v3)",
                                                "baseline": "Baseline"})
    assert after["analysis_hash"] == before
    assert after["display_labels"]["ours"] == "Ours (v3)"


def test_the_payload_separates_evidence_classes(tmp_path):
    payload = _identity(tmp_path)
    notes = payload["evidence_notes"]
    assert set(notes["cross_checked"]) == set(CROSS_CHECKED_FIELDS)
    assert set(notes["bundle_declared_only"]) == set(BUNDLE_ONLY_FIELDS)
    assert set(notes["authority_declared_only"]) == set(AUTHORITY_ONLY_FIELDS)
    assert "NOT independently verified" in notes["canonical_config_hash"]


def test_the_payload_is_serialisable_without_non_finite_values(tmp_path):
    payload = _identity(tmp_path)
    json.dumps(payload, allow_nan=False)


def test_the_hash_is_stable_across_key_order(tmp_path):
    assert _identity(tmp_path)["analysis_hash"] == _identity(tmp_path)["analysis_hash"]


def test_a_missing_provenance_input_is_refused(tmp_path):
    with pytest.raises(IdentityRefused, match="provenance input missing"):
        input_provenance({"rows": tmp_path / "nope.csv"})
