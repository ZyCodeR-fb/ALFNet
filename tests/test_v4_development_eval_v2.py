"""The evaluator must score each run at its own seed, and prove it.

The defect this file pins: `build_v4_session(config, device=...)` takes
`run_seed=None` and falls back to the config's `seed: 32`, so every seed-42 and
seed-52 checkpoint was re-scored against seed 32's node split and validation
schedule. It was visible only as an agreement pattern -- 4/4 seed-32 runs at max
difference 0, 8/8 others disagreeing on all 36 cells -- because nothing raised.

So the first test below drives the *production* helper, not a stub, and fails if
the seed is dropped anywhere along the way.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v4_runner import build_v4_session
from scripts import v4_development_eval_v2 as evaluator
from scripts.build_v4_development_plan import assert_plan_still_holds, load_plan
from scripts.v4_launcher import plan_runs

CONFIG = ROOT / "configs/paper/v4/PEMS08_MoGERNN.yaml"


@pytest.fixture(scope="module")
def plan():
    return assert_plan_still_holds(load_plan())


@pytest.fixture(scope="module")
def runs(plan):
    root = ROOT / "outputs_v4/development_base" / plan["plan_sha256"]
    return {run.name: run for run in plan_runs(root, plan)}


# --- the seed actually used ----------------------------------------------------


@pytest.mark.parametrize("seed", [32, 42, 52])
def test_the_session_uses_the_run_seed_not_the_config_default(seed):
    """The config says 32; the run says otherwise, and the run wins.

    The old call site would make all three of these identical, which is exactly
    what silently happened.
    """
    from args import load_args_from_config

    assert load_args_from_config(str(CONFIG), []).seed == 32
    session = build_v4_session(CONFIG, device="cpu", run_seed=seed)
    assert session.run_seed == seed
    assert session.args.seed == seed


def test_a_different_seed_is_a_different_split_and_schedule():
    """Otherwise the test above would pass with the seed still ignored."""
    a = build_v4_session(CONFIG, device="cpu", run_seed=32)
    b = build_v4_session(CONFIG, device="cpu", run_seed=42)
    assert a.validation_context.exposure_hash() != b.validation_context.exposure_hash()
    assert not (a.validation_context.validation_unseen
                == b.validation_context.validation_unseen).all()
    # The static graph is the same model; only the exposure moved.
    assert a.train_context.static_asset_hash() == b.train_context.static_asset_hash()


def test_the_evaluator_passes_the_run_seed(monkeypatch, runs, plan):
    """Capture what the evaluator asks for, without running a forward pass."""
    seen = {}
    original = evaluator.build_v4_session

    def capture(config_path, device="cpu", run_seed=None, **kwargs):
        seen["run_seed"] = run_seed
        raise RuntimeError("stop before the forward pass")

    monkeypatch.setattr(evaluator, "build_v4_session", capture)
    run = runs["PEMS08/MoGERNN/seed42"]
    with pytest.raises(RuntimeError, match="stop before"):
        evaluator.evaluate_run(run, plan, device="cpu")
    assert seen["run_seed"] == 42, "the evaluator dropped the run's seed"


def test_a_session_built_at_the_wrong_seed_is_refused(monkeypatch, runs, plan):
    """Belt and braces: even if the seed were dropped, the check catches it."""
    def wrong_seed(config_path, device="cpu", run_seed=None, **kwargs):
        return build_v4_session(config_path, device="cpu", run_seed=32)

    monkeypatch.setattr(evaluator, "build_v4_session", wrong_seed)
    run = runs["PEMS08/MoGERNN/seed52"]
    with pytest.raises(evaluator.EvalRefused, match="is not the run's"):
        evaluator.evaluate_run(run, plan, device="cpu")


# --- identity, checked before any forward pass ---------------------------------


def _identity(**overrides):
    base = {field: f"value-{field}" for field in evaluator.IDENTITY_FIELDS}
    base.update(overrides)
    return base


def test_identity_matches_when_every_field_agrees():
    evaluator.assert_identity(_identity(), _identity(), "unit")


@pytest.mark.parametrize("field", evaluator.IDENTITY_FIELDS)
def test_a_single_wrong_identity_field_is_refused(field):
    with pytest.raises(evaluator.EvalRefused, match=field):
        evaluator.assert_identity(_identity(**{field: "tampered"}), _identity(),
                                  "unit")


@pytest.mark.parametrize("field", ["seed", "config_sha256", "plan_sha256",
                                   "static_asset_hash",
                                   "validation_exposure_hash"])
def test_a_field_absent_from_the_checkpoint_is_refused(field):
    """`.get() == .get()` is True when both are absent -- that is the trap."""
    checkpoint = {name: value for name, value in _identity().items()
                  if name != field}
    with pytest.raises(evaluator.EvalRefused, match="absent from the checkpoint"):
        evaluator.assert_identity(checkpoint, _identity(), "unit")


def test_both_sides_absent_is_still_refused():
    partial = {field: "x" for field in evaluator.IDENTITY_FIELDS[:3]}
    with pytest.raises(evaluator.EvalRefused):
        evaluator.assert_identity(partial, partial, "unit")


def test_a_none_on_either_side_is_refused():
    with pytest.raises(evaluator.EvalRefused, match="None on one side"):
        evaluator.assert_identity(_identity(seed=None), _identity(), "unit")


# --- what the evaluator writes -------------------------------------------------


def test_writing_is_idempotent_but_never_silently_different(tmp_path):
    path = tmp_path / "record.json"
    payload = {"a": 1, "b": [2, 3]}
    evaluator.atomic_write_json(path, payload)
    evaluator.atomic_write_json(path, payload)          # identical: fine
    with pytest.raises(evaluator.EvalRefused, match="different content"):
        evaluator.atomic_write_json(path, {"a": 2})


def test_the_json_refuses_non_finite(tmp_path):
    with pytest.raises(ValueError):
        evaluator.atomic_write_json(tmp_path / "x.json", {"a": float("nan")})


# --- the twelve corrected evaluations on disk ----------------------------------


@pytest.fixture(scope="module")
def records(runs):
    found = {}
    for name, run in runs.items():
        path = run.directory / evaluator.EVAL_NAME
        if path.exists():
            found[name] = json.loads(path.read_text("utf-8"))
    return found


def test_every_corrected_evaluation_used_its_own_seed(records, runs):
    if len(records) < 12:
        pytest.skip("the corrected evaluations have not been produced here")
    for name, record in records.items():
        assert record["schema_version"] == evaluator.EVAL_SCHEMA, name
        assert record["run_seed_used"] == runs[name].seed, name
        assert record["seed"] == runs[name].seed, name
        assert record["partial_diagnostic_only"] is False, name
        assert record["agrees"] is True, name
        assert record["max_abs_cell_difference"] <= evaluator.TOLERANCE, name


def test_the_v1_evaluations_are_marked_invalid_and_kept(runs):
    for name, run in runs.items():
        path = run.directory / evaluator.V1_NAME
        if not path.exists():
            continue
        record = json.loads(path.read_text("utf-8"))
        assert record.get("invalid_wrong_seed") is True, name
        assert "without the run's seed" in record.get("invalid_reason", ""), name


def test_the_correction_record_describes_the_defect():
    path = ROOT / "configs/paper/v4_development_eval_correction.json"
    record = json.loads(path.read_text("utf-8"))
    assert record["kind"] == "post_registration_analysis_correction"
    assert record["plan_was_not_rewritten"] is True
    assert record["checkpoint_plan_hashes_unchanged"] is True
    assert record["correction"]["new_collector"]["reads_v1"] is False
    for untouched in ("model", "data", "endpoint", "aggregation rules"):
        assert untouched in record["correction"]["unchanged"]
    assert record["result_after_correction"]["agrees"] == 12


# --- the collector's intake ----------------------------------------------------


def test_the_collector_reads_only_v2(tmp_path, runs):
    from scripts import collect_v4_development_base_v2 as collector

    assert collector.EVAL_NAME == "development_validation_eval_v2.json"
    tables, missing = collector.load_tables(list(runs.values()))
    if missing:
        pytest.skip(f"corrected evaluations incomplete here: {missing[:2]}")
    for dataset in collector.DATASETS:
        for model in collector.MODELS:
            assert sorted(tables[dataset][model]) == [32, 42, 52]


@pytest.mark.parametrize("tamper, expected", [
    ({"schema_version": 1}, "not v2"),
    ({"partial_diagnostic_only": True}, "partial"),
    ({"invalid_wrong_seed": True}, "invalid_wrong_seed"),
    ({"run_seed_used": 32}, "evaluated at seed"),
    ({"agrees": False}, "disagrees"),
    ({"max_abs_cell_difference": 1.0}, "exceeds"),
])
def test_the_collector_refuses_a_bad_evaluation(tmp_path, plan, tamper, expected):
    from scripts import collect_v4_development_base_v2 as collector

    # A seed-42 run, so tampering `run_seed_used` to 32 is a genuine mismatch.
    root = ROOT / "outputs_v4/development_base" / plan["plan_sha256"]
    source = next(r for r in plan_runs(root, plan) if r.seed == 42)
    path = source.directory / evaluator.EVAL_NAME
    if not path.exists():
        pytest.skip("no corrected evaluation to tamper with here")
    record = dict(json.loads(path.read_text("utf-8")), **tamper)

    fake = next(r for r in plan_runs(tmp_path, plan) if r.name == source.name)
    fake.directory.mkdir(parents=True, exist_ok=True)
    (fake.directory / evaluator.EVAL_NAME).write_text(
        json.dumps(record, allow_nan=False), "utf-8")
    _, missing = collector.load_tables([fake])
    assert missing and expected in missing[0], missing


def test_a_partial_file_beside_a_full_one_is_refused(tmp_path, plan):
    from scripts import collect_v4_development_base_v2 as collector

    root = ROOT / "outputs_v4/development_base" / plan["plan_sha256"]
    source = plan_runs(root, plan)[0]
    path = source.directory / evaluator.EVAL_NAME
    if not path.exists():
        pytest.skip("no corrected evaluation here")
    fake = plan_runs(tmp_path, plan)[0]
    fake.directory.mkdir(parents=True, exist_ok=True)
    (fake.directory / evaluator.EVAL_NAME).write_text(path.read_text("utf-8"),
                                                      "utf-8")
    (fake.directory / evaluator.PARTIAL_NAME).write_text("{}", "utf-8")
    _, missing = collector.load_tables([fake])
    assert missing and "partial file is present" in missing[0]


def test_both_models_share_the_validation_design_at_each_seed():
    """Same dataset, same seed: the paired design must be identical."""
    for seed in (32, 42, 52):
        left = build_v4_session(ROOT / "configs/paper/v4/PEMS08_ALFNet.yaml",
                                device="cpu", run_seed=seed)
        right = build_v4_session(ROOT / "configs/paper/v4/PEMS08_MoGERNN.yaml",
                                 device="cpu", run_seed=seed)
        assert left.validation_context.exposure_hash() == \
               right.validation_context.exposure_hash(), seed
        assert (left.validation_dataset.origins ==
                right.validation_dataset.origins).all(), seed
        assert left.scaler == right.scaler, seed
