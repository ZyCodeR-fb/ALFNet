"""The five-dataset plan: reuse, readiness, and a test split that stays shut.

Nothing in this file reads a real test target. The one test that exercises the
test evaluator's own guard monkeypatches the context resolver to raise on
`phase="test"`, so a regression that reached for real data would fail loudly
rather than quietly succeed.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_v4_five_dataset_plan as five
from scripts import build_v4_readiness as readiness
from scripts import v4_launcher, v4_reuse_audit
from scripts.build_v4_readiness import ReadinessRefused
from scripts.v4_reuse_audit import CODE_ALLOWANCES

PLAN_PATH = ROOT / "configs/paper/v4_five_dataset_plan.json"


@pytest.fixture(scope="module")
def plan():
    return five.assert_plan_still_holds(five.load_plan())


@pytest.fixture(scope="module")
def attestation():
    return readiness.load()


# --- the plan's shape ----------------------------------------------------------


def test_thirty_slots_are_twelve_reuse_and_eighteen_train(plan):
    execution = plan["execution"]
    assert execution["total_result_slots"] == 30 == plan["trial_budget"]
    assert execution["provenance_verified_reused_runs"] == 12
    assert execution["new_training_runs"] == 18
    assert execution["gpu_jobs_to_execute"] == 18
    assert execution["reused_runs_are_not_runs_of_this_plan"] is True

    actions = {}
    for entry in plan["runs"]:
        actions.setdefault(entry["dataset"], []).append(entry["execution_action"])
    assert {d: sorted(set(a)) for d, a in actions.items()} == {
        "PEMS03": ["train"], "PEMS04": ["train"], "METRLA": ["train"],
        "PEMS08": ["reuse"], "PEMSBAY": ["reuse"]}


def test_every_reused_slot_carries_its_provenance(plan):
    for entry in plan["runs"]:
        if entry["execution_action"] != "reuse":
            continue
        for field in ("source_plan_sha256", "source_git_commit",
                      "source_run_directory", "checkpoint_sha256",
                      "reuse_compatibility_status"):
            assert entry.get(field), (entry["dataset"], entry["model"], field)
        assert entry["reuse_compatibility_status"] == "COMPATIBLE"
        assert entry["source_plan_sha256"] != plan["plan_sha256"]


def test_flow_and_speed_are_never_pooled(plan):
    rules = plan["cross_dataset_reporting"]
    assert rules["units_differ"] is True
    assert set(rules["flow_datasets"]) == {"PEMS03", "PEMS04", "PEMS08"}
    assert set(rules["speed_datasets"]) == {"METRLA", "PEMSBAY"}
    assert any("avg_mae" in item for item in rules["forbidden"])
    units = {d: plan["data"][d]["unit"] for d in plan["datasets"]}
    assert units == {"PEMS03": "veh_per_5min", "PEMS04": "veh_per_5min",
                     "PEMS08": "veh_per_5min", "METRLA": "mph", "PEMSBAY": "mph"}


def test_exposure_is_labelled_honestly(plan):
    status = {d: plan["data"][d]["test_exposure"]["status"] for d in plan["datasets"]}
    assert status == {"PEMS03": "confirmatory", "PEMS04": "confirmatory",
                      "PEMS08": "supportive", "METRLA": "supportive",
                      "PEMSBAY": "supportive"}
    assert plan["reporting_obligations"]["every_dataset_reported"] is True
    assert plan["reporting_obligations"][
        "36_of_36_is_an_engineering_target_not_a_stopping_rule"] is True


def test_both_models_share_the_node_split_at_every_seed(plan):
    """One dataset, one seed: K/V/T must be identical for the two models."""
    for dataset in plan["datasets"]:
        per_seed = plan["data"][dataset]["node_split_per_seed"]
        assert sorted(per_seed) == ["32", "42", "52"]
        for seed, entry in per_seed.items():
            for field in ("known_nodes_sha256", "validation_unseen_sha256",
                          "test_unseen_sha256", "static_asset_hash"):
                assert entry[field], (dataset, seed, field)
            assert entry["train_exposure_hash"] != entry["validation_exposure_hash"]
        # different seeds must actually give different held-out sets
        assert len({e["validation_unseen_sha256"] for e in per_seed.values()}) == 3


# --- the identity graph has no cycle -------------------------------------------


def test_the_plan_and_attestation_reference_each_other_one_way(plan, attestation):
    """The attestation names the plan; the plan names only a path and schema."""
    reference = plan["readiness_attestation"]
    assert reference["path"] == "configs/paper/v4_five_dataset_readiness.json"
    assert reference["must_reference_this_plan_sha256"] is True
    # Nothing in the plan is the attestation's digest -- that would be a cycle.
    serialised = json.dumps(plan)
    assert attestation["attestation_sha256"] not in serialised
    # ... and the attestation does carry the plan's.
    assert attestation["plan_sha256"] == plan["plan_sha256"]


def test_regenerating_either_one_converges_without_a_fixed_point(tmp_path):
    """Rebuild in both orders; neither needs the other's digest to settle.

    A cyclic design could only be satisfied by rewriting one file until the two
    agreed, which verifies nothing. Here the plan is a function of the
    experiment, and the attestation is a function of the plan.
    """
    plan_a = five.build_plan(include_commit=False)
    plan_a["plan_sha256"] = five.plan_hash(plan_a)
    plan_b = five.build_plan(include_commit=False)
    plan_b["plan_sha256"] = five.plan_hash(plan_b)
    assert plan_a["plan_sha256"] == plan_b["plan_sha256"], "the plan is not stable"

    live = five.load_plan()
    first = readiness.build(live, preflight={"all_passed": True, "combinations": 10})
    second = readiness.build(live, preflight={"all_passed": True, "combinations": 10})
    del first["generated_at"], second["generated_at"]
    first["attestation_sha256"] = readiness.attestation_hash(first)
    second["attestation_sha256"] = readiness.attestation_hash(second)
    assert first["attestation_sha256"] == second["attestation_sha256"]


# --- readiness is a gate -------------------------------------------------------


def test_the_attestation_reports_every_flag_true(attestation, plan):
    assert attestation["readiness"] == {"data_ready": True, "graph_ready": True,
                                        "exposure_ready": True,
                                        "training_ready": True}
    assert attestation["plan_sha256"] == plan["plan_sha256"]
    assert attestation["reuse_audit"]["all_compatible"] is True
    assert attestation["reuse_audit"]["runs_compatible"] == 12
    assert attestation["cpu_preflight"]["combinations"] == 10
    assert attestation["cpu_preflight"]["all_passed"] is True
    assert attestation["cpu_preflight"]["test_context_opened"] is False


@pytest.mark.parametrize("mutate, expected", [
    (lambda r: r["readiness"].__setitem__("training_ready", False), "false"),
    (lambda r: r["readiness"].__setitem__("exposure_ready", False), "false"),
    (lambda r: r.__setitem__("plan_sha256", "0" * 64), "permits plan"),
    (lambda r: r["reuse_audit"].__setitem__("all_compatible", False),
     "reuse audit"),
])
def test_readiness_refuses_before_anything_is_created(tmp_path, plan, mutate,
                                                      expected):
    record = json.loads(readiness.ATTESTATION_PATH.read_text("utf-8"))
    mutate(record)
    record["attestation_sha256"] = readiness.attestation_hash(record)
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(record), "utf-8")
    with pytest.raises(ReadinessRefused, match=expected):
        readiness.assert_ready(plan, path)


def test_a_missing_or_edited_attestation_is_refused(tmp_path, plan):
    with pytest.raises(ReadinessRefused, match="missing"):
        readiness.assert_ready(plan, tmp_path / "absent.json")
    record = json.loads(readiness.ATTESTATION_PATH.read_text("utf-8"))
    record["readiness"]["training_ready"] = False       # without rehashing
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(record), "utf-8")
    with pytest.raises(ReadinessRefused, match="does not describe its contents"):
        readiness.assert_ready(plan, path)


def test_the_launcher_checks_readiness_before_touching_anything(monkeypatch,
                                                                tmp_path):
    """The refusal must come before a run directory, a child or CUDA."""
    calls = []
    monkeypatch.setattr(v4_launcher, "acquire_lock",
                        lambda run: calls.append("lock"))
    monkeypatch.setattr(v4_launcher.subprocess, "Popen",
                        lambda *a, **k: calls.append("spawn"))
    record = json.loads(readiness.ATTESTATION_PATH.read_text("utf-8"))
    record["readiness"]["training_ready"] = False
    record["attestation_sha256"] = readiness.attestation_hash(record)
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(record), "utf-8")
    monkeypatch.setattr(readiness, "ATTESTATION_PATH", path)
    with pytest.raises(v4_launcher.LauncherRefused, match="training_ready"):
        v4_launcher.resolve_plan_and_root(str(PLAN_PATH))
    assert calls == [], "the launcher acted before the readiness gate"


# --- the reuse audit -----------------------------------------------------------


def test_the_reuse_audit_passes_for_all_twelve(plan):
    from scripts.build_v4_development_plan import (assert_plan_still_holds,
                                                   load_plan as load_development)

    records = v4_reuse_audit.audit_reusable(
        assert_plan_still_holds(load_development()), plan["data"], plan["configs"])
    assert len(records) == 12
    assert all(r["execution_action"] == "reuse" for r in records), [
        (r["run"], r["findings"]) for r in records if r["findings"]]
    for record in records:
        assert record["source_git_commit"]
        assert record["checkpoint_sha256"]


@pytest.mark.parametrize("field, value", [
    ("dataset", "PEMS03"), ("model", "MoGERNN"), ("seed", 99),
    ("config_sha256", "0" * 64), ("payload_sha256", "0" * 64),
    ("manifest_sha256", "0" * 64), ("parameter_count", 1),
    ("early_stopping_metric", "train_loss"), ("tie_rule", "keep_later_epoch"),
])
def test_one_wrong_checkpoint_field_rejects_only_that_run(tmp_path, plan, field,
                                                          value, monkeypatch):
    from scripts.build_v4_development_plan import (assert_plan_still_holds,
                                                   load_plan as load_development)
    from scripts.v4_launcher import plan_runs

    development = assert_plan_still_holds(load_development())
    root = ROOT / "outputs_v4/development_base" / development["plan_sha256"]
    run = plan_runs(root, development)[0]

    original = torch.load
    def tampered(path, *args, **kwargs):
        loaded = original(path, *args, **kwargs)
        if str(path).endswith("best.pt"):
            loaded = dict(loaded, **{field: value})
        return loaded

    monkeypatch.setattr(torch, "load", tampered)
    record = v4_reuse_audit.audit_run(
        run.dataset, run.model, run.seed, run.directory,
        ROOT / plan["configs"][f"{run.dataset}_{run.model}"]["path"],
        plan["data"][run.dataset], development["plan_sha256"])
    assert record["execution_action"] == "train"
    assert record["reuse_compatibility_status"] == "INCOMPATIBLE"
    assert record["findings"]


@pytest.mark.parametrize("field", ["dataset", "model", "seed",
                                   "static_asset_hash", "config_sha256"])
def test_a_missing_checkpoint_field_is_a_refusal(tmp_path, plan, field,
                                                 monkeypatch):
    from scripts.build_v4_development_plan import (assert_plan_still_holds,
                                                   load_plan as load_development)
    from scripts.v4_launcher import plan_runs

    development = assert_plan_still_holds(load_development())
    root = ROOT / "outputs_v4/development_base" / development["plan_sha256"]
    run = plan_runs(root, development)[0]
    original = torch.load

    def stripped(path, *args, **kwargs):
        loaded = original(path, *args, **kwargs)
        if str(path).endswith("best.pt"):
            loaded = {k: v for k, v in loaded.items() if k != field}
        return loaded

    monkeypatch.setattr(torch, "load", stripped)
    record = v4_reuse_audit.audit_run(
        run.dataset, run.model, run.seed, run.directory,
        ROOT / plan["configs"][f"{run.dataset}_{run.model}"]["path"],
        plan["data"][run.dataset], development["plan_sha256"])
    assert record["execution_action"] == "train"


def test_the_code_allowance_names_a_content_pair_not_a_file():
    """A blanket file whitelist would let the next edit ride along."""
    assert CODE_ALLOWANCES
    for allowance in CODE_ALLOWANCES:
        assert allowance["from_sha256"] and allowance["to_sha256"]
        assert allowance["from_sha256"] != allowance["to_sha256"]
        assert allowance["reason"] and allowance["reviewed_diff"]
    # An unrelated change to the same file is not covered.
    findings = v4_reuse_audit.code_findings(
        {"scripts/train_v4.py": "deadbeefdeadbeef" + "0" * 48})
    assert findings and "not a reviewed allowance" in findings[0]


# --- the launcher runs eighteen ------------------------------------------------


def test_the_launcher_enumerates_only_the_training_runs(plan, tmp_path):
    runs = v4_launcher.plan_runs(tmp_path, plan)
    counts = v4_launcher.summarise(runs)
    assert counts["total_result_slots"] == 30
    assert counts["reused_complete"] == 12
    assert counts["new_training"] == counts["gpu_jobs_to_execute"] == 18
    for run in counts["reused"]:
        assert v4_launcher.run_state(run) == "REUSED_COMPLETE"
        with pytest.raises(v4_launcher.LauncherRefused, match="not a training run"):
            v4_launcher.Launcher(tmp_path, plan, ["cpu"]).start(run, "cpu")


def test_a_reused_run_points_at_its_source_directory(plan, tmp_path):
    runs = v4_launcher.plan_runs(tmp_path, plan)
    for run in runs:
        if run.execution_action == "reuse":
            assert "development_base" in str(run.directory)
            assert not str(run.directory).startswith(str(tmp_path))
        else:
            assert str(run.directory).startswith(str(tmp_path))


# --- the validation evaluator --------------------------------------------------


def test_the_plan_evaluator_passes_the_run_seed(monkeypatch, plan, tmp_path):
    from scripts import eval_v4_plan_validation as evaluator

    seen = {}

    def capture(config_path, device="cpu", run_seed=None, **kwargs):
        seen["run_seed"] = run_seed
        raise RuntimeError("stop before the forward pass")

    monkeypatch.setattr(evaluator, "build_v4_session", capture)
    run = next(r for r in v4_launcher.plan_runs(tmp_path, plan)
               if r.execution_action == "reuse" and r.seed == 52)
    with pytest.raises(RuntimeError, match="stop before"):
        evaluator.evaluate_run(run, plan, device="cpu")
    assert seen["run_seed"] == 52


def test_the_plan_evaluator_refuses_a_refit_scaler(monkeypatch, plan, tmp_path):
    """The checkpoint's scaler, or nothing."""
    from scripts import eval_v4_plan_validation as evaluator

    run = next(r for r in v4_launcher.plan_runs(tmp_path, plan)
               if r.execution_action == "reuse")
    original = torch.load

    def moved_scaler(path, *args, **kwargs):
        loaded = original(path, *args, **kwargs)
        if str(path).endswith("best.pt"):
            loaded = dict(loaded)
            loaded["scaler"] = dict(loaded["scaler"], mean=float(
                loaded["scaler"]["mean"]) + 1.0)
        return loaded

    monkeypatch.setattr(torch, "load", moved_scaler)
    with pytest.raises(evaluator.PlanEvalRefused, match="refitted scaler"):
        evaluator.evaluate_run(run, plan, device="cpu")


def test_the_plan_collector_refuses_incomplete_evidence(plan):
    from scripts import collect_v4_plan_validation as collector

    with pytest.raises(SystemExit):
        collector.collect(str(PLAN_PATH))


# --- the test split stays shut -------------------------------------------------


def test_the_test_evaluator_refuses_while_runs_are_missing(monkeypatch, plan):
    """And it must refuse *before* resolving a single test context."""
    from lib import runtime_context
    from scripts import eval_v4_plan_test as tester

    def no_test(dataset, phase, **kwargs):
        assert phase != "test", f"a test context was built for {dataset}"
        return runtime_context.build_runtime_context(dataset, phase, **kwargs)

    monkeypatch.setattr(tester, "build_runtime_context", no_test)
    runs, root = tester.resolve_runs(plan)
    with pytest.raises(tester.TestAccessRefused) as error:
        tester.assert_may_open(plan, runs, root)
    message = str(error.value)
    assert "not COMPLETE" in message
    assert "validation summary" in message


def test_the_test_evaluator_names_every_gate():
    from scripts import eval_v4_plan_test as tester

    source = Path(tester.__file__).read_text("utf-8")
    for gate in ("readiness", "reuse audit", "not COMPLETE",
                 "validation recomputation", "validation summary",
                 "has changed since the plan was frozen"):
        assert gate in source, gate


def test_test_access_is_recorded_once_and_resumable_only_identically(tmp_path,
                                                                     plan):
    from scripts import eval_v4_plan_test as tester

    record = tester.open_test_access(plan, tmp_path)
    assert (tmp_path / tester.OPENED_MARKER).exists()
    assert tester.open_test_access(plan, tmp_path) == record   # idempotent

    marker = tmp_path / tester.OPENED_MARKER
    tampered = json.loads(marker.read_text("utf-8"))
    tampered["identity"]["plan_sha256"] = "0" * 64
    marker.write_text(json.dumps(tampered), "utf-8")
    with pytest.raises(tester.TestAccessRefused, match="different identity"):
        tester.open_test_access(plan, tmp_path)


def test_the_collector_refuses_before_the_test_audit_passes(plan):
    from scripts import collect_v4_plan_test as collector

    with pytest.raises(SystemExit):
        collector.collect(str(PLAN_PATH))


def test_no_real_test_target_is_read_by_this_module(monkeypatch, plan):
    """A blunt guard: make every test context raise, then run the gates."""
    from lib import runtime_context

    original = runtime_context.build_runtime_context

    def refuse(dataset, phase, **kwargs):
        if phase == "test":
            raise AssertionError(f"a real {dataset} test context was requested")
        return original(dataset, phase, **kwargs)

    monkeypatch.setattr(runtime_context, "build_runtime_context", refuse)
    from scripts import eval_v4_plan_test as tester

    monkeypatch.setattr(tester, "build_runtime_context", refuse)
    runs, root = tester.resolve_runs(plan)
    with pytest.raises(tester.TestAccessRefused):
        tester.assert_may_open(plan, runs, root)


def test_the_validation_collector_never_uses_a_test_context():
    from experiments.v4_validation_collector import CollectorRefused, \
        ValidationCellCollector
    from lib.runtime_context import build_runtime_context

    with pytest.raises(CollectorRefused, match="validation only"):
        ValidationCellCollector(build_runtime_context("PEMS08", "test"),
                                {"mean": 0.0, "std": 1.0})
    # ... and the test evaluator opts in by name, not by bypassing it.
    collector = ValidationCellCollector(
        build_runtime_context("PEMS08", "test"), {"mean": 0.0, "std": 1.0},
        allow_phase="test")
    assert collector.context.phase == "test"
