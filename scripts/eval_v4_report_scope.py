#!/usr/bin/env python
"""Evaluate the immutable four-dataset/11-model reporting scope.

This is a selector and gate around the already-frozen validation and test
evaluators.  It never creates a new training identity: every checkpoint remains
anchored to ``final_plan.json`` and is checked against that plan.  The scope
only determines which completed runs are eligible for the final report.
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v4_validation_collector import METRICS, cell_count  # noqa: E402
from scripts.eval_v4_plan_test import (  # noqa: E402
    TEST_EVAL_NAME, TEST_SCHEMA, evaluate_run as evaluate_test_run,
)
from scripts.eval_v4_plan_validation import (  # noqa: E402
    EVAL_NAME, EVAL_SCHEMA, PARTIAL_NAME,
    evaluate_run as evaluate_validation_run, load_plan, resolve_runs,
)
from scripts.train_v4 import COMPLETE_MARKER  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402


class ScopeRefused(RuntimeError):
    """The selected report scope is incomplete or has drifted."""


REPORT_ATOL = 1e-6
REPORT_RTOL = 1e-5


def _atomic_write(path, payload):
    path = Path(path)
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text("utf-8") == text:
            return path
        raise ScopeRefused(f"{path} exists under a different identity")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    os.close(handle)
    try:
        Path(temporary).write_text(text, "utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_scope(path):
    path = Path(path)
    scope = json.loads(path.read_text("utf-8"))
    if scope.get("schema_version") != 1:
        raise ScopeRefused("unsupported report-scope schema")
    datasets = tuple(scope.get("datasets", ()))
    models = tuple(scope.get("models", ()))
    seeds = tuple(int(seed) for seed in scope.get("seeds", ()))
    if len(set(datasets)) != len(datasets) or len(set(models)) != len(models):
        raise ScopeRefused("duplicate dataset or model in report scope")
    expected = len(datasets) * len(models) * len(seeds)
    if expected != int(scope.get("expected_runs", -1)):
        raise ScopeRefused(
            f"scope product is {expected}, expected_runs says "
            f"{scope.get('expected_runs')}")
    tolerance = scope.get("validation_reproducibility_tolerance", {})
    if (float(tolerance.get("absolute", -1)) != REPORT_ATOL
            or float(tolerance.get("relative", -1)) != REPORT_RTOL):
        raise ScopeRefused("report-scope reproducibility tolerance drifted")
    plan_path = ROOT / scope["source_plan"]
    plan = load_plan(plan_path)
    runs, root = resolve_runs(plan)
    selected = [run for run in runs if run.dataset in datasets
                and run.model in models and int(run.seed) in seeds]
    wanted = {(dataset, model, seed) for dataset in datasets
              for model in models for seed in seeds}
    got = {(run.dataset, run.model, int(run.seed)) for run in selected}
    if got != wanted or len(selected) != expected:
        raise ScopeRefused(
            f"resolved scope differs: missing={sorted(wanted-got)[:4]}, "
            f"extra={sorted(got-wanted)[:4]}")
    selected.sort(key=lambda run: (
        models.index(run.model), seeds.index(int(run.seed)),
        datasets.index(run.dataset)))
    return scope, path, plan, selected, root


def _replay_agrees_with_training_table(run, record):
    """Accept only float32 replay noise under the frozen abs+relative rule."""
    stored_path = run.directory / "development_validation_table.json"
    if not stored_path.exists():
        return False
    stored = json.loads(stored_path.read_text("utf-8"))
    replay = record.get("recomputed", {})
    try:
        for section in ("main", "phase"):
            if set(stored[section]) != set(replay[section]):
                return False
            for row in stored[section]:
                for metric in METRICS:
                    if not math.isclose(
                            float(stored[section][row][metric]),
                            float(replay[section][row][metric]),
                            rel_tol=REPORT_RTOL, abs_tol=REPORT_ATOL):
                        return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _validation_problem(run):
    path = run.directory / EVAL_NAME
    if not path.exists():
        return f"no {EVAL_NAME}"
    if (run.directory / PARTIAL_NAME).exists():
        return "partial validation file is present"
    record = json.loads(path.read_text("utf-8"))
    replay_agrees = _replay_agrees_with_training_table(run, record)
    checks = (
        (record.get("schema_version") == EVAL_SCHEMA, "wrong schema"),
        (record.get("split") == "validation", "wrong split"),
        (record.get("plan_sha256") is not None, "missing plan identity"),
        (record.get("dataset") == run.dataset, "dataset mismatch"),
        (record.get("model") == run.model, "model mismatch"),
        (int(record.get("run_seed_used", -1)) == int(run.seed), "seed mismatch"),
        (replay_agrees, "does not agree with training table"),
        (not record.get("partial_diagnostic_only"), "partial diagnostic"),
        (record.get("scaler", {}).get("refit") is False, "scaler was refit"),
        (cell_count(record.get("recomputed", {})) == 33, "not 33 cells"),
        (record.get("checkpoint_sha256") == sha256_file(run.directory / "best.pt"),
         "checkpoint digest mismatch"),
        (record.get("evaluator_sha256") == sha256_file(
            ROOT / "scripts/eval_v4_plan_validation.py"),
         "validation evaluator digest mismatch"),
    )
    for okay, reason in checks:
        if not okay:
            return reason
    return None


def _test_problem(run):
    path = run.directory / TEST_EVAL_NAME
    if not path.exists():
        return f"no {TEST_EVAL_NAME}"
    record = json.loads(path.read_text("utf-8"))
    checks = (
        (record.get("schema_version") == TEST_SCHEMA, "wrong schema"),
        (record.get("split") == "test", "wrong split"),
        (record.get("dataset") == run.dataset, "dataset mismatch"),
        (record.get("model") == run.model, "model mismatch"),
        (int(record.get("run_seed_used", -1)) == int(run.seed), "seed mismatch"),
        (record.get("scaler", {}).get("refit") is False, "scaler was refit"),
        (int(record.get("teacher_forcing_positions", -1)) == 0,
         "teacher forcing was non-zero"),
        (cell_count(record.get("cells", {})) == 33, "not 33 cells"),
        (record.get("checkpoint_sha256") == sha256_file(run.directory / "best.pt"),
         "checkpoint digest mismatch"),
        (record.get("evaluator_sha256") == sha256_file(
            ROOT / "scripts/eval_v4_plan_test.py"),
         "test evaluator digest mismatch"),
    )
    for okay, reason in checks:
        if not okay:
            return reason
    return None


def marker_identity(scope_path, plan, selected):
    return {
        "scope_file": str(scope_path.relative_to(ROOT)),
        "scope_sha256": sha256_file(scope_path),
        "report_replay_tolerance": {
            "absolute": REPORT_ATOL, "relative": REPORT_RTOL},
        "source_plan_sha256": plan["plan_sha256"],
        "validation_evaluator_sha256": sha256_file(
            ROOT / "scripts/eval_v4_plan_validation.py"),
        "test_evaluator_sha256": sha256_file(
            ROOT / "scripts/eval_v4_plan_test.py"),
        "collector_sha256": sha256_file(
            ROOT / "experiments/v4_validation_collector.py"),
        "runs": {run.name: sha256_file(run.directory / "best.pt")
                 for run in selected},
    }


def marker_path(root, scope_path):
    return root / f"TEST_REPORT_SCOPE_{sha256_file(scope_path)[:16]}_OPENED.json"


def prepare_test(scope_path, plan, selected, root):
    problems = []
    for run in selected:
        if not (run.directory / COMPLETE_MARKER).exists():
            problems.append(f"{run.name}: not COMPLETE")
        problem = _validation_problem(run)
        if problem:
            problems.append(f"{run.name}: {problem}")
    if problems:
        raise ScopeRefused(
            "test remains closed; validation gate failed:\n  - "
            + "\n  - ".join(problems[:30]))
    identity = marker_identity(scope_path, plan, selected)
    payload = {
        "schema_version": 1,
        "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "identity": identity,
        "note": "test opened for the exact four-dataset/11-model report scope",
    }
    path = marker_path(root, scope_path)
    _atomic_write(path, payload)
    return path


def require_marker(scope_path, plan, selected, root):
    path = marker_path(root, scope_path)
    if not path.exists():
        raise ScopeRefused("test scope has not passed prepare-test")
    recorded = json.loads(path.read_text("utf-8"))
    if recorded.get("identity") != marker_identity(scope_path, plan, selected):
        raise ScopeRefused("test marker identity has drifted")
    return path


def select_one(selected, name):
    matches = [run for run in selected if run.name == name]
    if len(matches) != 1:
        raise ScopeRefused(f"scope has no unique run named {name!r}")
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--phase", required=True,
                        choices=("status", "list-validation", "validation",
                                 "prepare-test", "list-test", "test",
                                 "complete-test"))
    parser.add_argument("--run")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    scope, scope_path, plan, selected, root = load_scope(ROOT / args.scope)

    if args.phase == "status":
        complete = sum((run.directory / COMPLETE_MARKER).exists()
                       for run in selected)
        validation = sum(_validation_problem(run) is None for run in selected)
        test = sum(_test_problem(run) is None for run in selected)
        print(f"scope {sha256_file(scope_path)[:16]} runs={len(selected)} "
              f"complete={complete} validation={validation} test={test}")
        return 0
    if args.phase == "list-validation":
        for run in selected:
            if _validation_problem(run):
                print(run.name)
        return 0
    if args.phase == "validation":
        run = select_one(selected, args.run)
        if not (run.directory / COMPLETE_MARKER).exists():
            raise ScopeRefused(f"{run.name}: not COMPLETE")
        if _validation_problem(run) is None:
            print(f"REUSE validation {run.name}")
            return 0
        record = evaluate_validation_run(run, plan, args.device)
        problem = _validation_problem(run)
        if problem:
            raise ScopeRefused(f"{run.name}: validation output {problem}")
        print(f"AGREES validation {run.name} "
              f"max|diff|={record['max_abs_cell_difference']:.3e}")
        return 0
    if args.phase == "prepare-test":
        path = prepare_test(scope_path, plan, selected, root)
        print(f"test gate passed; wrote {path}")
        return 0
    if args.phase == "list-test":
        require_marker(scope_path, plan, selected, root)
        for run in selected:
            if _test_problem(run):
                print(run.name)
        return 0
    if args.phase == "test":
        require_marker(scope_path, plan, selected, root)
        run = select_one(selected, args.run)
        if _test_problem(run) is None:
            print(f"REUSE test {run.name}")
            return 0
        evaluate_test_run(run, plan, args.device)
        problem = _test_problem(run)
        if problem:
            raise ScopeRefused(f"{run.name}: test output {problem}")
        print(f"DONE test {run.name}")
        return 0
    if args.phase == "complete-test":
        require_marker(scope_path, plan, selected, root)
        problems = [f"{run.name}: {_test_problem(run)}" for run in selected
                    if _test_problem(run)]
        if problems:
            raise ScopeRefused("test scope incomplete:\n  - "
                               + "\n  - ".join(problems[:30]))
        path = root / f"TEST_REPORT_SCOPE_{sha256_file(scope_path)[:16]}_COMPLETE.json"
        _atomic_write(path, {
            "schema_version": 1,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "identity": marker_identity(scope_path, plan, selected),
            "runs": len(selected),
        })
        print(f"test scope COMPLETE; wrote {path}")
        return 0
    raise AssertionError(args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
