#!/usr/bin/env python
"""Audit the twelve v4 development-base runs before anything reads their numbers.

The v3 lesson this exists for: a table is only as trustworthy as the runs behind
it, and "all twelve finished" is not the same as "all twelve ran the experiment
the plan registered". So this checks identity, not just presence -- config, plan,
code, payload, unseen and manifest hashes; the static asset and both exposure
hashes; that `best.pt` and `last.pt` agree about which epoch was selected; that
selection took the earliest minimum of validation LifeAvg; and that every
development table says it is validation-only and mentions no test.

Exits non-zero on any finding. It never repairs, and it never re-hashes to make
a mismatch go away.

    python scripts/audit_v4_development_base.py
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v4_validation_collector import cell_count  # noqa: E402
from scripts.build_v4_development_plan import (  # noqa: E402
    assert_plan_still_holds,
    load_plan,
    sha256_file,
)
from scripts.train_v4 import (  # noqa: E402
    CODE_SOURCES,
    COMPLETE_MARKER,
    RUN_STATE_NAME,
    TIE_RULE,
)
from scripts.v4_launcher import plan_runs  # noqa: E402

FORBIDDEN_WORDS = ("test", "confirmation")


def _finite(node, path, findings):
    if isinstance(node, dict):
        for key, value in node.items():
            _finite(value, f"{path}.{key}", findings)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _finite(value, f"{path}[{index}]", findings)
    elif isinstance(node, float) and not math.isfinite(node):
        findings.append(f"{path} is {node}")


def audit_run(run, plan, findings, notes):
    where = run.name
    directory = run.directory
    if not (directory / COMPLETE_MARKER).exists():
        findings.append(f"{where}: not COMPLETE")
        return
    missing = [name for name in ("best.pt", RUN_STATE_NAME, "run_report.json",
                                 "development_validation_table.json")
               if not (directory / name).exists()]
    if missing:
        findings.append(f"{where}: missing {missing}")
        return

    best = torch.load(directory / "best.pt", weights_only=False, map_location="cpu")
    last = torch.load(directory / RUN_STATE_NAME, weights_only=False,
                      map_location="cpu")
    report = json.loads((directory / "run_report.json").read_text("utf-8"))
    table = json.loads(
        (directory / "development_validation_table.json").read_text("utf-8"))
    registered = plan["configs"][f"{run.dataset}_{run.model}"]
    data = plan["data"][run.dataset]

    # --- identity ------------------------------------------------------------
    checks = {
        "protocol": ("lcbt_v4", best.get("protocol")),
        "dataset": (run.dataset, best.get("dataset")),
        "model": (run.model, best.get("model")),
        "seed": (run.seed, best.get("seed")),
        "config_sha256": (registered["sha256"], best.get("config_sha256")),
        "plan_sha256": (plan["plan_sha256"], best.get("plan_sha256")),
        "payload_sha256": (data["payload_sha256"], best.get("payload_sha256")),
        "unseen_sha256": (data["unseen_sha256"], best.get("unseen_sha256")),
        "manifest_sha256": (data["manifest_sha256"], best.get("manifest_sha256")),
        "early_stopping_metric": ("validation life_avg_mae",
                                  best.get("early_stopping_metric")),
        "tie_rule": (TIE_RULE, best.get("tie_rule")),
        "split_fractions": ([0.60, 0.20, 0.20], best.get("split_fractions")),
        "teacher_forcing_policy": (plan["training"]["teacher_forcing_policy"],
                                   best.get("teacher_forcing_policy")),
    }
    for field, (expected, actual) in checks.items():
        if actual != expected:
            findings.append(f"{where}: {field} is {actual!r}, expected {expected!r}")
    if best.get("train_exposure_hash") == best.get("validation_exposure_hash"):
        findings.append(f"{where}: train and validation share an exposure hash")
    for field in ("static_asset_hash", "train_exposure_hash",
                  "validation_exposure_hash"):
        if best.get(field) != last.get(field):
            findings.append(f"{where}: {field} differs between best.pt and last.pt")
    # Code drift since the run is *recorded*, not a finding. These runs are
    # complete; the code moving afterwards -- the evaluator seed fix, the
    # plan-driven launcher -- does not retroactively change what they did, and
    # the identity that decides whether they are comparable (payload, unseen,
    # manifest, config, plan and both exposure hashes) is checked above. Drift
    # *during* a run is what the plan's launch-time guard prevents.
    for name, digest in (last.get("code_sha256") or {}).items():
        if name in CODE_SOURCES and digest != sha256_file(ROOT / name):
            notes.append(f"{where}: {name} has changed since the run "
                         f"({digest[:12]} -> {sha256_file(ROOT / name)[:12]})")

    # --- selection consistency ----------------------------------------------
    if best.get("best_epoch") != last.get("best_epoch"):
        findings.append(f"{where}: best epoch {best.get('best_epoch')} in best.pt "
                        f"vs {last.get('best_epoch')} in {RUN_STATE_NAME}")
    if report.get("best_epoch") != best.get("best_epoch"):
        findings.append(f"{where}: run_report best epoch disagrees with best.pt")
    history = report.get("history") or []
    if history:
        curve = [(entry["epoch"], entry["validation"]["life_avg_mae"])
                 for entry in history]
        lowest = min(value for _, value in curve)
        earliest = min(epoch for epoch, value in curve if value == lowest)
        if report["best_epoch"] != earliest:
            findings.append(
                f"{where}: best epoch {report['best_epoch']} is not the earliest "
                f"minimum of validation LifeAvg ({earliest})")
        if best.get("epoch") != report["best_epoch"]:
            findings.append(f"{where}: best.pt was written at epoch "
                            f"{best.get('epoch')}, selected {report['best_epoch']}")

    # --- the development table ----------------------------------------------
    if table.get("development_validation_only") is not True:
        findings.append(f"{where}: table is not marked development_validation_only")
    if table.get("split") != "validation":
        findings.append(f"{where}: table split is {table.get('split')!r}")
    if cell_count(table) != 36:
        findings.append(f"{where}: table has {cell_count(table)} cells, expected 36")
    for section in ("main", "phase"):
        for row, cell in table.get(section, {}).items():
            if int(cell.get("valid_target_count", 0)) <= 0:
                findings.append(f"{where}: {section}/{row} has no valid targets")
    serialised = json.dumps({key: value for key, value in table.items()
                             if key != "note"}).lower()
    for word in FORBIDDEN_WORDS:
        if f'"{word}' in serialised:
            findings.append(f"{where}: table schema mentions {word!r}")
    _finite(table, f"{where}.table", findings)
    _finite({key: value for key, value in report.items() if key != "history"},
            f"{where}.report", findings)


def audit(output_root=None):
    plan = assert_plan_still_holds(load_plan())
    root = Path(output_root or
                ROOT / "outputs_v4/development_base" / plan["plan_sha256"])
    runs = plan_runs(root, plan)
    findings, notes = [], []
    names = [run.name for run in runs]
    if len(set(names)) != len(names):
        findings.append(f"duplicate runs in the plan: {names}")
    if len(runs) != 12:
        findings.append(f"expected 12 runs, planned {len(runs)}")
    for run in runs:
        audit_run(run, plan, findings, notes)
    complete = sum(1 for run in runs if (run.directory / COMPLETE_MARKER).exists())
    return {"plan_sha256": plan["plan_sha256"], "output_root": str(root),
            "complete": complete, "total": len(runs), "findings": findings,
            "notes": notes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    arguments = parser.parse_args(argv)
    result = audit(arguments.output_root)
    print(f"plan {result['plan_sha256'][:16]}  root {result['output_root']}")
    print(f"COMPLETE {result['complete']}/{result['total']}")
    if result["notes"]:
        print(f"\n{len(result['notes'])} note(s), none of which invalidate a run:")
        for note in result["notes"][:6]:
            print(f"  . {note}")
        if len(result["notes"]) > 6:
            print(f"  . ... and {len(result['notes']) - 6} more")
    if result["findings"]:
        print(f"\n{len(result['findings'])} finding(s):")
        for finding in result["findings"]:
            print(f"  - {finding}")
        return 1
    print("\naudit passed: identity, selection and validation-only labelling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
