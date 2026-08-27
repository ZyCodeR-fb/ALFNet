#!/usr/bin/env python
"""Audit every result slot in a plan before anything reads its numbers.

Plan-driven: the slot list, the datasets and the seeds come from `--plan`. It
covers both kinds of slot -- a `reuse` slot is audited against its source plan
and its recorded checkpoint digest, a `train` slot against this plan -- and it
requires the same evidence from each.

Exits non-zero on any finding. Code that moved *after* a completed run is
recorded as a note rather than a finding: a pre-registration describes the past,
and the identity that decides comparability is checked directly.
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
from scripts.v4_utils import sha256_file  # noqa: E402
from scripts.eval_v4_plan_validation import (  # noqa: E402
    EVAL_NAME, EVAL_SCHEMA, PARTIAL_NAME, TOLERANCE, load_plan, resolve_runs,
)
from scripts.train_v4 import COMPLETE_MARKER, RUN_STATE_NAME, TIE_RULE  # noqa: E402

TEST_MARKERS = ("TEST_OPENED.json", "TEST_COMPLETE", "plan_test_eval.json")


def _finite(node, path, out):
    if isinstance(node, dict):
        for key, value in node.items():
            _finite(value, f"{path}.{key}", out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _finite(value, f"{path}[{index}]", out)
    elif isinstance(node, float) and not math.isfinite(node):
        out.append(f"{path} is {node}")


def audit_run(run, plan, findings, notes, design):
    where = run.name
    directory = run.directory
    if not (directory / COMPLETE_MARKER).exists():
        findings.append(f"{where}: not COMPLETE ({run.execution_action})")
        return
    for name in ("best.pt", RUN_STATE_NAME):
        if not (directory / name).exists():
            findings.append(f"{where}: missing {name}")
            return

    best = torch.load(directory / "best.pt", weights_only=False, map_location="cpu")
    entry = plan["configs"][f"{run.dataset}_{run.model}"]
    data = plan["data"][run.dataset]

    for field, expected in (("protocol", "lcbt_v4"), ("dataset", run.dataset),
                            ("model", run.model), ("seed", run.seed),
                            ("config_sha256", entry["sha256"]),
                            ("payload_sha256", data["payload_sha256"]),
                            ("unseen_sha256", data["unseen_sha256"]),
                            ("manifest_sha256", data["manifest_sha256"]),
                            ("early_stopping_metric", "validation life_avg_mae"),
                            ("tie_rule", TIE_RULE)):
        if best.get(field) != expected:
            findings.append(f"{where}: {field} is {best.get(field)!r}, "
                            f"expected {expected!r}")
    if best.get("train_exposure_hash") == best.get("validation_exposure_hash"):
        findings.append(f"{where}: train and validation share an exposure hash")

    if run.execution_action == "reuse":
        for field in ("source_plan_sha256", "checkpoint_sha256"):
            if not run.entry.get(field):
                findings.append(f"{where}: reused slot records no {field}")
        recorded = run.entry.get("checkpoint_sha256")
        actual = sha256_file(directory / "best.pt")
        if recorded and recorded != actual:
            findings.append(f"{where}: checkpoint digest {actual[:12]} does not "
                            f"match the plan's {recorded[:12]}")
        if best.get("plan_sha256") != run.entry.get("source_plan_sha256"):
            findings.append(f"{where}: checkpoint plan {best.get('plan_sha256')} "
                            f"is not the recorded source plan")
    else:
        if best.get("plan_sha256") != plan["plan_sha256"]:
            findings.append(f"{where}: trained under plan "
                            f"{best.get('plan_sha256')}, not this one")

    evaluation_path = directory / EVAL_NAME
    if not evaluation_path.exists():
        findings.append(f"{where}: no {EVAL_NAME}; validation was not recomputed")
    else:
        record = json.loads(evaluation_path.read_text("utf-8"))
        if record.get("schema_version") != EVAL_SCHEMA:
            findings.append(f"{where}: validation evaluation is schema "
                            f"{record.get('schema_version')}")
        if record.get("partial_diagnostic_only"):
            findings.append(f"{where}: validation evaluation is partial")
        if int(record.get("run_seed_used", -1)) != int(run.seed):
            findings.append(f"{where}: validation used seed "
                            f"{record.get('run_seed_used')}, not {run.seed}")
        if not record.get("agrees"):
            findings.append(f"{where}: validation recomputation disagrees")
        if float(record.get("max_abs_cell_difference", 1.0)) > TOLERANCE:
            findings.append(f"{where}: validation max diff "
                            f"{record.get('max_abs_cell_difference')} > {TOLERANCE}")
        if record.get("scaler", {}).get("refit") is not False:
            findings.append(f"{where}: validation did not use the checkpoint scaler")
        if cell_count(record.get("recomputed", {"main": {}, "phase": {}})) != 33:
            findings.append(f"{where}: validation table is not 33 cells")
        _finite(record.get("recomputed"), f"{where}.table", findings)
        design.setdefault((run.dataset, run.seed), {})[run.model] = {
            row: record["recomputed"][section][row]["valid_target_count"]
            for section in ("main", "phase")
            for row in record["recomputed"][section]}
    if (directory / PARTIAL_NAME).exists():
        findings.append(f"{where}: a partial evaluation sits beside the full one")
    for marker in TEST_MARKERS:
        if (directory / marker).exists():
            findings.append(f"{where}: {marker} is present -- a test split was "
                            "opened for this run")
    for name, digest in (torch.load(directory / RUN_STATE_NAME, weights_only=False,
                                    map_location="cpu").get("code_sha256") or {}).items():
        path = ROOT / name
        if path.exists() and sha256_file(path) != digest:
            notes.append(f"{where}: {name} has changed since the run")


def audit(plan_path, output_root=None):
    plan = load_plan(plan_path)
    runs, root = resolve_runs(plan, output_root)
    findings, notes, design = [], [], {}
    declared = plan.get("execution", {})
    if declared:
        reused = sum(1 for run in runs if run.execution_action == "reuse")
        trained = sum(1 for run in runs if run.execution_action == "train")
        if reused != declared.get("provenance_verified_reused_runs", 0):
            findings.append(f"{reused} reused slots, plan declares "
                            f"{declared['provenance_verified_reused_runs']}")
        if trained != declared["new_training_runs"]:
            findings.append(f"{trained} training slots, plan declares "
                            f"{declared['new_training_runs']}")
    if len(runs) != plan["trial_budget"]:
        findings.append(f"{len(runs)} slots, plan declares {plan['trial_budget']}")
    for run in runs:
        audit_run(run, plan, findings, notes, design)
    # Every model must have scored the same valid cells as ALFNet.
    for (dataset, seed), models in sorted(design.items()):
        if "ALFNet" in models:
            for model, counts in models.items():
                if counts != models["ALFNet"]:
                    findings.append(
                        f"{dataset}/{model}/seed{seed}: valid-cell counts differ "
                        "from ALFNet")
    complete = sum(1 for run in runs if (run.directory / COMPLETE_MARKER).exists())
    return {"plan_sha256": plan["plan_sha256"], "output_root": str(root),
            "slots": len(runs), "complete": complete,
            "findings": findings, "notes": notes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root", default=None)
    arguments = parser.parse_args(argv)
    result = audit(arguments.plan, arguments.output_root)
    print(f"plan {result['plan_sha256'][:16]}  root {result['output_root']}")
    print(f"COMPLETE {result['complete']}/{result['slots']}")
    if result["notes"]:
        print(f"\n{len(result['notes'])} note(s), none invalidating:")
        for note in result["notes"][:5]:
            print(f"  . {note}")
    if result["findings"]:
        print(f"\n{len(result['findings'])} finding(s):")
        for finding in result["findings"][:40]:
            print(f"  - {finding}")
        return 1
    print("\naudit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
