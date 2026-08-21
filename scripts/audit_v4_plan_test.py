#!/usr/bin/env python
"""Audit a plan's test results, after they exist. Not a gate for opening them.

The gate that decides whether the confirmation splits may be opened lives in
`eval_v4_plan_test.assert_may_open`, before any target is read. This runs after,
and checks that what was written is what the plan asked for: every slot present,
scored at its own seed, with the checkpoint's scaler, no teacher forcing, the
right exposure hash, and 36 cells.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v4_validation_collector import cell_count  # noqa: E402
from scripts.eval_v4_plan_test import (  # noqa: E402
    COMPLETE_TEST_MARKER, OPENED_MARKER, TEST_EVAL_NAME, TEST_SCHEMA,
    access_identity,
)
from scripts.eval_v4_plan_validation import load_plan, resolve_runs  # noqa: E402


def audit(plan_path, output_root=None):
    plan = load_plan(plan_path)
    runs, root = resolve_runs(plan, output_root)
    findings = []

    opened = Path(root) / OPENED_MARKER
    if not opened.exists():
        findings.append(f"{OPENED_MARKER} is absent; no test access was recorded")
    else:
        record = json.loads(opened.read_text("utf-8"))
        if record.get("identity") != access_identity(plan):
            findings.append("the recorded test-access identity no longer matches "
                            "the plan, attestation, configs, data or code")

    for run in runs:
        path = run.directory / TEST_EVAL_NAME
        if not path.exists():
            findings.append(f"{run.name}: no {TEST_EVAL_NAME}")
            continue
        record = json.loads(path.read_text("utf-8"))
        where = run.name
        if record.get("schema_version") != TEST_SCHEMA:
            findings.append(f"{where}: schema {record.get('schema_version')}")
        if record.get("split") != "test":
            findings.append(f"{where}: split is {record.get('split')!r}")
        if int(record.get("run_seed_used", -1)) != int(run.seed):
            findings.append(f"{where}: scored at seed {record.get('run_seed_used')}")
        if record.get("scaler", {}).get("refit") is not False:
            findings.append(f"{where}: the scaler was refit on test")
        if record.get("teacher_forcing_positions") != 0:
            findings.append(f"{where}: teacher forcing was not zero")
        if record.get("plan_sha256") != plan["plan_sha256"]:
            findings.append(f"{where}: produced under another plan")
        if cell_count(record.get("cells", {"main": {}, "phase": {}})) != 36:
            findings.append(f"{where}: not 36 cells")
        expected = plan["data"][run.dataset]["test_exposure"]["status"]
        if record.get("exposure_status") != expected:
            findings.append(f"{where}: exposure status {record.get('exposure_status')!r} "
                            f"!= {expected!r}")

    if not (Path(root) / COMPLETE_TEST_MARKER).exists():
        findings.append(f"{COMPLETE_TEST_MARKER} is absent; the test pass did not finish")
    return {"plan_sha256": plan["plan_sha256"], "slots": len(runs),
            "findings": findings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root", default=None)
    arguments = parser.parse_args(argv)
    result = audit(arguments.plan, arguments.output_root)
    print(f"plan {result['plan_sha256'][:16]}  slots {result['slots']}")
    if result["findings"]:
        print(f"\n{len(result['findings'])} finding(s):")
        for finding in result["findings"][:30]:
            print(f"  - {finding}")
        return 1
    print("test audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
