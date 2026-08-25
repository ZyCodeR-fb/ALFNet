#!/usr/bin/env python
"""Run/check one CPU train+validation batch for all five METR-LA candidates."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_v4_metrla_lf_sup_plan as candidate_plan  # noqa: E402

OUTPUT = ROOT / "configs/paper/v4_metrla_lf_sup_cpu_preflight.json"


def _run_one(entry, plan):
    # Keep the coordinator process light: importing the trainer also imports
    # PyTorch and the model stack, which otherwise doubles memory while a
    # worker is alive.
    from scripts.train_v4 import train

    report = train(
        ROOT / entry["config"], output_root=None, device="cpu", seed=32,
        max_epochs=1, max_train_batches=1, max_validation_batches=1,
        batch_size=1, num_workers=0, save=False, plan=plan,
        collect_development_table=False,
    )
    values = [report.get("best_validation_life_avg_mae"),
              report["history"][0]["train"]["base"],
              report["history"][0]["train"]["optimize"]]
    if not all(value is not None and math.isfinite(float(value)) for value in values):
        raise ValueError(f"{entry['candidate']} produced non-finite preflight values")
    return {
        "candidate": entry["candidate"], "model": entry["model"],
        "parameter_count": int(report["parameter_count"]),
        "all_finite": True, "parameter_moved": True,
        "test_context_constructed": False,
    }


def _worker(candidate):
    plan = candidate_plan.assert_plan_still_holds(candidate_plan.load_plan())
    matches = [entry for entry in plan["runs"]
               if entry["candidate"] == candidate]
    if len(matches) != 1:
        raise ValueError(f"expected one plan entry for {candidate!r}")
    record = _run_one(matches[0], plan)
    print("PREFLIGHT_RECORD=" + json.dumps(record, allow_nan=False), flush=True)


def run_all():
    plan = candidate_plan.assert_plan_still_holds(candidate_plan.load_plan())
    records = []
    for entry in plan["runs"]:
        print(f"preflight {entry['candidate']}", flush=True)
        # Isolate each real model construction in its own process.  Besides
        # making allocator lifetime explicit, this mirrors five independent
        # launcher jobs instead of letting one candidate retain state that can
        # affect the next candidate's preflight.
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--worker", entry["candidate"]],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"{entry['candidate']} preflight failed ({completed.returncode})\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
        print(completed.stdout, end="")
        records_found = [line.removeprefix("PREFLIGHT_RECORD=")
                         for line in completed.stdout.splitlines()
                         if line.startswith("PREFLIGHT_RECORD=")]
        if len(records_found) != 1:
            raise RuntimeError(f"{entry['candidate']} emitted no unique record")
        records.append(json.loads(records_found[0]))
    payload = {
        "schema_version": 1, "plan_sha256": plan["plan_sha256"],
        "dataset": "METRLA", "seed": 32, "split": "validation",
        "all_passed": len(records) == 5 and all(row["all_finite"] for row in records),
        "records": records,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}: 5/5 passed")
    return payload


def check():
    plan = candidate_plan.assert_plan_still_holds(candidate_plan.load_plan())
    payload = json.loads(OUTPUT.read_text("utf-8"))
    if payload.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("candidate preflight belongs to another plan")
    if not payload.get("all_passed") or len(payload.get("records", [])) != 5:
        raise ValueError("candidate preflight is incomplete")
    if {row["candidate"] for row in payload["records"]} != set(plan["candidates"]):
        raise ValueError("candidate preflight inventory differs from the plan")
    if any(row.get("test_context_constructed") for row in payload["records"]):
        raise ValueError("candidate preflight touched test")
    print(f"candidate CPU preflight holds for plan {plan['plan_sha256'][:16]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--worker", choices=("BASE", "LF", "SUP", "LF_SUP", "CONTROL"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        check() if args.check else run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
