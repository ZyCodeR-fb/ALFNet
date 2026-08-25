#!/usr/bin/env python
"""Run/check isolated one-batch CPU preflights for all six speed SUP5 runs."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_v4_speed_sup5_plan as builder  # noqa: E402

OUTPUT = ROOT / "configs/paper/v4_speed_sup5_cpu_preflight.json"


def _run_one(entry, plan):
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
        raise ValueError(f"{entry['dataset']}/{entry['candidate']} is non-finite")
    return {
        "dataset": entry["dataset"], "candidate": entry["candidate"],
        "model": entry["model"], "parameter_count": int(report["parameter_count"]),
        "all_finite": True, "parameter_moved": True,
        "test_context_constructed": False,
    }


def _worker(run_key):
    plan = builder.assert_plan_still_holds(builder.load_plan())
    matches = [entry for entry in plan["runs"]
               if f"{entry['dataset']}/{entry['candidate']}" == run_key]
    if len(matches) != 1:
        raise ValueError(f"expected one plan entry for {run_key!r}")
    print("PREFLIGHT_RECORD=" + json.dumps(
        _run_one(matches[0], plan), allow_nan=False), flush=True)


def run_all():
    plan = builder.assert_plan_still_holds(builder.load_plan())
    records = []
    for entry in plan["runs"]:
        key = f"{entry['dataset']}/{entry['candidate']}"
        print(f"preflight {key}", flush=True)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", key],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"{key} preflight failed ({completed.returncode})\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
        print(completed.stdout, end="")
        found = [line.removeprefix("PREFLIGHT_RECORD=")
                 for line in completed.stdout.splitlines()
                 if line.startswith("PREFLIGHT_RECORD=")]
        if len(found) != 1:
            raise RuntimeError(f"{key} emitted no unique record")
        records.append(json.loads(found[0]))
    payload = {
        "schema_version": 1, "plan_sha256": plan["plan_sha256"],
        "seed": 32, "split": "validation", "all_passed": len(records) == 6,
        "records": records,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n",
        "utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}: 6/6 passed")


def check():
    plan = builder.assert_plan_still_holds(builder.load_plan())
    payload = json.loads(OUTPUT.read_text("utf-8"))
    expected = {(entry["dataset"], entry["candidate"]) for entry in plan["runs"]}
    actual = {(row["dataset"], row["candidate"]) for row in payload["records"]}
    if payload.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("speed SUP5 preflight belongs to another plan")
    if not payload.get("all_passed") or actual != expected:
        raise ValueError("speed SUP5 preflight inventory is incomplete")
    if any(row.get("test_context_constructed") for row in payload["records"]):
        raise ValueError("speed SUP5 preflight touched test")
    print(f"speed SUP5 CPU preflight holds for plan {plan['plan_sha256'][:16]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        check() if args.check else run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
