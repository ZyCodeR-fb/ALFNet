#!/usr/bin/env python
"""Run the ten final one-batch CPU smoke tests and write their attestation."""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_v4_five_dataset_plan as builder  # noqa: E402
from scripts.train_v4 import train  # noqa: E402

OUTPUT = ROOT / "configs/paper/v4_cpu_preflight.json"


def expected_combinations(plan):
    return [f"{dataset}/{model}"
            for dataset in plan["datasets"] for model in plan["models"]]


def validate(report, plan):
    expected = expected_combinations(plan)
    if report.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("CPU preflight belongs to another plan")
    if report.get("combinations") != expected:
        raise ValueError("CPU preflight combination inventory is incomplete or reordered")
    results = report.get("results", {})
    if set(results) != set(expected):
        raise ValueError("CPU preflight result keys do not match the ten final combinations")
    failures = [name for name in expected if results[name].get("status") != "PASS"]
    if failures or report.get("passed") != len(expected) or report.get("failed") != 0 \
            or report.get("all_passed") is not True:
        raise ValueError(f"CPU preflight has failures: {failures}")
    for name in expected:
        item = results[name]
        for field in ("parameter_count", "train_loss", "validation_life_avg_mae"):
            if not math.isfinite(float(item[field])):
                raise ValueError(f"{name}.{field} is not finite")
        if item["train_batches"] != 1 or item["validation_batches"] != 1:
            raise ValueError(f"{name} was not a one-train/one-validation preflight")
        if item["validation_teacher_forcing_ratio"] != 0.0:
            raise ValueError(f"{name} used teacher forcing in validation")
    return report


def run(plan):
    results = {}
    for dataset in plan["datasets"]:
        for model in plan["models"]:
            name = f"{dataset}/{model}"
            config = ROOT / plan["configs"][f"{dataset}_{model}"]["path"]
            print(f"preflight {name}", flush=True)
            try:
                report = train(
                    config, output_root=None, device="cpu", seed=32,
                    max_epochs=1, max_train_batches=1, max_validation_batches=1,
                    batch_size=1, num_workers=0, save=False)
                epoch = report["history"][0]
                results[name] = {
                    "status": "PASS",
                    "parameter_count": report["parameter_count"],
                    "train_batches": epoch["train"]["batches"],
                    "validation_batches": epoch["validation"]["batches"],
                    "train_loss": epoch["train"]["optimize"],
                    "validation_life_avg_mae": epoch["validation"]["life_avg_mae"],
                    "validation_teacher_forcing_ratio":
                        epoch["validation"]["teacher_forcing_ratio"],
                }
            except Exception as error:  # recorded, then validate fails closed
                results[name] = {"status": "FAIL", "error": repr(error)}
    expected = expected_combinations(plan)
    passed = sum(results[name]["status"] == "PASS" for name in expected)
    report = {
        "schema_version": 2,
        "kind": "v4_cpu_preflight",
        "plan_sha256": plan["plan_sha256"],
        "seed": 32,
        "scope": "one train batch and one validation batch on CPU",
        "combinations": expected,
        "results": results,
        "passed": passed,
        "failed": len(expected) - passed,
        "all_passed": passed == len(expected),
    }
    validate(report, plan)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        "utf-8")
    temporary.replace(OUTPUT)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    plan = builder.assert_plan_still_holds(builder.load_plan())
    if args.check:
        validate(json.loads(OUTPUT.read_text("utf-8")), plan)
        print(f"CPU preflight holds for {len(expected_combinations(plan))} combinations")
    else:
        report = run(plan)
        print(f"wrote {OUTPUT.relative_to(ROOT)}: {report['passed']}/"
              f"{len(report['combinations'])} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
