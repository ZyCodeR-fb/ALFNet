"""Is the frozen 18-run training matrix complete, consistent and reusable?

This runs between training and evaluation, and it is the gate: the test split is
spent exactly once, so it must not be spent on a matrix that is missing a seed,
carries a stale directory, or mixes protocols.

It reads **no test data and computes no test metric**. It also makes no model
choice -- it checks completeness and provenance, nothing else. A model decision
on the basis of validation is out of scope here and out of scope generally: the
architecture is frozen.

What it checks, per run, against configs/paper/final_runs.json:

  * the run finished (`stop_reason` is a completed one), and the checkpoint and
    summary both exist and parse;
  * `model` and `model_variant` match the manifest;
  * `canonical_config_hash` matches -- which covers seed, dataset, config file
    and every hyper-parameter, since all of them enter that hash;
  * structural provenance passes the same validator the loader uses;
  * validation metrics are present and finite.

And across runs:

  * all 18 are present, exactly once, with seeds {32, 42, 52} complete per cell;
  * every dataset's runs agree on one `training_protocol_hash`, and different
    datasets do not share one;
  * no two runs share a `canonical_config_hash`.

    conda run -n COMG python scripts/audit_final_18.py
    conda run -n COMG python scripts/audit_final_18.py --only alfnet-metrla-s32
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final_runs import (  # noqa: E402
    MANIFEST,
    load_manifest,
    run_dir,
    verify_run,
)


def _protocol_hash(run, output_root):
    summary = run_dir(run, output_root) / "checkpoints" / "training_summary.json"
    if not summary.exists():
        return None
    try:
        return json.loads(summary.read_text(encoding="utf-8")).get(
            "training_protocol_hash"
        )
    except (ValueError, OSError):
        return None


def _validation_row(run, output_root):
    summary = run_dir(run, output_root) / "checkpoints" / "training_summary.json"
    if not summary.exists():
        return None
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    metrics = payload.get("best_val_lifecycle_metrics") or {}
    return {
        "epoch": payload.get("best_val_metric_epoch"),
        "finished_epoch": payload.get("finished_epoch"),
        "stop_reason": payload.get("stop_reason"),
        "lifeavg": payload.get("best_val_metric"),
        "unseen": metrics.get("unseen_mae"),
        "steady": metrics.get("steady_mae"),
        "join": metrics.get("join_mae"),
        "fault": metrics.get("fault_mae"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--only", help="audit a single run_id")
    parser.add_argument("--quiet", action="store_true",
                        help="exit code only, for use by the launcher")
    parser.add_argument("--skip-structure", action="store_true",
                        help="skip the structural provenance check (faster)")
    parser.add_argument("--json", dest="json_out",
                        help="write the full report to this path")
    options = parser.parse_args()

    payload = load_manifest(options.manifest)
    runs = payload["runs"]
    if options.only:
        runs = [run for run in runs if run["run_id"] == options.only]
        if not runs:
            print(f"no run named {options.only!r} in the manifest", file=sys.stderr)
            return 2

    report = {"manifest": str(options.manifest), "output_root": options.output_root,
              "runs": {}, "matrix_problems": []}
    failures = {}

    if not options.quiet:
        print(f"auditing {len(runs)} run(s) from {options.manifest}")
        print(f"{'run_id':<22}{'result':<10}{'epoch':>6}{'LifeAvg':>10}"
              f"{'Unseen':>9}{'Steady':>9}{'Join':>9}{'Fault':>9}")
        print("-" * 84)

    for run in runs:
        try:
            problems = verify_run(run, options.output_root,
                                  check_structure=not options.skip_structure)
        except Exception as exc:                              # noqa: BLE001
            # Exit 2, not 1. "The checker could not run" is not "the run is
            # wrong": a missing dependency or an unreadable data file would
            # otherwise surface as every run being refused, which reads like a
            # corrupt matrix and sends you looking in the wrong place.
            print(
                f"CANNOT VERIFY {run['run_id']}: {type(exc).__name__}: {exc}\n"
                "This is an environment problem, not a verdict on the run. "
                "Check that PYTHON points at the project environment (COMG).",
                file=sys.stderr,
            )
            return 2
        row = _validation_row(run, options.output_root)
        report["runs"][run["run_id"]] = {"problems": problems, "validation": row}
        if problems:
            failures[run["run_id"]] = problems
        if options.quiet:
            continue

        def fmt(value, width=9):
            return f"{value:>{width}.4f}" if isinstance(value, (int, float)) else f"{'-':>{width}}"

        verdict = "OK" if not problems else "FAIL"
        if row:
            print(f"{run['run_id']:<22}{verdict:<10}{str(row['epoch']):>6}"
                  f"{fmt(row['lifeavg'], 10)}{fmt(row['unseen'])}{fmt(row['steady'])}"
                  f"{fmt(row['join'])}{fmt(row['fault'])}")
        else:
            print(f"{run['run_id']:<22}{verdict:<10}{'-':>6}{'-':>10}"
                  f"{'-':>9}{'-':>9}{'-':>9}{'-':>9}")

    # --- matrix-level checks, only meaningful over the whole set -------------
    matrix_problems = []
    if not options.only:
        by_dataset = defaultdict(dict)
        by_cell = defaultdict(set)
        canonical_seen = defaultdict(list)
        for run in runs:
            digest = _protocol_hash(run, options.output_root)
            if digest:
                by_dataset[run["dataset"]][run["run_id"]] = digest
            by_cell[(run["model"], run["dataset"])].add(run["seed"])
            canonical_seen[run["canonical_config_hash"]].append(run["run_id"])

        expected_seeds = set(payload["seeds"])
        for (model, dataset), seeds in sorted(by_cell.items()):
            if seeds != expected_seeds:
                matrix_problems.append(
                    f"{model}/{dataset}: seeds {sorted(seeds)} != {sorted(expected_seeds)}"
                )

        for dataset, mapping in sorted(by_dataset.items()):
            distinct = set(mapping.values())
            if len(distinct) > 1:
                matrix_problems.append(
                    f"{dataset}: runs disagree on training_protocol_hash -- "
                    f"{ {k: v[:12] for k, v in mapping.items()} }"
                )
            known = (payload.get("known_training_protocol_hashes") or {}).get(dataset)
            if known and distinct and known not in distinct:
                matrix_problems.append(
                    f"{dataset}: training_protocol_hash {sorted(distinct)[0][:12]}... "
                    f"does not match the frozen {known[:12]}..."
                )

        # Two datasets sharing a protocol hash means one of them reused the
        # other's manifest, which silently evaluates the wrong graph.
        collapsed = defaultdict(list)
        for dataset, mapping in by_dataset.items():
            for digest in set(mapping.values()):
                collapsed[digest].append(dataset)
        for digest, datasets in collapsed.items():
            if len(datasets) > 1:
                matrix_problems.append(
                    f"datasets {sorted(datasets)} share training_protocol_hash "
                    f"{digest[:12]}... -- one of them reused the other's manifest"
                )

        for digest, ids in canonical_seen.items():
            if len(ids) > 1:
                matrix_problems.append(
                    f"runs {sorted(ids)} share canonical_config_hash {digest[:12]}..."
                )

    report["matrix_problems"] = matrix_problems

    if options.json_out:
        Path(options.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(options.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if options.quiet:
        return 1 if (failures or matrix_problems) else 0

    print()
    if failures:
        print(f"{len(failures)} run(s) failed verification:")
        for run_id, problems in failures.items():
            print(f"  {run_id}")
            for problem in problems:
                print(f"    - {problem}")
    if matrix_problems:
        print("matrix problems:")
        for problem in matrix_problems:
            print(f"  - {problem}")

    if failures or matrix_problems:
        print()
        print("AUDIT FAILED. Do not run the evaluation: the test split is spent")
        print("once and must not be spent on an incomplete or mixed matrix.")
        return 1

    print(f"AUDIT PASSED: {len(runs)} runs complete, provenance consistent, "
          "metrics finite.")
    print("No test data was read. Evaluation may now proceed:")
    print("  bash scripts/run_final_eval_18.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
