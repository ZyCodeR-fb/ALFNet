"""The paper's tables, from exactly the 18 frozen runs and nothing else.

Why this exists rather than `collect_lifecycle_results.py`: that script walks a
result root, so pointing it at `outputs/ALFNet` sweeps up every historical run
sitting there -- `fg1-s32`, `ad-s32`, `kt-s32`, the r1 diagnostics -- all of
them produced under superseded protocols. Mixing one of those into a mean is
silent and unrecoverable. This reads the exact directories named in
configs/paper/final_runs.json, and refuses anything else.

It refuses, rather than works around:

  * a missing run, or a seed that is not exactly {32, 42, 52} per cell;
  * a run whose result set is incomplete or duplicated;
  * two runs of one dataset disagreeing on a protocol hash, or two datasets
    sharing one;
  * a result whose model, variant, seed or dataset does not match the manifest.

Output, per dataset: the six-row main table, the six-phase table and the PLES
effects, each as ALFNet and MoGERNN three-seed mean +/- std.

    conda run -n COMG python scripts/collect_final_18.py \
        --output-dir ./outputs/reports/final
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final_runs import MANIFEST, load_manifest, verify_eval  # noqa: E402

MAIN_ROWS = ("Unseen", "Steady", "Join", "Fault", "LifeAvg", "WorstGroup")
PHASE_ROWS = ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage")
METRICS = ("MAPE", "MAE", "RMSE")
PLES_EFFECTS = ("JoinLift", "FaultPenalty", "OutageFaultPenalty")


def _read(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _mean_std(values):
    """mean +/- population-free std. None if any seed is missing."""
    clean = [v for v in values if isinstance(v, (int, float))]
    if len(clean) != len(values) or not clean:
        return None
    if len(clean) == 1:
        return {"mean": float(clean[0]), "std": 0.0, "n": 1}
    return {
        "mean": float(statistics.fmean(clean)),
        "std": float(statistics.stdev(clean)),
        "n": len(clean),
    }


def collect(payload, output_root):
    """{(dataset, model): {seed: {...tables...}}}, or raise on any inconsistency."""
    problems = []
    loaded = defaultdict(dict)
    protocol_by_dataset = defaultdict(set)

    for run in payload["runs"]:
        run_problems, directory = verify_eval(run, output_root)
        if run_problems:
            problems.append(f"{run['run_id']}: " + "; ".join(run_problems))
            continue

        main = _read(directory, "lce_main_summary.json")
        phase = _read(directory, "lce_phase_summary.json")
        ples = _read(directory, "ples_effects_summary.json")

        # The result files carry their own identity. If it disagrees with the
        # manifest, the directory holds someone else's run.
        for label, found, expected in (
            ("model", main.get("model"), run["model"]),
            ("model_variant", main.get("model_variant"), run["model_variant"]),
            ("seed", main.get("seed"), run["seed"]),
            ("dataset", (main.get("dataset") or {}).get("name"), run["dataset"]),
        ):
            if found != expected:
                problems.append(
                    f"{run['run_id']}: result says {label}={found!r}, "
                    f"manifest says {expected!r}"
                )

        if not main.get("formal_protocol_match", False):
            problems.append(
                f"{run['run_id']}: formal_protocol_match is false -- the result "
                "was produced against a different protocol than it requires"
            )
        for name, table in (("main", main), ("phase", phase), ("ples", ples)):
            digest = table.get("training_protocol_hash")
            if digest:
                protocol_by_dataset[run["dataset"]].add(digest)
            if table.get("lce_protocol_hash") != main.get("lce_protocol_hash"):
                problems.append(
                    f"{run['run_id']}: {name} table has a different lce_protocol_hash"
                )

        loaded[(run["dataset"], run["model"])][run["seed"]] = {
            "run_id": run["run_id"],
            "result_dir": str(directory),
            "main": main.get("metrics") or {},
            "phase": phase.get("metrics") or {},
            "ples": ples.get("paired_effects") or {},
            "ples_intervals": ples.get("paired_effect_intervals") or {},
        }

    # --- matrix-level consistency -------------------------------------------
    expected_seeds = set(payload["seeds"])
    for dataset in payload["datasets"]:
        for model in payload["models"]:
            seeds = set(loaded.get((dataset, model), {}))
            if seeds != expected_seeds:
                missing = sorted(expected_seeds - seeds)
                extra = sorted(seeds - expected_seeds)
                detail = []
                if missing:
                    detail.append(f"missing {missing}")
                if extra:
                    detail.append(f"unexpected {extra}")
                problems.append(f"{dataset}/{model}: " + ", ".join(detail))

    for dataset, digests in protocol_by_dataset.items():
        if len(digests) > 1:
            problems.append(
                f"{dataset}: results disagree on training_protocol_hash "
                f"({sorted(d[:12] for d in digests)})"
            )
    seen = defaultdict(list)
    for dataset, digests in protocol_by_dataset.items():
        for digest in digests:
            seen[digest].append(dataset)
    for digest, datasets in seen.items():
        if len(datasets) > 1:
            problems.append(
                f"datasets {sorted(datasets)} share training_protocol_hash "
                f"{digest[:12]}... -- one reused the other's manifest"
            )

    if problems:
        raise SystemExit(
            "REFUSING to aggregate:\n" + "\n".join(f"  - {p}" for p in problems)
        )
    return loaded


def _table(loaded, dataset, models, rows, key):
    out = {}
    for row in rows:
        out[row] = {}
        for metric in METRICS:
            out[row][metric] = {}
            for model in models:
                cell = loaded.get((dataset, model), {})
                values = [
                    (cell[seed][key].get(row) or {}).get(metric)
                    for seed in sorted(cell)
                ]
                out[row][metric][model] = _mean_std(values)
    return out


def _ples_table(loaded, dataset, models):
    out = {}
    for effect in PLES_EFFECTS:
        out[effect] = {}
        for metric in METRICS:
            out[effect][metric] = {}
            for model in models:
                cell = loaded.get((dataset, model), {})
                values = [
                    (cell[seed]["ples"].get(metric) or {}).get(effect)
                    for seed in sorted(cell)
                ]
                out[effect][metric][model] = _mean_std(values)
    return out


def _format(entry):
    if entry is None:
        return "-"
    return f"{entry['mean']:.4f}+/-{entry['std']:.4f}"


def _print_table(title, table, rows, models):
    print(f"\n{title}")
    header = f"{'row':<12}{'metric':<8}" + "".join(f"{m:>22}" for m in models)
    print(header)
    print("-" * len(header))
    for row in rows:
        for metric in METRICS:
            cells = "".join(f"{_format(table[row][metric][m]):>22}" for m in models)
            print(f"{row:<12}{metric:<8}{cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--output-dir", default="./outputs/reports/final")
    options = parser.parse_args()

    payload = load_manifest(options.manifest)
    loaded = collect(payload, options.output_root)
    models = list(payload["models"])

    report = {"manifest": str(options.manifest), "datasets": {}}
    for dataset in payload["datasets"]:
        main_table = _table(loaded, dataset, models, MAIN_ROWS, "main")
        phase_table = _table(loaded, dataset, models, PHASE_ROWS, "phase")
        ples_table = _ples_table(loaded, dataset, models)
        report["datasets"][dataset] = {
            "main": main_table, "phase": phase_table, "ples": ples_table,
            "runs": {
                model: {
                    str(seed): loaded[(dataset, model)][seed]["run_id"]
                    for seed in sorted(loaded[(dataset, model)])
                }
                for model in models
            },
        }
        print(f"\n{'=' * 70}\n{dataset}   (three-seed mean +/- std)\n{'=' * 70}")
        _print_table("main", main_table, MAIN_ROWS, models)
        _print_table("phases", phase_table, PHASE_ROWS, models)
        _print_table("PLES paired effects", ples_table, PLES_EFFECTS, models)

    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "final_18_tables.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
