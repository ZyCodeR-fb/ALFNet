import argparse
import csv
import json
import statistics
from collections import OrderedDict
from pathlib import Path


MAIN_ORDER = ("Unseen", "Steady", "Join", "Fault", "LifeAvg")
PHASE_ORDER = ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage")
METRICS = ("MAPE", "MAE", "RMSE")
HASH_FIELDS = (
    "protocol_hash",
    "training_protocol_hash",
    "lce_protocol_hash",
    "ples_protocol_hash",
)
PROVENANCE_FIELDS = (
    "required_training_protocol_hash",
    "formal_lcbt_v2",
)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_run(result_dir):
    result_dir = Path(result_dir)
    if result_dir.is_file():
        result_dir = result_dir.parent
    main = _read_json(result_dir / "lce_main_summary.json")
    phases = _read_json(result_dir / "lce_phase_summary.json")
    if int(main.get("schema_version", 0)) != 2 or int(phases.get("schema_version", 0)) != 2:
        raise ValueError(f"LCE aggregation only accepts lifecycle protocol v2 outputs: {result_dir}")
    for field in HASH_FIELDS + PROVENANCE_FIELDS + (
        "model",
        "model_variant",
        "seed",
        "dataset",
    ):
        if main.get(field) != phases.get(field):
            raise ValueError(f"LCE main/phase metadata mismatch for {field}: {result_dir}")
    if main.get("formal_lcbt_v2") is not True:
        raise ValueError(
            f"LCE aggregation rejects diagnostic or legacy checkpoints: {result_dir}"
        )
    if main.get("training_protocol_hash") != main.get(
        "required_training_protocol_hash"
    ):
        raise ValueError(
            f"LCE checkpoint training protocol does not match the frozen manifest: {result_dir}"
        )
    if set(main.get("metrics", {})) != set(MAIN_ORDER):
        raise ValueError(f"LCE main table has unexpected groups: {result_dir}")
    if set(phases.get("metrics", {})) != set(PHASE_ORDER):
        raise ValueError(f"LCE phase table has unexpected phases: {result_dir}")
    return {
        "result_dir": str(result_dir.resolve()),
        "model": str(main["model"]),
        "model_variant": str(main["model_variant"]),
        "seed": int(main["seed"]),
        "dataset": main["dataset"],
        "hashes": {field: main.get(field) for field in HASH_FIELDS},
        "main": main["metrics"],
        "phases": phases["metrics"],
    }


def _metric_summary(runs, table, label, metric):
    values = [float(run[table][label][metric]) for run in runs]
    return float(statistics.mean(values)), float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _validate_common_protocol(runs):
    reference = runs[0]
    reference_dataset = json.dumps(reference["dataset"], sort_keys=True)
    for run in runs[1:]:
        if json.dumps(run["dataset"], sort_keys=True) != reference_dataset:
            raise ValueError("LCE aggregation requires one dataset")
        if run["hashes"] != reference["hashes"]:
            raise ValueError("LCE aggregation requires identical protocol hashes")


def aggregate_results(result_dirs, expected_seeds=(32, 42, 52)):
    runs = [_load_run(path) for path in result_dirs]
    if not runs:
        raise ValueError("No LCE v1 result directories were provided")
    _validate_common_protocol(runs)
    expected = sorted(int(seed) for seed in expected_seeds)

    grouped = OrderedDict()
    for run in runs:
        key = (run["model"], run["model_variant"])
        grouped.setdefault(key, []).append(run)

    model_entries = []
    for (model, variant), model_runs in grouped.items():
        seeds = [run["seed"] for run in model_runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{model}/{variant} contains a duplicate seed")
        if sorted(seeds) != expected:
            raise ValueError(
                f"{model}/{variant} expected seeds {expected}, got {sorted(seeds)}"
            )
        model_runs.sort(key=lambda run: run["seed"])
        model_entries.append(
            {
                "model": model,
                "model_variant": variant,
                "seeds": sorted(seeds),
                "result_dirs": [run["result_dir"] for run in model_runs],
                "runs": model_runs,
            }
        )

    def build_rows(table, order, label_key):
        rows = []
        for label in order:
            for entry in model_entries:
                row = {
                    label_key: label,
                    "model": entry["model"],
                    "model_variant": entry["model_variant"],
                    "seeds": "/".join(str(seed) for seed in entry["seeds"]),
                }
                for metric in METRICS:
                    mean, std = _metric_summary(entry["runs"], table, label, metric)
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_std"] = std
                rows.append(row)
        return rows

    return {
        "schema_version": 2,
        "table": "LCE v1 three-seed mean and sample standard deviation",
        "dataset": runs[0]["dataset"],
        **runs[0]["hashes"],
        "seeds": expected,
        "models": [
            {
                "model": entry["model"],
                "model_variant": entry["model_variant"],
                "result_dirs": entry["result_dirs"],
            }
            for entry in model_entries
        ],
        "main_rows": build_rows("main", MAIN_ORDER, "group"),
        "phase_rows": build_rows("phases", PHASE_ORDER, "phase"),
    }


def _write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path, rows, label_key, title):
    lines = [
        f"# {title}",
        "",
        f"| {label_key.title()} | Model | MAPE | MAE | RMSE | Seeds |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        values = [f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}" for metric in METRICS]
        lines.append(
            f"| {row[label_key]} | {row['model']} | {values[0]} | {values[1]} | "
            f"{values[2]} | {row['seeds']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "lce_three_seed_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    for stem, rows, label_key, title in (
        ("lce_main_mean_std", summary["main_rows"], "group", "LCE v1 Main Results"),
        ("lce_phase_mean_std", summary["phase_rows"], "phase", "LCE v1 Phase Results"),
    ):
        _write_csv(output_dir / f"{stem}.csv", rows)
        _write_markdown(output_dir / f"{stem}.md", rows, label_key, title)


def discover_result_dirs(paths):
    discovered = []
    for value in paths:
        path = Path(value)
        if path.is_file() and path.name == "lce_main_summary.json":
            discovered.append(path.parent)
        elif path.is_dir() and (path / "lce_main_summary.json").exists():
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(candidate.parent for candidate in path.rglob("lce_main_summary.json"))
        else:
            raise FileNotFoundError(path)
    unique = OrderedDict((str(path.resolve()), path) for path in discovered)
    if not unique:
        raise ValueError("No lce_main_summary.json files were found")
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser(description="Aggregate LCE v1 results across frozen training seeds.")
    parser.add_argument("results", nargs="+", help="Result directories or roots to scan recursively")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[32, 42, 52])
    args = parser.parse_args()

    result_dirs = discover_result_dirs(args.results)
    summary = aggregate_results(result_dirs, expected_seeds=args.expected_seeds)
    write_outputs(summary, args.output_dir)
    print(
        f"Collected {len(summary['models'])} models over seeds "
        f"{','.join(str(seed) for seed in summary['seeds'])}"
    )


if __name__ == "__main__":
    main()
