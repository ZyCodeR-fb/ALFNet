#!/usr/bin/env python
"""Create the four final tables for a validated report scope."""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.v4_metrics import PHASE_ORDER  # noqa: E402
from experiments.v4_validation_collector import (  # noqa: E402
    MAIN_TABLE_ROWS, METRICS, aggregate_seeds,
)
from scripts.eval_v4_plan_test import TEST_EVAL_NAME  # noqa: E402
from scripts.eval_v4_plan_validation import EVAL_NAME  # noqa: E402
from scripts.eval_v4_report_scope import (  # noqa: E402
    _test_problem, _validation_problem, load_scope, require_marker,
)
from scripts.v4_utils import sha256_file  # noqa: E402


DISPLAY = {"FCLSTM": "FC-LSTM", "MCPST_FSL": "MCPST-FSL"}
UNITS = {"PEMS03": "veh/5min", "PEMS04": "veh/5min",
         "PEMS08": "veh/5min", "PEMSBAY": "mph"}


class CollectRefused(RuntimeError):
    """The final evidence set is incomplete."""


def atomic_write(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text("utf-8") == text:
            return path
        raise CollectRefused(f"{path} exists with different content")
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


def aggregate(scope, selected):
    datasets = list(scope["datasets"])
    models = list(scope["models"])
    evidence = {split: {dataset: {model: {} for model in models}
                        for dataset in datasets}
                for split in ("validation", "test")}
    hashes = {}
    problems = []
    for run in selected:
        for split, filename, field, checker in (
            ("validation", EVAL_NAME, "recomputed", _validation_problem),
            ("test", TEST_EVAL_NAME, "cells", _test_problem),
        ):
            problem = checker(run)
            if problem:
                problems.append(f"{split}/{run.name}: {problem}")
                continue
            path = run.directory / filename
            record = json.loads(path.read_text("utf-8"))
            evidence[split][run.dataset][run.model][run.seed] = record[field]
            hashes[f"{split}/{run.name}"] = sha256_file(path)
    if problems:
        raise CollectRefused("evidence incomplete:\n  - "
                             + "\n  - ".join(problems[:40]))
    aggregated = {
        split: {dataset: {
            model: aggregate_seeds(evidence[split][dataset][model])
            for model in models} for dataset in datasets}
        for split in evidence}
    return aggregated, hashes


def number(value):
    return f"{float(value):.3f}"


def best_mark(value, best):
    rendered = number(value)
    return f"**{rendered}**" if math.isclose(
        float(value), float(best), rel_tol=0, abs_tol=1e-12) else rendered


def table_text(scope, aggregated, split, section, rows):
    datasets, models = scope["datasets"], scope["models"]
    title = "Main" if section == "main" else "Phase"
    lines = [f"# {title} table — {split}", "",
             "> Three-seed mean ± sample standard deviation (ddof=1). "
             "Lower is better; bold is best among the 11 models for that "
             "dataset/scenario/metric.", "",
             "> Each result cell is `MAE / RMSE / MAPE`. PEMS03/04/08 "
             "MAE and RMSE use veh/5min; PEMS-BAY uses mph; MAPE is %.", ""]
    for dataset in datasets:
        lines += [f"## {dataset}", "",
                  "| Model | " + " | ".join(rows) + " |",
                  "|---|" + "|".join("---" for _ in rows) + "|"]
        best = {row: {metric: min(
            aggregated[split][dataset][model][section][row][metric]["mean"]
            for model in models) for metric in METRICS} for row in rows}
        for model in models:
            cells = []
            for row in rows:
                record = aggregated[split][dataset][model][section][row]
                cells.append(" / ".join(
                    f"{best_mark(record[metric]['mean'], best[row][metric])}"
                    f"±{number(record[metric]['std'])}" for metric in METRICS))
            lines.append(f"| {DISPLAY.get(model, model)} | "
                         + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def csv_text(scope, aggregated, split, section, rows):
    from io import StringIO
    handle = StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(("split", "dataset", "unit", "scenario", "model",
                     "n_seeds", "mae_mean", "mae_std", "rmse_mean",
                     "rmse_std", "mape_mean", "mape_std"))
    for dataset in scope["datasets"]:
        for row in rows:
            for model in scope["models"]:
                record = aggregated[split][dataset][model][section][row]
                writer.writerow((split, dataset, UNITS[dataset], row,
                                 DISPLAY.get(model, model), 3,
                                 record["MAE"]["mean"], record["MAE"]["std"],
                                 record["RMSE"]["mean"], record["RMSE"]["std"],
                                 record["MAPE"]["mean"], record["MAPE"]["std"]))
    return handle.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args(argv)
    scope, scope_path, plan, selected, root = load_scope(ROOT / args.scope)
    require_marker(scope_path, plan, selected, root)
    aggregated, hashes = aggregate(scope, selected)
    scope_hash = sha256_file(scope_path)
    output = ROOT / "outputs_v4/reports/four_dataset_11_models_final" / scope_hash
    outputs = {}
    for split in ("validation", "test"):
        for section, rows in (("main", list(MAIN_TABLE_ROWS)),
                              ("phase", list(PHASE_ORDER))):
            stem = f"{split}_{section}_table"
            md = output / f"{stem}.md"
            csv_path = output / f"{stem}.csv"
            atomic_write(md, table_text(scope, aggregated, split, section, rows))
            atomic_write(csv_path, csv_text(scope, aggregated, split, section, rows))
            outputs[md.name] = sha256_file(md)
            outputs[csv_path.name] = sha256_file(csv_path)
    summary = {
        "schema_version": 1,
        "status": "final_validation_and_test_report",
        "scope_sha256": scope_hash,
        "source_plan_sha256": plan["plan_sha256"],
        "datasets": scope["datasets"], "models": scope["models"],
        "seeds": scope["seeds"], "n": len(scope["seeds"]), "std_ddof": 1,
        "excluded": scope["excluded"],
        "input_evidence_sha256": hashes,
        "aggregated": aggregated,
        "table_sha256": outputs,
    }
    summary_path = output / "summary.json"
    atomic_write(summary_path, json.dumps(
        summary, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {output}")
    for name, digest in sorted(outputs.items()):
        print(f"  {name}: {digest}")
    print(f"  summary.json: {sha256_file(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
