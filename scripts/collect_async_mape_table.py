import argparse
import csv
import json
from pathlib import Path


MAIN_METRICS = ("Join", "Fault", "LifeMacro")
PHASE_METRICS = ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage")
ERROR_METRICS = ("MAPE", "MAE", "RMSE")


def _parse_label_rules(values):
    rules = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --label rule: {value!r}; expected NAME=SUBSTRING")
        name, needle = (part.strip() for part in value.split("=", 1))
        if not name or not needle:
            raise ValueError(f"Invalid --label rule: {value!r}; expected NAME=SUBSTRING")
        rules.append((name, needle))
    return rules


def _validate_metric_group(payload, key, names):
    group = payload.get(key)
    if not isinstance(group, dict):
        raise ValueError(f"PLES summary is missing {key!r}")
    result = {}
    for name in names:
        metrics = group.get(name)
        if not isinstance(metrics, dict):
            raise ValueError(f"PLES summary is missing {key}.{name}")
        result[name] = {metric: float(metrics[metric]) for metric in ERROR_METRICS}
    return result


def _method_name(path, payload, label_rules):
    full_path = str(path)
    for name, needle in label_rules:
        if needle in full_path:
            return name
    model = str(payload.get("model") or "").strip()
    variant = str(payload.get("model_variant") or "default").strip()
    if model == "ALFNet" and variant == "rf_alf_sd":
        return "ALFNet (ours)"
    if model:
        return model if variant == "default" else f"{model} [{variant}]"
    return path.parent.parent.name


def _read_summary(path, label_rules):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("protocol") != "Paired Lifecycle Event Study":
        raise ValueError(f"Not a PLES summary: {path}")
    return {
        "Models": _method_name(path, payload, label_rules),
        "Dataset": payload["dataset"]["name"],
        "DatasetFingerprint": payload["dataset"]["fingerprint"],
        "ProtocolHash": payload["protocol_hash"],
        "Main": _validate_metric_group(payload, "main_metrics", MAIN_METRICS),
        "Phases": _validate_metric_group(payload, "phase_metrics", PHASE_METRICS),
        "Source": str(path),
    }


def _collect_rows(args):
    root = Path(args.predictions)
    rules = _parse_label_rules(args.label)
    paths = sorted(root.rglob("paired_event_summary.json"))
    if args.include:
        paths = [path for path in paths if any(token in str(path) for token in args.include)]
    rows = [_read_summary(path, rules) for path in paths]
    if not rows:
        raise ValueError(f"No paired_event_summary.json found under {root}")

    dataset_keys = {(row["Dataset"], row["DatasetFingerprint"], row["ProtocolHash"]) for row in rows}
    if len(dataset_keys) != 1:
        details = sorted((dataset, protocol[:12]) for dataset, _, protocol in dataset_keys)
        raise ValueError(
            "Selected PLES results do not share one dataset manifest; collect each protocol separately. "
            f"found={details}"
        )

    grouped = {}
    for row in rows:
        grouped.setdefault(row["Models"], []).append(row)
    duplicates = {name: values for name, values in grouped.items() if len(values) > 1}
    if duplicates and not args.best_per_method:
        names = ", ".join(f"{name} ({len(values)})" for name, values in sorted(duplicates.items()))
        raise ValueError(
            "Multiple PLES runs map to the same model label. Use --include/--label to select one run, "
            f"or explicitly pass --best-per-method. duplicates={names}"
        )
    if args.best_per_method:
        rows = [
            min(values, key=lambda row: row["Main"]["LifeMacro"]["MAPE"])
            for values in grouped.values()
        ]
    rows.sort(key=lambda row: (row["Main"]["LifeMacro"]["MAPE"], row["Models"]))
    return rows


def _format(value):
    return f"{float(value):.4f}"


def _table_lines(rows, groups):
    headers = ["Models"] + [f"{group} {metric}" for group in groups for metric in ERROR_METRICS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    source_key = "Main" if groups == MAIN_METRICS else "Phases"
    for row in rows:
        values = [row["Models"]]
        values.extend(_format(row[source_key][group][metric]) for group in groups for metric in ERROR_METRICS)
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_markdown(rows, path):
    dataset = rows[0]["Dataset"]
    protocol_hash = rows[0]["ProtocolHash"]
    lines = [
        f"# PLES Metrics: {dataset}",
        "",
        f"Protocol manifest: `{protocol_hash}`",
        "",
        "## Main Table",
        "",
        *_table_lines(rows, MAIN_METRICS),
        "",
        "## Phase Table",
        "",
        *_table_lines(rows, PHASE_METRICS),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_csv(rows, path):
    if not path:
        return
    headers = ["Models", "Dataset", "ProtocolHash"]
    headers.extend(f"{group} {metric}" for group in MAIN_METRICS + PHASE_METRICS for metric in ERROR_METRICS)
    headers.append("Source")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in ("Models", "Dataset", "ProtocolHash", "Source")}
            for group in MAIN_METRICS:
                for metric in ERROR_METRICS:
                    flat[f"{group} {metric}"] = row["Main"][group][metric]
            for group in PHASE_METRICS:
                for metric in ERROR_METRICS:
                    flat[f"{group} {metric}"] = row["Phases"][group][metric]
            writer.writerow(flat)


def main():
    parser = argparse.ArgumentParser(description="Collect tables from one frozen PLES manifest.")
    parser.add_argument("--predictions", default="predictions")
    parser.add_argument("--out", default="reports_ples_metrics_summary.md")
    parser.add_argument("--csv", default="")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--label", action="append", default=[], help="Map output name: NAME=SUBSTRING")
    parser.add_argument(
        "--best-per-method",
        action="store_true",
        help="Explicitly select the lowest test LifeMacro when duplicate labels exist (not recommended for final reporting)",
    )
    args = parser.parse_args()
    rows = _collect_rows(args)
    _write_markdown(rows, args.out)
    _write_csv(rows, args.csv)
    print(f"wrote {len(rows)} PLES rows to {args.out}")
    if args.csv:
        print(f"wrote csv to {args.csv}")


if __name__ == "__main__":
    main()
