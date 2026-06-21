import argparse
import csv
import json
import re
from pathlib import Path


ROLES = ("VS", "AAS", "NAS", "FS")


def _parse_label_rules(values):
    rules = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --label rule: {value!r}; expected NAME=SUBSTRING")
        name, needle = value.split("=", 1)
        name = name.strip()
        needle = needle.strip()
        if not name or not needle:
            raise ValueError(f"Invalid --label rule: {value!r}; expected NAME=SUBSTRING")
        rules.append((name, needle))
    return rules


def _parse_manual_rows(values):
    rows = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 5:
            raise ValueError(
                f"Invalid --manual-row: {value!r}; expected METHOD,VS,AAS,NAS,FS"
            )
        method = parts[0]
        metrics = {role: float(raw) for role, raw in zip(ROLES, parts[1:])}
        rows.append(_build_row(method, metrics, source="manual"))
    return rows


def _metric_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_json_metrics(path):
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    candidates = []
    for key in ("summary_metrics", "role_metrics"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)

    for candidate in candidates:
        metrics = {}
        for role in ROLES:
            role_data = candidate.get(role)
            if not isinstance(role_data, dict):
                break
            mape = _metric_float(role_data.get("MAPE"))
            if mape is None:
                break
            metrics[role] = mape
        if len(metrics) == len(ROLES):
            return metrics
    return None


def _extract_text_metrics(path):
    pattern = re.compile(r"^(VS|AAS|NAS|FS):\s+MAPE=([0-9.]+)", re.MULTILINE)
    text = path.read_text(encoding="utf-8", errors="ignore")
    metrics = {role: float(value) for role, value in pattern.findall(text)}
    return metrics if len(metrics) == len(ROLES) else None


def _infer_method(exp_dir, async_dir, label_rules):
    full = f"{exp_dir.name}/{async_dir.name}"
    for name, needle in label_rules:
        if needle in full:
            return name

    name = exp_dir.name
    known = (
        ("MCPST-FSL", ("MCPST", "mcpst")),
        ("RoleFusion-final-p2groc05", ("p2groc05",)),
        ("RoleFusion-e2e-route", ("e2e_route",)),
        ("MoGERNN", ("MoGERNN",)),
        ("PatchSTG", ("PatchSTG",)),
        ("STBP", ("STBP",)),
        ("STGODE", ("STGODE",)),
        ("STID", ("STID",)),
        ("STGCN", ("STGCN",)),
        ("SLPF", ("SLPF",)),
    )
    for label, needles in known:
        if any(needle in name for needle in needles):
            return label
    return name


def _build_row(method, metrics, source):
    macro = sum(metrics[role] for role in ROLES) / len(ROLES)
    row = {"Method": method, "Macro MAPE": macro, "Source": source}
    for role in ROLES:
        row[f"{role} MAPE"] = metrics[role]
    return row


def _collect_rows(args):
    label_rules = _parse_label_rules(args.label)
    rows = _parse_manual_rows(args.manual_row)
    prediction_root = Path(args.predictions)

    json_paths = prediction_root.rglob("async_lifecycle_summary.json")
    text_paths = prediction_root.rglob("async_lifecycle_results.txt")
    for path in sorted([*json_paths, *text_paths]):
        async_dir = path.parent
        if not args.all_async and args.async_filter not in async_dir.name:
            continue
        exp_dir = async_dir.parent
        full = f"{exp_dir.name}/{async_dir.name}"
        if args.include and not any(needle in full for needle in args.include):
            continue

        metrics = _extract_json_metrics(path) if path.suffix == ".json" else _extract_text_metrics(path)
        if metrics is None:
            continue

        method = _infer_method(exp_dir, async_dir, label_rules)
        rows.append(_build_row(method, metrics, source=str(path)))

    if args.best_per_method:
        best = {}
        for row in rows:
            key = row["Method"]
            if key not in best or row["Macro MAPE"] < best[key]["Macro MAPE"]:
                best[key] = row
        rows = list(best.values())

    rows.sort(key=lambda row: (row["Macro MAPE"], row["Method"]))
    return rows


def _format_number(value):
    return f"{value:.4f}"


def _write_markdown(rows, out_path):
    headers = ("Method", "VS MAPE", "AAS MAPE", "NAS MAPE", "FS MAPE", "Macro MAPE")
    lines = [
        "# Async Lifecycle MAPE Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["Method"],
                    _format_number(row["VS MAPE"]),
                    _format_number(row["AAS MAPE"]),
                    _format_number(row["NAS MAPE"]),
                    _format_number(row["FS MAPE"]),
                    _format_number(row["Macro MAPE"]),
                ]
            )
            + " |"
        )
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(rows, out_path):
    if not out_path:
        return
    headers = ("Method", "VS MAPE", "AAS MAPE", "NAS MAPE", "FS MAPE", "Macro MAPE", "Source")
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in headers})


def main():
    parser = argparse.ArgumentParser(description="Collect async lifecycle role MAPE tables.")
    parser.add_argument("--predictions", default="predictions", help="Prediction root directory.")
    parser.add_argument("--out", default="reports_async_lifecycle_mape_summary_auto.md")
    parser.add_argument("--csv", default="")
    parser.add_argument("--async-filter", default="base", help="Substring in async result dir.")
    parser.add_argument("--all-async", action="store_true", help="Include every async adaptor.")
    parser.add_argument("--best-per-method", action="store_true", default=True)
    parser.add_argument("--include", action="append", default=[], help="Only include paths containing this substring.")
    parser.add_argument("--label", action="append", default=[], help="Map output name: NAME=SUBSTRING.")
    parser.add_argument("--manual-row", action="append", default=[], help="Add row: METHOD,VS,AAS,NAS,FS.")
    args = parser.parse_args()

    rows = _collect_rows(args)
    _write_markdown(rows, args.out)
    _write_csv(rows, args.csv)
    print(f"wrote {len(rows)} rows to {args.out}")
    if args.csv:
        print(f"wrote csv to {args.csv}")


if __name__ == "__main__":
    main()
