import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path


ROLE_ORDER = ["VS", "AAS", "NAS", "FS"]
DEFAULT_AGE_BINS = [0, 3, 6, 12, 24, 48, 96, math.inf]
DEFAULT_VISIBILITY_BINS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_results(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("rb") as handle:
        return pickle.load(handle)


def parse_bins(value, default_bins):
    if value is None:
        return list(default_bins)
    bins = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        bins.append(math.inf if item.lower() in {"inf", "infinity"} else float(item))
    if len(bins) < 2:
        raise ValueError("At least two bin edges are required")
    return bins


def _event_lookup(results):
    return {str(event["event_id"]): event for event in results.get("event_plan", [])}


def _role_sensor_state(role, window_offset, event):
    if role == "VS":
        return "Cold-start"
    if role == "AAS":
        return "Steady"
    evaluation_start, evaluation_end = event.get("evaluation_window_range", [0, 0])
    evaluation_len = max(1, int(evaluation_end) - int(evaluation_start) + 1)
    early_horizon = max(1, min(6, evaluation_len // 4))
    if role == "NAS":
        return "Warm-up" if window_offset < early_horizon else "Adapted"
    if role == "FS":
        return "Degrading" if window_offset < early_horizon else "Outage"
    return role


def _role_visibility(role, event, window_index, input_len, slide_step):
    if role == "AAS":
        return 1.0
    if role == "VS":
        return 0.0
    event_time = int(event.get("event_time_index", 0))
    input_begin = int(window_index) * int(slide_step)
    visible = 0
    for time_index in range(input_begin, input_begin + int(input_len)):
        if role == "NAS":
            visible += int(time_index >= event_time)
        elif role == "FS":
            visible += int(time_index < event_time)
    return visible / float(input_len)


def _routing_weight(results, role):
    routing = results.get("routing_diagnostics") or {}
    role_diag = routing.get(role) or {}
    alpha = role_diag.get("alpha_mean")
    if alpha is None:
        return None, None
    graph_weight = float(alpha)
    return graph_weight, 1.0 - graph_weight


def build_stress_rows(results, model_name):
    config = results.get("config") or {}
    input_len = int(config.get("input_len", 12))
    slide_step = int(config.get("slide_step", 12))
    events = _event_lookup(results)
    rows = []
    for event_id, event_roles in (results.get("event_metrics") or {}).items():
        event = events.get(str(event_id))
        if event is None:
            continue
        evaluation_start = int(event.get("evaluation_window_range", [0, 0])[0])
        for role in ROLE_ORDER:
            role_metrics = (event_roles or {}).get(role) or {}
            graph_weight, patch_residual_weight = _routing_weight(results, role)
            for metric in role_metrics.get("window_metrics") or []:
                window_index = int(metric["window_index"])
                window_offset = int(metric.get("window_offset", window_index - evaluation_start))
                visibility = _role_visibility(role, event, window_index, input_len, slide_step)
                rows.append(
                    {
                        "model": model_name,
                        "event_id": int(event.get("event_id", event_id)),
                        "role": role,
                        "sensor_state": _role_sensor_state(role, window_offset, event),
                        "window_index": window_index,
                        "window_offset": window_offset,
                        "event_age": float(max(0, window_offset)),
                        "history_age": float(visibility),
                        "visibility": float(visibility),
                        "mape": float(metric["MAPE"]),
                        "mae": float(metric["MAE"]),
                        "rmse": float(metric["RMSE"]),
                        "graph_weight": graph_weight,
                        "patch_residual_weight": patch_residual_weight,
                    }
                )
    return rows


def _alignment_key(row):
    return (row["event_id"], row["role"], row["window_index"], row["window_offset"])


def align_with_baseline(ours_rows, baseline_rows):
    baseline_by_key = {_alignment_key(row): row for row in baseline_rows}
    aligned = []
    for row in ours_rows:
        base = baseline_by_key.get(_alignment_key(row))
        if base is None:
            continue
        item = dict(row)
        item["baseline_mape"] = base["mape"]
        item["baseline_mae"] = base["mae"]
        item["baseline_rmse"] = base["rmse"]
        item["improvement_pct"] = (
            None
            if base["mape"] == 0
            else (base["mape"] - row["mape"]) / base["mape"] * 100.0
        )
        aligned.append(item)
    return aligned


def _bin_index(value, bins):
    for index in range(len(bins) - 1):
        lo, hi = bins[index], bins[index + 1]
        if value >= lo and (value < hi or (index == len(bins) - 2 and value <= hi)):
            return index
    return None


def _bin_label(bins, index):
    lo, hi = bins[index], bins[index + 1]
    if math.isinf(hi):
        return f"{lo:g}+"
    return f"{lo:g}-{hi:g}"


def _mean(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def aggregate_grid(aligned_rows, age_bins, visibility_bins, x_axis="event_age"):
    if x_axis not in {"event_age", "history_age"}:
        raise ValueError("x_axis must be 'event_age' or 'history_age'")
    grouped = defaultdict(list)
    for row in aligned_rows:
        age_index = _bin_index(float(row[x_axis]), age_bins)
        visibility_index = _bin_index(float(row["visibility"]), visibility_bins)
        if age_index is None or visibility_index is None:
            continue
        grouped[(age_index, visibility_index)].append(row)

    cells = []
    for (age_index, visibility_index), rows in sorted(grouped.items()):
        cells.append(
            {
                "age_bin": _bin_label(age_bins, age_index),
                "visibility_bin": _bin_label(visibility_bins, visibility_index),
                "count": len(rows),
                "mape": _mean([row["mape"] for row in rows]),
                "mae": _mean([row["mae"] for row in rows]),
                "rmse": _mean([row["rmse"] for row in rows]),
                "baseline_mape": _mean([row["baseline_mape"] for row in rows]),
                "improvement_pct": _mean([row["improvement_pct"] for row in rows]),
                "graph_weight": _mean([row.get("graph_weight") for row in rows]),
                "patch_residual_weight": _mean([row.get("patch_residual_weight") for row in rows]),
            }
        )
    return cells


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(cells, age_bins, visibility_bins, value_key, output_path, title):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    age_labels = [_bin_label(age_bins, index) for index in range(len(age_bins) - 1)]
    visibility_labels = [_bin_label(visibility_bins, index) for index in range(len(visibility_bins) - 1)]
    matrix = np.full((len(visibility_labels), len(age_labels)), np.nan, dtype=float)
    beta_matrix = np.full_like(matrix, np.nan)
    age_to_index = {label: index for index, label in enumerate(age_labels)}
    visibility_to_index = {label: index for index, label in enumerate(visibility_labels)}
    for cell in cells:
        y = visibility_to_index[cell["visibility_bin"]]
        x = age_to_index[cell["age_bin"]]
        value = cell.get(value_key)
        if value is not None:
            matrix[y, x] = float(value)
        beta = cell.get("patch_residual_weight")
        if beta is not None:
            beta_matrix[y, x] = float(beta)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = "RdYlGn" if value_key == "improvement_pct" else "viridis_r"
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(age_labels)), age_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(visibility_labels)), visibility_labels)
    ax.set_xlabel("Event age bin" if value_key == "improvement_pct" else "Age bin")
    ax.set_ylabel("Visibility bin")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(value_key)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isnan(matrix[y, x]):
                continue
            text = f"{matrix[y, x]:.1f}"
            if not np.isnan(beta_matrix[y, x]):
                text += f"\nbeta={beta_matrix[y, x]:.2f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Build Lifecycle Stress Map tables and heatmaps.")
    parser.add_argument("--ours", required=True, help="Path to ALFNet async_lifecycle_results.pkl/json")
    parser.add_argument("--baseline", required=True, help="Path to baseline async_lifecycle_results.pkl/json")
    parser.add_argument("--out-dir", default="reports/lifecycle_stress_map")
    parser.add_argument("--ours-name", default="ALFNet")
    parser.add_argument("--baseline-name", default="MoGERNN")
    parser.add_argument("--age-bins", default=None, help="Comma-separated event-age/history-age bin edges")
    parser.add_argument("--visibility-bins", default=None, help="Comma-separated visibility bin edges")
    parser.add_argument("--x-axis", choices=["event_age", "history_age"], default="event_age")
    args = parser.parse_args()

    age_bins = parse_bins(args.age_bins, DEFAULT_AGE_BINS)
    visibility_bins = parse_bins(args.visibility_bins, DEFAULT_VISIBILITY_BINS)
    ours_rows = build_stress_rows(load_results(args.ours), args.ours_name)
    baseline_rows = build_stress_rows(load_results(args.baseline), args.baseline_name)
    aligned_rows = align_with_baseline(ours_rows, baseline_rows)
    grid = aggregate_grid(aligned_rows, age_bins, visibility_bins, x_axis=args.x_axis)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "lifecycle_stress_rows.csv", aligned_rows)
    write_csv(out_dir / "lifecycle_stress_grid.csv", grid)
    plotted_improvement = plot_heatmap(
        grid,
        age_bins,
        visibility_bins,
        "improvement_pct",
        out_dir / "lifecycle_stress_improvement.png",
        f"{args.ours_name} improvement over {args.baseline_name}",
    )
    plotted_mape = plot_heatmap(
        grid,
        age_bins,
        visibility_bins,
        "mape",
        out_dir / "lifecycle_stress_mape.png",
        f"{args.ours_name} MAPE",
    )
    print(f"rows={len(aligned_rows)} grid_cells={len(grid)}")
    print(f"wrote {out_dir / 'lifecycle_stress_rows.csv'}")
    print(f"wrote {out_dir / 'lifecycle_stress_grid.csv'}")
    if plotted_improvement:
        print(f"wrote {out_dir / 'lifecycle_stress_improvement.png'}")
    if plotted_mape:
        print(f"wrote {out_dir / 'lifecycle_stress_mape.png'}")


if __name__ == "__main__":
    main()
