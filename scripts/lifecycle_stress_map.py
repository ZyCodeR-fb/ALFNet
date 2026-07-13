import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path


ROLE_ORDER = ["VS", "AAS", "NAS", "FS"]
SENSOR_STATES = ["Cold-start", "Warm-up", "Adapted", "Degrading", "Outage", "Steady"]
DEFAULT_AGE_BINS = [0, 4, 10, math.inf]
DEFAULT_VISIBILITY_BINS = [0.0, 0.25, 0.75, 1.0]


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


def _routing_weight(results, role, window_metric=None):
    window_metric = window_metric or {}
    alpha = window_metric.get("alpha_mean", window_metric.get("graph_weight"))
    if alpha is not None:
        graph_weight = float(alpha)
        patch_weight = window_metric.get("patch_residual_weight")
        return graph_weight, float(1.0 - graph_weight if patch_weight is None else patch_weight)
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
            for metric in role_metrics.get("window_metrics") or []:
                window_index = int(metric["window_index"])
                window_offset = int(metric.get("window_offset", window_index - evaluation_start))
                visibility = _role_visibility(role, event, window_index, input_len, slide_step)
                graph_weight, patch_residual_weight = _routing_weight(results, role, metric)
                lifecycle_age = float(max(0, window_offset))
                history_steps = float(round(visibility * input_len))
                rows.append(
                    {
                        "model": model_name,
                        "event_id": int(event.get("event_id", event_id)),
                        "role": role,
                        "sensor_state": _role_sensor_state(role, window_offset, event),
                        "window_index": window_index,
                        "window_offset": window_offset,
                        "lifecycle_age": lifecycle_age,
                        "event_age": lifecycle_age,
                        "history_steps": history_steps,
                        "history_age": history_steps,
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


def _age_bin_label(bins, index):
    lo, hi = bins[index], bins[index + 1]
    if math.isinf(hi):
        return f"{lo:g}+"
    if float(lo).is_integer() and float(hi).is_integer():
        lo_int, hi_int = int(lo), int(hi)
        if hi_int == lo_int + 1:
            return str(lo_int)
        return f"{lo_int}-{hi_int - 1}"
    return f"{lo:g}-{hi:g}"


def _visibility_bin_label(bins, index):
    lo, hi = bins[index], bins[index + 1]
    return f"{100 * lo:g}-{100 * hi:g}%"


def _mean(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def aggregate_grid(aligned_rows, age_bins, visibility_bins, x_axis="lifecycle_age"):
    if x_axis not in {"lifecycle_age", "event_age", "history_steps", "history_age"}:
        raise ValueError("x_axis must be lifecycle_age/event_age or history_steps/history_age")
    grouped = defaultdict(list)
    for row in aligned_rows:
        age_index = _bin_index(float(row[x_axis]), age_bins)
        visibility_index = _bin_index(float(row["visibility"]), visibility_bins)
        if age_index is None or visibility_index is None:
            continue
        grouped[(age_index, visibility_index)].append(row)

    cells = []
    for (age_index, visibility_index), rows in sorted(grouped.items()):
        mape = _mean([row["mape"] for row in rows])
        baseline_mape = _mean([row["baseline_mape"] for row in rows])
        improvement_pct = None
        if mape is not None and baseline_mape not in {None, 0.0}:
            improvement_pct = (baseline_mape - mape) / baseline_mape * 100.0
        cells.append(
            {
                "age_bin": _age_bin_label(age_bins, age_index),
                "visibility_bin": _visibility_bin_label(visibility_bins, visibility_index),
                "count": len(rows),
                "mape": mape,
                "mae": _mean([row["mae"] for row in rows]),
                "rmse": _mean([row["rmse"] for row in rows]),
                "baseline_mape": baseline_mape,
                "improvement_pct": improvement_pct,
                "graph_weight": _mean([row.get("graph_weight") for row in rows]),
                "patch_residual_weight": _mean([row.get("patch_residual_weight") for row in rows]),
            }
        )
    return cells


def _quantile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def aggregate_transition_profile(rows_by_model, roles=("NAS", "FS"), block_size=12):
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    block_size = int(block_size)
    role_set = set(roles)
    event_windows = defaultdict(list)
    observed_ages = defaultdict(list)

    for model_name, rows in rows_by_model.items():
        for row in rows:
            role = row.get("role")
            if role not in role_set:
                continue
            age = int(math.floor(float(row["lifecycle_age"])))
            if age < 0:
                continue
            block = age // block_size
            event_key = (model_name, role, block, int(row["event_id"]))
            profile_key = (model_name, role, block)
            event_windows[event_key].append(float(row["mape"]))
            observed_ages[profile_key].append(age)

    event_means = defaultdict(list)
    window_counts = defaultdict(int)
    for (model_name, role, block, _event_id), values in event_windows.items():
        profile_key = (model_name, role, block)
        event_means[profile_key].append(_mean(values))
        window_counts[profile_key] += len(values)

    profile = []
    for (model_name, role, block), values in sorted(event_means.items()):
        ages = observed_ages[(model_name, role, block)]
        age_start = block * block_size
        age_end = max(ages)
        profile.append(
            {
                "model": model_name,
                "role": role,
                "age_block": block,
                "age_start": age_start,
                "age_end": age_end,
                "age_center": (age_start + age_end) / 2.0,
                "age_width": age_end - age_start + 1,
                "event_count": len(values),
                "window_count": window_counts[(model_name, role, block)],
                "mape": _mean(values),
                "mape_q25": _quantile(values, 0.25),
                "mape_q75": _quantile(values, 0.75),
            }
        )
    return profile


def summarize_transition_profile(profile, ours_name, baseline_names):
    indexed = {
        (row["model"], row["role"], int(row["age_block"])): row
        for row in profile
    }
    roles = sorted({row["role"] for row in profile if row["model"] == ours_name})
    summary = []
    for role in roles:
        ours_blocks = {
            int(row["age_block"]): row
            for row in profile
            if row["model"] == ours_name and row["role"] == role
        }
        for baseline_name in baseline_names:
            common_blocks = sorted(
                block
                for block in ours_blocks
                if (baseline_name, role, block) in indexed
            )
            if not common_blocks:
                continue
            ours_auc = 0.0
            baseline_auc = 0.0
            win_blocks = 0
            for block in common_blocks:
                ours = ours_blocks[block]
                baseline = indexed[(baseline_name, role, block)]
                width = min(int(ours["age_width"]), int(baseline["age_width"]))
                ours_auc += width * float(ours["mape"])
                baseline_auc += width * float(baseline["mape"])
                win_blocks += int(float(ours["mape"]) < float(baseline["mape"]))
            summary.append(
                {
                    "role": role,
                    "baseline": baseline_name,
                    "integrated_gain_pct": (
                        None
                        if baseline_auc == 0.0
                        else (baseline_auc - ours_auc) / baseline_auc * 100.0
                    ),
                    "transition_win_rate": win_blocks / len(common_blocks),
                    "win_blocks": win_blocks,
                    "total_blocks": len(common_blocks),
                    "ours_error_auc": ours_auc,
                    "baseline_error_auc": baseline_auc,
                }
            )
    return summary


def idw_fill_matrix(matrix, power=2.0):
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    measured = []
    for y, row in enumerate(matrix):
        if len(row) != cols:
            raise ValueError("All matrix rows must have the same length")
        for x, value in enumerate(row):
            if value is not None and math.isfinite(float(value)):
                measured.append((x, y, float(value)))
    if not measured:
        return [[math.nan for _ in range(cols)] for _ in range(rows)]

    filled = []
    for y, row in enumerate(matrix):
        filled_row = []
        for x, value in enumerate(row):
            if value is not None and math.isfinite(float(value)):
                filled_row.append(float(value))
                continue
            weighted_sum = 0.0
            weight_sum = 0.0
            for measured_x, measured_y, measured_value in measured:
                distance_sq = float((x - measured_x) ** 2 + (y - measured_y) ** 2)
                weight = 1.0 / max(distance_sq ** (power / 2.0), 1e-12)
                weighted_sum += weight * measured_value
                weight_sum += weight
            filled_row.append(weighted_sum / weight_sum)
        filled.append(filled_row)
    return filled


def build_role_trajectory(aligned_rows, role, age_bins, x_axis="lifecycle_age", overlay_key="patch_residual_weight"):
    grouped = defaultdict(list)
    for row in aligned_rows:
        if row.get("role") != role:
            continue
        age_index = _bin_index(float(row[x_axis]), age_bins)
        if age_index is not None:
            grouped[age_index].append(row)

    points = []
    for age_index, rows in sorted(grouped.items()):
        points.append(
            {
                "x": age_index + 0.5,
                "visibility": _mean([row.get("visibility") for row in rows]),
                "improvement_pct": _mean([row.get("improvement_pct") for row in rows]),
                "overlay_weight": _mean([row.get(overlay_key) for row in rows]) if overlay_key != "none" else None,
                "count": len(rows),
            }
        )
    return points


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


def plot_heatmap(
    cells,
    age_bins,
    visibility_bins,
    value_key,
    output_path,
    title,
    x_axis="lifecycle_age",
    overlay_key="patch_residual_weight",
):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    age_labels = [_age_bin_label(age_bins, index) for index in range(len(age_bins) - 1)]
    visibility_labels = [_visibility_bin_label(visibility_bins, index) for index in range(len(visibility_bins) - 1)]
    matrix = np.full((len(visibility_labels), len(age_labels)), np.nan, dtype=float)
    overlay_matrix = np.full_like(matrix, np.nan)
    count_matrix = np.zeros_like(matrix, dtype=int)
    age_to_index = {label: index for index, label in enumerate(age_labels)}
    visibility_to_index = {label: index for index, label in enumerate(visibility_labels)}
    for cell in cells:
        y = visibility_to_index[cell["visibility_bin"]]
        x = age_to_index[cell["age_bin"]]
        value = cell.get(value_key)
        if value is not None:
            matrix[y, x] = float(value)
        overlay = cell.get(overlay_key) if overlay_key != "none" else None
        if overlay is not None:
            overlay_matrix[y, x] = float(overlay)
        count_matrix[y, x] = int(cell.get("count", 0))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = "RdYlGn" if value_key == "improvement_pct" else "viridis_r"
    norm = None
    if value_key == "improvement_pct":
        finite = matrix[np.isfinite(matrix)]
        if finite.size and finite.min() < 0.0 < finite.max():
            from matplotlib.colors import TwoSlopeNorm

            norm = TwoSlopeNorm(vmin=float(finite.min()), vcenter=0.0, vmax=float(finite.max()))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, origin="lower")
    ax.set_xticks(range(len(age_labels)), age_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(visibility_labels)), visibility_labels)
    ax.set_xlabel("Lifecycle age (evaluation windows since event)" if x_axis in {"lifecycle_age", "event_age"} else "Observed history steps")
    ax.set_ylabel("Visibility bin")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(value_key)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isnan(matrix[y, x]):
                continue
            text = f"{matrix[y, x]:.1f}\nn={count_matrix[y, x]}"
            if not np.isnan(overlay_matrix[y, x]):
                overlay_name = "beta" if overlay_key == "patch_residual_weight" else "alpha"
                text += f"\n{overlay_name}={overlay_matrix[y, x]:.2f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def _matrix_from_cells(cells, age_bins, visibility_bins, value_key):
    age_labels = [_age_bin_label(age_bins, index) for index in range(len(age_bins) - 1)]
    visibility_labels = [_visibility_bin_label(visibility_bins, index) for index in range(len(visibility_bins) - 1)]
    matrix = [[None for _ in age_labels] for _ in visibility_labels]
    count_matrix = [[0 for _ in age_labels] for _ in visibility_labels]
    age_to_index = {label: index for index, label in enumerate(age_labels)}
    visibility_to_index = {label: index for index, label in enumerate(visibility_labels)}
    for cell in cells:
        x = age_to_index.get(cell.get("age_bin"))
        y = visibility_to_index.get(cell.get("visibility_bin"))
        value = cell.get(value_key)
        if x is None or y is None or value is None:
            continue
        matrix[y][x] = float(value)
        count_matrix[y][x] = int(cell.get("count", 0))
    return matrix, count_matrix


def _visibility_position(value, bins):
    value = max(float(bins[0]), min(float(value), float(bins[-1])))
    index = _bin_index(value, bins)
    if index is None:
        index = len(bins) - 2
    lo, hi = float(bins[index]), float(bins[index + 1])
    fraction = 0.5 if math.isinf(hi) or hi <= lo else (value - lo) / (hi - lo)
    return index + min(max(fraction, 0.08), 0.92)


def _paper_axis_labels(age_bins, visibility_bins):
    if len(age_bins) == 4:
        age_labels = ["Short\n0-3", "Medium\n4-9", "Long\n10+"]
    else:
        age_labels = [_age_bin_label(age_bins, index) for index in range(len(age_bins) - 1)]
    if len(visibility_bins) == 4:
        visibility_labels = ["Low\n0-25%", "Medium\n25-75%", "High\n75-100%"]
    else:
        visibility_labels = [_visibility_bin_label(visibility_bins, index) for index in range(len(visibility_bins) - 1)]
    return age_labels, visibility_labels


def plot_paper_stress_map(
    cells,
    aligned_rows,
    age_bins,
    visibility_bins,
    value_key,
    output_path,
    title,
    x_axis="lifecycle_age",
    overlay_key="patch_residual_weight",
):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.patheffects as path_effects
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
        from matplotlib.lines import Line2D
    except ImportError:
        return False

    matrix, count_matrix = _matrix_from_cells(cells, age_bins, visibility_bins, value_key)
    measured = [
        float(value)
        for row in matrix
        for value in row
        if value is not None and math.isfinite(float(value))
    ]
    if not measured:
        return False

    filled = np.asarray(idw_fill_matrix(matrix), dtype=float)
    row_count, col_count = filled.shape
    if value_key == "improvement_pct":
        limit = max(max(abs(value) for value in measured), 1.0)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        cmap = LinearSegmentedColormap.from_list(
            "lifecycle_gain",
            ["#1D4F91", "#79A7D8", "#F7F7F4", "#F3A27C", "#C73E36"],
        )
        colorbar_label = r"$\Delta$MAPE vs MoGERNN (%)  $\uparrow$"
    else:
        norm = Normalize(vmin=min(measured), vmax=max(measured) if max(measured) > min(measured) else min(measured) + 1.0)
        cmap = LinearSegmentedColormap.from_list(
            "lifecycle_error",
            ["#EEF6F8", "#8AC4C8", "#287D8E", "#173F5F"],
        )
        colorbar_label = f"{value_key.upper()}  " + r"$\downarrow$"

    fig, ax = plt.subplots(figsize=(9.2, 6.6), facecolor="white")
    image = ax.imshow(
        filled,
        origin="lower",
        extent=(0, col_count, 0, row_count),
        aspect="auto",
        interpolation="bicubic",
        cmap=cmap,
        norm=norm,
        alpha=0.97,
    )

    max_count = max(max(row) for row in count_matrix) or 1
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if value is None:
                continue
            count = count_matrix[y][x]
            marker_size = 82.0 + 115.0 * math.log1p(count) / math.log1p(max_count)
            ax.scatter(
                x + 0.5,
                y + 0.5,
                s=marker_size,
                c=[float(value)],
                cmap=cmap,
                norm=norm,
                edgecolors="white",
                linewidths=2.0,
                zorder=5,
            )
            ax.scatter(
                x + 0.5,
                y + 0.5,
                s=marker_size + 20.0,
                facecolors="none",
                edgecolors="#263238",
                linewidths=0.8,
                zorder=4,
            )
            label = f"{float(value):+.1f}" if value_key == "improvement_pct" else f"{float(value):.1f}"
            label_offset = 18 if y >= row_count / 2 else -18
            ax.annotate(
                label,
                xy=(x + 0.5, y + 0.5),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#17202A",
                weight="bold",
                zorder=8,
                bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )

    role_styles = {
        "NAS": {"label": "Warm-up  ->  Adapted", "linestyle": "-", "offset": -0.10},
        "FS": {"label": "Degrading  ->  Outage", "linestyle": "--", "offset": 0.10},
    }
    for role, style in role_styles.items():
        points = build_role_trajectory(aligned_rows, role, age_bins, x_axis=x_axis, overlay_key=overlay_key)
        if not points:
            continue
        coords = [
            (
                point["x"],
                min(
                    row_count - 0.12,
                    max(0.12, _visibility_position(point["visibility"], visibility_bins) + style["offset"]),
                ),
            )
            for point in points
        ]
        for index in range(len(coords) - 1):
            weight_values = [points[index].get("overlay_weight"), points[index + 1].get("overlay_weight")]
            weight = _mean(weight_values)
            line_width = 2.0 if weight is None else 1.6 + 8.0 * max(0.0, min(1.0, weight))
            start, end = coords[index], coords[index + 1]
            line = ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                linewidth=line_width,
                linestyle=style["linestyle"],
                color="white",
                alpha=0.92,
                zorder=7,
                solid_capstyle="round",
                dash_capstyle="round",
            )[0]
            line.set_path_effects([path_effects.withStroke(linewidth=line_width + 2.2, foreground=(0.08, 0.12, 0.16, 0.34))])
            arrow_size = 54.0 + 14.0 * line_width
            ax.scatter(
                end[0],
                end[1],
                marker=">",
                s=arrow_size,
                facecolor="white",
                edgecolor="#34495E",
                linewidth=1.1,
                alpha=0.96,
                zorder=8,
            )
        ax.text(
            coords[0][0] + 0.10,
            coords[0][1] + (-0.45 if role == "NAS" else 0.45),
            style["label"],
            color="white",
            fontsize=9,
            weight="bold",
            ha="left",
            va="center",
            zorder=8,
            path_effects=[path_effects.withStroke(linewidth=2.4, foreground=(0.05, 0.08, 0.12, 0.55))],
        )

    age_labels, visibility_labels = _paper_axis_labels(age_bins, visibility_bins)
    ax.set_xticks(np.arange(col_count) + 0.5, age_labels)
    ax.set_yticks(np.arange(row_count) + 0.5, visibility_labels)
    ax.set_xlim(0, col_count)
    ax.set_ylim(0, row_count)
    ax.set_xlabel("Age (time since lifecycle event)", fontsize=12, labelpad=10)
    ax.set_ylabel("Visibility", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=18, weight="bold", pad=17, color="#17202A")
    ax.text(
        0.5,
        1.015,
        "Circles are measured bins (size proportional to sample count); background is an IDW visual guide",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#566573",
    )
    ax.grid(color="white", linewidth=1.0, alpha=0.30)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#34495E")

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    colorbar.set_label(colorbar_label, fontsize=11)
    colorbar.ax.tick_params(labelsize=9)

    overlay_symbol = r"$\beta$" if overlay_key == "patch_residual_weight" else r"$\alpha$"
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#F7F7F4", markeredgecolor="#263238", markersize=8, label="Measured bin"),
        Line2D([0], [0], color="#566573", linewidth=1.8, label=f"Low {overlay_symbol}"),
        Line2D([0], [0], color="#566573", linewidth=3.3, label=f"Medium {overlay_symbol}"),
        Line2D([0], [0], color="#566573", linewidth=5.0, label=f"High {overlay_symbol}"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=2.6,
        columnspacing=1.5,
    )

    fig.subplots_adjust(left=0.12, right=0.89, top=0.85, bottom=0.23)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


def _strongest_transition_baseline(summary, role):
    candidates = [row for row in summary if row["role"] == role]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["baseline_error_auc"]))


def _transition_advantage(profile, ours_name, baseline_name, role):
    indexed = {
        (row["model"], row["role"], int(row["age_block"])): row
        for row in profile
    }
    ours_rows = sorted(
        (
            row
            for row in profile
            if row["model"] == ours_name and row["role"] == role
        ),
        key=lambda row: int(row["age_block"]),
    )
    result = []
    for ours in ours_rows:
        block = int(ours["age_block"])
        baseline = indexed.get((baseline_name, role, block))
        if baseline is None or float(baseline["mape"]) == 0.0:
            continue
        result.append(
            {
                "age_start": float(ours["age_start"]),
                "age_end": float(ours["age_end"]),
                "age_center": float(ours["age_center"]),
                "gain_pct": (
                    float(baseline["mape"]) - float(ours["mape"])
                )
                / float(baseline["mape"])
                * 100.0,
            }
        )
    return result


def plot_transition_response(
    profile,
    summary,
    output_path,
    ours_name="ALFNet",
    baseline_names=("MoGERNN", "PatchSTG"),
    early_horizon=6,
):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        from matplotlib.lines import Line2D
    except ImportError:
        return False

    roles = [role for role in ("NAS", "FS") if any(row["role"] == role for row in profile)]
    if not roles:
        return False

    models = [ours_name, *baseline_names]
    colors = {
        ours_name: "#126E5A",
        baseline_names[0] if baseline_names else "MoGERNN": "#4C5967",
    }
    if len(baseline_names) > 1:
        colors[baseline_names[1]] = "#D17632"
    line_styles = {ours_name: "-"}
    for index, name in enumerate(baseline_names):
        line_styles[name] = "--" if index == 0 else ":"

    profile_index = defaultdict(list)
    for row in profile:
        profile_index[(row["model"], row["role"])].append(row)
    for rows in profile_index.values():
        rows.sort(key=lambda row: int(row["age_block"]))

    strongest = {
        role: _strongest_transition_baseline(summary, role)
        for role in roles
    }
    advantage = {
        role: _transition_advantage(
            profile,
            ours_name,
            strongest[role]["baseline"],
            role,
        )
        for role in roles
        if strongest[role] is not None
    }
    gain_values = [
        item["gain_pct"]
        for role_items in advantage.values()
        for item in role_items
    ]
    gain_limit = max(10.0, max((abs(value) for value in gain_values), default=10.0))
    gain_cmap = LinearSegmentedColormap.from_list(
        "lifecycle_advantage",
        ["#B84848", "#F7F6F1", "#198A70"],
    )
    gain_norm = TwoSlopeNorm(vmin=-gain_limit, vcenter=0.0, vmax=gain_limit)

    fig = plt.figure(figsize=(13.2, 8.2), facecolor="white")
    grid = fig.add_gridspec(
        2,
        len(roles),
        height_ratios=[4.5, 0.72],
        hspace=0.16,
        wspace=0.10,
    )
    main_axes = [fig.add_subplot(grid[0, index]) for index in range(len(roles))]
    ribbon_axes = [fig.add_subplot(grid[1, index]) for index in range(len(roles))]

    all_mape = [float(row["mape"]) for row in profile if row["model"] in models]
    y_min = max(0.0, min(all_mape) * 0.82)
    y_max = max(all_mape) * 1.12
    max_age = max(float(row["age_end"]) for row in profile) + 1.0
    role_titles = {
        "NAS": "New-sensor adaptation",
        "FS": "Failure resilience",
    }
    phase_labels = {
        "NAS": ("Warm-up", "Adapted"),
        "FS": ("Degrading", "Outage"),
    }

    for axis_index, (role, ax, ribbon_ax) in enumerate(zip(roles, main_axes, ribbon_axes)):
        ax.axvspan(0, early_horizon, color="#DDEBF3", alpha=0.62, zorder=0)
        ax.axvspan(early_horizon, max_age, color="#F8F5EE", alpha=0.62, zorder=0)
        ax.axvline(early_horizon, color="#A6B0B8", linewidth=1.0, linestyle="--", zorder=1)
        first_phase, second_phase = phase_labels[role]
        ax.text(
            early_horizon / 2.0,
            0.965,
            first_phase,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9.5,
            weight="bold",
            color="#466779",
        )
        ax.text(
            early_horizon + (max_age - early_horizon) / 2.0,
            0.965,
            second_phase,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9.5,
            weight="bold",
            color="#78684E",
        )

        for model_name in models:
            rows = profile_index.get((model_name, role), [])
            if not rows:
                continue
            x = np.asarray([float(row["age_center"]) for row in rows])
            y = np.asarray([float(row["mape"]) for row in rows])
            lower = np.asarray([float(row.get("mape_q25", row["mape"])) for row in rows])
            upper = np.asarray([float(row.get("mape_q75", row["mape"])) for row in rows])
            color = colors.get(model_name, "#6B7280")
            is_ours = model_name == ours_name
            ax.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.16 if is_ours else 0.07,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                x,
                y,
                color=color,
                linewidth=3.2 if is_ours else 1.9,
                linestyle=line_styles.get(model_name, "-"),
                marker="o" if is_ours else None,
                markersize=4.2 if is_ours else 0,
                markerfacecolor="white",
                markeredgewidth=1.5,
                zorder=4 if is_ours else 3,
            )

        selected = strongest.get(role)
        if selected is not None:
            gain = float(selected["integrated_gain_pct"])
            win_rate = float(selected["transition_win_rate"]) * 100.0
            ax.text(
                0.975,
                0.80,
                f"vs strongest baseline: {selected['baseline']}\n"
                f"Integrated gain  {gain:+.1f}%\n"
                f"Transition win rate  {win_rate:.0f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9.4,
                linespacing=1.45,
                color="#1F2933",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": "white",
                    "edgecolor": "#D3D8DC",
                    "alpha": 0.92,
                },
                zorder=8,
            )

        ax.set_title(f"{chr(97 + axis_index)}) {role_titles[role]} ({role})", loc="left", fontsize=14, weight="bold", pad=10)
        ax.set_xlim(0, max_age)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", color="#D8DEE3", linewidth=0.8, alpha=0.72)
        ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9AA4AC")
        if axis_index == 0:
            ax.set_ylabel("MAPE (%)", fontsize=11.5, weight="bold")
        else:
            ax.tick_params(axis="y", labelleft=False)

        selected_advantage = advantage.get(role, [])
        if selected_advantage:
            edges = [selected_advantage[0]["age_start"]]
            edges.extend(item["age_end"] + 1.0 for item in selected_advantage)
            values = np.asarray([[item["gain_pct"] for item in selected_advantage]])
            ribbon_ax.pcolormesh(
                np.asarray(edges),
                np.asarray([0.0, 1.0]),
                values,
                cmap=gain_cmap,
                norm=gain_norm,
                shading="flat",
            )
            for edge in edges[1:-1]:
                ribbon_ax.axvline(edge, color="white", linewidth=0.45, alpha=0.55)
        ribbon_ax.axvline(early_horizon, color="#5E6C76", linewidth=1.0, linestyle="--")
        ribbon_ax.set_xlim(0, max_age)
        ribbon_ax.set_ylim(0, 1)
        ribbon_ax.set_yticks([])
        ribbon_ax.set_xlabel("Relative lifecycle age (evaluation windows)", fontsize=10.5, labelpad=7)
        ribbon_ax.tick_params(axis="x", labelsize=8.5)
        ribbon_ax.set_title(
            f"ALFNet advantage over {selected['baseline']} (%)" if selected is not None else "ALFNet advantage (%)",
            fontsize=9.5,
            loc="left",
            pad=5,
            color="#48545D",
        )
        for spine in ribbon_ax.spines.values():
            spine.set_color("#AAB2B9")
            spine.set_linewidth(0.7)

    legend_handles = [
        Line2D([0], [0], color=colors.get(model, "#6B7280"), linewidth=3.0 if model == ours_name else 2.0, linestyle=line_styles.get(model, "-"), marker="o" if model == ours_name else None, markerfacecolor="white", label=model)
        for model in models
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.892),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=10.5,
        handlelength=3.2,
        columnspacing=2.0,
    )
    fig.suptitle("Event-aligned Lifecycle Transition Response", fontsize=21, weight="bold", y=0.982, color="#17212B")
    event_count = max((int(row.get("event_count", 0)) for row in profile), default=0)
    role_window_count = max(
        (
            sum(int(row.get("window_count", 0)) for row in profile if row["model"] == ours_name and row["role"] == role)
            for role in roles
        ),
        default=0,
    )
    age_block_size = max((int(row.get("age_width", 0)) for row in profile), default=0)
    fig.text(
        0.5,
        0.935,
        f"{event_count} asynchronous events | {role_window_count} measured windows per dynamic role | "
        f"{age_block_size}-window age blocks | shaded bands: event-wise IQR",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#5B6872",
    )
    colorbar_axis = fig.add_axes([0.925, 0.145, 0.014, 0.64])
    colorbar = fig.colorbar(ScalarMappable(norm=gain_norm, cmap=gain_cmap), cax=colorbar_axis)
    colorbar.set_label("ALFNet MAPE reduction (%)", fontsize=9.5, labelpad=8)
    colorbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.075, right=0.905, top=0.815, bottom=0.12)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Build lifecycle transition diagnostics and stress maps.")
    parser.add_argument("--ours", required=True, help="Path to ALFNet async_lifecycle_results.pkl/json")
    parser.add_argument("--baseline", required=True, help="Path to baseline async_lifecycle_results.pkl/json")
    parser.add_argument("--secondary-baseline", default=None, help="Path to a second baseline for transition response")
    parser.add_argument("--out-dir", default="reports/lifecycle_stress_map")
    parser.add_argument("--ours-name", default="ALFNet")
    parser.add_argument("--baseline-name", default="MoGERNN")
    parser.add_argument("--secondary-baseline-name", default="PatchSTG")
    parser.add_argument("--transition-block-size", type=int, default=12)
    parser.add_argument("--early-horizon", type=int, default=6)
    parser.add_argument("--age-bins", default=None, help="Comma-separated event-age/history-age bin edges")
    parser.add_argument("--visibility-bins", default=None, help="Comma-separated visibility bin edges")
    parser.add_argument(
        "--x-axis",
        choices=["lifecycle_age", "event_age", "history_steps", "history_age"],
        default="lifecycle_age",
        help="Use lifecycle_age for the paper figure; history_steps is correlated with visibility.",
    )
    parser.add_argument("--sensor-state", choices=["all", *SENSOR_STATES], default="all")
    parser.add_argument(
        "--overlay",
        choices=["patch_residual_weight", "graph_weight", "none"],
        default="patch_residual_weight",
    )
    parser.add_argument(
        "--style",
        choices=["transition", "paper", "matrix"],
        default="transition",
        help="transition draws event-aligned response curves; paper/matrix retain the earlier stress-map diagnostics",
    )
    args = parser.parse_args()

    ours_rows = build_stress_rows(load_results(args.ours), args.ours_name)
    baseline_rows = build_stress_rows(load_results(args.baseline), args.baseline_name)
    out_dir = Path(args.out_dir)
    if args.style == "transition":
        if not args.secondary_baseline:
            parser.error("--secondary-baseline is required when --style transition")
        secondary_rows = build_stress_rows(
            load_results(args.secondary_baseline),
            args.secondary_baseline_name,
        )
        rows_by_model = {
            args.ours_name: ours_rows,
            args.baseline_name: baseline_rows,
            args.secondary_baseline_name: secondary_rows,
        }
        profile = aggregate_transition_profile(
            rows_by_model,
            roles=("NAS", "FS"),
            block_size=args.transition_block_size,
        )
        summary = summarize_transition_profile(
            profile,
            args.ours_name,
            [args.baseline_name, args.secondary_baseline_name],
        )
        write_csv(out_dir / "lifecycle_transition_profile.csv", profile)
        write_csv(out_dir / "lifecycle_transition_summary.csv", summary)
        plotted = plot_transition_response(
            profile,
            summary,
            out_dir / "lifecycle_transition_response.png",
            ours_name=args.ours_name,
            baseline_names=[args.baseline_name, args.secondary_baseline_name],
            early_horizon=args.early_horizon,
        )
        print(f"profile_cells={len(profile)} summary_rows={len(summary)}")
        print(f"wrote {out_dir / 'lifecycle_transition_profile.csv'}")
        print(f"wrote {out_dir / 'lifecycle_transition_summary.csv'}")
        if plotted:
            print(f"wrote {out_dir / 'lifecycle_transition_response.png'}")
        return

    age_bins = parse_bins(args.age_bins, DEFAULT_AGE_BINS)
    visibility_bins = parse_bins(args.visibility_bins, DEFAULT_VISIBILITY_BINS)
    aligned_rows = align_with_baseline(ours_rows, baseline_rows)
    if args.sensor_state != "all":
        aligned_rows = [row for row in aligned_rows if row["sensor_state"] == args.sensor_state]
    grid = aggregate_grid(aligned_rows, age_bins, visibility_bins, x_axis=args.x_axis)

    write_csv(out_dir / "lifecycle_stress_rows.csv", aligned_rows)
    write_csv(out_dir / "lifecycle_stress_grid.csv", grid)
    plot_fn = plot_paper_stress_map if args.style == "paper" else plot_heatmap
    improvement_kwargs = {"aligned_rows": aligned_rows} if args.style == "paper" else {}
    plotted_improvement = plot_fn(
        grid,
        **improvement_kwargs,
        age_bins=age_bins,
        visibility_bins=visibility_bins,
        value_key="improvement_pct",
        output_path=out_dir / "lifecycle_stress_improvement.png",
        title="Lifecycle Stress Map",
        x_axis=args.x_axis,
        overlay_key=args.overlay,
    )
    plotted_mape = plot_fn(
        grid,
        **improvement_kwargs,
        age_bins=age_bins,
        visibility_bins=visibility_bins,
        value_key="mape",
        output_path=out_dir / "lifecycle_stress_mape.png",
        title=f"{args.ours_name} Lifecycle Error Map",
        x_axis=args.x_axis,
        overlay_key=args.overlay,
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
