import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


CELL_KEYS = ("trajectory", "tau", "phase", "hop")
PAIR_KEYS = ("event_id", "trajectory", "source_sensor", "tau", "phase", "hop")
NUMERIC_INT_FIELDS = {
    "event_id",
    "source_sensor",
    "window_index",
    "tau",
    "hop",
    "target_node_count",
    "count",
}
NUMERIC_FLOAT_FIELDS = {
    "treated_ape_sum",
    "control_ape_sum",
    "treated_ae_sum",
    "control_ae_sum",
    "treated_se_sum",
    "control_se_sum",
}
PHASE_COLORS = {
    "Cold-start": "#DDEAF2",
    "Warm-up": "#F2D6AA",
    "Adapted": "#CDE4D3",
    "Steady": "#DCE6D9",
    "Degrading": "#F3D1B5",
    "Outage": "#DDD9E8",
}


def _row_key(row, fields):
    return tuple(row[field] for field in fields)


def read_shockwave_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row = dict(raw_row)
            for field in NUMERIC_INT_FIELDS:
                if field in row:
                    row[field] = int(row[field])
            for field in NUMERIC_FLOAT_FIELDS:
                if field in row:
                    row[field] = float(row[field])
            rows.append(row)
    if not rows:
        raise ValueError(f"Shockwave export is empty: {path}")
    return rows


def _index_rows(rows):
    indexed = {}
    for row in rows:
        key = _row_key(row, PAIR_KEYS)
        if key in indexed:
            raise ValueError(f"Duplicate shockwave row: {key}")
        indexed[key] = row
    return indexed


def _aggregate_model_cell(rows):
    count = sum(float(row["count"]) for row in rows)
    if count <= 0:
        raise ValueError("Shockwave cell has no valid targets")
    treated_ape = sum(float(row["treated_ape_sum"]) for row in rows)
    control_ape = sum(float(row["control_ape_sum"]) for row in rows)
    treated_ae = sum(float(row["treated_ae_sum"]) for row in rows)
    control_ae = sum(float(row["control_ae_sum"]) for row in rows)
    treated_se = sum(float(row["treated_se_sum"]) for row in rows)
    control_se = sum(float(row["control_se_sum"]) for row in rows)
    treated_mape = 100.0 * treated_ape / count
    control_mape = 100.0 * control_ape / count
    treated_mae = treated_ae / count
    control_mae = control_ae / count
    treated_rmse = math.sqrt(treated_se / count)
    control_rmse = math.sqrt(control_se / count)
    return {
        "count": count,
        "treated_mape": treated_mape,
        "control_mape": control_mape,
        "ace_mape": treated_mape - control_mape,
        "treated_mae": treated_mae,
        "control_mae": control_mae,
        "ace_mae": treated_mae - control_mae,
        "treated_rmse": treated_rmse,
        "control_rmse": control_rmse,
        "ace_rmse": treated_rmse - control_rmse,
    }


def _bootstrap_cell_gain(ours_rows, baseline_rows, bootstrap_samples, rng):
    cluster_count = len(ours_rows)
    if cluster_count < 2 or bootstrap_samples <= 0:
        return None, None
    draws = np.empty(bootstrap_samples, dtype=np.float64)
    for draw_index in range(bootstrap_samples):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        sampled_ours = [ours_rows[index] for index in sampled]
        sampled_baseline = [baseline_rows[index] for index in sampled]
        ours_ace = _aggregate_model_cell(sampled_ours)["ace_mape"]
        baseline_ace = _aggregate_model_cell(sampled_baseline)["ace_mape"]
        draws[draw_index] = baseline_ace - ours_ace
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def aggregate_counterfactual_gain(ours_rows, baseline_rows, bootstrap_samples=2000, seed=0):
    ours_by_pair = _index_rows(ours_rows)
    baseline_by_pair = _index_rows(baseline_rows)
    if set(ours_by_pair) != set(baseline_by_pair):
        missing_ours = sorted(set(baseline_by_pair) - set(ours_by_pair))[:3]
        missing_baseline = sorted(set(ours_by_pair) - set(baseline_by_pair))[:3]
        raise ValueError(
            "Shockwave exports do not contain the same paired cells. "
            f"missing_ours={missing_ours}, missing_baseline={missing_baseline}"
        )

    grouped = {}
    for pair_key in sorted(ours_by_pair):
        ours_row = ours_by_pair[pair_key]
        baseline_row = baseline_by_pair[pair_key]
        if int(ours_row["count"]) != int(baseline_row["count"]):
            raise ValueError(f"Paired valid-target count differs for {pair_key}")
        cell_key = _row_key(ours_row, CELL_KEYS)
        group = grouped.setdefault(cell_key, {"ours": [], "baseline": []})
        group["ours"].append(ours_row)
        group["baseline"].append(baseline_row)

    rng = np.random.default_rng(seed)
    cells = []
    for cell_key in sorted(grouped, key=lambda key: (key[0], int(key[1]), int(key[3]))):
        group = grouped[cell_key]
        ours_metrics = _aggregate_model_cell(group["ours"])
        baseline_metrics = _aggregate_model_cell(group["baseline"])
        ci_low, ci_high = _bootstrap_cell_gain(group["ours"], group["baseline"], bootstrap_samples, rng)
        cells.append(
            {
                **dict(zip(CELL_KEYS, cell_key)),
                "cluster_count": len(group["ours"]),
                "count": int(ours_metrics["count"]),
                **{f"ours_{key}": value for key, value in ours_metrics.items() if key != "count"},
                **{f"baseline_{key}": value for key, value in baseline_metrics.items() if key != "count"},
                "attenuation_gain_mape": baseline_metrics["ace_mape"] - ours_metrics["ace_mape"],
                "attenuation_gain_mae": baseline_metrics["ace_mae"] - ours_metrics["ace_mae"],
                "attenuation_gain_rmse": baseline_metrics["ace_rmse"] - ours_metrics["ace_rmse"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "significant": bool(ci_low is not None and (ci_low > 0 or ci_high < 0)),
            }
        )
    return cells


def _write_cells(path, cells):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)


def _weighted_mean(cells, field):
    total_count = sum(cell["count"] for cell in cells)
    if total_count <= 0:
        return None
    return float(sum(cell[field] * cell["count"] for cell in cells) / total_count)


def summarize_cells(cells):
    summary = {
        "definition": {
            "model_effect": "ACE = MAPE_treated - MAPE_control",
            "attenuation_gain": "ACE_MoGERNN - ACE_ALFNet",
            "positive_meaning": "ALFNet is better",
        },
        "trajectories": {},
        "phases": {},
    }
    for trajectory in sorted({cell["trajectory"] for cell in cells}):
        trajectory_cells = [cell for cell in cells if cell["trajectory"] == trajectory]
        pre_cells = [cell for cell in trajectory_cells if cell["tau"] < 0]
        post_cells = [cell for cell in trajectory_cells if cell["tau"] >= 0]
        significant_positive = [
            cell for cell in post_cells if cell["significant"] and cell["attenuation_gain_mape"] > 0
        ]
        peak = max(post_cells, key=lambda cell: cell["attenuation_gain_mape"])
        significant_hops = [
            int(cell["hop"])
            for cell in significant_positive
        ]
        summary["trajectories"][trajectory] = {
            "pre_event_placebo_max_abs_mape": float(
                max((abs(cell["attenuation_gain_mape"]) for cell in pre_cells), default=0.0)
            ),
            "post_event_weighted_mean_gain_mape": _weighted_mean(post_cells, "attenuation_gain_mape"),
            "post_event_weighted_mean_gain_mae": _weighted_mean(post_cells, "attenuation_gain_mae"),
            "post_event_weighted_mean_gain_rmse": _weighted_mean(post_cells, "attenuation_gain_rmse"),
            "positive_cell_fraction": float(
                sum(cell["attenuation_gain_mape"] > 0 for cell in post_cells) / max(1, len(post_cells))
            ),
            "significant_positive_cell_fraction": float(len(significant_positive) / max(1, len(post_cells))),
            "farthest_significant_positive_hop": max(significant_hops) if significant_hops else None,
            "peak_gain": {
                "mape_points": float(peak["attenuation_gain_mape"]),
                "tau": int(peak["tau"]),
                "hop": int(peak["hop"]),
                "phase": peak["phase"],
            },
        }
    for phase in sorted({cell["phase"] for cell in cells}):
        phase_cells = [cell for cell in cells if cell["phase"] == phase]
        summary["phases"][phase] = {
            "weighted_mean_gain_mape": _weighted_mean(phase_cells, "attenuation_gain_mape"),
            "weighted_mean_gain_mae": _weighted_mean(phase_cells, "attenuation_gain_mae"),
            "weighted_mean_gain_rmse": _weighted_mean(phase_cells, "attenuation_gain_rmse"),
        }
    return summary


def _phase_segments(panel_cells, taus):
    phase_by_tau = {int(cell["tau"]): cell["phase"] for cell in panel_cells}
    segments = []
    start = taus[0]
    current = phase_by_tau[start]
    previous = start
    for tau in taus[1:]:
        phase = phase_by_tau[tau]
        if phase != current:
            segments.append((start, previous, current))
            start = tau
            current = phase
        previous = tau
    segments.append((start, previous, current))
    return segments


def plot_shockwave(cells, output_dir, dpi=300, color_limit=None):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    trajectories = [trajectory for trajectory in ("commissioning", "failure") if any(cell["trajectory"] == trajectory for cell in cells)]
    if len(trajectories) != 2:
        raise ValueError("Shockwave figure requires both commissioning and failure trajectories")
    finite_gains = np.array([cell["attenuation_gain_mape"] for cell in cells], dtype=np.float64)
    finite_gains = finite_gains[np.isfinite(finite_gains)]
    if color_limit is None:
        color_limit = max(0.1, float(np.max(np.abs(finite_gains))))
    color_limit = max(0.1, abs(float(color_limit)))
    cmap = LinearSegmentedColormap.from_list("shockwave", ["#B64B4A", "#F7F6F1", "#176C5B"])
    cmap.set_bad("#ECEDEB")
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.4), sharey=True, constrained_layout=True)
    panel_titles = {"commissioning": "(a) Sensor commissioning", "failure": "(b) Sensor failure"}
    mesh = None
    for axis, trajectory in zip(axes, trajectories):
        panel_cells = [cell for cell in cells if cell["trajectory"] == trajectory]
        taus = sorted({int(cell["tau"]) for cell in panel_cells})
        hops = sorted({int(cell["hop"]) for cell in panel_cells})
        tau_index = {tau: index for index, tau in enumerate(taus)}
        hop_index = {hop: index for index, hop in enumerate(hops)}
        matrix = np.full((len(hops), len(taus)), np.nan, dtype=np.float64)
        significant = np.zeros_like(matrix, dtype=bool)
        for cell in panel_cells:
            row = hop_index[int(cell["hop"])]
            column = tau_index[int(cell["tau"])]
            matrix[row, column] = float(cell["attenuation_gain_mape"])
            significant[row, column] = bool(cell["significant"])
        mesh = axis.pcolormesh(
            np.arange(len(taus) + 1) - 0.5,
            np.arange(len(hops) + 1) - 0.5,
            matrix,
            cmap=cmap,
            norm=norm,
            shading="flat",
            linewidth=0.25,
            edgecolors=(1, 1, 1, 0.25),
        )
        sig_rows, sig_columns = np.where(significant)
        axis.scatter(sig_columns, sig_rows, s=9, c="#202020", marker="o", linewidths=0, zorder=3)
        tick_step = 6 if len(taus) > 18 else max(1, len(taus) // 6)
        tick_positions = list(range(0, len(taus), tick_step))
        if tick_positions[-1] != len(taus) - 1:
            tick_positions.append(len(taus) - 1)
        axis.set_xticks(tick_positions, [str(taus[index]) for index in tick_positions])
        max_hop = max(hops)
        axis.set_yticks(range(len(hops)), [f"{hop}+ hops" if hop == max_hop else f"{hop} hop" for hop in hops])
        axis.set_xlabel("Event-relative time (5-min steps)")
        axis.set_title(panel_titles[trajectory], pad=38, fontweight="semibold")
        axis.axvline(tau_index.get(0, 0) - 0.5, color="#2B2B2B", linewidth=1.0, linestyle="--")
        axis.set_xlim(-0.5, len(taus) - 0.5)
        axis.set_ylim(len(hops) - 0.5, -0.5)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#555555")

        phase_axis = axis.inset_axes([0.0, 1.015, 1.0, 0.065], transform=axis.transAxes)
        phase_axis.set_xlim(-0.5, len(taus) - 0.5)
        phase_axis.set_ylim(0, 1)
        for start_tau, end_tau, phase in _phase_segments(panel_cells, taus):
            start_column = tau_index[start_tau] - 0.5
            end_column = tau_index[end_tau] + 0.5
            phase_axis.axvspan(start_column, end_column, color=PHASE_COLORS[phase], ec="white", lw=0.8)
            phase_axis.text(
                (start_column + end_column) / 2,
                0.5,
                phase,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="semibold",
                color="#303030",
            )
        phase_axis.axis("off")

    axes[0].set_ylabel("Distance from changed sensor")
    colorbar = fig.colorbar(mesh, ax=axes, fraction=0.028, pad=0.02)
    colorbar.set_label("Counterfactual attenuation gain (MAPE points)\npositive = ALFNet better")
    fig.suptitle("Counterfactual Lifecycle Shockwave", fontsize=17, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        -0.02,
        "Difference-in-differences against a matched no-current-shock replay; dots denote paired 95% bootstrap CI excluding zero.",
        ha="center",
        fontsize=9,
        color="#4A4A4A",
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "counterfactual_lifecycle_shockwave.png"
    pdf_path = output_dir / "counterfactual_lifecycle_shockwave.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(png_path), str(pdf_path)


def _load_optional_metadata(rows_path):
    rows_path = Path(rows_path)
    candidate = rows_path.with_name(rows_path.name.replace("_rows.csv", "_meta.json"))
    if not candidate.exists():
        return None
    with open(candidate, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_metadata(ours_meta, baseline_meta):
    if ours_meta is None or baseline_meta is None:
        return
    comparable_keys = (
        "input_len",
        "pred_len",
        "dense_replay_slide_step",
        "pre_steps",
        "post_steps",
        "max_hop_bucket",
        "async_num_events",
        "async_commissioning_k",
        "async_event_seed",
    )
    mismatches = []
    for key in comparable_keys:
        ours_value = ours_meta.get("config", {}).get(key)
        baseline_value = baseline_meta.get("config", {}).get(key)
        if ours_value != baseline_value:
            mismatches.append(f"{key}: ours={ours_value}, baseline={baseline_value}")
    ours_events = ours_meta.get("events")
    baseline_events = baseline_meta.get("events")
    if ours_events != baseline_events:
        mismatches.append("event plans differ")
    if mismatches:
        raise ValueError("Shockwave protocols are not paired: " + "; ".join(mismatches))


def _write_summary_text(path, summary):
    lines = [
        "Counterfactual Lifecycle Shockwave",
        "ACE = MAPE_treated - MAPE_control",
        "AttenuationGain = ACE_MoGERNN - ACE_ALFNet (positive = ALFNet better)",
        "",
    ]
    for trajectory, metrics in summary["trajectories"].items():
        peak = metrics["peak_gain"]
        lines.append(
            f"{trajectory}: mean_gain={metrics['post_event_weighted_mean_gain_mape']:.4f} MAPE points | "
            f"positive_cells={metrics['positive_cell_fraction']:.3f} | "
            f"significant_positive_cells={metrics['significant_positive_cell_fraction']:.3f} | "
            f"placebo_max_abs={metrics['pre_event_placebo_max_abs_mape']:.6f} | "
            f"peak={peak['mape_points']:.4f}@tau{peak['tau']}/hop{peak['hop']}"
        )
    lines.append("")
    for phase, metrics in summary["phases"].items():
        lines.append(f"{phase}: mean_gain={metrics['weighted_mean_gain_mape']:.4f} MAPE points")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot paired Counterfactual Lifecycle Shockwave exports")
    parser.add_argument("--ours", required=True, help="ALFNet shockwave rows CSV")
    parser.add_argument("--baseline", required=True, help="MoGERNN shockwave rows CSV")
    parser.add_argument("--output-dir", default="./figures/shockwave")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--color-limit", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ours_meta = _load_optional_metadata(args.ours)
    baseline_meta = _load_optional_metadata(args.baseline)
    _validate_metadata(ours_meta, baseline_meta)
    ours_rows = read_shockwave_rows(args.ours)
    baseline_rows = read_shockwave_rows(args.baseline)
    cells = aggregate_counterfactual_gain(
        ours_rows,
        baseline_rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cells_path = os.path.join(output_dir, "counterfactual_lifecycle_shockwave_cells.csv")
    summary_path = os.path.join(output_dir, "counterfactual_lifecycle_shockwave_summary.json")
    summary_text_path = os.path.join(output_dir, "counterfactual_lifecycle_shockwave_summary.txt")
    _write_cells(cells_path, cells)
    summary = summarize_cells(cells)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_summary_text(summary_text_path, summary)
    png_path, pdf_path = plot_shockwave(cells, output_dir, dpi=args.dpi, color_limit=args.color_limit)
    print(f"-> Shockwave cells saved to {cells_path}")
    print(f"-> Shockwave summary saved to {summary_path}")
    print(f"-> Shockwave figure saved to {png_path}")
    print(f"-> Shockwave vector figure saved to {pdf_path}")


if __name__ == "__main__":
    main()
