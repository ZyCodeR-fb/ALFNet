import argparse
import csv
from pathlib import Path

import numpy as np


def read_curves(path):
    curves = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            curves.append(
                {
                    "track": row["track"],
                    "tau": int(row["tau"]),
                    "ace_mape": float(row["ace_mape"]),
                }
            )
    if not curves:
        raise ValueError(f"PLES curve export is empty: {path}")
    return curves


def plot_trajectory(curves, output_path, input_len, dpi=220):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    panels = (
        ("join", "Sensor joining", "#2673B8"),
        ("fault", "Sensor fault", "#C4473A"),
    )
    for axis, (track, title, color) in zip(axes, panels):
        cells = sorted(
            (cell for cell in curves if cell["track"] == track),
            key=lambda cell: cell["tau"],
        )
        if not cells:
            raise ValueError(f"PLES curve export has no {track!r} track")
        tau = np.asarray([cell["tau"] for cell in cells], dtype=np.int64)
        ace = np.asarray([cell["ace_mape"] for cell in cells], dtype=np.float64)
        axis.plot(tau, np.zeros_like(ace), color="#555555", linewidth=1.0)
        axis.plot([0, 0], [min(0.0, float(ace.min())), max(0.0, float(ace.max()))],
                  color="#222222", linestyle="--", linewidth=1.0)
        axis.plot(tau, ace, color=color, linewidth=2.3)
        axis.fill_between(tau, 0.0, ace, color=color, alpha=0.15)
        axis.axvspan(0, 1, color="#DCEAF4", alpha=0.6)
        axis.axvspan(1, input_len, color="#F4DFB5", alpha=0.35)
        axis.axvspan(input_len, int(tau.max()), color="#DCEBDC", alpha=0.35)
        axis.set_title(title, fontsize=13, weight="bold")
        axis.set_xlabel("Event-relative time (steps)")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("ACE MAPE (percentage points)")
    fig.suptitle("Paired Lifecycle Event Study", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot PLES event-relative treatment effects.")
    parser.add_argument("--curves", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    plot_trajectory(read_curves(args.curves), args.output, args.input_len, args.dpi)


if __name__ == "__main__":
    main()
