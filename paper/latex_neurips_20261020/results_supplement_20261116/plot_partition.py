"""Render appendix Figure A3 from paired seed-32 Unseen test errors."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
METRICS = ("MAE", "RMSE", "MAPE")


def load(path):
    return json.loads(path.read_text())


def test_pairs():
    plan = load(ROOT / "configs/paper/supplement_test_20261116/plan.json")
    runs = ROOT / "artifacts/evaluation/supplement_test_20261116" / plan["plan_sha256"]
    with (ROOT / "paper/pems08_wd_ablation_test_20261012/four_dataset_test.csv").open() as handle:
        anchors = {(r["dataset"], r["metric"]): float(r["value"]) for r in csv.DictReader(handle)
                   if r["condition"] == "Unseen" and r["variant"] == "ALFNet"}
    pairs = {}
    for key, entry in plan["entries"].items():
        if entry["ablation"] == "bfs":
            record = load(runs / f"{key}.json")
            assert record["split"] == "test" and record["seed"] == 32
            for metric in METRICS:
                ds = entry["dataset"]
                pairs[ds, metric] = anchors[ds, metric], record["table"]["main"]["Unseen"][metric]
    return pairs


def main():
    data = test_pairs()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(6.25, 3.05))
    fig.subplots_adjust(left=.32, right=.95, top=.85, bottom=.20)
    teal, slate = "#116960", "#46566b"
    ax.set(xlim=(-.3, 12.5), ylim=(-.6, 6.65), xticks=[0, 2, 4, 6, 8, 10, 12], yticks=[],
           xlabel="Change in Unseen test error relative to KD-tree (%)")
    ax.grid(axis="x", color="#e5e9ea", linewidth=.6)
    ax.set_axisbelow(True)
    for ds, name, top in [("METRLA", "METR-LA", 5.9), ("PEMSBAY", "PEMS-BAY", 2.4)]:
        ax.text(-.47, top - 1, name, transform=ax.get_yaxis_transform(), va="center",
                weight="bold", color=slate)
        for i, metric in enumerate(METRICS):
            y = top - i
            anchor, bfs = data[ds, metric]
            change = 100 * (bfs / anchor - 1)
            ax.text(-.055, y, metric, transform=ax.get_yaxis_transform(), ha="right", va="center")
            ax.plot([0, change], [y, y], color="#9eadaf", linewidth=1.7, zorder=2)
            ax.plot(0, y, "o", mfc="white", mec=slate, mew=1.4, ms=6, zorder=3)
            ax.plot(change, y, "D", color=teal, ms=5.6, zorder=3)
            ax.annotate(f"{change:+.2f}%", (change, y), (7, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=teal)
    ax.plot([], [], "o", mfc="white", mec=slate, mew=1.4, label="KD-tree")
    ax.plot([], [], "D", color=teal, label="BFS")
    ax.legend(loc="lower left", bbox_to_anchor=(-.03, 1), ncol=2, frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(PAPER / "figures/figA3_partition_unseen.pdf")
    fig.savefig(PAPER / "figures/figA3_partition_unseen.svg")
    fig.savefig(PAPER / "figures/figA3_partition_unseen.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
