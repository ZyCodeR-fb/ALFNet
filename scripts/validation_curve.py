"""Compact per-epoch validation curve for a run, in ~30 lines instead of ~500.

`training_summary.json` keeps only the aggregated volatility, and the per-epoch
lifecycle numbers used to exist solely on stdout. `train.py` now writes
`reports/validation_history.json`; this prints the part a human reads.

The Join k1->k12 slope is included on every row because it is the diagnostic
that decides whether the temporal conditioner is using accumulating own-history
at all. A slope that is flat at every epoch cannot be fixed by training longer.

    python scripts/validation_curve.py outputs/ALFNet/gs-s32
    python scripts/validation_curve.py outputs/ALFNet/gs-s32 --phases
"""

import argparse
import json
from pathlib import Path

GROUPS = ("unseen_mae", "steady_mae", "join_mae", "fault_mae")
PHASES = ("Cold_MAE", "Warm_MAE", "Ready_MAE", "Onset_MAE", "Decay_MAE", "Outage_MAE")


def _join_slope(row):
    """Join_k12 - Join_k1: negative means evidence is being used."""
    first, last = row.get("Join_k1_MAE"), row.get("Join_k12_MAE")
    if first is None or last is None:
        return None
    return last - first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--phases", action="store_true",
                        help="also print the six phase MAEs per epoch")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    for candidate in (run_dir / "reports" / "validation_history.json",
                      run_dir / "checkpoints" / "validation_history.json"):
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            break
    else:
        raise SystemExit(
            f"no validation_history.json under {run_dir}; the run predates it -- "
            "read logs/train_*.log instead"
        )

    best = payload.get("best_val_metric_epoch")
    rows = payload["epochs"]
    print(f"{run_dir}   epochs={len(rows)}   best={best}")
    header = f"{'ep':>3} {'unseen':>7} {'steady':>7} {'join':>7} {'fault':>7} {'S+J+F':>7} {'joinK':>7}"
    if args.phases:
        header += "".join(f"{name.replace('_MAE', ''):>8}" for name in PHASES)
    print(header)
    for row in rows:
        values = [row.get(name) for name in GROUPS]
        sjf = [v for name, v in zip(GROUPS, values) if name != "unseen_mae" and v is not None]
        slope = _join_slope(row)
        marker = " *" if row.get("epoch") == best else ""
        line = (
            f"{row.get('epoch', 0):>3}"
            + "".join(f"{v:>8.4f}" if v is not None else f"{'-':>8}" for v in values)
            + (f"{sum(sjf) / len(sjf):>8.4f}" if sjf else f"{'-':>8}")
            + (f"{slope:>+8.3f}" if slope is not None else f"{'-':>8}")
        )
        if args.phases:
            line += "".join(
                f"{row[name]:>8.4f}" if row.get(name) is not None else f"{'-':>8}"
                for name in PHASES
            )
        print(line + marker)

    slopes = [s for s in (_join_slope(r) for r in rows) if s is not None]
    if slopes:
        print()
        print(f"Join k1->k12 slope: min={min(slopes):+.3f} max={max(slopes):+.3f} "
              f"mean={sum(slopes) / len(slopes):+.3f}   "
              f"(negative = Join MAE falls as own-history accumulates)")
        # Report what was observed. Whether a longer run would have developed a
        # response is not something this file can establish, so it does not say.
        clearly_negative = [s for s in slopes if s <= -0.05]
        print(f"  epochs with slope <= -0.05: {len(clearly_negative)} / {len(slopes)}")
        if not clearly_negative:
            print("  no epoch in this run reached a clearly negative slope")


if __name__ == "__main__":
    main()
