"""Per-layer, per-phase and per-k routing report for a trained ALFGraph run.

Why post-hoc rather than instrumented into training: the router's only input is
the lifecycle state at the last input step, and under LCBT that state is an
exact function of (scenario, evidence length k). So the full phase-by-k routing
table can be reconstructed from `w_gate` alone, with no need to thread per-node
gates out of a 24-step recurrence or to slow the training loop down.

The exact states, from `data_provider/graph_data_loader.py`:

    Unseen / zero-history   role=VS   age=0.0        mask=0
    Steady                  role=AAS  age=1.0        mask=1
    Join,  evidence k       role=NAS  age=k/L        mask=1   (k1 Cold, k12 Ready)
    Fault, evidence k       role=FS   age=k/L        mask=0   (k1 Onset, k12 Outage)

What to read it for:

* `entropy_ratio` near 1.0 means the router is not committing -- it is spreading
  mass evenly and carries no information about the lifecycle state.
* an identical expert across every phase means the router has collapsed, and the
  mixture is a single fixed operator wearing five weight matrices.
* `top1_margin` near zero means the choice is a coin flip. On the shipped
  `fg1-s32` checkpoint the margin was ~1e-3 against an injected training noise
  of ~0.70, which is why that router routed at chance.

Usage:
    python scripts/router_report.py outputs/ALFNet/gs-s32
    python scripts/router_report.py outputs/ALFNet/gs-s32 --output-dir some/dir
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

# Direct execution sets ``sys.path[0]`` to ``scripts/`` rather than the
# repository root, so make the project's layers package importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layers.lifecycle_moge import SUPPORT_NAMES

ROLE_AAS, ROLE_NAS, ROLE_VS, ROLE_FS = 0, 1, 2, 3
INPUT_LEN = 12


def protocol_states(input_len=INPUT_LEN):
    """Every lifecycle state the router can see, labelled by phase and k."""
    states = [
        ("Unseen", None, ROLE_VS, 0.0, 0.0),
        ("Steady", None, ROLE_AAS, 1.0, 1.0),
    ]
    for k in range(1, input_len + 1):
        phase = "Cold" if k == 1 else ("Ready" if k == input_len else "Warm")
        states.append((phase, k, ROLE_NAS, k / input_len, 1.0))
    for k in range(1, input_len + 1):
        phase = "Onset" if k == 1 else ("Outage" if k == input_len else "Decay")
        states.append((phase, k, ROLE_FS, k / input_len, 0.0))
    return states


def _vector(role, age, mask):
    vector = torch.zeros(6)
    vector[role] = 1.0
    vector[4] = age
    vector[5] = mask
    return vector


def analyse(state_dict, mode):
    """Routing distribution per layer per lifecycle state."""
    states = protocol_states()
    inputs = torch.stack([_vector(role, age, mask) for _, _, role, age, mask in states])
    report = {}
    for key in sorted(k for k in state_dict if k.endswith("router.w_gate")):
        w_gate = state_dict[key]
        logits = inputs @ w_gate
        probs = torch.softmax(logits, dim=1)
        top2 = probs.topk(2, dim=1).values
        rows = []
        for index, (phase, k, _, age, mask) in enumerate(states):
            distribution = probs[index]
            entropy = float(-(distribution.clamp_min(1e-12).log() * distribution).sum())
            rows.append({
                "phase": phase,
                "k": k,
                "state_age": round(age, 4),
                "obs_mask": mask,
                # `soft` mixes every expert, so the whole distribution matters;
                # the hard modes only ever apply the argmax.
                "expert": SUPPORT_NAMES[int(distribution.argmax())],
                "probs": {n: round(float(p), 5) for n, p in zip(SUPPORT_NAMES, distribution)},
                "entropy_ratio": round(entropy / math.log(len(SUPPORT_NAMES)), 5),
                "top1_margin": round(float(top2[index, 0] - top2[index, 1]), 6),
            })
        chosen = {row["expert"] for row in rows}
        report[key.replace(".router.w_gate", "")] = {
            "router_mode": mode,
            "distinct_experts_across_phases": sorted(chosen),
            "collapsed": len(chosen) == 1,
            "mean_entropy_ratio": round(
                sum(r["entropy_ratio"] for r in rows) / len(rows), 5
            ),
            "mean_top1_margin": round(sum(r["top1_margin"] for r in rows) / len(rows), 6),
            "states": rows,
        }
    return report


def _phase_table(report):
    """One line per phase per layer, which is what a human actually reads."""
    lines = []
    for layer in sorted(report):
        entry = report[layer]
        lines.append(
            f"{layer}  mode={entry['router_mode']}  "
            f"entropy_ratio={entry['mean_entropy_ratio']:.4f}  "
            f"top1_margin={entry['mean_top1_margin']:.6f}  "
            f"{'COLLAPSED' if entry['collapsed'] else 'experts=' + ','.join(entry['distinct_experts_across_phases'])}"
        )
        for row in entry["states"]:
            label = row["phase"] if row["k"] is None else f"{row['phase']}(k={row['k']})"
            lines.append(
                f"    {label:<14} -> {row['expert']:<20}"
                f" p={row['probs'][row['expert']]:.4f}"
                f" margin={row['top1_margin']:.6f}"
                f" H={row['entropy_ratio']:.4f}"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="outputs/<model>/<run_id>")
    parser.add_argument("--output-dir", default=None,
                        help="defaults to <run_dir>/reports")
    parser.add_argument("--mode", default=None,
                        help="router mode label for the report; only cosmetic, "
                             "the analysis reads w_gate directly")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "checkpoints" / "best_val_checkpoint.pth"
    if not checkpoint_path.exists():
        raise SystemExit(f"no checkpoint at {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"]
    if not any(k.endswith("router.w_gate") for k in state_dict):
        raise SystemExit("this run has no lifecycle-routed graph convolutions")

    mode = args.mode or str(
        (payload.get("config") or {}).get("moge_router_mode", "unknown")
    )
    report = analyse(state_dict, mode)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "router_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    text = _phase_table(report)
    (output_dir / "router_report.txt").write_text(text + "\n", encoding="utf-8")

    collapsed = [name for name, entry in report.items() if entry["collapsed"]]
    print(text)
    print()
    print(f"layers: {len(report)}   collapsed: {len(collapsed)}")
    if collapsed:
        print("  collapsed layers route every lifecycle phase to one operator:")
        for name in collapsed:
            print(f"    {name}")
    print(f"written to {output_dir}")


if __name__ == "__main__":
    main()
