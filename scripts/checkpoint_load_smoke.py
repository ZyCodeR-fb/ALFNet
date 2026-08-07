"""Can a finished run's checkpoint still be loaded for evaluation, on this machine?

This is the check that the provenance work exists for. The r1 checkpoints were
trained on the server and are evaluated here, and until the schema-2 fix every
one of them was refused: `build_structural_features` sums rows of the adjacency,
a reduction's order depends on the BLAS build, and the resulting 1.31e-07
difference changed every bit of the schema-1 digest.

It exercises the real loader -- `load_pretrained_model`, so the same provenance,
shape and structural checks that `--mode test_async` would run -- and stops
before the test data provider is built. **No test split is read and no metric is
computed**, so this is safe to run at any time; the test split stays unspent.

    conda run -n COMG python scripts/checkpoint_load_smoke.py
    conda run -n COMG python scripts/checkpoint_load_smoke.py \
        --run outputs/ALFNet/gs-r1-s32 --config configs/ALFNet/gs.yaml
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (label, run dir, config, training flags that identify the run)
DEFAULT_CASES = [
    ("gs  (mainline, r1)", "outputs/ALFNet/gs-r1-s32", "configs/ALFNet/gs.yaml"),
    ("fg1 (ablation, r1)", "outputs/ALFNet/fg1-r1-s32", "configs/ALFNet/fg1.yaml"),
    ("g1  (ablation, r1)", "outputs/ALFNet/g1-r1-s32", "configs/ALFNet/g1.yaml"),
    ("gm  (ablation, r1)", "outputs/ALFNet/gm-r1-s32", "configs/ALFNet/gm.yaml"),
    ("MoGERNN baseline, r1", "outputs/MoGERNN/mg-r1-s32",
     "configs/MoGERNN/METRLA_lcbt_v3.yaml"),
]


def load_one(run, config, batch_size):
    import torch

    from args import apply_output_root, initialize_args
    from model.regist_model import load_model_class
    import main as main_module

    run_dir = ROOT / run
    sys.argv = [
        "main.py",
        "-c", config,
        "--batch_size", str(batch_size),
        "--output_root", "./outputs",
        "--run_id", run_dir.name,
        # The evaluation mode, so the loader takes the path test_async takes.
        # It changes which checks run; it does not read any data by itself.
        "--mode", "test_async",
    ]
    args = initialize_args()
    apply_output_root(args)
    args.device = "cpu"
    args.num_workers = 0

    # `--mode test_async` normally binds these from the evaluation manifest.
    # Binding from the run's own training manifest instead keeps the protocol
    # check live without constructing the test dataset.
    manifest = run_dir / "manifests"
    candidates = sorted(manifest.glob("*.json")) if manifest.exists() else []
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        found = payload.get("training_protocol_hash")
        if found:
            args.training_protocol_hash = found
            break

    model = load_model_class(args.model)(args).float()
    model, _path = main_module.load_pretrained_model(model, args)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return args, trainable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--config")
    parser.add_argument("--batch_size", type=int, default=64)
    options = parser.parse_args()

    if options.run and options.config:
        cases = [("requested", options.run, options.config)]
    elif options.run or options.config:
        raise SystemExit("--run and --config must be given together")
    else:
        cases = DEFAULT_CASES

    failures, skipped = [], []
    for label, run, config in cases:
        checkpoint = ROOT / run / "checkpoints" / "best_val_checkpoint.pth"
        if not checkpoint.exists():
            skipped.append(f"{label}: no checkpoint at {run}")
            print(f"{label:<22} SKIP (no checkpoint)")
            continue
        try:
            args, trainable = load_one(run, config, options.batch_size)
            print(f"{label:<22} OK   {args.model}/{args.model_variant} "
                  f"params={trainable:,}")
        except Exception as exc:
            print(f"{label:<22} FAIL {type(exc).__name__}: {exc}")
            failures.append(f"{label}: {exc}")

    print()
    for note in skipped:
        print(f"skipped -- {note}")
    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(f"  - {item}")
        return 1
    if not skipped or len(skipped) < len(cases):
        print("every available checkpoint loaded through the real evaluation path.")
        print("No test split was read; no metric was computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
