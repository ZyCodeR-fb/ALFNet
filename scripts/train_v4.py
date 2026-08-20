#!/usr/bin/env python
"""The v4 training entry point. Train and validation only, never a test split.

Separate from `main.py` on purpose. That entry point carries the v3 protocol,
the async test mode, the PLES/LCE evaluators and eight objective knobs; adding
v4 to it would make every one of those a branch, and the two protocols are meant
to be separable rather than interleaved.

What this refuses, by name rather than by omission:

* any `test`, `evaluate` or `confirmation` argument -- the confirmation splits
  are read once, by the evaluator, and never during training or selection;
* a config that is not one of the eight v4 development-base configs;
* an override of anything but seed, device, output root and the preflight caps.
  Split, target, layout and capacity are properties of the experiment, not of
  the command line that happened to launch it;
* overwriting a run directory already marked COMPLETE.

Checkpoint selection is validation LifeAvg and nothing else. Ties keep the
*earlier* epoch -- fixed here, before any number exists, and written into the
checkpoint so the rule cannot be chosen after the fact.

    python scripts/train_v4.py --config configs/paper/v4/PEMS08_ALFNet.yaml \
        --device cuda:0 --seed 32 --output_root ./outputs_v4
    python scripts/train_v4.py --config ... --preflight     # 1 batch, CPU, no checkpoint
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v4_runner import (  # noqa: E402
    CONFIG_DIR,
    MANIFEST_DIR,
    RunnerRefused,
    build_v4_session,
    early_stopping_metric,
    train_one_epoch,
    validate,
)
from lib.lcbt_v4 import (  # noqa: E402
    SPLIT_FRACTIONS,
    TEACHER_FORCING_POLICY,
    teacher_forcing_ratio,
)

CHECKPOINT_SCHEMA = 1
TIE_RULE = "keep_earlier_epoch"
COMPLETE_MARKER = "COMPLETE"
# Arguments whose presence means the caller is asking this entry point for
# something it does not do. Refused by name so the refusal survives a rename of
# whatever module would otherwise have implemented them.
FORBIDDEN_ARGUMENT_SUBSTRINGS = ("test", "evaluate", "confirmation")


def v4_config_paths():
    return sorted(path for path in CONFIG_DIR.glob("*.yaml"))


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def assert_no_forbidden_arguments(argv):
    """A test-shaped flag is an error here, not something to ignore."""
    offending = [token for token in argv if token.startswith("-") and any(
        word in token.lower() for word in FORBIDDEN_ARGUMENT_SUBSTRINGS)]
    if offending:
        raise RunnerRefused(
            f"{offending} asks the v4 training entry point for evaluation; the "
            "confirmation splits are read once, by the evaluator, and never "
            "during training or model selection")


def resolve_run_directory(output_root, session, config_path, config_sha):
    """dataset / model / config hash / seed. Identity, not a timestamp."""
    return (Path(output_root) / session.dataset / session.model_name
            / config_sha[:16] / f"seed{session.run_seed}")


def checkpoint_payload(session, config_path, config_sha, epoch, best_epoch,
                       best_validation, optimizer, scheduler, train_report,
                       validation_report):
    manifest = json.loads(
        (MANIFEST_DIR / f"{session.dataset}_run_manifest.json").read_text("utf-8"))
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "protocol": "lcbt_v4",
        "dataset": session.dataset,
        "model": session.model_name,
        "model_variant": getattr(session.args, "model_variant", None),
        "seed": session.run_seed,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "model_state": session.model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "config_path": str(Path(config_path).relative_to(ROOT)),
        "config_sha256": config_sha,
        "git_commit": _git_commit(),
        "payload_sha256": manifest["payload"]["sha256"],
        "unseen_sha256": manifest["unseen"]["sha256"],
        "manifest_sha256": _sha256_file(
            MANIFEST_DIR / f"{session.dataset}_run_manifest.json"),
        "static_asset_hash": session.train_context.static_asset_hash(),
        "train_exposure_hash": session.train_context.exposure_hash(),
        "validation_exposure_hash": session.validation_context.exposure_hash(),
        "scaler": dict(session.scaler),
        "split_fractions": list(SPLIT_FRACTIONS),
        "teacher_forcing_policy": TEACHER_FORCING_POLICY,
        "teacher_forcing_ratio": teacher_forcing_ratio(epoch),
        "loss": getattr(session.args, "loss", "mse"),
        "objective_terms": {
            "base": "masked MSE over loss_mask & isfinite",
            "role_balanced_mix": train_report.get("role_mix") is not None,
            "model_aux_losses": train_report.get("aux") is not None,
        },
        "train_report": {key: value for key, value in train_report.items()
                         if key != "model_state"},
        "validation_report": validation_report,
        "best_validation_life_avg_mae": best_validation,
        "early_stopping_metric": "validation life_avg_mae",
        "tie_rule": TIE_RULE,
        "parameter_count": sum(p.numel() for p in session.model.parameters()),
    }


def atomic_save(payload, path):
    """Write beside the target, fsync, then rename. A half-written checkpoint
    that loads is worse than one that is absent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    os.close(handle)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as opened:
            os.fsync(opened.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def assert_not_clobbering(run_directory, resume):
    marker = Path(run_directory) / COMPLETE_MARKER
    if marker.exists():
        raise RunnerRefused(
            f"{run_directory} is marked {COMPLETE_MARKER}; refusing to overwrite "
            "a finished run. Delete it deliberately or choose another output root")
    if Path(run_directory).exists() and any(Path(run_directory).iterdir()) and not resume:
        raise RunnerRefused(
            f"{run_directory} holds an unfinished run; pass --resume to continue "
            "it rather than silently overwriting")


def build_optimizer(session):
    """Each model's existing optimiser. No learning rate is changed here."""
    args = session.args
    groups = (session.model.get_optimizer_param_groups(args.lr)
              if hasattr(session.model, "get_optimizer_param_groups")
              else [{"params": list(session.model.parameters()), "lr": args.lr}])
    return torch.optim.AdamW(groups, lr=args.lr,
                             weight_decay=float(args.weight_decay))


def build_scheduler(session, optimizer):
    setting = str(getattr(session.args, "scheduler", "None"))
    if setting in ("None", "none", ""):
        return None
    if setting == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            patience=int(getattr(session.args, "scheduler_patience", 1)))
    raise RunnerRefused(f"unknown scheduler {setting!r}; none is added here")


def train(config_path, output_root, device="cuda:0", seed=None, resume=False,
          max_epochs=None, max_train_batches=None, max_validation_batches=None,
          batch_size=None, num_workers=None, save=True):
    """One v4 run. Returns the run report; writes nothing when `save` is false."""
    config_path = Path(config_path).resolve()
    if config_path not in {path.resolve() for path in v4_config_paths()}:
        raise RunnerRefused(
            f"{config_path} is not one of the eight v4 configs in "
            f"{CONFIG_DIR.relative_to(ROOT)}")
    config_sha = _sha256_file(config_path)
    session = build_v4_session(config_path, device=device, run_seed=seed,
                               batch_size=batch_size, num_workers=num_workers)
    if output_root is None:
        # A preflight writes nothing, and what it would have written goes to a
        # temporary directory rather than anywhere a checkpoint is looked for.
        output_root = tempfile.mkdtemp(prefix="v4-preflight-")
    run_directory = resolve_run_directory(output_root, session, config_path, config_sha)
    if save:
        assert_not_clobbering(run_directory, resume)

    optimizer = build_optimizer(session)
    scheduler = build_scheduler(session, optimizer)
    epochs = int(max_epochs if max_epochs is not None else session.args.max_epoch)
    patience = int(session.args.patience)

    history, best = [], {"epoch": None, "life_avg_mae": None}
    for epoch in range(epochs):
        train_report = train_one_epoch(session, optimizer, epoch,
                                       max_batches=max_train_batches,
                                       loss_name=session.args.loss)
        validation_report = validate(session, epoch=epoch,
                                     max_batches=max_validation_batches,
                                     loss_name=session.args.loss)
        candidate = early_stopping_metric(validation_report)
        history.append({"epoch": epoch, "train": train_report,
                        "validation": validation_report})
        # Strictly less than: a tie keeps the earlier epoch, decided before any
        # number existed and recorded in the checkpoint.
        improved = best["life_avg_mae"] is None or candidate < best["life_avg_mae"]
        if improved:
            best = {"epoch": epoch, "life_avg_mae": candidate}
            if save:
                atomic_save(
                    checkpoint_payload(session, config_path, config_sha, epoch,
                                       best["epoch"], best["life_avg_mae"],
                                       optimizer, scheduler, train_report,
                                       validation_report),
                    run_directory / "best.pt")
        if scheduler is not None:
            scheduler.step(candidate)
        if epoch - best["epoch"] >= patience:
            break
    report = {
        "dataset": session.dataset, "model": session.model_name,
        "seed": session.run_seed, "config": str(config_path.relative_to(ROOT)),
        "config_sha256": config_sha, "run_directory": str(run_directory),
        "epochs_run": len(history), "best_epoch": best["epoch"],
        "best_validation_life_avg_mae": best["life_avg_mae"],
        "early_stopping_metric": "validation life_avg_mae", "tie_rule": TIE_RULE,
        "parameter_count": sum(p.numel() for p in session.model.parameters()),
        "history": history,
    }
    if save:
        (run_directory / "run_report.json").write_text(
            json.dumps(report, indent=1, default=str), "utf-8")
        (run_directory / COMPLETE_MARKER).write_text(
            json.dumps({"best_epoch": best["epoch"]}), "utf-8")
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    assert_no_forbidden_arguments(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help=f"one of the v4 configs in {CONFIG_DIR.relative_to(ROOT)}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", default="./outputs_v4")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true",
                        help="one train batch, one validation batch, no checkpoint")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_validation_batches", type=int, default=None)
    arguments = parser.parse_args(argv)

    if arguments.preflight:
        report = train(arguments.config, output_root=None, device="cpu",
                       seed=arguments.seed, max_epochs=1, max_train_batches=1,
                       max_validation_batches=1, batch_size=1, num_workers=0,
                       save=False)
        print(json.dumps({key: report[key] for key in
                          ("dataset", "model", "parameter_count", "best_epoch")},
                         indent=1))
        return 0
    report = train(arguments.config, arguments.output_root, device=arguments.device,
                   seed=arguments.seed, resume=arguments.resume,
                   max_train_batches=arguments.max_train_batches,
                   max_validation_batches=arguments.max_validation_batches)
    print(json.dumps({key: value for key, value in report.items()
                      if key != "history"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
