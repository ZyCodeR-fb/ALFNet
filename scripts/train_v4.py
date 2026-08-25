#!/usr/bin/env python
"""The v4 training entry point. Train and validation only, never a test split.

Separate from `main.py` on purpose. That entry point carries the v3 protocol,
the async test mode, the PLES/LCE evaluators and eight objective knobs; adding
v4 to it would make every one of those a branch, and the two protocols are meant
to be separable rather than interleaved.

What this refuses, by name rather than by omission:

* any `test`, `evaluate` or `confirmation` argument -- the confirmation splits
  are read once, by the evaluator, and never during training or selection;
* a config that is not one of the ten final v4 configs;
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
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
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
from experiments.v4_validation_collector import (  # noqa: E402
    cell_count,
    collect_validation_table,
)
from lib.lcbt_v4 import (  # noqa: E402
    SHUFFLE_POLICY,
    SPLIT_FRACTIONS,
    TEACHER_FORCING_POLICY,
    teacher_forcing_ratio,
)
from scripts.v4_utils import sha256_file as _plan_sha256_file  # noqa: E402

CHECKPOINT_SCHEMA = 2
RUN_STATE_SCHEMA = 2
RUN_STATE_NAME = "last.pt"
TIE_RULE = "keep_earlier_epoch"
COMPLETE_MARKER = "COMPLETE"
FAILED_MARKER = "FAILED.json"
LAUNCHER_VERSION = "v2_parallel_tmux"
# Files the launcher owns. They are not evidence that a run produced anything.
LAUNCHER_BOOKKEEPING = frozenset({".lock", FAILED_MARKER})
# Code whose change would make a resumed run a different experiment.
CODE_SOURCES = ("experiments/v4_runner.py", "experiments/v4_validation_collector.py",
                "scripts/train_v4.py", "lib/lcbt_v4.py", "lib/v4_seeds.py")
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


def model_method_contract(session):
    graph = getattr(session.model, "graph_expert", None)
    return {
        "safe_relative_weight": float(getattr(
            session.model, "forecast_safe_relative_weight", 0.0)),
        "safe_relative_floor_policy": getattr(
            session.model, "forecast_safe_relative_floor_policy", None),
        "graph_support_set": getattr(graph, "moge_support_set", None),
        "graph_support_names": list(getattr(
            graph, "resolved_support_names", ())),
    }


def checkpoint_payload(session, config_path, config_sha, epoch, best_epoch,
                       best_validation, optimizer, scheduler, train_report,
                       validation_report, plan=None):
    manifest = json.loads(
        (MANIFEST_DIR / f"{session.dataset}_run_manifest.json").read_text("utf-8"))
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "protocol": "lcbt_v4",
        "dataset": session.dataset,
        "model": session.model_name,
        "model_variant": getattr(session.args, "model_variant", None),
        "seed": session.run_seed,
        "seed_contract": session.seed_contract,
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
            "safe_relative": train_report.get("safe_relative") is not None,
            "model_aux_losses": train_report.get("aux") is not None,
        },
        "model_method_contract": model_method_contract(session),
        "train_report": {key: value for key, value in train_report.items()
                         if key != "model_state"},
        "validation_report": validation_report,
        "best_validation_life_avg_mae": best_validation,
        "early_stopping_metric": "validation life_avg_mae",
        "tie_rule": TIE_RULE,
        "parameter_count": sum(p.numel() for p in session.model.parameters()),
        "plan_sha256": None if plan is None else plan["plan_sha256"],
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


def rng_state():
    """Everything a resumed epoch would otherwise draw differently."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu()
                        if torch.is_tensor(state["torch_cpu"]) else state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
        except (RuntimeError, ValueError):
            # A different device count than the run was saved on. The training
            # schedule and the shuffle are both stateless functions of
            # (seed, epoch), so this affects nothing the protocol depends on --
            # but it is recorded rather than swallowed silently.
            print("warning: could not restore CUDA RNG state (device count "
                  "differs); (seed, epoch)-derived schedules are unaffected",
                  flush=True)


def run_state_payload(session, config_path, config_sha, epoch, best, wait,
                      history, optimizer, scheduler, plan):
    """`last.pt`: everything needed to continue, plus everything to verify it.

    The identity fields are not decoration. Resuming into a different config, a
    different payload or a different protocol produces a model whose first half
    was trained on one experiment and whose second half on another, and nothing
    downstream could tell.
    """
    manifest = json.loads(
        (MANIFEST_DIR / f"{session.dataset}_run_manifest.json").read_text("utf-8"))
    return {
        "schema_version": RUN_STATE_SCHEMA,
        "protocol": "lcbt_v4",
        "dataset": session.dataset,
        "model": session.model_name,
        "model_variant": getattr(session.args, "model_variant", None),
        "seed": session.run_seed,
        "seed_contract": session.seed_contract,
        "completed_epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "best_epoch": best["epoch"],
        "best_validation_life_avg_mae": best["life_avg_mae"],
        "patience_wait": int(wait),
        "history": history,
        "model_state": session.model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "rng_state": rng_state(),
        "config_path": str(Path(config_path).relative_to(ROOT)),
        "config_sha256": config_sha,
        "plan_sha256": None if plan is None else plan["plan_sha256"],
        "code_sha256": {name: _sha256_file(ROOT / name) for name in CODE_SOURCES},
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
        "shuffle_policy": SHUFFLE_POLICY,
        "loss": getattr(session.args, "loss", "mse"),
        "model_method_contract": model_method_contract(session),
        "parameter_count": sum(p.numel() for p in session.model.parameters()),
    }


RESUME_IDENTITY_FIELDS = (
    "protocol", "dataset", "model", "seed", "config_sha256", "plan_sha256",
    "payload_sha256", "unseen_sha256", "manifest_sha256", "static_asset_hash",
    "train_exposure_hash", "validation_exposure_hash", "split_fractions",
    "teacher_forcing_policy", "shuffle_policy", "seed_contract", "loss",
    "parameter_count", "model_method_contract",
)


def load_run_state(run_directory, session, config_path, config_sha, plan):
    """Read `last.pt` and refuse it unless it describes *this* run."""
    path = Path(run_directory) / RUN_STATE_NAME
    if not path.exists():
        raise RunnerRefused(
            f"{run_directory} has no {RUN_STATE_NAME}; there is no epoch "
            "boundary to resume from. Use --restart-incomplete to archive it and "
            "start over")
    try:
        state = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as error:                       # noqa: BLE001 - reported
        raise RunnerRefused(f"{path} could not be read: {error}") from error
    if not isinstance(state, dict) or state.get("schema_version") != RUN_STATE_SCHEMA:
        raise RunnerRefused(
            f"{path} is not a schema {RUN_STATE_SCHEMA} run state")
    expected = run_state_payload(session, config_path, config_sha, -1,
                                 {"epoch": None, "life_avg_mae": None}, 0, [],
                                 build_optimizer(session), None, plan)
    differing = {field: (expected[field], state.get(field))
                 for field in RESUME_IDENTITY_FIELDS
                 if state.get(field) != expected[field]}
    if differing:
        raise RunnerRefused(
            f"{path} describes a different run and may not be resumed into this "
            "one: " + ", ".join(f"{name} expected {want!r} got {got!r}"
                                for name, (want, got) in sorted(differing.items())))
    for field in ("completed_epoch", "next_epoch", "history", "model_state",
                  "optimizer_state"):
        if field not in state:
            raise RunnerRefused(f"{path} is missing {field}")
    return state


def archive_incomplete(run_directory):
    """Move an unfinished run aside. Never delete, never overwrite."""
    run_directory = Path(run_directory)
    if not run_directory.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%S")
    target = (ROOT / "outputs_v4/aborted_runs"
              / f"restart-{run_directory.name}-{stamp}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_directory), str(target))
    print(f"archived the incomplete run to {target}", flush=True)
    return target


def run_artefacts(directory):
    """What the run itself produced, ignoring the launcher's bookkeeping.

    The launcher takes its per-run lock before starting the trainer, so a fresh
    run's directory already contains `.lock` by the time the trainer looks at
    it. Counting that as "an unfinished run is here" made every new run refuse
    to start -- caught by the health phase in ten seconds, which is what the
    health phase is for.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    return [entry for entry in directory.iterdir()
            if entry.name not in LAUNCHER_BOOKKEEPING]


def assert_not_clobbering(run_directory, resume, restart_incomplete=False):
    marker = Path(run_directory) / COMPLETE_MARKER
    if marker.exists():
        raise RunnerRefused(
            f"{run_directory} is marked {COMPLETE_MARKER}; refusing to overwrite "
            "a finished run. Delete it deliberately or choose another output root")
    directory = Path(run_directory)
    if not run_artefacts(directory):
        return
    if restart_incomplete:
        archive_incomplete(directory)
        return
    if not resume:
        raise RunnerRefused(
            f"{run_directory} holds an unfinished run; pass --resume to continue "
            f"it from {RUN_STATE_NAME}, or --restart-incomplete to archive it and "
            "start over. Neither silently overwrites")


def _table_from_best(session, run_directory, best_epoch):
    """Recompute the 33 cells from stored best weights after a resume.

    Used only on resume. The weights in memory are the *last* epoch's, not the
    selected one, so the table has to come from best.pt or it would describe a
    different model from the one the run reports.
    """
    path = Path(run_directory) / "best.pt"
    if not path.exists():
        return None
    live = {name: tensor.detach().clone()
            for name, tensor in session.model.state_dict().items()}
    stored = torch.load(path, weights_only=False, map_location="cpu")
    session.model.load_state_dict(stored["model_state"])
    session.model.to(session.device)
    try:
        return collect_validation_table(session, epoch=int(best_epoch))
    finally:
        session.model.load_state_dict(live)
        session.model.to(session.device)


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


def load_plan_authority(plan_path=None):
    """Load a supported active plan; archived plans are never authorities."""
    if plan_path is None:
        plan_path = ROOT / "configs/paper/v4_five_dataset_plan.json"
    plan_path = Path(plan_path)
    if not plan_path.exists():
        raise RunnerRefused(f"{plan_path} does not exist")
    active = (ROOT / "configs/paper/v4_five_dataset_plan.json").resolve()
    speed_sup5 = (ROOT / "configs/paper/v4_speed_sup5_candidate_plan.json").resolve()
    if plan_path.resolve() not in {active, speed_sup5}:
        raise RunnerRefused(
            f"{plan_path} is neither the active final plan nor a bounded "
            "validation-only candidate plan; archived plans are evidence")
    kind = json.loads(plan_path.read_text("utf-8")).get("kind")
    if kind == "v4_five_dataset_plan":
        from scripts import build_v4_five_dataset_plan as five
        from scripts.build_v4_readiness import ReadinessRefused, assert_ready
        plan = five.assert_plan_still_holds(five.load_plan(plan_path))
        try:
            assert_ready(plan)
        except ReadinessRefused as error:
            raise RunnerRefused(f"{plan_path}: {error}") from error
        return plan
    if kind == "v4_speed_sup5_candidate_plan":
        from scripts import build_v4_speed_sup5_plan as speed_plan
        plan = speed_plan.assert_plan_still_holds(speed_plan.load_plan(plan_path))
        if plan.get("selection", {}).get("split") != "validation" or not \
                plan.get("selection", {}).get("test_forbidden"):
            raise RunnerRefused("speed SUP5 candidate plan is not validation-only")
        return plan
    raise RunnerRefused(f"{plan_path} declares unsupported kind={kind!r}")


def resolve_plan(config_path, plan=None):
    """Check the config belongs to *this* plan. Fails closed on drift.

    `plan` is the already-verified plan the caller loaded. It is not reloaded
    here and never falls back to another one -- that fallback is the bug this
    signature exists to prevent.
    """
    if plan is None:
        plan = load_plan_authority()
    relative = str(Path(config_path).resolve().relative_to(ROOT))
    registered = {entry["path"]: entry for entry in plan["configs"].values()}
    if relative not in registered:
        raise RunnerRefused(
            f"{relative} is not one of the {len(registered)} configs registered "
            f"by plan {plan['plan_sha256'][:16]} "
            f"({plan.get('kind')}): {sorted(registered)}")
    return plan, registered[relative]


def train(config_path, output_root, device="cuda:0", seed=None, resume=False,
          max_epochs=None, max_train_batches=None, max_validation_batches=None,
          batch_size=None, num_workers=None, save=True, plan=None,
          collect_development_table=False, restart_incomplete=False):
    """One v4 run. Returns the run report; writes nothing when `save` is false."""
    config_path = Path(config_path).resolve()
    canonical = {path.resolve() for path in v4_config_paths()}
    known = (canonical if plan is None else {
        (ROOT / entry["path"]).resolve() for entry in plan["configs"].values()
    })
    if config_path not in known:
        raise RunnerRefused(
            f"{config_path} is not one of the {len(known)} v4 configs in "
            f"{CONFIG_DIR.relative_to(ROOT)}")
    config_sha = _sha256_file(config_path)
    plan_entry = None
    if plan is not None or collect_development_table:
        plan, plan_entry = resolve_plan(config_path, plan)
        if plan_entry["sha256"] != config_sha:
            raise RunnerRefused(
                f"{config_path.name} hashes to {config_sha} but plan "
                f"{plan['plan_sha256'][:16]} registered {plan_entry['sha256']}")
    session = build_v4_session(config_path, device=device, run_seed=seed,
                               batch_size=batch_size, num_workers=num_workers)
    if output_root is None:
        # A preflight writes nothing, and what it would have written goes to a
        # temporary directory rather than anywhere a checkpoint is looked for.
        output_root = tempfile.mkdtemp(prefix="v4-preflight-")
    run_directory = resolve_run_directory(output_root, session, config_path, config_sha)
    if save:
        assert_not_clobbering(run_directory, resume, restart_incomplete)

    optimizer = build_optimizer(session)
    scheduler = build_scheduler(session, optimizer)
    epochs = int(max_epochs if max_epochs is not None else session.args.max_epoch)
    patience = int(session.args.patience)

    history, best, best_table = [], {"epoch": None, "life_avg_mae": None}, None
    start_epoch, wait = 0, 0
    if resume and run_artefacts(run_directory):
        state = load_run_state(run_directory, session, config_path, config_sha, plan)
        session.model.load_state_dict(state["model_state"])
        session.model.to(device)
        optimizer = build_optimizer(session)
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        restore_rng_state(state.get("rng_state"))
        history = list(state["history"])
        best = {"epoch": state["best_epoch"],
                "life_avg_mae": state["best_validation_life_avg_mae"]}
        wait = int(state.get("patience_wait", 0))
        start_epoch = int(state["next_epoch"])
        print(f"resumed from {RUN_STATE_NAME}: completed through epoch "
              f"{state['completed_epoch']}, best epoch {best['epoch']} "
              f"({best['life_avg_mae']}), continuing at epoch {start_epoch}",
              flush=True)
        if collect_development_table and best["epoch"] is not None:
            # The table belongs to the selected epoch. After a resume the
            # selected epoch may be one already behind us, so it is rebuilt from
            # best.pt rather than left absent or attributed to the wrong epoch.
            best_table = _table_from_best(session, run_directory, best["epoch"])

    for epoch in range(start_epoch, epochs):
        train_report = train_one_epoch(session, optimizer, epoch,
                                       max_batches=max_train_batches,
                                       loss_name=session.args.loss)
        validation_result = validate(
            session, epoch=epoch, max_batches=max_validation_batches,
            loss_name=session.args.loss,
            return_cell_table=collect_development_table)
        if collect_development_table:
            validation_report, candidate_table = validation_result
        else:
            validation_report, candidate_table = validation_result, None
        candidate = early_stopping_metric(validation_report)
        history.append({"epoch": epoch, "train": train_report,
                        "validation": validation_report})
        # One line per epoch, flushed. A multi-day queue with no output is
        # indistinguishable from a hung one, and the per-epoch validation curve
        # is not recoverable from the checkpoint afterwards.
        print(f"epoch {epoch:3d}  train_base {train_report['base']:.6f}  "
              f"optimize {train_report['optimize']:.6f}  "
              f"val_mae {validation_report['mae']:.4f}  "
              f"val_lifeavg {candidate:.4f}  "
              f"tf_ratio {train_report['ratio']:.3f}  "
              f"({time.strftime('%H:%M:%S')})", flush=True)
        # Strictly less than: a tie keeps the earlier epoch, decided before any
        # number existed and recorded in the checkpoint.
        improved = best["life_avg_mae"] is None or candidate < best["life_avg_mae"]
        wait = 0 if improved else wait + 1
        if improved:
            best = {"epoch": epoch, "life_avg_mae": candidate}
            if collect_development_table:
                # The table and early-stopping score come from the same model
                # forward.  Do not run validation a second time at a best point.
                best_table = candidate_table
            if save:
                atomic_save(
                    checkpoint_payload(session, config_path, config_sha, epoch,
                                       best["epoch"], best["life_avg_mae"],
                                       optimizer, scheduler, train_report,
                                       validation_report, plan),
                    run_directory / "best.pt")
        if scheduler is not None:
            scheduler.step(candidate)
        if save:
            # The epoch boundary, written after best.pt so a crash between the
            # two leaves last.pt describing the older, already-durable state.
            atomic_save(
                run_state_payload(session, config_path, config_sha, epoch, best,
                                  wait, history, optimizer, scheduler, plan),
                run_directory / RUN_STATE_NAME)
        if wait >= patience:
            print(f"early stop at epoch {epoch}; best epoch {best['epoch']} "
                  f"lifeavg {best['life_avg_mae']:.4f}", flush=True)
            break
    report = {
        "dataset": session.dataset, "model": session.model_name,
        "seed": session.run_seed, "config": str(config_path.relative_to(ROOT)),
        "seed_contract": session.seed_contract,
        "config_sha256": config_sha, "run_directory": str(run_directory),
        "epochs_run": len(history), "best_epoch": best["epoch"],
        "best_validation_life_avg_mae": best["life_avg_mae"],
        "early_stopping_metric": "validation life_avg_mae", "tie_rule": TIE_RULE,
        "parameter_count": sum(p.numel() for p in session.model.parameters()),
        "model_method_contract": model_method_contract(session),
        "history": history,
    }
    if plan is not None:
        report["plan_sha256"] = plan["plan_sha256"]
        report["plan_status"] = plan["status"]
    if collect_development_table:
        if best_table is None or cell_count(best_table) != 33:
            raise RunnerRefused(
                "the development table is missing or is not 33 cells; refusing "
                "to report a shrunken table")
        report["development_validation_table"] = best_table
    if save:
        (run_directory / "run_report.json").write_text(
            json.dumps(report, indent=1, default=str, allow_nan=False), "utf-8")
        if collect_development_table:
            (run_directory / "development_validation_table.json").write_text(
                json.dumps(best_table, indent=1, sort_keys=True,
                           allow_nan=False) + "\n", "utf-8")
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
    parser.add_argument("--resume", action="store_true",
                        help=f"continue from {RUN_STATE_NAME} at its next epoch")
    parser.add_argument("--restart-incomplete", dest="restart_incomplete",
                        action="store_true",
                        help="archive an unfinished run and start it over")
    parser.add_argument("--preflight", action="store_true",
                        help="one train batch, one validation batch, no checkpoint")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_validation_batches", type=int, default=None)
    parser.add_argument(
        "--plan", default="configs/paper/v4_five_dataset_plan.json",
        help=("path to the frozen plan this run belongs to, e.g. "
              "configs/paper/v4_five_dataset_plan.json"))
    arguments = parser.parse_args(argv)

    if arguments.preflight:
        # Preflight must be authorised by the same plan as a real run.  Without
        # this, candidate configs are either refused as non-final configs or,
        # worse, never exercise the production plan-membership checks.
        plan = load_plan_authority(arguments.plan)
        report = train(arguments.config, output_root=None, device="cpu",
                       seed=arguments.seed, max_epochs=1, max_train_batches=1,
                       max_validation_batches=1, batch_size=1, num_workers=0,
                       save=False, plan=plan,
                       collect_development_table=False)
        print(json.dumps({key: report[key] for key in
                          ("dataset", "model", "parameter_count", "best_epoch")},
                         indent=1))
        return 0
    plan = load_plan_authority(arguments.plan)
    report = train(arguments.config, arguments.output_root, device=arguments.device,
                   seed=arguments.seed, resume=arguments.resume,
                   max_train_batches=arguments.max_train_batches,
                   max_validation_batches=arguments.max_validation_batches,
                   plan=plan, restart_incomplete=arguments.restart_incomplete,
                   collect_development_table=True)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("history", "development_validation_table")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
