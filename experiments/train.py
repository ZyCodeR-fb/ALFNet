"""LCBT training and lifecycle-aware checkpoint selection."""

import json
import math
import os
import statistics
import time

import torch

from .inductive_epoch import shared_step


def _checkpoint_path(args):
    path = os.path.join(args.best_checkpoint_path, "best_val_checkpoint.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No validation checkpoint found at {path}")
    return path


def _accept_checkpoint(now_metric, best_metric, now_worst, best_worst):
    if not math.isfinite(best_metric):
        return True
    tolerance = max(abs(best_metric), 1e-12) * 0.001
    if now_metric < best_metric - tolerance:
        return True
    return (
        abs(now_metric - best_metric) <= tolerance
        and math.isfinite(now_worst)
        and now_worst < best_worst
    )


def _save_checkpoint(model, optimizer, args, epoch, lifecycle_metrics):
    lifecycle_mae = float(lifecycle_metrics["mae"])
    worst_group_mae = float(lifecycle_metrics["worst_group_mae"])
    if not math.isfinite(lifecycle_mae) or not math.isfinite(worst_group_mae):
        print("-> Skip checkpoint: val_lifecycle_mae or worst_group_mae is non-finite")
        return False
    if not _accept_checkpoint(
        lifecycle_mae,
        args.best_val_metric,
        worst_group_mae,
        args.best_val_worst_group_metric,
    ):
        return False

    args.best_val_metric = lifecycle_mae
    args.best_val_worst_group_metric = worst_group_mae
    args.best_val_metric_epoch = int(epoch)
    args.best_val_lifecycle_metrics = {
        key: float(value)
        for key, value in lifecycle_metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }
    torch.save(
        {
            "epoch": int(epoch),
            "val_metric": lifecycle_mae,
            "val_worst_group_metric": worst_group_mae,
            "val_lifecycle_metrics": lifecycle_metrics,
            "training_protocol_hash": args.training_protocol_hash,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "enc_in": args.enc_in,
                "raw_enc_in": args.raw_enc_in,
                "c_out": args.c_out,
                "lifecycle_training_strategy": args.lifecycle_training_strategy,
                "lifecycle_role_features": args.lifecycle_role_features,
                "lifecycle_history_age_feature": args.lifecycle_history_age_feature,
                "lifecycle_observation_mask_feature": args.lifecycle_observation_mask_feature,
                "lifecycle_age_mode": args.lifecycle_age_mode,
                "max_epoch": args.max_epoch,
                "patience": args.patience,
                "stop_based": "val_lifecycle",
            },
        },
        _checkpoint_path_for_save(args),
    )
    print(
        f"-> New best val_lifecycle at epoch {epoch}: "
        f"MAE={lifecycle_mae:.6f}, worst_group_MAE={worst_group_mae:.6f}"
    )
    return True


def _checkpoint_path_for_save(args):
    return os.path.join(args.best_checkpoint_path, "best_val_checkpoint.pth")


def run_epoch(model, data_loaders, optimizer, scheduler, criterions, args, epoch):
    metrics = {}
    timings = {}
    for mode in ("train", "val"):
        start = time.time()
        model, mode_metrics = shared_step(
            model,
            data_loaders[mode],
            optimizer,
            criterions,
            args,
            mode,
            train=True,
            epoch=epoch,
        )
        metrics[mode] = mode_metrics
        timings[mode] = time.time() - start
        print(f"-> {mode} finished, took {timings[mode]:.2f}s")
    args.epoch_seconds.append(timings)

    improved = _save_checkpoint(
        model,
        optimizer,
        args,
        epoch,
        metrics["val"]["lifecycle"],
    )
    if improved:
        args.seconds_to_best = sum(
            entry["train"] + entry["val"] for entry in args.epoch_seconds
        )
    routing = metrics["val"].get("routing") or {}
    if routing:
        # Printed every epoch so router collapse is caught while it is happening
        # rather than after a full run: a flat beta across availability groups
        # or a route accuracy stuck at the base rate is the signature.
        args.last_routing = routing
        if improved:
            args.best_routing = dict(routing)
        print("-> routing: " + " | ".join(
            f"{name}={value:.4f}" for name, value in sorted(routing.items())
        ))
    monitored = metrics["val"]["lifecycle"]["mae"]
    if args.scheduler == "plateau":
        scheduler.step(monitored)
    elif args.scheduler in {"cosine", "step"}:
        scheduler.step()
    print(f'-> Learning rate: {optimizer.param_groups[0]["lr"]:.5f}')
    return model, metrics, improved


def _finite_or_none(value):
    return float(value) if math.isfinite(value) else None


def _efficiency_report(args, data_loaders):
    """Cost has to be recorded per run, not inferred from wall-clock afterwards.

    Batch size, optimizer steps and model size differ between the models being
    compared, so "it took N hours" is not a comparison. These fields let the
    paper report samples/s, steps/s and peak memory alongside accuracy.
    """
    train_loader, val_loader = data_loaders["train"], data_loaders["val"]
    train_seconds = [entry["train"] for entry in args.epoch_seconds]
    val_seconds = [entry["val"] for entry in args.epoch_seconds]
    total = sum(train_seconds) + sum(val_seconds)
    steps = len(train_loader)
    samples = len(train_loader.dataset)
    mean_train = statistics.fmean(train_seconds) if train_seconds else float("nan")
    peak_bytes = None
    if torch.cuda.is_available():
        peak_bytes = int(torch.cuda.max_memory_allocated())
    return {
        "trainable_parameters": int(getattr(args, "num_params", 0)),
        "batch_size": int(args.batch_size),
        "optimizer_steps_per_epoch": int(steps),
        "train_samples_per_epoch": int(samples),
        "validation_records": int(len(val_loader.dataset)),
        "epochs_run": len(train_seconds),
        "mean_train_seconds_per_epoch": mean_train,
        "mean_val_seconds_per_epoch": (
            statistics.fmean(val_seconds) if val_seconds else float("nan")
        ),
        "total_seconds": total,
        "seconds_to_best_checkpoint": _finite_or_none(args.seconds_to_best),
        "train_samples_per_second": samples / mean_train if mean_train else None,
        "optimizer_steps_per_second": steps / mean_train if mean_train else None,
        "peak_gpu_memory_bytes": peak_bytes,
        "device": str(args.device),
    }


def _write_training_summary(args, finished_epoch, stop_reason, data_loaders):
    summary = {
        "finished_epoch": int(finished_epoch),
        "stop_reason": stop_reason,
        "max_epoch": int(args.max_epoch),
        "patience": int(args.patience),
        "stop_based": "val_lifecycle",
        "best_val_metric_epoch": int(args.best_val_metric_epoch),
        "best_val_metric": _finite_or_none(args.best_val_metric),
        "best_val_worst_group_metric": _finite_or_none(
            args.best_val_worst_group_metric
        ),
        "best_val_lifecycle_metrics": args.best_val_lifecycle_metrics,
        "validation_group_volatility": args.validation_group_volatility,
        "routing_diagnostics": getattr(args, "best_routing", {}),
        "efficiency": _efficiency_report(args, data_loaders),
        "training_protocol_hash": args.training_protocol_hash,
    }
    path = os.path.join(args.best_checkpoint_path, "training_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"-> Training summary saved to: {path}")


def _print_metrics(metrics):
    for mode, groups in metrics.items():
        for group_name, values in groups.items():
            if not values:
                continue
            description = f"-> {mode}_{group_name}: "
            description += " | ".join(
                f"{name}={value:.4f}" for name, value in values.items()
            )
            print(description)


def train_model(model, optimizer, scheduler, data_loaders, criterions, args, wandb=None):
    if (
        args.lifecycle_training_strategy != "lcbt_v3"
        or args.stop_based != "val_lifecycle"
    ):
        raise ValueError("The streamlined trainer requires LCBT and val_lifecycle")
    if set(data_loaders) != {"train", "val"}:
        raise ValueError("Training must receive only train and validation loaders")

    # With --output_root the run folder already identifies the run, so appending
    # the 176-character exp_id again would double the path length for no gain.
    if not getattr(args, "paths_are_run_scoped", False):
        args.best_checkpoint_path = os.path.join(args.best_checkpoint_path, args.exp_id)
        args.best_prediction_path = os.path.join(args.best_prediction_path, args.exp_id)
    if getattr(args, "paths_are_run_scoped", False):
        args.fig_path = args.best_fig_path
    else:
        args.fig_path = os.path.join(args.best_fig_path, args.exp_id)
    for path in (args.best_checkpoint_path, args.best_prediction_path, args.fig_path):
        os.makedirs(path, exist_ok=True)

    args.best_val_metric = float("inf")
    args.best_val_worst_group_metric = float("inf")
    args.best_val_metric_epoch = -1
    args.best_val_lifecycle_metrics = {}
    args.epoch_seconds = []
    args.seconds_to_best = float("nan")
    args.validation_group_history = {}
    args.validation_group_volatility = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    args.training_protocol_hash = getattr(
        data_loaders["train"].dataset,
        "training_protocol_hash",
        None,
    )
    print(f"-> Start {args.lifecycle_training_strategy} training...")

    wait_count = 0
    stopped_early = False
    finished_epoch = 0
    for epoch in range(1, args.max_epoch + 1):
        finished_epoch = epoch
        print(f"-> Epoch {epoch} started...")
        train_dataset = data_loaders["train"].dataset
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        start = time.time()
        _, metrics, improved = run_epoch(
            model,
            data_loaders,
            optimizer,
            scheduler,
            criterions,
            args,
            epoch,
        )
        for name in ("unseen_mae", "steady_mae", "join_mae", "fault_mae"):
            value = metrics["val"]["lifecycle"].get(name)
            if value is not None and math.isfinite(value):
                args.validation_group_history.setdefault(name, []).append(float(value))
        wait_count = 0 if improved else wait_count + 1
        if not improved:
            print(f"-> Early stopping count: {wait_count}/{args.patience}")
        _print_metrics(metrics)

        if wandb is not None:
            flat = {}
            for mode, groups in metrics.items():
                for group_name, values in groups.items():
                    for name, value in values.items():
                        flat[f"{mode}/{group_name}/{name}"] = value
            flat["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(flat, step=epoch)
        print(f"-> Epoch {epoch} finished, took {time.time() - start:.2f}s")
        if wait_count >= args.patience:
            print(f"Early stopping at epoch {epoch}!")
            stopped_early = True
            break

    if args.best_val_metric_epoch < 0:
        raise RuntimeError("Training finished without a finite lifecycle checkpoint")
    # Reported, not used for selection: a group whose score swings between
    # epochs makes the checkpoint metric noisier than its value suggests.
    args.validation_group_volatility = {
        name: {
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "max_epoch_to_epoch_jump": (
                max(abs(b - a) for a, b in zip(values, values[1:])) if len(values) > 1 else 0.0
            ),
        }
        for name, values in args.validation_group_history.items()
    }
    stop_reason = "early_stop" if stopped_early else "max_epoch"
    print(f"-> Training stopped at epoch {finished_epoch}/{args.max_epoch} ({stop_reason})")
    print(f"-> Best val_lifecycle checkpoint: epoch {args.best_val_metric_epoch}")
    _write_training_summary(args, finished_epoch, stop_reason, data_loaders)

    checkpoint = torch.load(_checkpoint_path(args), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model
