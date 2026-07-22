"""LCBT v2 training and lifecycle-aware checkpoint selection."""

import json
import math
import os
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
                "lifecycle_training_strategy": "lcbt_v2",
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
        print(f"├── {mode} finished, took {time.time() - start:.2f}s")

    improved = _save_checkpoint(
        model,
        optimizer,
        args,
        epoch,
        metrics["val"]["lifecycle"],
    )
    monitored = metrics["val"]["lifecycle"]["mae"]
    if args.scheduler == "plateau":
        scheduler.step(monitored)
    elif args.scheduler in {"cosine", "step"}:
        scheduler.step()
    print(f'-> Learning rate: {optimizer.param_groups[0]["lr"]:.5f}')
    return model, metrics, improved


def _finite_or_none(value):
    return float(value) if math.isfinite(value) else None


def _write_training_summary(args, finished_epoch, stop_reason):
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
            description = f"├── {mode}_{group_name}: "
            description += " | ".join(
                f"{name}={value:.4f}" for name, value in values.items()
            )
            print(description)


def train_model(model, optimizer, scheduler, data_loaders, criterions, args, wandb=None):
    if args.lifecycle_training_strategy != "lcbt_v2" or args.stop_based != "val_lifecycle":
        raise ValueError("The streamlined trainer requires LCBT v2 and val_lifecycle")
    if set(data_loaders) != {"train", "val"}:
        raise ValueError("Training must receive only train and validation loaders")

    args.best_checkpoint_path = os.path.join(args.best_checkpoint_path, args.exp_id)
    args.best_prediction_path = os.path.join(args.best_prediction_path, args.exp_id)
    args.fig_path = os.path.join(args.best_fig_path, args.exp_id)
    for path in (args.best_checkpoint_path, args.best_prediction_path, args.fig_path):
        os.makedirs(path, exist_ok=True)

    args.best_val_metric = float("inf")
    args.best_val_worst_group_metric = float("inf")
    args.best_val_metric_epoch = -1
    args.best_val_lifecycle_metrics = {}
    args.training_protocol_hash = getattr(
        data_loaders["train"].dataset,
        "training_protocol_hash",
        None,
    )
    print("-> Start LCBT v2 training...")

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
        print(f"├── Epoch {epoch} finished, took {time.time() - start:.2f}s")
        if wait_count >= args.patience:
            print(f"Early stopping at epoch {epoch}!")
            stopped_early = True
            break

    if args.best_val_metric_epoch < 0:
        raise RuntimeError("Training finished without a finite lifecycle checkpoint")
    stop_reason = "early_stop" if stopped_early else "max_epoch"
    print(f"-> Training stopped at epoch {finished_epoch}/{args.max_epoch} ({stop_reason})")
    print(f"-> Best val_lifecycle checkpoint: epoch {args.best_val_metric_epoch}")
    _write_training_summary(args, finished_epoch, stop_reason)

    checkpoint = torch.load(_checkpoint_path(args), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model
