"""
Training functions and helpers
"""

import json
import math
import pickle
import os
import time

import torch
from .test import read_out

from .inductive_epoch import shared_step


def _resolve_return_checkpoint_path(args):
    candidates = []
    if "val" in args.stop_based:
        candidates.extend(
            [
                args.best_checkpoint_path + "best_val_checkpoint.pth",
                args.best_checkpoint_path + "best_train_checkpoint.pth",
            ]
        )
    else:
        candidates.extend(
            [
                args.best_checkpoint_path + "best_train_checkpoint.pth",
                args.best_checkpoint_path + "best_val_checkpoint.pth",
            ]
        )
    for check_point in candidates:
        if os.path.exists(check_point):
            return check_point
    raise FileNotFoundError(f"No checkpoint found in {args.best_checkpoint_path}")


def run_epoch(model, data_loaders, optimizer, scheduler, criterions, args, epoch, train=True):
    metrics = {}
    total_y = {mode: {} for mode in data_loaders.keys()}

    for mode, dataloader in data_loaders.items():
        if "train" in args.stop_based and mode == "val":
            continue
        # you can define your own shared_step function here to set the desired training process
        st = time.time()
        model, mode_metrics, total_y_mode = shared_step(model, dataloader, optimizer, criterions, args, mode, train, epoch)
        total_time = time.time() - st
        print(f"├── {mode} finished, took {total_time:.2f}s")
        if mode == "test":
            metrics.update(
                {
                    "test_always_known_nodes": mode_metrics[0],
                    "test_never_known_nodes ": mode_metrics[1],
                    "test_new_add_nodes     ": mode_metrics[2],
                    "test_failed_nodes      ": mode_metrics[3],
                }
            )
        else:
            metrics.update(
                {
                    f"{mode}_total ": mode_metrics[0],
                    f"{mode}_mask  ": mode_metrics[1],
                    f"{mode}_unmask": mode_metrics[2],
                }
            )
            if len(mode_metrics) > 4:
                metrics.update({f"{mode}_role  ": mode_metrics[4]})
        total_y[mode] = total_y_mode

    if train:
        if "val" in args.stop_based:
            metric_mode = "val"
            if args.stop_based == "val_total":
                metric_name = "val_total "
            elif args.stop_based == "val_role":
                metric_name = "val_role  "
            else:
                metric_name = "val_mask  "
        elif "train" in args.stop_based:
            metric_mode = "train"
            if args.stop_based == "train_total":
                metric_name = "train_total "
            elif args.stop_based == "train_role":
                metric_name = "train_role  "
            else:
                metric_name = "train_mask  "
        save_checkpoint(model, optimizer, args, epoch, metric_mode, metrics[metric_name][args.loss], total_y, metric_name)
        save_checkpoint(model, optimizer, args, epoch, "test", metrics["test_never_known_nodes "][args.loss], total_y, "test_never_known_nodes")
        if args.scheduler == "plateau":
            scheduler.step(metrics[metric_name][args.loss])
        elif args.scheduler == "cosine" or args.scheduler == "step":
            scheduler.step()

        print(f'-> Learning rate: {optimizer.param_groups[0]["lr"]:.5f}')
    return model, metrics, total_y


def better_metric(metric_a, metric_b):
    return metric_a < metric_b


def save_checkpoint(model, optimizer, args, epoch, mode, now_metric, total_y, metric_name):
    checkpoint_path = args.best_checkpoint_path + f"best_{mode}_checkpoint.pth"
    best_metric = getattr(args, f"best_{mode}_metric")
    if not math.isfinite(now_metric):
        print(f"-> Skip saving best {metric_name}: metric is non-finite ({now_metric})")
        return
    try:  # try-except here because checkpoint fname could be too long
        if better_metric(now_metric, best_metric) or epoch == 0:
            setattr(args, f"best_{mode}_metric", now_metric)
            setattr(args, f"best_{mode}_metric_epoch", epoch)
            torch.save(
                {
                    "epoch": epoch,
                    "val_metric": now_metric,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "enc_in": args.enc_in,
                        "raw_enc_in": args.raw_enc_in,
                        "c_out": args.c_out,
                        "lifecycle_training_strategy": args.lifecycle_training_strategy,
                        "lifecycle_keep_random_mask": args.lifecycle_keep_random_mask,
                        "train_init_checkpoint_path": args.train_init_checkpoint_path,
                        "lifecycle_aas_loss_weight": args.lifecycle_aas_loss_weight,
                        "lifecycle_nas_loss_weight": args.lifecycle_nas_loss_weight,
                        "lifecycle_vs_loss_weight": args.lifecycle_vs_loss_weight,
                        "lifecycle_fs_loss_weight": args.lifecycle_fs_loss_weight,
                        "lifecycle_role_features": args.lifecycle_role_features,
                        "lifecycle_history_age_feature": args.lifecycle_history_age_feature,
                        "max_epoch": args.max_epoch,
                        "patience": args.patience,
                        "stop_based": args.stop_based,
                    },
                },
                checkpoint_path,
            )
            args.best_total_y.update({f"{mode}": total_y["test"]})
            print(f"-> New best {metric_name} loss at epoch {epoch}! ({metric_name} loss: {now_metric:.4f})")
    except Exception as e:
        print(e)


def _finite_or_none(value):
    return float(value) if math.isfinite(value) else None


def _write_training_summary(args, finished_epoch, stop_reason):
    summary = {
        "finished_epoch": int(finished_epoch),
        "stop_reason": stop_reason,
        "max_epoch": int(args.max_epoch),
        "patience": int(args.patience),
        "stop_based": args.stop_based,
        "best_train_metric_epoch": int(args.best_train_metric_epoch),
        "best_val_metric_epoch": int(args.best_val_metric_epoch),
        "best_test_metric_epoch": int(args.best_test_metric_epoch),
        "best_train_metric": _finite_or_none(args.best_train_metric),
        "best_val_metric": _finite_or_none(args.best_val_metric),
        "best_test_metric": _finite_or_none(args.best_test_metric),
    }
    summary_path = os.path.join(args.best_checkpoint_path, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"-> Training summary saved to: {summary_path}")


def train_model(model, optimizer, scheduler, data_loaders, criterions, args, wandb=None):

    if args.read_only:
        check_point_path = args.best_checkpoint_path + args.exp_id + "/"
        prediction_path = args.best_prediction_path + args.exp_id + "/"
        with open(prediction_path + "best_total_y.pkl", "rb") as f:
            args.best_total_y = pickle.load(f)
        read_out(data_loaders, check_point_path, criterions, args.exp_id, args)
        args.best_checkpoint_path = check_point_path
        check_point = _resolve_return_checkpoint_path(args)
        best_model_dict = torch.load(check_point, weights_only=False)
        best_epoch = best_model_dict["epoch"]
        print(f"Returning best train model from epoch {best_epoch}")
        model.load_state_dict(best_model_dict["state_dict"])
        return model

    # set file path
    args.best_checkpoint_path = args.best_checkpoint_path + args.exp_id + "/"
    if not os.path.exists(args.best_checkpoint_path):
        os.makedirs(args.best_checkpoint_path)
    args.best_prediction_path = args.best_prediction_path + args.exp_id + "/"
    if not os.path.exists(args.best_prediction_path):
        os.makedirs(args.best_prediction_path)
    args.fig_path = args.best_fig_path + args.exp_id + "/"
    if not os.path.exists(args.fig_path):
        os.makedirs(args.fig_path)

    args.best_val_metric = 1e10
    args.best_val_metric_epoch = -1
    args.best_train_metric = 1e10  # Interpolation / fitting also good to test
    args.best_train_metric_epoch = -1
    args.best_test_metric = 1e10
    args.best_test_metric_epoch = -1
    args.best_total_y = {}  # save best epoch test results
    print("-> Start training...")
    wait_count = 0
    stopped_early = False
    for epoch in range(1, args.max_epoch + 1):
        print(f"-> Epoch {epoch} started...")
        train_dataset = data_loaders["train"].dataset
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        st = time.time()
        _, metrics, _ = run_epoch(model, data_loaders, optimizer, scheduler, criterions, args, epoch)

        # Reset early stopping count if epoch improved
        if "val" in args.stop_based:
            if args.best_val_metric_epoch == epoch or epoch < 20:
                wait_count = 0
            else:
                wait_count += 1
                print(f"-> Early stopping count: {wait_count}/{args.patience}")
        else:
            if args.best_train_metric_epoch == epoch or epoch < 20:
                wait_count = 0
            else:
                wait_count += 1
                print(f"-> Early stopping count: {wait_count}/{args.patience}")

        if wandb is not None:
            log_metrics = {}
            for mode in metrics.keys():
                for metric_name, metric_value in metrics[mode].items():
                    log_metrics[f"{mode}/{metric_name}"] = metric_value
            wandb.log(log_metrics, step=epoch)
            wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch)

        print(f"├── Epoch {epoch} finished, took {time.time() - st:.2f}s")
        for mode in metrics:
            if "train" in mode and "val" in args.stop_based:
                continue
            if "val" in mode and "train" in args.stop_based:
                continue
            description = f"├── {mode} metrics:"
            for metric_name, metric in metrics[mode].items():
                description += f"{metric_name}: {metric:.4f}| "
            print(description)

        if wait_count == args.patience:
            print(f"Early stopping at epoch {epoch}!")
            stopped_early = True
            break  # Exit for loop and do early stopping
    finished_epoch = epoch if args.max_epoch > 0 else 0
    stop_reason = "early_stop" if stopped_early else "max_epoch"
    print("-> Training finished!")
    print(f"-> Training stopped at epoch {finished_epoch}/{args.max_epoch} ({stop_reason})")
    if "train" in args.stop_based:
        print(f"-> Saved best train model checkpoint at epoch {args.best_train_metric_epoch}!")
    elif "val" in args.stop_based:
        print(f"-> Saved best val model checkpoint at epoch {args.best_val_metric_epoch}!")
    print(f"-> Saved best test model checkpoint at epoch {args.best_test_metric_epoch}!")
    print(f"   - Saved to: {args.best_checkpoint_path}")
    _write_training_summary(args, finished_epoch, stop_reason)

    if "val" in args.stop_based and args.best_val_metric_epoch < 0:
        raise RuntimeError("Training finished without any finite validation checkpoint. The training loss likely became NaN or Inf.")
    if "train" in args.stop_based and args.best_train_metric_epoch < 0:
        raise RuntimeError("Training finished without any finite training checkpoint. The training loss likely became NaN or Inf.")

    print("-> save predictions and metrics...")
    with open(args.best_prediction_path + "best_total_y.pkl", "wb") as f:
        (
            args.best_total_y["always_known_nodes"],
            args.best_total_y["never_known_nodes"],
            args.best_total_y["new_add_nodes"],
            args.best_total_y["failed_nodes"],
        ) = data_loaders["test"].dataset.get_know_unknow_nodes()
        pickle.dump(args.best_total_y, f)
    read_out(data_loaders, args.best_checkpoint_path, criterions, args.exp_id, args)
    check_point = _resolve_return_checkpoint_path(args)
    best_model_dict = torch.load(check_point, weights_only=False)
    epoch = best_model_dict["epoch"]
    best_epoch = best_model_dict["epoch"]
    print(f"Returning best train model from epoch {best_epoch}")
    model.load_state_dict(best_model_dict["state_dict"])
    return model
