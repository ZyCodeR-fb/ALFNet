import os
import hashlib
import json

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch import nn

from args import initialize_args
from data_provider.data_factory import data_provider
from experiments import async_lifecycle_inference, inference_model, train_model
from model.regist_model import model_dict
from utils.evaluation import MSEPinballLoss, MultiPinBallLoss, SlitPhyLoss, masked_gaussian_nll_loss, rmse_loss
from utils.logtool import print_config, print_header, print_train, set_wandb
from utils.report_results import load_prediction, report_error


def _format_float_token(value):
    token = f"{value:.0e}" if value != 0 and (abs(value) <= 1e-3 or abs(value) >= 1e2) else f"{value:g}"
    return token.replace("+0", "").replace("+", "").replace("-0", "-")


def _build_train_signature(args):
    expert_bits = f"{int(args.mean_expert)}{int(args.weight_expert)}{int(args.max_expert)}{int(args.min_expert)}{int(args.diffusion_expert)}"
    warm_start_path = os.path.normpath(args.train_init_checkpoint_path) if args.train_init_checkpoint_path else ""
    model_config_path = os.path.normpath(args.model_config_path) if args.model_config_path else ""
    signature_payload = {
        "model_variant": args.model_variant,
        "model_recipe": args.model_recipe,
        "model_config_path": model_config_path,
        "model_family_cfg": getattr(args, "model_family_cfg", {}),
        "d_model": args.d_model,
        "k_hop": args.k_hop,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "activation": args.activation,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "scheduler_patience": args.scheduler_patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epoch": args.max_epoch,
        "patience": args.patience,
        "stop_based": args.stop_based,
        "num_used_experts": args.num_used_experts,
        "expert_bits": expert_bits,
        "train_init_checkpoint_path": warm_start_path,
        "lifecycle_aas_loss_weight": args.lifecycle_aas_loss_weight,
        "lifecycle_nas_loss_weight": args.lifecycle_nas_loss_weight,
        "lifecycle_vs_loss_weight": args.lifecycle_vs_loss_weight,
        "lifecycle_fs_loss_weight": args.lifecycle_fs_loss_weight,
    }
    signature_hash = hashlib.md5(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    readable = (
        f"dm{args.d_model}-kh{args.k_hop}-"
        f"lr{_format_float_token(args.lr)}-wd{_format_float_token(args.weight_decay)}-"
        f"bs{args.batch_size}-ws{int(bool(args.train_init_checkpoint_path))}"
    )
    return f"{readable}-cfg{signature_hash}"


def build_exp_id(args):
    exp_id = (
        f"{args.input_len}-{args.pred_len}-{args.look_back}-{args.slide_step}-"
        f"{args.input_features}-{args.target_features}-t_mark-{args.t_mark}-"
        f"pos_mark-{args.pos_mark}-seed-{args.seed}-"
        f"{_build_train_signature(args)}"
    )
    if args.unknown_nodes_path:
        per = args.unknown_nodes_path.split("/")[-1].split("_")[-1].split(".")[0]
        if "%" in per:
            exp_id = f"{args.dloader_name}-{per}-{exp_id}"
        else:
            exp_id = f"{args.dloader_name}-{exp_id}"

    if args.model == "woMoERNN":
        if args.mean_expert and args.weight_expert and args.max_expert and args.min_expert and args.diffusion_expert:
            model_wo = "avg"
        elif args.mean_expert:
            model_wo = "mean"
        elif args.weight_expert:
            model_wo = "weight"
        elif args.max_expert:
            model_wo = "max"
        elif args.min_expert:
            model_wo = "min"
        elif args.diffusion_expert:
            model_wo = "diffusion"
        else:
            raise NotImplementedError
        exp_prefix = f"{args.model}-{model_wo}"
    else:
        exp_prefix = args.model

    if args.model_variant != "default":
        exp_prefix = f"{exp_prefix}-var-{args.model_variant}"
    if args.model_recipe != "base":
        exp_prefix = f"{exp_prefix}-recipe-{args.model_recipe}"

    use_lifecycle_train_prefix = args.train_for_async_lifecycle or (
        args.test_for_async_lifecycle
        and (
            args.lifecycle_role_features
            or args.lifecycle_history_age_feature
            or args.lifecycle_training_strategy != "v1"
            or args.lifecycle_keep_random_mask
        )
    )
    if use_lifecycle_train_prefix:
        strategy_prefix = "" if args.lifecycle_training_strategy == "v1" else f"strategy-{args.lifecycle_training_strategy}-"
        exp_prefix = (
            f"{exp_prefix}-lifecycle_train-"
            f"{strategy_prefix}"
            f"sampling-{args.lifecycle_event_sampling}-"
            f"role-{int(args.lifecycle_role_features)}-"
            f"age-{int(args.lifecycle_history_age_feature)}-"
            f"randmask-{int(args.lifecycle_keep_random_mask)}"
        )
    elif args.lifecycle_role_features or args.lifecycle_history_age_feature:
        exp_prefix = (
            f"{exp_prefix}-lifecycle_input-"
            f"role-{int(args.lifecycle_role_features)}-"
            f"age-{int(args.lifecycle_history_age_feature)}"
        )
    return f"{exp_prefix}-{exp_id}"


def _validate_checkpoint_compatibility(checkpoint, args):
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        return
    expected_pairs = {
        "enc_in": args.enc_in,
        "raw_enc_in": args.raw_enc_in,
        "c_out": args.c_out,
        "lifecycle_training_strategy": args.lifecycle_training_strategy,
        "lifecycle_keep_random_mask": args.lifecycle_keep_random_mask,
        "lifecycle_role_features": args.lifecycle_role_features,
        "lifecycle_history_age_feature": args.lifecycle_history_age_feature,
    }
    mismatches = []
    for key, expected in expected_pairs.items():
        if key in checkpoint_config and checkpoint_config[key] != expected:
            mismatches.append(f"{key}: checkpoint={checkpoint_config[key]} current={expected}")
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Checkpoint lifecycle/input configuration does not match the current CLI flags. "
            f"Please reuse the same lifecycle input switches during async test. {mismatch_text}"
        )


def _validate_checkpoint_shape_compatibility(checkpoint, args):
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        return
    expected_pairs = {
        "enc_in": args.enc_in,
        "raw_enc_in": args.raw_enc_in,
        "c_out": args.c_out,
    }
    mismatches = []
    for key, expected in expected_pairs.items():
        if key in checkpoint_config and checkpoint_config[key] != expected:
            mismatches.append(f"{key}: checkpoint={checkpoint_config[key]} current={expected}")
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(f"Checkpoint model dimensions are incompatible with the current training configuration. {mismatch_text}")


def _load_model_state(model, loaded, args, validator):
    if isinstance(loaded, dict) and "state_dict" in loaded:
        validator(loaded, args)
        state_dict = _patch_compatible_state_dict(model, loaded["state_dict"])
        model.load_state_dict(state_dict)
        return model
    return loaded


def _patch_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    compatible_state = dict(state_dict)
    optional_runtime_buffers = {
        "graph_recovery_expert_role_scale",
        "role_residual_scale",
        "role_graph_prior",
        "role_graph_alpha_floor",
        "pstm_mae_role_scale",
    }
    for key in optional_runtime_buffers:
        if key not in compatible_state and key in model_state:
            compatible_state[key] = model_state[key]
    return compatible_state


_CHECKPOINT_VARIANT_ALIASES = {
    "rf_s_skan": ["rf_s_p2groc05sourcekan"],
    "rf_s_smh": ["rf_s_p2groc05sourcemixhop"],
    "rf_s_skan_smh": ["rf_s_p2groc05sourcekansourcemixhop"],
    "rf_s_smh_bf": ["rf_s_p2groc05sourcemixhopspecbf"],
    "rf_s_skan_smh_bf": ["rf_s_p2groc05sourcekansourcemixhopspecbf"],
}


def _variant_aliases(model_variant):
    aliases = [model_variant]
    aliases.extend(_CHECKPOINT_VARIANT_ALIASES.get(model_variant, []))
    for short_name, long_names in _CHECKPOINT_VARIANT_ALIASES.items():
        if model_variant in long_names:
            aliases.append(short_name)
    return list(dict.fromkeys(aliases))


def _checkpoint_files(checkpoint_dir):
    return [
        os.path.join(checkpoint_dir, "best_val_checkpoint.pth"),
        os.path.join(checkpoint_dir, "best_train_checkpoint.pth"),
        os.path.join(checkpoint_dir, "best_test_checkpoint.pth"),
    ]


def _select_checkpoint_file(checkpoint_dir):
    for checkpoint_file in _checkpoint_files(checkpoint_dir):
        if os.path.exists(checkpoint_file):
            return checkpoint_file
    return None


def _exp_id_without_cfg_hash(exp_id):
    marker_index = exp_id.rfind("-cfg")
    if marker_index == -1:
        return exp_id
    return exp_id[:marker_index]


def _compatible_checkpoint_prefixes(args):
    base_prefix = _exp_id_without_cfg_hash(args.exp_id)
    variant_marker = f"var-{args.model_variant}"
    prefixes = [base_prefix]
    if variant_marker in base_prefix:
        for variant_alias in _variant_aliases(args.model_variant):
            prefixes.append(base_prefix.replace(variant_marker, f"var-{variant_alias}", 1))
    return list(dict.fromkeys(prefixes))


def _find_compatible_checkpoint_dir(args, requested_checkpoint_dir):
    root_dir = args.best_checkpoint_path
    if not os.path.isdir(root_dir):
        return None

    requested_checkpoint_dir = os.path.normpath(requested_checkpoint_dir)
    prefixes = _compatible_checkpoint_prefixes(args)
    candidates = []
    for dirname in os.listdir(root_dir):
        checkpoint_dir = os.path.join(root_dir, dirname)
        if not os.path.isdir(checkpoint_dir):
            continue
        if os.path.normpath(checkpoint_dir) == requested_checkpoint_dir:
            continue
        checkpoint_file = _select_checkpoint_file(checkpoint_dir)
        if checkpoint_file is None:
            continue
        if not any(dirname.startswith(prefix) for prefix in prefixes):
            continue
        candidates.append((os.path.getmtime(checkpoint_file), checkpoint_dir, checkpoint_file))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _load_checkpoint_from_dir(args, checkpoint_dir):
    checkpoint_file = _select_checkpoint_file(checkpoint_dir)
    if checkpoint_file is None:
        fallback_dir = _find_compatible_checkpoint_dir(args, checkpoint_dir)
        if fallback_dir is not None:
            print(f"-> Current exp_id checkpoint not found: {checkpoint_dir}")
            print(f"-> Use compatible checkpoint directory: {fallback_dir}")
            checkpoint_dir = fallback_dir
            checkpoint_file = _select_checkpoint_file(checkpoint_dir)

    if checkpoint_file is None:
        expected_files = ", ".join(_checkpoint_files(checkpoint_dir))
        raise FileNotFoundError(f"No checkpoint file found. Expected one of: {expected_files}")

    checkpoint = torch.load(checkpoint_file, weights_only=False)
    print(f"├── load pretrained from {checkpoint_dir}/")
    return checkpoint, checkpoint_dir


def load_pretrained_model(model, args):
    checkpoint_dir = os.path.join(args.best_checkpoint_path, args.exp_id)
    if args.pretrained_model_path is None:
        checkpoint, checkpoint_dir = _load_checkpoint_from_dir(args, checkpoint_dir)
        model = _load_model_state(model, checkpoint, args, _validate_checkpoint_compatibility)
        return model, checkpoint_dir + os.sep

    loaded = torch.load(args.pretrained_model_path, weights_only=False)
    model = _load_model_state(model, loaded, args, _validate_checkpoint_compatibility)
    return model, checkpoint_dir + os.sep


def _safe_pretrained_snapshot_path(args):
    os.makedirs("./pretrained", exist_ok=True)
    exp_hash = hashlib.md5(args.exp_id.encode("utf-8")).hexdigest()[:12]
    variant = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in args.model_variant)
    filename = f"{args.model}-{variant}-cfg{exp_hash}.pth"
    if len(filename) > 160:
        filename = f"{args.model}-cfg{exp_hash}.pth"
    return os.path.join("./pretrained", filename)


def _pretrained_snapshot_payload(pretrained, args):
    if isinstance(pretrained, nn.Module):
        return {
            "state_dict": pretrained.state_dict(),
            "config": {
                "model": args.model,
                "model_variant": args.model_variant,
                "model_recipe": args.model_recipe,
                "exp_id": args.exp_id,
                "d_model": args.d_model,
                "k_hop": args.k_hop,
                "input_len": args.input_len,
                "pred_len": args.pred_len,
                "look_back": args.look_back,
                "slide_step": args.slide_step,
                "enc_in": args.enc_in,
                "raw_enc_in": args.raw_enc_in,
                "c_out": args.c_out,
                "lifecycle_training_strategy": args.lifecycle_training_strategy,
                "lifecycle_keep_random_mask": args.lifecycle_keep_random_mask,
                "lifecycle_role_features": args.lifecycle_role_features,
                "lifecycle_history_age_feature": args.lifecycle_history_age_feature,
            },
            "model_class": pretrained.__class__.__name__,
            "model_module": pretrained.__class__.__module__,
        }
    return pretrained


def save_pretrained_snapshot(pretrained, args):
    snapshot_path = _safe_pretrained_snapshot_path(args)
    try:
        torch.save(_pretrained_snapshot_payload(pretrained, args), snapshot_path)
        with open(snapshot_path + ".exp_id.txt", "w", encoding="utf-8") as fp:
            fp.write(args.exp_id + "\n")
        print(f"-> Saved pretrained state snapshot to {snapshot_path}")
    except Exception as exc:
        print(f"-> Skip pretrained snapshot save to {snapshot_path}: {exc}")
        print(f"   Best checkpoints remain available under {args.best_checkpoint_path}")


def initialize_training_from_checkpoint(model, args):
    if args.train_init_checkpoint_path is None:
        return model
    loaded = torch.load(args.train_init_checkpoint_path, weights_only=False)
    model = _load_model_state(model, loaded, args, _validate_checkpoint_shape_compatibility)
    print(f"-> Initialize training weights from {args.train_init_checkpoint_path}")
    return model


def should_skip_model_checkpoint_load(args):
    return args.model == "RoleFusion" and args.model_recipe == "hard_route"


if __name__ == "__main__":
    args = initialize_args()
    args.exp_id = build_exp_id(args)
    wandb = set_wandb(args)
    configs = OmegaConf.create(vars(args))
    pl.seed_everything(args.seed)
    loss_candidate = dict(
        zip(
            ["mse", "huber", "mae", "rmse", "msepinball", "multipinball", "nll"],
            [nn.MSELoss(), nn.HuberLoss(), nn.L1Loss(), rmse_loss, MSEPinballLoss(), MultiPinBallLoss(), masked_gaussian_nll_loss],
        )
    )
    loss = loss_candidate[args.loss]
    criterions = dict(zip([args.loss, "split_phy_loss"], [loss, SlitPhyLoss(features=args.target_features)]))
    model = model_dict[args.model].Model(args).float()
    if not (args.test_for_async_lifecycle or args.test_for_changed or args.inference_only):
        model = initialize_training_from_checkpoint(model, args)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    args.num_params = sum([np.prod(p.size()) for p in model_parameters])
    if args.verbose:
        print_header("*** Model Configurations ***")
        print(model)
    print_header("*** Experiment Configurations *** ")
    print_config(configs)
    print_train(args)

    if args.test_for_async_lifecycle:
        if should_skip_model_checkpoint_load(args):
            print("-> RoleFusion hard_route uses expert checkpoints from family config; skip self-checkpoint loading")
        else:
            model, args.best_checkpoint_path = load_pretrained_model(model, args)
        args.slide_step = 12
        test_loader = data_provider(args, "test")
        print(f"├── async test data dim: {len(test_loader.dataset)}")  # type: ignore
        model.to(args.device).float()
        async_lifecycle_inference(model, test_loader, args, criterions)
        exit()

    if args.test_for_changed or args.inference_only:
        model, args.best_checkpoint_path = load_pretrained_model(model, args)
        args.slide_step = 12
        test_loader = data_provider(args, "test")
        print(f"├── test  data dim: {len(test_loader.dataset)}")  # type: ignore
        model.to(args.device).float()
        inference_model(model, test_loader, args, criterions)
        exit()

    slide_step = args.slide_step
    args.slide_step = 1  # use 1 for train to collect more data
    train_loader = data_provider(args, "train")
    args.slide_step = slide_step
    val_loader = data_provider(args, "val")
    test_loader = data_provider(args, "test")
    data_loaders = dict(zip(["train", "val", "test"], [train_loader, val_loader, test_loader]))
    optimizers = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optimizer_params = model.get_optimizer_param_groups(args.lr) if hasattr(model, "get_optimizer_param_groups") else model.parameters()
    optimizer = optimizers[args.optimizer](optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    schdulers = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "step": torch.optim.lr_scheduler.StepLR,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }
    if args.scheduler == "plateau":
        scheduler = schdulers[args.scheduler](optimizer, mode="min", factor=0.5, min_lr=1e-5, patience=args.scheduler_patience)
    elif args.scheduler == "cosine":
        scheduler = schdulers[args.scheduler](optimizer, T_max=5, eta_min=1e-7)
    elif args.scheduler == "step":
        scheduler = schdulers[args.scheduler](
            optimizer,
            step_size=args.scheduler_patience,
            gamma=0.5,
        )
    else:

        class NullScheduler:

            def step(self, loss):
                pass

        scheduler = NullScheduler()
    if args.verbose:
        print_header("*** Model Configurations ***")
        print(model)
    print_header("*** Experiment Configurations *** ")
    print_config(configs)
    print_train(args)
    print(f"├── train data dim: {len(train_loader.dataset)}")  # type: ignore
    print(f"├── val   data dim: {len(val_loader.dataset)}")  # type: ignore
    print(f"└── test  data dim: {len(test_loader.dataset)}")  # type: ignore
    pretrained = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loaders=data_loaders,
        criterions=criterions,
        args=args,
        wandb=wandb,
    )
    if args.save_model:
        save_pretrained_snapshot(pretrained, args)

    path = f"{args.best_prediction_path}best_total_y.pkl"
    y_pred, y, AAS, VS, FS, NAS = load_prediction(path, "val")
    report_error(y_pred, y, AAS, VS, FS, NAS, f"{args.model}-{args.dloader_name}", "./reports.csv")
    print("-> Done!")
