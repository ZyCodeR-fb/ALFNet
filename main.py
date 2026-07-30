import os
import hashlib
import json

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch import nn

from args import apply_output_root, initialize_args
from data_provider.data_factory import data_provider
from experiments import async_lifecycle_inference, train_model
from model.regist_model import load_model_class
from utils.evaluation import MSEPinballLoss, MultiPinBallLoss, SlitPhyLoss, masked_gaussian_nll_loss, rmse_loss
from utils.logtool import print_config, print_header, print_train, set_wandb
from utils.run_logger import install_run_log


MAX_EXP_ID_COMPONENT_LEN = 250

# name -> value that reproduces the historical experiment id
TRAINING_OBJECTIVE_KNOBS = {
    "lcbt_context_loss_weight": 0.10,
    "lcbt_stratum_balanced_loss": 0.0,
    "lcbt_include_stress_cohort": False,
    # Tail and worst-group terms change what is optimised, so two runs that
    # differ only in these must not share an experiment directory.
    "lcbt_tail_weight": 0.0,
    "lcbt_tail_quantile": 0.1,
    "lcbt_worst_group_weight": 0.0,
    "lcbt_worst_group_temperature": 0.1,
}


def _shorten_exp_id_component(exp_id):
    if len(exp_id.encode("utf-8")) <= MAX_EXP_ID_COMPONENT_LEN:
        return exp_id
    full_hash = hashlib.md5(exp_id.encode("utf-8")).hexdigest()[:10]
    suffix = f"-cfg{full_hash}"
    prefix_budget = MAX_EXP_ID_COMPONENT_LEN - len(suffix)
    return f"{exp_id[:prefix_budget].rstrip('-_')}{suffix}"


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
    }
    # Knobs that change what the model is optimising. The LCBT manifest hash
    # covers the protocol-shaped ones (cohort tiers, ratios, hidden fraction),
    # but the loss weights live only on the command line, so two runs with
    # different objectives used to land in the same experiment directory and
    # overwrite each other's checkpoint. Only non-default values are folded in,
    # which keeps every existing run's signature byte-identical.
    for name, default in TRAINING_OBJECTIVE_KNOBS.items():
        value = getattr(args, name, default)
        if value != default:
            signature_payload[name] = value
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

    use_lifecycle_train_prefix = (
        args.train_for_async_lifecycle or args.test_for_async_lifecycle
    )
    if use_lifecycle_train_prefix:
        exp_prefix = (
            f"{exp_prefix}-lifecycle_train-"
            f"strategy-{args.lifecycle_training_strategy}-"
            f"role-{int(args.lifecycle_role_features)}-"
            f"age-{int(args.lifecycle_history_age_feature)}-"
            f"mask-{int(args.lifecycle_observation_mask_feature)}"
        )
    return _shorten_exp_id_component(f"{exp_prefix}-{exp_id}")


def _validate_checkpoint_compatibility(checkpoint, args):
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        return
    expected_pairs = {
        "enc_in": args.enc_in,
        "raw_enc_in": args.raw_enc_in,
        "c_out": args.c_out,
        "lifecycle_training_strategy": args.lifecycle_training_strategy,
        "lifecycle_role_features": args.lifecycle_role_features,
        "lifecycle_history_age_feature": args.lifecycle_history_age_feature,
        "lifecycle_observation_mask_feature": args.lifecycle_observation_mask_feature,
        "lifecycle_age_mode": args.lifecycle_age_mode,
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


def _load_model_state(model, loaded, args, validator, allow_new_modules=False):
    """Load a checkpoint. Strict unless the caller is warm-starting.

    Warm-starting a candidate from its anchor's checkpoint necessarily has
    missing keys -- the anchor never had the new module. Those are allowed and
    reported; unexpected keys never are, because they mean the checkpoint and
    the model disagree about something that already existed.
    """
    if not isinstance(loaded, dict) or "state_dict" not in loaded:
        raise TypeError("Checkpoint must be a state_dict payload produced by the LCBT v3 trainer")
    validator(loaded, args)
    if not allow_new_modules:
        model.load_state_dict(loaded["state_dict"])
        return model
    missing, unexpected = model.load_state_dict(loaded["state_dict"], strict=False)
    if unexpected:
        raise ValueError(
            f"Checkpoint carries parameters this model does not have: {sorted(unexpected)[:5]}"
        )
    if missing:
        print(
            f"-> Warm start: {len(missing)} freshly initialised tensors kept "
            f"(e.g. {sorted(missing)[:3]})"
        )
    return model


def _select_checkpoint_file(checkpoint_dir):
    checkpoint_file = os.path.join(checkpoint_dir, "best_val_checkpoint.pth")
    return checkpoint_file if os.path.exists(checkpoint_file) else None


def _load_checkpoint_from_dir(args, checkpoint_dir):
    checkpoint_file = _select_checkpoint_file(checkpoint_dir)
    if checkpoint_file is None:
        raise FileNotFoundError(
            f"No LCBT v3 validation checkpoint found at "
            f"{os.path.join(checkpoint_dir, 'best_val_checkpoint.pth')}"
        )

    checkpoint = torch.load(checkpoint_file, weights_only=False)
    print(f"├── load pretrained from {checkpoint_dir}/")
    return checkpoint, checkpoint_dir, checkpoint_file


def _to_plain_number(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)):
        return value
    return None


def _checkpoint_run_metadata(checkpoint, checkpoint_dir, checkpoint_file):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": checkpoint_file,
        "loaded_checkpoint_epoch": _to_plain_number(checkpoint.get("epoch")) if isinstance(checkpoint, dict) else None,
        "loaded_checkpoint_metric": _to_plain_number(checkpoint.get("val_metric")) if isinstance(checkpoint, dict) else None,
        "max_epoch": config.get("max_epoch") if isinstance(config, dict) else None,
        "patience": config.get("patience") if isinstance(config, dict) else None,
        "stop_based": config.get("stop_based") if isinstance(config, dict) else None,
        "training_protocol_hash": (
            config.get("training_protocol_hash")
            if isinstance(config, dict)
            else None
        ),
    }


def load_pretrained_model(model, args):
    # Run-scoped output paths already identify the experiment via ``run_id``.
    # Appending the long exp_id again would make test_async look in a directory
    # that training never created.
    checkpoint_dir = (
        args.best_checkpoint_path
        if getattr(args, "paths_are_run_scoped", False)
        else os.path.join(args.best_checkpoint_path, args.exp_id)
    )
    if args.pretrained_model_path is None:
        checkpoint, checkpoint_dir, checkpoint_file = _load_checkpoint_from_dir(args, checkpoint_dir)
        args.loaded_checkpoint_metadata = _checkpoint_run_metadata(checkpoint, checkpoint_dir, checkpoint_file)
        model = _load_model_state(model, checkpoint, args, _validate_checkpoint_compatibility)
        return model, checkpoint_dir + os.sep

    loaded = torch.load(args.pretrained_model_path, weights_only=False)
    args.loaded_checkpoint_metadata = _checkpoint_run_metadata(loaded, checkpoint_dir, args.pretrained_model_path)
    model = _load_model_state(model, loaded, args, _validate_checkpoint_compatibility)
    return model, checkpoint_dir + os.sep


def initialize_training_from_checkpoint(model, args):
    if args.train_init_checkpoint_path is None:
        return model
    loaded = torch.load(args.train_init_checkpoint_path, weights_only=False)
    model = _load_model_state(
        model, loaded, args, _validate_checkpoint_shape_compatibility,
        allow_new_modules=True,
    )
    print(f"-> Initialize training weights from {args.train_init_checkpoint_path}")
    return model


if __name__ == "__main__":
    args = initialize_args()
    args.exp_id = build_exp_id(args)
    # Must run after exp_id exists: the default run folder is the experiment id.
    apply_output_root(args)
    # Install before anything prints: the per-epoch lifecycle group numbers are
    # only ever written to stdout, and they are the diagnostic that
    # training_summary.json cannot reconstruct.
    run_log_path = install_run_log(args)
    if run_log_path:
        print(f"-> Console log: {run_log_path}")
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
    model = load_model_class(args.model)(args).float()
    if not args.test_for_async_lifecycle:
        model = initialize_training_from_checkpoint(model, args)
    if getattr(model, "freeze_anchor", False):
        frozen = model.freeze_anchor_parameters()
        print(f"-> Anchor frozen: {frozen} tensors held fixed, training the residual only")
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    args.num_params = sum([np.prod(p.size()) for p in model_parameters])
    if args.verbose:
        print_header("*** Model Configurations ***")
        print(model)
    print_header("*** Experiment Configurations *** ")
    print_config(configs)
    print_train(args)

    if args.test_for_async_lifecycle:
        model, args.best_checkpoint_path = load_pretrained_model(model, args)
        args.slide_step = 1
        test_loader = data_provider(args, "test")
        manifest = test_loader.dataset.get_protocol_manifest()  # type: ignore
        print(
            f"PLES test data: {manifest['dataset']['time_steps']} time steps | "
            f"{manifest['resolved']['origin_count']} origins | "
            f"{manifest['resolved']['cohort_count']} cohorts"
        )
        print(f"-> async test data dim: {len(test_loader.dataset)}")  # type: ignore
        model.to(args.device).float()
        async_lifecycle_inference(model, test_loader, args, criterions)
        exit()

    slide_step = args.slide_step
    args.slide_step = 1  # use 1 for train to collect more data
    train_loader = data_provider(args, "train")
    args.slide_step = slide_step
    val_loader = data_provider(args, "val")
    data_loaders = {"train": train_loader, "val": val_loader}
    optimizers = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optimizer_params = model.get_optimizer_param_groups(args.lr) if hasattr(model, "get_optimizer_param_groups") else model.parameters()
    optimizer = optimizers[args.optimizer](optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    schedulers = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "step": torch.optim.lr_scheduler.StepLR,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }
    if args.scheduler == "plateau":
        scheduler = schedulers[args.scheduler](optimizer, mode="min", factor=0.5, min_lr=1e-5, patience=args.scheduler_patience)
    elif args.scheduler == "cosine":
        scheduler = schedulers[args.scheduler](optimizer, T_max=5, eta_min=1e-7)
    elif args.scheduler == "step":
        scheduler = schedulers[args.scheduler](
            optimizer,
            step_size=args.scheduler_patience,
            gamma=0.5,
        )
    else:

        class NullScheduler:

            def step(self, loss):
                pass

        scheduler = NullScheduler()
    print(f"├── train data dim: {len(train_loader.dataset)}")  # type: ignore
    print(f"├── val   data dim: {len(val_loader.dataset)}")  # type: ignore
    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loaders=data_loaders,
        criterions=criterions,
        args=args,
        wandb=wandb,
    )
    print("-> Done!")
