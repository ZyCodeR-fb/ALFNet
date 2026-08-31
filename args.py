"""Strict YAML loader for the single active v4 experiment.

Runtime code does not repair or override experiment semantics. Values such as
window stride, horizon, target and split ratios come from YAML and are checked
against the v4 contract here.
"""

import argparse
from copy import deepcopy

from omegaconf import OmegaConf

from lib import lcbt_v4
from utils.canonical_config import freeze_canonical_config

SECTIONS = {"runtime", "dataset", "training", "lifecycle"}
DATASETS = ("PEMS03", "PEMS04", "PEMS08", "METRLA", "PEMSBAY")


def _add_arguments(parser):
    # Runtime/model
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("--model", default="MoGERNN")
    parser.add_argument("--model_variant", default="default")
    parser.add_argument("--model_recipe", default="base")
    parser.add_argument("--model_config_path", default=None)
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=32)

    # Dataset/window: every field is consumed by the v4 loader.
    parser.add_argument("--dloader_name", choices=DATASETS, default="PEMS08")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--unknown_nodes_path", default=None)
    parser.add_argument("--unknown_nodes_list", default=[])
    parser.add_argument("--unknown_nodes_num", type=int, default=0)
    parser.add_argument("--input_len", type=int, default=None)
    parser.add_argument("--pred_len", type=int, default=None)
    parser.add_argument("--look_back", type=int, default=0)
    parser.add_argument("--slide_step", type=int, default=None)
    parser.add_argument("--input_features", choices=["q", "o", "v"], default=None)
    parser.add_argument("--target_features", choices=["q", "o", "v"], default=None)
    parser.add_argument("--freq", default="min")

    # Training
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--teacher_forcing_mode",
                        choices=["linear_decay", "free_run"],
                        default="linear_decay")
    parser.add_argument("--teacher_forcing_decay_epochs", type=int, default=40)
    parser.add_argument("--early_stopping_start_epoch", type=int, default=0)
    parser.add_argument("--stop_based", choices=["val_lifecycle"],
                        default="val_lifecycle")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--loss", choices=["mse", "mae", "rmse", "huber"],
                        default="mse")
    parser.add_argument("--optimizer", choices=["adamw"], default="adamw")
    parser.add_argument("--scheduler", choices=["None", "plateau"], default="None")
    parser.add_argument("--scheduler_patience", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--k_hop", type=int, default=1)
    parser.add_argument("--t_mark", action="store_true")
    parser.add_argument("--pos_mark", action="store_true")

    # MoGERNN expert declarations.
    for name in ("mean", "weight", "max", "min", "diffusion"):
        parser.add_argument(f"--{name}_expert", action="store_true")
    parser.add_argument("--num_used_experts", type=int, default=3)

    # Lifecycle protocol and ALFNet-only method inputs.
    parser.add_argument("--train_for_async_lifecycle", action="store_true")
    parser.add_argument("--lifecycle_role_features", action="store_true")
    parser.add_argument("--lifecycle_history_age_feature", action="store_true")
    parser.add_argument("--lifecycle_observation_mask_feature", action="store_true")
    parser.add_argument("--lifecycle_age_mode", choices=["state_duration"],
                        default="state_duration")
    parser.add_argument("--lifecycle_training_strategy", choices=[lcbt_v4.STRATEGY],
                        default=lcbt_v4.STRATEGY)
    parser.add_argument("--lcbt_validation_seed", type=int, default=12026)
    parser.add_argument("--lcbt_train_end_ratio", type=float,
                        default=lcbt_v4.TRAIN_END_RATIO)
    parser.add_argument("--lcbt_val_end_ratio", type=float,
                        default=lcbt_v4.VAL_END_RATIO)
    parser.add_argument("--lcbt_test_start_ratio", type=float,
                        default=lcbt_v4.TEST_START_RATIO)
    parser.add_argument("--lcbt_scaler_end_ratio", type=float,
                        default=lcbt_v4.SCALER_END_RATIO)
    for name in ("zero_history", "steady", "join", "fault"):
        parser.add_argument(f"--lcbt_{name}_ratio", type=float, default=0.25)


def _family(raw):
    return deepcopy(raw.get("family", {})) if isinstance(raw.get("family"), dict) else {}


def _defaults(path, parser):
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("config YAML must be a mapping")
    known = {action.dest for action in parser._actions}
    defaults = {}
    for key, value in raw.items():
        if key in known:
            defaults[key] = value
        elif key in SECTIONS and isinstance(value, dict):
            defaults.update({name: item for name, item in value.items()
                             if name in known})
    defaults["config"] = str(path)
    return defaults, _family(raw)


def _validate(args):
    required = ("data_path", "unknown_nodes_path", "input_len", "pred_len",
                "slide_step", "input_features", "target_features")
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        raise ValueError("missing v4 config fields: " + ", ".join(missing))
    if min(args.input_len, args.pred_len, args.slide_step) <= 0:
        raise ValueError("input_len, pred_len and slide_step must be positive")
    if args.teacher_forcing_decay_epochs <= 0:
        raise ValueError("teacher_forcing_decay_epochs must be positive")
    if args.early_stopping_start_epoch < 0:
        raise ValueError("early_stopping_start_epoch must be non-negative")
    if args.input_features != args.target_features:
        raise ValueError("v4 uses one identical input/target variable")
    if args.lifecycle_training_strategy != lcbt_v4.STRATEGY:
        raise ValueError(f"only {lcbt_v4.STRATEGY} is supported")
    expected = {
        "lcbt_train_end_ratio": lcbt_v4.TRAIN_END_RATIO,
        "lcbt_val_end_ratio": lcbt_v4.VAL_END_RATIO,
        "lcbt_test_start_ratio": lcbt_v4.TEST_START_RATIO,
        "lcbt_scaler_end_ratio": lcbt_v4.SCALER_END_RATIO,
        **lcbt_v4.SCENARIO_RATIOS,
    }
    wrong = [f"{name}={getattr(args, name)}" for name, value in expected.items()
             if abs(float(getattr(args, name)) - float(value)) > 1e-9]
    if wrong:
        raise ValueError("v4 split/scenario contract mismatch: " + ", ".join(wrong))
    if args.model == "ALFNet":
        missing = [name for name in (
            "lifecycle_role_features", "lifecycle_history_age_feature",
            "lifecycle_observation_mask_feature") if not getattr(args, name)]
        if missing:
            raise ValueError("ALFNet lifecycle inputs missing: " + ", ".join(missing))

    args.raw_enc_in = 1
    args.c_out = 1
    args.enc_in = 1 + 4 * int(args.lifecycle_role_features)
    args.enc_in += int(args.lifecycle_history_age_feature)
    args.enc_in += int(args.lifecycle_observation_mask_feature)
    args.enc_mark_in = 3 * int(args.t_mark) + 2 * int(args.pos_mark)
    args.num_experts = sum(int(getattr(args, f"{name}_expert"))
                           for name in ("mean", "weight", "max", "min", "diffusion"))
    return args


def load_args_from_config(config_path, overrides=None):
    parser = argparse.ArgumentParser(add_help=False)
    _add_arguments(parser)
    defaults, family = _defaults(config_path, parser)
    parser.set_defaults(**defaults)
    argv = []
    for key, value in (overrides or {}).items():
        if key not in {action.dest for action in parser._actions}:
            raise ValueError(f"unknown v4 override {key!r}")
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
        elif value is not None:
            argv.extend((f"--{key}", str(value)))
    args = parser.parse_args(argv)
    args.model_family_cfg = family
    freeze_canonical_config(_validate(args))
    return args
