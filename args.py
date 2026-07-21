import argparse
from copy import deepcopy

from omegaconf import OmegaConf


CONFIG_SECTION_KEYS = {
    "runtime",
    "dataset",
    "training",
    "lifecycle",
    "async_test",
    "save",
    "logging",
    "wandb_cfg",
}


def _add_arguments(parser):
    parser.add_argument("-c", "--config", type=str, default=None, help="single-file experiment yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["train", "test_async", "test_shockwave", "test_changed", "inference"],
        help="optional high-level run mode override",
    )
    parser.add_argument("--project_name", type=str, default="MoGERNN")

    # Model
    parser.add_argument("--model", type=str, default="MoGERNN")
    parser.add_argument("--model_variant", type=str, default="default", help="optional model-family-specific variant name")
    parser.add_argument("--model_recipe", type=str, default="base", help="optional model-family-specific training recipe name")
    parser.add_argument("--model_config_path", default=None, help="optional external model yaml/config path used to build family-specific hyperparameters")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--k_hop", type=int, default=1, help="k_hop for graph")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--mean_expert", action="store_true", help="use mean expert")
    parser.add_argument("--weight_expert", action="store_true", help="use distane-based weighted expert")
    parser.add_argument("--max_expert", action="store_true", help="use max pooling expert")
    parser.add_argument("--min_expert", action="store_true", help="use min pooling expert")
    parser.add_argument("--diffusion_expert", action="store_true", help="use diffusion expert")
    parser.add_argument("--num_used_experts", type=int, default=3)

    parser.add_argument("--t_mark", action="store_true", help="do not add time stamp to feature")
    parser.add_argument("--pos_mark", action="store_true", help="do not add pos stamp to feature")

    # train
    parser.add_argument("--step_type", type=str, default="step", help="step type for the model")
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "mae", "rmse", "huber", "msepinball", "multipinball", "nll"],
        help="loss function used for backpropagation",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device to train on")
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "sgd", "adamw"], help="optimizer")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "step", "cosine", "None"], help="scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=1, help="scheduler patience")
    parser.add_argument("--patience", type=int, default=3, help="early stopping epochs")
    parser.add_argument(
        "--stop_based",
        type=str,
        default="train_mask",
        choices=["val_total", "val_mask", "val_role", "val_phase", "train_total", "train_mask", "train_role"],
        help="early stopping criteria",
    )
    parser.add_argument("--return_best", action="store_true", help="return best model")
    parser.add_argument("--test_for_changed", action="store_true", help="test pretrained model in changed sensor network")
    parser.add_argument("--test_for_async_lifecycle", action="store_true", help="test pretrained model in async lifecycle sensor network")
    parser.add_argument("--test_for_shockwave", action="store_true", help="legacy alias for the PLES async test")
    parser.add_argument("--train_for_async_lifecycle", action="store_true", help="train model with lifecycle-aware window simulation")
    parser.add_argument("--train_init_checkpoint_path", default=None, help="optional checkpoint path used to initialize model weights before training/fine-tuning")
    parser.add_argument("--read_only", action="store_true", help="read only")
    parser.add_argument("--inference_only", action="store_true", help="inference only")
    parser.add_argument("--pretrained_model_path", default=None, help="path to pretrained model")
    parser.add_argument("--async_num_events", type=int, default=4, help="event-count divisor used by lifecycle-aware training simulation")
    parser.add_argument("--ples_manifest_root", type=str, default="./protocol_manifests", help="dataset-specific PLES manifest root")
    parser.add_argument("--ples_pre_steps", type=int, default=12, help="dense pre-event steps in PLES")
    parser.add_argument("--ples_post_steps", type=int, default=36, help="dense post-event steps in PLES")
    parser.add_argument("--ples_temporal_blocks", type=int, default=4, help="dataset-adaptive temporal blocks in PLES")
    parser.add_argument("--ples_cohort_count", type=int, default=5, help="requested disjoint node cohorts in PLES")
    parser.add_argument("--ples_cohort_fraction", type=float, default=0.05, help="requested changed-node fraction per PLES cohort")
    parser.add_argument("--ples_min_event_gap", type=int, default=72, help="minimum raw time-step gap between PLES origins")
    parser.add_argument("--ples_seed", type=int, default=2026, help="PLES protocol seed recorded in the manifest")
    parser.add_argument("--ples_bootstrap_samples", type=int, default=2000, help="hierarchical bootstrap samples for PLES effects")
    parser.add_argument("--ples_bootstrap_seed", type=int, default=2026, help="hierarchical bootstrap seed for PLES effects")
    parser.add_argument("--ples_overwrite_manifest", action="store_true", help="overwrite a mismatched canonical PLES manifest intentionally")
    parser.add_argument("--lifecycle_role_features", action="store_true", help="append lifecycle role one-hot features to async lifecycle inputs")
    parser.add_argument("--lifecycle_history_age_feature", action="store_true", help="append lifecycle history-age feature to async lifecycle inputs")
    parser.add_argument(
        "--lifecycle_observation_mask_feature",
        action="store_true",
        help="append one effective-observation mask channel to lifecycle inputs",
    )
    parser.add_argument(
        "--lifecycle_event_sampling",
        type=str,
        default="mixed",
        choices=["random", "clustered", "mixed"],
        help="node sampling strategy for lifecycle-aware training events",
    )
    parser.add_argument(
        "--lifecycle_keep_random_mask",
        action="store_true",
        help="keep the legacy extra random masking on the raw observation channel during lifecycle-aware train/val",
    )
    parser.add_argument(
        "--lifecycle_training_strategy",
        type=str,
        default="v1",
        choices=["v1", "round1", "round2", "round3", "round3_event_only", "round3_weight_only", "round4_dense_event_only", "round4_dense_weighted", "lcbt_v1"],
        help="lifecycle-aware training recipe: v1=original heavy local lifecycle simulation, round1=lighter events + keep legacy random mask, round2=round1 without local VS nodes, round3=combined event+weight tuning, round3_event_only=only the event-profile changes from round3, round3_weight_only=only the optimizer weighting changes on top of round2-style events",
    )
    parser.add_argument("--lifecycle_aas_loss_weight", type=float, default=None, help="optional training loss weight override for AAS nodes in lifecycle-aware training")
    parser.add_argument("--lifecycle_nas_loss_weight", type=float, default=None, help="optional training loss weight override for NAS nodes in lifecycle-aware training")
    parser.add_argument("--lifecycle_vs_loss_weight", type=float, default=None, help="optional training loss weight override for VS nodes in lifecycle-aware training")
    parser.add_argument("--lifecycle_fs_loss_weight", type=float, default=None, help="optional training loss weight override for FS nodes in lifecycle-aware training")
    parser.add_argument("--lcbt_manifest_root", type=str, default="./protocol_manifests", help="dataset-specific LCBT manifest root")
    parser.add_argument("--lcbt_protocol_seed", type=int, default=2026, help="LCBT training schedule seed")
    parser.add_argument("--lcbt_validation_seed", type=int, default=12026, help="fixed LCBT validation schedule seed")
    parser.add_argument("--lcbt_train_end_ratio", type=float, default=0.50)
    parser.add_argument("--lcbt_val_end_ratio", type=float, default=0.60)
    parser.add_argument("--lcbt_test_start_ratio", type=float, default=0.60)
    parser.add_argument("--lcbt_scaler_end_ratio", type=float, default=0.50)
    parser.add_argument("--lcbt_join_ratio", type=float, default=0.40)
    parser.add_argument("--lcbt_fault_ratio", type=float, default=0.40)
    parser.add_argument("--lcbt_steady_ratio", type=float, default=0.20)
    parser.add_argument("--lcbt_cohort_fractions", type=float, nargs="+", default=[0.025, 0.05, 0.10])
    parser.add_argument("--lcbt_cohort_weights", type=float, nargs="+", default=[0.25, 0.50, 0.25])
    parser.add_argument("--lcbt_clustered_ratio", type=float, default=0.50)
    parser.add_argument("--lcbt_overwrite_manifest", action="store_true", help="overwrite a mismatched LCBT manifest intentionally")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_scale", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)

    # task
    parser.add_argument("--input_len", type=int, default=None)
    parser.add_argument("--pred_len", type=int, default=None)
    parser.add_argument("--look_back", type=int, default=0)
    parser.add_argument("--slide_step", type=int, default=1, help="step for sliding window")
    parser.add_argument("--target_features", type=str, default=None, choices=["q", "o", "v", "qv", "qov"], help="q is flow, v is speed, o is occupancy")
    parser.add_argument("--input_features", type=str, default=None, choices=["q", "o", "v", "qv", "qov"], help="q is flow, v is speed, o is occupancy")
    parser.add_argument("--train_ratio", type=float, default=0.6, help="train ratio")  # 当前 loader 实际写死 70/30，val 与 train 同段
    parser.add_argument("--valid_ratio", type=float, default=0.2, help="validation ratio")

    parser.add_argument("--inverse", action="store_true", help="inverse the data for better results presentation")

    # save and logging settings
    parser.add_argument("--wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--wandb_entity", type=str, default="zeng699304-xidian-university", help="user name of wandb")
    parser.add_argument("--best_checkpoint_path", type=str, default="./check_points/", help="best val checkpoint path")
    parser.add_argument("--best_prediction_path", type=str, default="./predictions/", help="results path")
    parser.add_argument("--best_metrics_path", type=str, default="./", help="results path")
    parser.add_argument("--best_fig_path", type=str, default="./figures/", help="figures path")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="log path")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("--save_model", action="store_true", help="save model not model state dict")
    # dataset
    parser.add_argument("--unknown_nodes_path", type=str, default="", help="path for unobserved sensors id in the train data")
    parser.add_argument("--unknown_nodes_list", type=str, default="", help="unobserved sensors in the train data")
    parser.add_argument("--unknown_nodes_num", type=int, default=0, help="number of unobserved sensors  in the training data")
    parser.add_argument("--ori_unknown_nodes_list", type=str, default="", help="unobserved sensors in the original data")
    parser.add_argument(
        "--dloader_name",
        type=str,
        default="METRLA",
        choices=["METRLA", "PEMSBAY", "PEMS04", "PEMS08", "METRLA-dynamic", "PEMSBAY-dynamic", "PEMS04-dynamic", "PEMS08-dynamic"],
        help="dynamic denotes the scenario where the scensor is changing",
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--freq", type=str, default="min")

def _collect_known_dests(parser):
    return {action.dest for action in parser._actions if action.dest != "help"}


def _merge_family_sections(raw_config):
    family_cfg = deepcopy(raw_config.get("family", {})) if isinstance(raw_config.get("family"), dict) else {}
    for key in ("model_cfg", "recipe_cfg", "assets"):
        value = raw_config.get(key)
        if isinstance(value, dict):
            merged_key = key.replace("_cfg", "")
            current = family_cfg.get(merged_key, {})
            if not isinstance(current, dict):
                current = {}
            current = deepcopy(current)
            current.update(value)
            family_cfg[merged_key] = current
    return family_cfg


def _flatten_config_defaults(raw_config, known_dests):
    defaults = {}
    for key, value in raw_config.items():
        if key in {"family", "model_cfg", "recipe_cfg", "assets"}:
            continue
        if key in known_dests:
            defaults[key] = value
            continue
        if isinstance(value, dict) and key in CONFIG_SECTION_KEYS:
            for sub_key, sub_value in value.items():
                if sub_key in known_dests:
                    defaults[sub_key] = sub_value
    return defaults


def _load_config_payload(config_path, parser):
    if not config_path:
        return {}, {}
    raw_cfg = OmegaConf.load(config_path)
    raw_cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raise ValueError("config yaml must resolve to a mapping/dict")
    defaults = _flatten_config_defaults(raw_cfg, _collect_known_dests(parser))
    defaults["config"] = config_path
    return defaults, _merge_family_sections(raw_cfg)


def _normalize_node_lists(args):
    if isinstance(args.unknown_nodes_list, str):
        args.unknown_nodes_list = [int(x) for x in args.unknown_nodes_list.split(",") if x]
    else:
        args.unknown_nodes_list = [int(x) for x in args.unknown_nodes_list]
    if isinstance(args.ori_unknown_nodes_list, str):
        args.ori_unknown_nodes_list = [int(x) for x in args.ori_unknown_nodes_list.split(",") if x]
    else:
        args.ori_unknown_nodes_list = [int(x) for x in args.ori_unknown_nodes_list]
    args.unknown_nodes_list = [x for x in args.unknown_nodes_list if x not in args.ori_unknown_nodes_list]


def _apply_mode_overrides(args):
    if args.mode is None:
        return
    args.test_for_changed = False
    args.test_for_async_lifecycle = False
    args.test_for_shockwave = False
    args.inference_only = False
    if args.mode == "test_async":
        args.test_for_async_lifecycle = True
    elif args.mode == "test_shockwave":
        args.test_for_async_lifecycle = True
        args.test_for_shockwave = True
    elif args.mode == "test_changed":
        args.test_for_changed = True
    elif args.mode == "inference":
        args.inference_only = True


def _validate_required_fields(args):
    required = ["input_len", "pred_len", "input_features", "target_features", "data_path"]
    missing = [name for name in required if getattr(args, name) in {None, ""}]
    if missing:
        raise ValueError(f"Missing required arguments after config merge: {', '.join(missing)}")


def finalize_args(args):
    _normalize_node_lists(args)
    _apply_mode_overrides(args)
    if args.test_for_shockwave:
        args.test_for_async_lifecycle = True
    _validate_required_fields(args)

    if args.test_for_async_lifecycle:
        if args.ples_pre_steps < 1:
            raise ValueError("ples_pre_steps must be positive")
        if args.ples_post_steps < args.input_len:
            raise ValueError("ples_post_steps must cover at least one complete input history")
        if args.ples_temporal_blocks < 1 or args.ples_cohort_count < 1:
            raise ValueError("ples_temporal_blocks and ples_cohort_count must be positive")
        if not 0 < args.ples_cohort_fraction <= 1:
            raise ValueError("ples_cohort_fraction must be in (0, 1]")
        if args.ples_min_event_gap < 0 or args.ples_bootstrap_samples < 0:
            raise ValueError("ples_min_event_gap and ples_bootstrap_samples must be non-negative")

    assert args.input_features in args.target_features, "input_features should be in target_features"
    lifecycle_feature_enabled = (
        args.lifecycle_role_features
        or args.lifecycle_history_age_feature
        or args.lifecycle_observation_mask_feature
    )
    if lifecycle_feature_enabled and not (args.train_for_async_lifecycle or args.test_for_async_lifecycle):
        raise ValueError("lifecycle input augmentation flags require train_for_async_lifecycle or test_for_async_lifecycle")
    if args.train_for_async_lifecycle or lifecycle_feature_enabled:
        if args.input_features != args.target_features or args.input_features not in {"q", "o", "v"}:
            raise ValueError("lifecycle-aware training/input augmentation currently supports one same input/target feature among q/o/v")
    if args.train_for_async_lifecycle and args.lifecycle_training_strategy in {"round1", "round2", "round3", "round3_event_only", "round3_weight_only"}:
        args.lifecycle_keep_random_mask = True
    if args.lifecycle_training_strategy == "lcbt_v1":
        required_lcbt_features = {
            "lifecycle_role_features": args.lifecycle_role_features,
            "lifecycle_history_age_feature": args.lifecycle_history_age_feature,
            "lifecycle_observation_mask_feature": args.lifecycle_observation_mask_feature,
        }
        missing_features = [
            name for name, enabled in required_lcbt_features.items() if not enabled
        ]
        if missing_features:
            raise ValueError(
                "lcbt_v1 requires raw + role(4) + age(1) + mask(1); missing: "
                + ", ".join(missing_features)
            )
        fixed_ratios = {
            "lcbt_train_end_ratio": 0.50,
            "lcbt_val_end_ratio": 0.60,
            "lcbt_test_start_ratio": 0.60,
            "lcbt_scaler_end_ratio": 0.50,
            "lcbt_join_ratio": 0.40,
            "lcbt_fault_ratio": 0.40,
            "lcbt_steady_ratio": 0.20,
            "lcbt_clustered_ratio": 0.50,
        }
        changed_ratios = [
            f"{name}={getattr(args, name)}"
            for name, expected in fixed_ratios.items()
            if abs(float(getattr(args, name)) - expected) > 1e-9
        ]
        if changed_ratios:
            raise ValueError(
                "lcbt_v1 uses fixed split/scenario/topology ratios; unexpected values: "
                + ", ".join(changed_ratios)
            )
        ratios = [args.lcbt_join_ratio, args.lcbt_fault_ratio, args.lcbt_steady_ratio]
        if any(value < 0 for value in ratios) or sum(ratios) <= 0:
            raise ValueError("LCBT scenario ratios must be non-negative with a positive sum")
        if len(args.lcbt_cohort_fractions) != len(args.lcbt_cohort_weights):
            raise ValueError("lcbt_cohort_fractions and lcbt_cohort_weights must have equal length")
        if any(not 0 < value <= 1 for value in args.lcbt_cohort_fractions):
            raise ValueError("LCBT cohort fractions must be in (0, 1]")
        if any(value < 0 for value in args.lcbt_cohort_weights) or sum(args.lcbt_cohort_weights) <= 0:
            raise ValueError("LCBT cohort weights must be non-negative with a positive sum")
        expected_cohort_fractions = (0.025, 0.05, 0.10)
        expected_cohort_weights = (0.25, 0.50, 0.25)
        if len(args.lcbt_cohort_fractions) != len(expected_cohort_fractions) or any(
            abs(float(actual) - expected) > 1e-9
            for actual, expected in zip(
                args.lcbt_cohort_fractions,
                expected_cohort_fractions,
            )
        ) or len(args.lcbt_cohort_weights) != len(expected_cohort_weights) or any(
            abs(float(actual) - expected) > 1e-9
            for actual, expected in zip(
                args.lcbt_cohort_weights,
                expected_cohort_weights,
            )
        ):
            raise ValueError(
                "lcbt_v1 requires cohort fractions [0.025, 0.05, 0.10] "
                "with weights [0.25, 0.50, 0.25]"
            )
        if not 0 <= args.lcbt_clustered_ratio <= 1:
            raise ValueError("lcbt_clustered_ratio must be in [0, 1]")
        args.lifecycle_keep_random_mask = False
    strategy_weight_defaults = {
        "v1": {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.0},
        "round1": {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.0},
        "round2": {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.0},
        "round3": {"aas": 1.0, "nas": 1.0, "vs": 1.35, "fs": 1.60},
        "round3_event_only": {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.0},
        "round3_weight_only": {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.60},
        "round4_dense_event_only": {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.0},
        "round4_dense_weighted": {"aas": 1.0, "nas": 1.20, "vs": 1.20, "fs": 1.30},
        "lcbt_v1": {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.0},
    }
    default_weights = strategy_weight_defaults[args.lifecycle_training_strategy]
    if args.lifecycle_aas_loss_weight is None:
        args.lifecycle_aas_loss_weight = default_weights["aas"]
    if args.lifecycle_nas_loss_weight is None:
        args.lifecycle_nas_loss_weight = default_weights["nas"]
    if args.lifecycle_vs_loss_weight is None:
        args.lifecycle_vs_loss_weight = default_weights["vs"]
    if args.lifecycle_fs_loss_weight is None:
        args.lifecycle_fs_loss_weight = default_weights["fs"]
    args.c_out = len(args.target_features)
    args.raw_enc_in = len(args.input_features)
    args.enc_in = args.raw_enc_in
    if args.lifecycle_role_features:
        args.enc_in += 4
    if args.lifecycle_history_age_feature:
        args.enc_in += 1
    if args.lifecycle_observation_mask_feature:
        args.enc_in += 1
    args.enc_mark_in = 3 * (args.t_mark) + 2 * (args.pos_mark)
    print(
        f"enc_in enc_mark_in, and c_out is set automatically based on input_features,target_features and t_mark,pos_mark",
    )
    args.num_experts = 1 * (args.mean_expert) + 1 * (args.weight_expert) + 1 * (args.max_expert) + 1 * (args.min_expert) + 1 * (args.diffusion_expert)
    return args


def load_args_from_config(config_path, overrides=None):
    parser = argparse.ArgumentParser(description="Arguments")
    _add_arguments(parser)
    config_defaults, family_cfg = _load_config_payload(config_path, parser)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    override_list = []
    for key, value in (overrides or {}).items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                override_list.append(flag)
        elif value is not None:
            override_list.extend([flag, str(value)])
    args = parser.parse_args(override_list if override_list else [])
    args.model_family_cfg = family_cfg
    return finalize_args(args)


def initialize_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-c", "--config", type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Arguments")
    _add_arguments(parser)
    config_defaults, family_cfg = _load_config_payload(pre_args.config, parser)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining if pre_args.config else None)
    args.model_family_cfg = family_cfg
    return finalize_args(args)
