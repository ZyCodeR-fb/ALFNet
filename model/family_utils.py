from copy import deepcopy

from omegaconf import OmegaConf

from lib.utils import infer_project_num_nodes, resolve_known_unknown_nodes


def deep_merge_dict(base, override):
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml_dict(path):
    if not path:
        return {}
    raw = OmegaConf.load(path)
    raw = OmegaConf.to_container(raw, resolve=True)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"yaml config at {path} must resolve to a dict")
    return raw


def build_project_family_context(args):
    num_nodes = infer_project_num_nodes({"data_path": args.data_path})
    known_nodes, unknown_nodes = resolve_known_unknown_nodes(
        {
            "data_path": args.data_path,
            "unknown_nodes_path": args.unknown_nodes_path,
            "unknown_nodes_list": args.unknown_nodes_list,
            "unknown_nodes_num": args.unknown_nodes_num,
            "seed": args.seed,
        }
    )
    train_node_ratio = len(known_nodes) / float(num_nodes)
    return {
        "dataset_name": args.dloader_name.replace("-dynamic", ""),
        "data_path": args.data_path,
        "unknown_nodes_path": args.unknown_nodes_path,
        "unknown_nodes": unknown_nodes,
        "unknown_nodes_list": unknown_nodes,
        "unknown_nodes_num": len(unknown_nodes),
        "seed": args.seed,
        "device": args.device,
        "num_nodes": num_nodes,
        "train_node_ratio": train_node_ratio,
        "val_node_ratio": 0.0,
        "known_nodes": known_nodes,
    }


def build_stfit_like_cfg(args, recipe_defaults):
    project_ctx = build_project_family_context(args)
    reference_cfg = load_yaml_dict(args.model_config_path)
    family_cfg = getattr(args, "model_family_cfg", {}) or {}

    cfg = deep_merge_dict(reference_cfg, {})
    cfg = deep_merge_dict(cfg, recipe_defaults)
    cfg = deep_merge_dict(cfg, family_cfg.get("recipe", {}))
    cfg["assets"] = deep_merge_dict(cfg.get("assets", {}), family_cfg.get("assets", {}))

    model_cfg = cfg.get("model", {})
    model_cfg = deep_merge_dict(model_cfg, family_cfg.get("model", {}))
    model_cfg.setdefault("hidden_channels", args.d_model)
    model_cfg["num_nodes"] = project_ctx["num_nodes"]
    model_cfg["in_channels"] = args.enc_in
    model_cfg["out_channels"] = args.pred_len
    model_cfg.setdefault("kernel_size", 3)
    model_cfg.setdefault("K", 3)
    model_cfg.setdefault("normalization", "sym")
    model_cfg.setdefault("bias", True)
    cfg["model"] = model_cfg

    cfg["dataset_name"] = project_ctx["dataset_name"]
    cfg["data_path"] = args.data_path
    cfg["unknown_nodes_path"] = args.unknown_nodes_path
    cfg["unknown_nodes_list"] = project_ctx["unknown_nodes"]
    cfg["unknown_nodes_num"] = project_ctx["unknown_nodes_num"]
    cfg["seed"] = args.seed
    cfg["device"] = args.device
    cfg["num_nodes"] = project_ctx["num_nodes"]
    cfg["train_node_ratio"] = project_ctx["train_node_ratio"]
    cfg["val_node_ratio"] = project_ctx["val_node_ratio"]
    cfg["window_size"] = args.input_len + args.pred_len
    cfg["horizon"] = args.pred_len
    cfg["input_size"] = args.enc_in
    cfg["hidden_channels"] = model_cfg["hidden_channels"]
    cfg["out_channels"] = model_cfg["out_channels"]
    cfg.setdefault("node_seed", args.seed)
    cfg.setdefault("dropout", 0.0)
    cfg.setdefault("batch_norm", False)
    cfg.setdefault("drop_node", False)
    cfg.setdefault("drop_node_ratio", 0.1)
    cfg.setdefault("is_vae", False)
    cfg.setdefault("graph_learning", False)
    cfg.setdefault("dynamic_graph", False)
    cfg.setdefault("dynamic_mix", True)
    cfg.setdefault("anchor_lambda", 0.5)
    cfg.setdefault("sim_loss", False)
    cfg.setdefault("fst_loss", False)
    cfg.setdefault("aug_node", False)
    cfg.setdefault("vae_loss_weight", 2.0)
    cfg.setdefault("aug_loss", "MSE")
    graph_cfg = cfg.get("graph", {})
    graph_cfg.setdefault("hidden_dim", args.d_model)
    graph_cfg.setdefault("node2edge_type", "cosine")
    cfg["graph"] = graph_cfg
    cfg.setdefault("random_graph", False)
    cfg.setdefault("gumbel_tau", 1.0)
    cfg.setdefault(
        "vae",
        {
            "input_dim": args.enc_in,
            "hidden_dim": args.d_model,
            "latent_dim": args.d_model,
            "num_layers": 2,
            "dropout": 0.1,
            "variance": 0.1,
        },
    )
    return cfg


def build_stgode_cfg(args):
    project_ctx = build_project_family_context(args)
    reference_cfg = load_yaml_dict(args.model_config_path)
    family_cfg = getattr(args, "model_family_cfg", {}) or {}
    cfg = deep_merge_dict(reference_cfg, {})
    cfg = deep_merge_dict(cfg, family_cfg.get("recipe", {}))
    cfg["assets"] = deep_merge_dict(cfg.get("assets", {}), family_cfg.get("assets", {}))

    model_cfg = cfg.get("model", {})
    model_cfg = deep_merge_dict(model_cfg, family_cfg.get("model", {}))
    model_cfg["num_nodes"] = project_ctx["num_nodes"]
    model_cfg["num_features"] = args.enc_in
    model_cfg["window_size"] = args.input_len + args.pred_len
    model_cfg["horizon"] = args.pred_len
    cfg["model"] = model_cfg

    cfg["dataset_name"] = project_ctx["dataset_name"]
    cfg["data_path"] = args.data_path
    cfg["unknown_nodes_path"] = args.unknown_nodes_path
    cfg["unknown_nodes_list"] = project_ctx["unknown_nodes"]
    cfg["unknown_nodes_num"] = project_ctx["unknown_nodes_num"]
    cfg["seed"] = args.seed
    cfg["device"] = args.device
    cfg["num_nodes"] = project_ctx["num_nodes"]
    cfg["train_node_ratio"] = project_ctx["train_node_ratio"]
    cfg["val_node_ratio"] = project_ctx["val_node_ratio"]
    cfg.setdefault("assets", {})
    return cfg
