"""Configuration adapter for the active ALFTemporal expert only."""

from copy import deepcopy

import numpy as np
from omegaconf import OmegaConf


def deep_merge_dict(base, override):
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml_dict(path):
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"yaml config at {path} must resolve to a dict")
    return raw


def _family_cfg(args):
    cfg = ({} if not args.model_config_path
           else load_yaml_dict(args.model_config_path))
    family = getattr(args, "model_family_cfg", {}) or {}
    cfg["assets"] = deep_merge_dict(cfg.get("assets", {}), family.get("assets", {}))
    cfg["model"] = deep_merge_dict(cfg.get("model", {}), family.get("model", {}))
    cfg["recipe"] = deep_merge_dict(cfg.get("recipe", {}), family.get("recipe", {}))
    return cfg


def _context_fields(args, runtime_context):
    if runtime_context is None:
        raise ValueError("ALFTemporal configuration requires a RuntimeDataContext")
    identity = runtime_context.static_asset_identity()
    declared = getattr(args, "num_nodes", None) or getattr(args, "node_num", None)
    if declared is not None and int(declared) != identity["full_num_nodes"]:
        raise ValueError(
            f"config declares node_num={declared} but the runtime context has "
            f"{identity['full_num_nodes']}")
    known = [int(node) for node in runtime_context.known_nodes]
    unknown = [int(node) for node in np.setdiff1d(
        runtime_context.node_ids, runtime_context.known_nodes)]
    return {
        "dataset_name": runtime_context.dataset,
        "num_nodes": int(identity["full_num_nodes"]),
        "known_nodes": known,
        "unknown_nodes": unknown,
    }


def build_patchstg_cfg(args, runtime_context=None):
    """Bind the nested YAML to the graph context without hidden defaults."""
    context = _context_fields(args, runtime_context)
    cfg = _family_cfg(args)
    model = cfg.get("model", {})
    required = {
        "layers", "tem_patchsize", "tem_patchnum", "recur_times",
        "spa_patchsize", "factors", "tod", "dow", "input_dims",
        "node_dims", "tod_dims", "dow_dims",
    }
    missing = sorted(required - set(model))
    if missing:
        raise ValueError(
            "ALFTemporal YAML is missing runtime parameters: " + ", ".join(missing))
    if int(model["tem_patchsize"]) * int(model["tem_patchnum"]) != int(args.input_len):
        raise ValueError(
            "tem_patchsize * tem_patchnum must equal the configured input_len")
    model["spa_patchnum"] = 2 ** int(model["recur_times"])
    cfg["model"] = model
    cfg.update({
        "dataset_name": context["dataset_name"],
        "known_nodes": context["known_nodes"],
        "unknown_nodes_list": context["unknown_nodes"],
        "node_num": context["num_nodes"],
        "input_dim": int(args.raw_enc_in),
        "output_dim": int(args.c_out),
        "seq_len": int(args.input_len),
        "horizon": int(args.pred_len),
    })
    return cfg
