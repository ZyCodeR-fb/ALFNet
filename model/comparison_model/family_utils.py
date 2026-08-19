from copy import deepcopy

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
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



# Baselines published before this benchmark existed take the traffic series and
# whatever calendar features their own paper defines. The lifecycle role, age
# and observation-mask channels are *this method's* inputs; handing them to a
# baseline silently upgrades it into a different model and stops the comparison
# measuring what it claims to. Every original baseline therefore slices back to
# the raw channels here, and only ALFNet's own modules read the rest.
def source_native_input(x_enc, raw_enc_in):
    """`[B, L, N, enc_in] -> [B, L, N, raw_enc_in]`, the masked traffic series."""
    raw = int(raw_enc_in)
    if x_enc.shape[-1] < raw:
        raise ValueError(
            f"input has {x_enc.shape[-1]} channels, fewer than the {raw} raw ones"
        )
    return x_enc[..., :raw]


def source_native_dim(args):
    """Channel count a source-native baseline actually consumes."""
    return int(getattr(args, "raw_enc_in", args.enc_in))


def build_family_context_from_runtime(args, runtime_context):
    """The v4 path: every graph fact comes from the context, none from a path.

    Deleting ALFTemporal's explicit `load_project_adjacency` call is not enough
    -- `build_project_family_context` reaches `data_path` through
    `infer_project_num_nodes` and `resolve_known_unknown_nodes`, so the private
    resolution survives the obvious fix and stays invisible.
    """
    identity = runtime_context.static_asset_identity()
    declared = getattr(args, "num_nodes", None) or getattr(args, "node_num", None)
    if declared is not None and int(declared) != identity["full_num_nodes"]:
        raise ValueError(
            f"config declares node_num={declared} but the runtime context has "
            f"{identity['full_num_nodes']}; refusing rather than silently winning")
    known_nodes = [int(node) for node in runtime_context.known_nodes]
    num_nodes = identity["full_num_nodes"]
    return {
        "dataset_name": runtime_context.dataset,
        "data_path": None,          # v4 resolves nothing from a path
        "unknown_nodes_path": None,
        "num_nodes": num_nodes,
        "known_nodes": known_nodes,
        "unknown_nodes": [int(n) for n in np.setdiff1d(
            runtime_context.node_ids, runtime_context.known_nodes)],
        "unknown_nodes_list": [int(n) for n in np.setdiff1d(
            runtime_context.node_ids, runtime_context.known_nodes)],
        "unknown_nodes_num": int(num_nodes - len(known_nodes)),
        "seed": getattr(args, "seed", 0),
        "device": getattr(args, "device", "cpu"),
        "train_node_ratio": len(known_nodes) / float(num_nodes),
        "val_node_ratio": 0.0,
        "static_asset_hash": runtime_context.static_asset_hash(),
    }


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


def _build_family_cfg(args):
    reference_cfg = load_yaml_dict(args.model_config_path)
    family_cfg = getattr(args, "model_family_cfg", {}) or {}
    cfg = deep_merge_dict(reference_cfg, {})
    cfg["assets"] = deep_merge_dict(cfg.get("assets", {}), family_cfg.get("assets", {}))
    model_cfg = deep_merge_dict(cfg.get("model", {}), family_cfg.get("model", {}))
    recipe_cfg = deep_merge_dict(cfg.get("recipe", {}), family_cfg.get("recipe", {}))
    cfg["model"] = model_cfg
    cfg["recipe"] = recipe_cfg
    return cfg


def build_stfit_like_cfg(args, recipe_defaults):
    project_ctx = build_project_family_context(args)
    cfg = _build_family_cfg(args)
    family_cfg = getattr(args, "model_family_cfg", {}) or {}
    cfg = deep_merge_dict(cfg, recipe_defaults)
    cfg = deep_merge_dict(cfg, family_cfg.get("recipe", {}))
    cfg["assets"] = deep_merge_dict(cfg.get("assets", {}), family_cfg.get("assets", {}))

    model_cfg = cfg.get("model", {})
    model_cfg = deep_merge_dict(model_cfg, family_cfg.get("model", {}))
    model_cfg.setdefault("hidden_channels", args.d_model)
    model_cfg["num_nodes"] = project_ctx["num_nodes"]
    model_cfg["in_channels"] = source_native_dim(args)
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
            "input_dim": source_native_dim(args),
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
    cfg = _build_family_cfg(args)
    family_cfg = getattr(args, "model_family_cfg", {}) or {}
    cfg = deep_merge_dict(cfg, family_cfg.get("recipe", {}))
    cfg["assets"] = deep_merge_dict(cfg.get("assets", {}), family_cfg.get("assets", {}))

    model_cfg = cfg.get("model", {})
    model_cfg = deep_merge_dict(model_cfg, family_cfg.get("model", {}))
    model_cfg["num_nodes"] = project_ctx["num_nodes"]
    model_cfg["num_features"] = source_native_dim(args)
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


def build_ha_cfg(args):
    cfg = _build_family_cfg(args)
    model_cfg = cfg.get("model", {})
    model_cfg.setdefault("window_size", args.input_len + args.pred_len)
    model_cfg.setdefault("horizon", args.pred_len)
    model_cfg.setdefault("channel", None)
    model_cfg.setdefault("reverse", False)
    cfg["model"] = model_cfg
    return cfg


def build_lstm_cfg(args, recipe_defaults):
    cfg = build_stfit_like_cfg(args, recipe_defaults)
    model_cfg = cfg.get("model", {})
    # Not setdefault: a leftover input_dim in a YAML would silently re-admit
    # the lifecycle channels this baseline is not supposed to see.
    model_cfg["input_dim"] = source_native_dim(args)
    model_cfg.setdefault("embed_dim", args.d_model)
    model_cfg.setdefault("hidden_dim", args.d_model)
    model_cfg.setdefault("end_dim", args.pred_len)
    model_cfg.setdefault("num_layer", 2)
    model_cfg.setdefault("dropout", 0.1)
    model_cfg["horizon"] = args.pred_len
    cfg["model"] = model_cfg
    return cfg


def calculate_normalized_laplacian(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(adj_mx.shape[0]) - d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt).tocoo()


def calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which="LM")
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    identity = sp.identity(lap.shape[0], format="csr", dtype=lap.dtype)
    return (2 / lambda_max * lap) - identity


def calculate_sym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1))
    try:
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    except FloatingPointError:
        rowsum = rowsum.astype(np.float64) + 1e-10
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt)


def calculate_asym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1)).flatten()
    try:
        d_inv = np.power(rowsum, -1).flatten()
    except FloatingPointError:
        rowsum = rowsum.astype(np.float64) + 1e-10
        d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj_mx)


def normalize_adj_mx(adj_mx, adj_type, return_type="dense"):
    if adj_type == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx)]
    elif adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adj_type == "symadj":
        adj = [calculate_sym_adj(adj_mx)]
    elif adj_type == "transition":
        adj = [calculate_asym_adj(adj_mx)]
    elif adj_type == "doubletransition":
        adj = [calculate_asym_adj(adj_mx), calculate_asym_adj(np.transpose(adj_mx))]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        return []

    if return_type == "dense":
        return [np.asarray(a.astype(np.float32).todense()) if hasattr(a, "todense") else np.asarray(a, dtype=np.float32) for a in adj]
    if return_type == "coo":
        return [a.tocoo() for a in adj]
    raise ValueError(f"Unsupported return_type: {return_type}")


def build_gwnet_cfg(args):
    project_ctx = build_project_family_context(args)
    cfg = _build_family_cfg(args)
    model_cfg = cfg.get("model", {})
    model_cfg.setdefault("adj_type", "doubletransition")
    model_cfg.setdefault("adp_adj", True)
    model_cfg.setdefault("dropout", 0.3)
    model_cfg.setdefault("residual_channels", 32)
    model_cfg.setdefault("dilation_channels", 32)
    model_cfg.setdefault("skip_channels", 256)
    model_cfg.setdefault("end_channels", 512)
    model_cfg.setdefault("kernel_size", 2)
    model_cfg.setdefault("blocks", 4)
    model_cfg.setdefault("layers", 2)
    cfg["model"] = model_cfg
    cfg["dataset_name"] = project_ctx["dataset_name"]
    cfg["known_nodes"] = project_ctx["known_nodes"]
    cfg["unknown_nodes_list"] = project_ctx["unknown_nodes"]
    cfg["node_num"] = project_ctx["num_nodes"]
    cfg["input_dim"] = source_native_dim(args)
    cfg["output_dim"] = args.c_out
    cfg["seq_len"] = args.input_len
    cfg["horizon"] = args.pred_len
    return cfg


def _resolve_time_of_day_size(freq):
    freq = (freq or "").lower()
    if freq in {"min", "t"}:
        return 24 * 60
    if freq == "h":
        return 24
    return 24 * 60


def build_stid_cfg(args):
    project_ctx = build_project_family_context(args)
    cfg = _build_family_cfg(args)
    model_cfg = cfg.get("model", {})
    model_cfg.setdefault("node_dim", 32)
    model_cfg.setdefault("embed_dim", 32)
    model_cfg.setdefault("num_layer", 3)
    model_cfg.setdefault("temp_dim_tid", 32)
    model_cfg.setdefault("temp_dim_diw", 32)
    model_cfg.setdefault("time_of_day_size", _resolve_time_of_day_size(args.freq))
    model_cfg.setdefault("day_of_week_size", 7)
    model_cfg.setdefault("if_T_i_D", True)
    model_cfg.setdefault("if_D_i_W", False)
    model_cfg.setdefault("if_node", True)
    model_cfg["input_len"] = args.input_len
    model_cfg["output_len"] = args.pred_len
    model_cfg["input_dim"] = (
        source_native_dim(args) + int(model_cfg["if_T_i_D"]) + int(model_cfg["if_D_i_W"])
    )
    cfg["model"] = model_cfg
    cfg["dataset_name"] = project_ctx["dataset_name"]
    cfg["known_nodes"] = project_ctx["known_nodes"]
    cfg["unknown_nodes_list"] = project_ctx["unknown_nodes"]
    cfg["node_num"] = project_ctx["num_nodes"]
    cfg["output_dim"] = args.c_out
    cfg["seq_len"] = args.input_len
    cfg["horizon"] = args.pred_len
    return cfg


def build_patchstg_cfg(args, runtime_context=None):
    project_ctx = (build_project_family_context(args) if runtime_context is None
                   else build_family_context_from_runtime(args, runtime_context))
    cfg = _build_family_cfg(args)
    model_cfg = cfg.get("model", {})
    model_cfg.setdefault("layers", 5)
    model_cfg.setdefault("tem_patchsize", args.pred_len)
    model_cfg.setdefault("tem_patchnum", 1)
    model_cfg.setdefault("recur_times", 5)
    model_cfg.setdefault("spa_patchsize", 8)
    model_cfg.setdefault("factors", 1)
    model_cfg.setdefault("tod", 288)
    model_cfg.setdefault("dow", 7)
    model_cfg.setdefault("input_dims", args.d_model)
    model_cfg.setdefault("node_dims", max(16, args.d_model // 2))
    model_cfg.setdefault("tod_dims", max(16, args.d_model // 2))
    model_cfg.setdefault("dow_dims", max(16, args.d_model // 2))
    model_cfg["spa_patchnum"] = int(model_cfg.get("spa_patchnum", 2 ** int(model_cfg["recur_times"])))
    cfg["model"] = model_cfg
    cfg["dataset_name"] = project_ctx["dataset_name"]
    cfg["known_nodes"] = project_ctx["known_nodes"]
    cfg["unknown_nodes_list"] = project_ctx["unknown_nodes"]
    cfg["node_num"] = project_ctx["num_nodes"]
    cfg["input_dim"] = source_native_dim(args)
    cfg["output_dim"] = args.c_out
    cfg["seq_len"] = args.input_len
    cfg["horizon"] = args.pred_len
    return cfg


def build_slpf_cfg(args):
    project_ctx = build_project_family_context(args)
    cfg = _build_family_cfg(args)
    model_cfg = cfg.get("model", {})
    model_cfg.setdefault("embed_dim", args.d_model)
    model_cfg.setdefault("num_layer", 3)
    cfg["model"] = model_cfg
    cfg["dataset_name"] = project_ctx["dataset_name"]
    cfg["known_nodes"] = project_ctx["known_nodes"]
    cfg["unknown_nodes_list"] = project_ctx["unknown_nodes"]
    cfg["num_locs"] = project_ctx["num_nodes"]
    cfg["num_unsensed_locs"] = project_ctx["unknown_nodes_num"]
    cfg["lag"] = args.input_len
    cfg["horizon"] = args.pred_len
    cfg["input_dim"] = 4
    cfg["output_dim"] = args.c_out
    cfg["device"] = args.device
    cfg["seed"] = args.seed
    return cfg


def build_stbp_cfg(args):
    project_ctx = build_project_family_context(args)
    cfg = _build_family_cfg(args)
    model_cfg = cfg.get("model", {})
    model_cfg.setdefault("in_channel", args.input_len)
    model_cfg.setdefault("out_channel", args.input_len)
    model_cfg.setdefault("hidden_channel", max(192, args.d_model * 3))
    cfg["model"] = model_cfg
    cfg["dataset_name"] = project_ctx["dataset_name"]
    cfg["known_nodes"] = project_ctx["known_nodes"]
    cfg["unknown_nodes_list"] = project_ctx["unknown_nodes"]
    cfg["base_node_size"] = project_ctx["num_nodes"]
    cfg["dropout"] = float(cfg.get("dropout", 0.0))
    cfg["y_len"] = args.pred_len
    cfg["x_len"] = args.input_len
    cfg["year"] = 0
    return cfg


def build_mcpst_fsl_cfg(args):
    project_ctx = build_project_family_context(args)
    cfg = _build_family_cfg(args)
    family_cfg = getattr(args, "model_family_cfg", {}) or {}
    cfg = deep_merge_dict(cfg, family_cfg.get("recipe", {}))

    model_cfg = cfg.get("model", {})
    model_cfg = deep_merge_dict(model_cfg, family_cfg.get("model", {}))
    model_cfg.setdefault("input_dim", args.c_out)
    model_cfg.setdefault("hidden_dim", args.d_model)
    model_cfg.setdefault("pred_len", args.pred_len)
    model_cfg.setdefault("spatial_dim", 4)
    model_cfg.setdefault("num_diffusion_steps", 3)
    model_cfg.setdefault("num_oscillator_steps", 4)
    model_cfg.setdefault("num_eigen_vectors", 6)
    model_cfg.setdefault("node_minibatch_cpu", 2048)
    model_cfg.setdefault("node_minibatch_gpu", 1024)
    cfg["model"] = model_cfg

    dataset_key = str(cfg.get("training", {}).get("test_dataset", project_ctx["dataset_name"])).lower()
    dataset_cfg = cfg.get("data", {})
    dataset_cfg.setdefault(dataset_key, {})
    dataset_cfg[dataset_key]["node_num"] = project_ctx["num_nodes"]
    cfg["data"] = dataset_cfg

    training_cfg = cfg.get("training", {})
    training_cfg["test_dataset"] = dataset_key
    cfg["training"] = training_cfg
    cfg["dataset_name"] = project_ctx["dataset_name"]
    cfg["known_nodes"] = project_ctx["known_nodes"]
    cfg["unknown_nodes_list"] = project_ctx["unknown_nodes"]
    cfg["num_nodes"] = project_ctx["num_nodes"]
    cfg["input_dim"] = args.c_out
    cfg["output_dim"] = args.c_out
    cfg["seq_len"] = args.input_len
    cfg["horizon"] = args.pred_len
    return cfg
