import os
import pickle
from pathlib import Path

import numpy as np
import torch


from lib.v4_payload import load_raw_payload as _load_raw_payload


def load_project_adjacency(cfg):
    raw = _load_raw_payload(cfg["data_path"])
    return torch.tensor(np.asarray(raw["adj"]), dtype=torch.float32)


def load_project_positions(cfg):
    """Coordinates, or None when the payload ships none.

    A v4 payload without coordinates omits `pos_stamp` entirely rather than
    writing zeros or node indices, which would look like an answer. Callers must
    handle None; ALFTemporal falls back to a graph-native layout and
    StructuralNodeEncoder already has a topology-only path.
    """
    raw = _load_raw_payload(cfg["data_path"])
    positions = raw.get("pos_stamp") if hasattr(raw, "get") else None
    if positions is None:
        return None
    return np.asarray(positions, dtype=np.float32)


def infer_project_num_nodes(cfg):
    raw = _load_raw_payload(cfg["data_path"])
    return int(raw["data"].shape[1])


def resolve_known_unknown_nodes(cfg):
    num_nodes = infer_project_num_nodes(cfg)
    unknown_nodes_path = cfg.get("unknown_nodes_path") or ""
    unknown_nodes_list = cfg.get("unknown_nodes_list", []) or []
    unknown_nodes_num = int(cfg.get("unknown_nodes_num", 0) or 0)
    seed = int(cfg.get("seed", 0))

    if unknown_nodes_path and os.path.exists(unknown_nodes_path):
        unknown_nodes = np.asarray(np.load(unknown_nodes_path), dtype=int)
    elif unknown_nodes_list:
        unknown_nodes = np.asarray(unknown_nodes_list, dtype=int)
    elif unknown_nodes_num > 0:
        rng = np.random.default_rng(seed)
        unknown_nodes = rng.choice(np.arange(num_nodes), size=min(unknown_nodes_num, num_nodes), replace=False)
    else:
        unknown_nodes = np.asarray([], dtype=int)

    known_nodes = np.setdiff1d(np.arange(num_nodes), unknown_nodes)
    return known_nodes.tolist(), unknown_nodes.tolist()


def _default_stgode_asset_dir(cfg):
    return Path(cfg["data_path"]).resolve().parent / "stgode"


def resolve_stgode_matrix_paths(cfg):
    assets = cfg.get("assets", {}) or {}
    semantic_matrix_path = (
        assets.get("semantic_matrix_path")
        or assets.get("dtw_matrix_path")
        or (_default_stgode_asset_dir(cfg) / "dtw_matrix_normalized.pkl").as_posix()
    )
    spatial_matrix_path = (
        assets.get("spatial_matrix_path")
        or assets.get("sp_matrix_path")
        or (_default_stgode_asset_dir(cfg) / "sp_matrix_normalized.pkl").as_posix()
    )
    return semantic_matrix_path, spatial_matrix_path


def _load_matrix(path):
    try:
        matrix = np.load(path, allow_pickle=True)
        if isinstance(matrix, np.ndarray):
            return torch.tensor(matrix, dtype=torch.float32)
        if isinstance(matrix, dict) and "matrix" in matrix:
            return torch.tensor(np.asarray(matrix["matrix"]), dtype=torch.float32)
    except Exception:
        pass
    with open(path, "rb") as handle:
        matrix = pickle.load(handle)
    return torch.tensor(np.asarray(matrix), dtype=torch.float32)


class DataTrimer:

    def __init__(self, cfg):
        self.cfg = cfg
        self.random_seed = int(cfg.get("seed", 0))
        self.num_nodes = int(cfg.get("num_nodes", infer_project_num_nodes(cfg)))

    def __call__(self):
        adj = load_project_adjacency(self.cfg)
        train_nodes, unknown_nodes = resolve_known_unknown_nodes(self.cfg)
        val_nodes = train_nodes.copy()
        train_num_nodes = len(train_nodes)
        val_num_nodes = len(val_nodes)

        train_graph = adj[train_nodes, :][:, train_nodes]
        val_graph = adj[val_nodes, :][:, val_nodes]
        test_graph = adj

        return (
            [train_num_nodes, val_num_nodes, self.num_nodes],
            [train_nodes, val_nodes],
            [train_graph, val_graph, test_graph],
        )


class DataTrimer_STGODE:

    def __init__(self, cfg):
        self.cfg = cfg
        self.random_seed = int(cfg.get("seed", 0))
        self.num_nodes = int(cfg.get("num_nodes", infer_project_num_nodes(cfg)))
        semantic_matrix_path, spatial_matrix_path = resolve_stgode_matrix_paths(cfg)
        if not os.path.exists(semantic_matrix_path) or not os.path.exists(spatial_matrix_path):
            raise FileNotFoundError(
                "STGODE matrix assets are missing. "
                f"semantic_matrix_path={semantic_matrix_path}, spatial_matrix_path={spatial_matrix_path}"
            )
        self.se = _load_matrix(semantic_matrix_path)
        self.sp = _load_matrix(spatial_matrix_path)

    def __call__(self):
        train_nodes, unknown_nodes = resolve_known_unknown_nodes(self.cfg)
        val_nodes = train_nodes.copy()
        train_num_nodes = len(train_nodes)
        val_num_nodes = len(val_nodes)

        se_train_graph = self.se[train_nodes, :][:, train_nodes]
        se_val_graph = self.se[val_nodes, :][:, val_nodes]
        se_test_graph = self.se

        sp_train_graph = self.sp[train_nodes, :][:, train_nodes]
        sp_val_graph = self.sp[val_nodes, :][:, val_nodes]
        sp_test_graph = self.sp

        return (
            [train_num_nodes, val_num_nodes, self.num_nodes],
            [train_nodes, val_nodes],
            [se_train_graph, se_val_graph, se_test_graph],
            [sp_train_graph, sp_val_graph, sp_test_graph],
        )
