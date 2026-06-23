import math

import numpy as np
import torch
import torch.nn as nn

from lib.utils import load_project_adjacency, load_project_positions
from model.family_utils import build_patchstg_cfg
from layers.source_graph_unet import graph_unet_parts
from .patchstg import PatchRevIN, PatchSTG as PatchSTGBackbone


def _augment_align(dist_matrix, auglen):
    sorted_idx = np.argsort(dist_matrix.reshape(-1) * -1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(int(idx))
        if len(augidx) == auglen:
            break
    return np.asarray(augidx, dtype=int)


def _reorder_data(parts_idx, adj, patch_size):
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if patch_size - part_idx.shape[0] > 0:
            local_part_idx = _augment_align(part_dist, patch_size - part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx

        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0]) + patch_size * i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

    return ori_parts_idx, reo_parts_idx, reo_all_idx


def _kd_tree(locations, times, axis):
    sorted_idx = np.argsort(locations[axis])
    part1 = np.sort(sorted_idx[: locations.shape[1] // 2])
    part2 = np.sort(sorted_idx[locations.shape[1] // 2 :])
    if times == 1:
        return [part1, part2]
    parts = []
    left_parts = _kd_tree(locations[:, part1], times - 1, axis ^ 1)
    right_parts = _kd_tree(locations[:, part2], times - 1, axis ^ 1)
    for part in left_parts:
        parts.append(part1[part])
    for part in right_parts:
        parts.append(part2[part])
    return parts


def _split_order_to_parts(order, part_count):
    return [np.sort(part.astype(int)) for part in np.array_split(np.asarray(order, dtype=int), part_count) if part.size > 0]


def _spectral_graph_parts(adj, part_count):
    adj_sym = (adj + adj.T) / 2.0
    degree = adj_sym.sum(axis=1)
    if np.count_nonzero(degree) <= 1:
        return _split_order_to_parts(np.arange(adj.shape[0]), part_count)
    degree_safe = np.where(degree > 0, degree, 1.0)
    d_inv_sqrt = np.diag(np.power(degree_safe, -0.5))
    laplacian = np.eye(adj.shape[0]) - d_inv_sqrt @ adj_sym @ d_inv_sqrt
    _, eigvecs = np.linalg.eigh(laplacian)
    fiedler_idx = 1 if eigvecs.shape[1] > 1 else 0
    order = np.lexsort((np.arange(adj.shape[0]), eigvecs[:, fiedler_idx]))
    return _split_order_to_parts(order, part_count)


def _adjacency_walk_parts(adj, part_count):
    adj_sym = (adj + adj.T) / 2.0
    degree = adj_sym.sum(axis=1)
    visited = np.zeros(adj.shape[0], dtype=bool)
    order = []
    while len(order) < adj.shape[0]:
        remaining = np.where(~visited)[0]
        if remaining.size == 0:
            break
        seed = int(remaining[np.argmax(degree[remaining])])
        queue = [seed]
        visited[seed] = True
        while queue:
            node = queue.pop(0)
            order.append(node)
            neighbors = np.where((adj_sym[node] > 0) & (~visited))[0]
            if neighbors.size == 0:
                continue
            neighbors = neighbors[np.argsort(adj_sym[node, neighbors] * -1)]
            for neighbor in neighbors:
                visited[neighbor] = True
                queue.append(int(neighbor))
    return _split_order_to_parts(order, part_count)


def _build_spatial_layout(positions, adj, recur_times, spa_patchsize, layout_type="kd_tree"):
    layout_type = str(layout_type).lower()
    part_count = 2 ** int(recur_times)
    if layout_type in {"kd_tree", "kdtree", "position"}:
        parts_idx = _kd_tree(positions.T, recur_times, 0)
    elif layout_type in {"spectral", "graph_spectral"}:
        parts_idx = _spectral_graph_parts(adj, part_count)
    elif layout_type in {"adjacency", "adjacency_walk", "graph_walk"}:
        parts_idx = _adjacency_walk_parts(adj, part_count)
    elif layout_type in {"graph_unet", "gunet", "gpool"}:
        parts_idx = graph_unet_parts(adj, recur_times)
    else:
        raise ValueError(f"Unsupported PatchSTG spatial_layout: {layout_type}")
    ori_parts_idx, reo_parts_idx, reo_all_idx = _reorder_data(parts_idx, adj, spa_patchsize)
    return ori_parts_idx.tolist(), reo_parts_idx.tolist(), reo_all_idx.tolist(), len(parts_idx)


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.cfg = build_patchstg_cfg(configs)
        model_cfg = self.cfg["model"]
        self.full_num_nodes = int(self.cfg["node_num"])
        self.known_nodes = list(self.cfg["known_nodes"])
        self.output_dim = configs.c_out
        self.use_revin = bool(model_cfg.get("use_revin", False))
        self.revin = PatchRevIN(
            self.output_dim,
            eps=float(model_cfg.get("revin_eps", 1e-5)),
            affine=bool(model_cfg.get("revin_affine", True)),
        ) if self.use_revin else None

        full_adj = load_project_adjacency({"data_path": configs.data_path}).cpu().numpy()
        positions = load_project_positions({"data_path": configs.data_path})
        ori_parts_idx, reo_parts_idx, reo_all_idx, spa_patchnum = _build_spatial_layout(
            positions,
            full_adj,
            int(model_cfg["recur_times"]),
            int(model_cfg["spa_patchsize"]),
            model_cfg.get("spatial_layout", "kd_tree"),
        )
        self.ori_parts_idx = ori_parts_idx
        self.reo_parts_idx = reo_parts_idx
        self.reo_all_idx = reo_all_idx

        tem_patchsize = int(model_cfg["tem_patchsize"])
        tem_patchnum = int(model_cfg["tem_patchnum"])
        if tem_patchsize * tem_patchnum != configs.pred_len:
            raise ValueError("PatchSTG requires tem_patchsize * tem_patchnum == pred_len under the current adapter")

        self.backbone = PatchSTGBackbone(
            tem_patchsize=tem_patchsize,
            tem_patchnum=tem_patchnum,
            node_num=self.full_num_nodes,
            spa_patchsize=int(model_cfg["spa_patchsize"]),
            spa_patchnum=spa_patchnum,
            tod=int(model_cfg["tod"]),
            dow=int(model_cfg["dow"]),
            layers=int(model_cfg["layers"]),
            factors=int(model_cfg["factors"]),
            input_dims=int(model_cfg["input_dims"]),
            node_dims=int(model_cfg["node_dims"]),
            tod_dims=int(model_cfg["tod_dims"]),
            dow_dims=int(model_cfg["dow_dims"]),
            ori_parts_idx=self.ori_parts_idx,
            reo_parts_idx=self.reo_parts_idx,
            reo_all_idx=self.reo_all_idx,
            attention_order=model_cfg.get("attention_order", "depth_breadth"),
            shift_size=int(model_cfg.get("attention_shift_size", 0)),
            attention_type=model_cfg.get("attention_type", "timm"),
            lepe_kernel_size=int(model_cfg.get("lepe_kernel_size", 5)),
            crossformer_router_factor=int(model_cfg.get("crossformer_router_factor", 4)),
            crossformer_dropout=float(model_cfg.get("crossformer_dropout", 0.1)),
            use_timefilter_patch_mixer=bool(model_cfg.get("use_timefilter_patch_mixer", False)),
            timefilter_n_heads=int(model_cfg.get("timefilter_n_heads", 4)),
            timefilter_layers=int(model_cfg.get("timefilter_layers", 1)),
            timefilter_d_ff=int(model_cfg.get("timefilter_d_ff", 0) or 0),
            timefilter_top_p=float(model_cfg.get("timefilter_top_p", 0.5)),
            timefilter_alpha=float(model_cfg.get("timefilter_alpha", 0.1)),
            timefilter_dropout=float(model_cfg.get("timefilter_dropout", 0.1)),
            timefilter_residual_scale=float(model_cfg.get("timefilter_residual_scale", 0.10)),
            use_duet_patch_router=bool(model_cfg.get("use_duet_patch_router", False)),
            duet_num_experts=int(model_cfg.get("duet_num_experts", 2)),
            duet_k=int(model_cfg.get("duet_k", 1)),
            duet_hidden_size=int(model_cfg.get("duet_hidden_size", 64)),
            duet_moving_avg=int(model_cfg.get("duet_moving_avg", 3)),
            duet_residual_scale=float(model_cfg.get("duet_residual_scale", 0.10)),
            duet_noisy_gating=bool(model_cfg.get("duet_noisy_gating", True)),
            use_staeformer_adapter=bool(model_cfg.get("use_staeformer_adapter", False)),
            staeformer_adapter_mode=model_cfg.get("staeformer_adapter_mode", "temporal"),
            staeformer_adapter_layers=int(model_cfg.get("staeformer_adapter_layers", 1)),
            staeformer_num_heads=int(model_cfg.get("staeformer_num_heads", 4)),
            staeformer_feed_forward_dim=int(model_cfg.get("staeformer_feed_forward_dim", 256)),
            staeformer_dropout=float(model_cfg.get("staeformer_dropout", 0.1)),
        )

    def _resolve_current_node_ids(self, current_num_nodes):
        if current_num_nodes == len(self.known_nodes):
            return self.known_nodes
        if current_num_nodes == self.full_num_nodes:
            return list(range(self.full_num_nodes))
        return list(range(current_num_nodes))

    def _build_temporal_features(self, x_enc, x_t_mark):
        if x_t_mark is None or x_t_mark.shape[-1] < 3:
            raise ValueError("PatchSTG requires x_t_mark with minute/hour/day-of-week features")
        minute = x_t_mark[..., 0].to(x_enc.device, dtype=x_enc.dtype)
        hour = x_t_mark[..., 1].to(x_enc.device, dtype=x_enc.dtype)
        dow = x_t_mark[..., 2].to(x_enc.device, dtype=x_enc.dtype)
        tod = hour * 12.0 + torch.floor(minute / 5.0)
        te = torch.stack([tod, dow], dim=-1).unsqueeze(2).expand(-1, -1, self.full_num_nodes, -1)
        return te

    def _scatter_to_full_nodes(self, x_enc, current_node_ids):
        if x_enc.shape[2] == self.full_num_nodes:
            return x_enc[..., : self.output_dim]
        full_x = torch.zeros(x_enc.shape[0], x_enc.shape[1], self.full_num_nodes, self.output_dim, device=x_enc.device, dtype=x_enc.dtype)
        full_x[..., current_node_ids, :] = x_enc[..., : self.output_dim]
        return full_x

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        current_node_ids = self._resolve_current_node_ids(x_enc.shape[2])
        full_x = self._scatter_to_full_nodes(x_enc, current_node_ids)
        full_te = self._build_temporal_features(full_x, x_t_mark)
        backbone_x = self.revin(full_x, "norm") if self.revin is not None else full_x
        pred = self.backbone(backbone_x, full_te).contiguous()
        if self.revin is not None:
            pred = self.revin(pred, "denorm").contiguous()
        if x_enc.shape[2] != self.full_num_nodes:
            pred = pred[..., current_node_ids, :]
        return {"pred": pred}
