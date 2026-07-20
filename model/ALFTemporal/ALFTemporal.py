import numpy as np
import torch
import torch.nn as nn

from lib.utils import load_project_adjacency, load_project_positions
from model.comparison_model.family_utils import build_patchstg_cfg
from .patchstg import PatchSTG as PatchSTGBackbone


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
            local_part_idx = _augment_align(
                part_dist, patch_size - part_idx.shape[0]
            )
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx

        reo_parts_idx = np.concatenate([
            reo_parts_idx,
            np.arange(part_idx.shape[0]) + patch_size * i,
        ])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])
    return ori_parts_idx, reo_parts_idx, reo_all_idx


def _kd_tree(locations, times, axis):
    sorted_idx = np.argsort(locations[axis])
    part1 = np.sort(sorted_idx[:locations.shape[1] // 2])
    part2 = np.sort(sorted_idx[locations.shape[1] // 2:])
    if times == 1:
        return [part1, part2]
    parts = []
    for part in _kd_tree(locations[:, part1], times - 1, axis ^ 1):
        parts.append(part1[part])
    for part in _kd_tree(locations[:, part2], times - 1, axis ^ 1):
        parts.append(part2[part])
    return parts


def _build_spatial_layout(
    positions,
    adj,
    recur_times,
    spa_patchsize,
    layout_type="kd_tree",
):
    if str(layout_type).lower() not in {"kd_tree", "kdtree", "position"}:
        raise ValueError(
            "ALFTemporal only retains the final KDTree spatial layout, "
            f"got spatial_layout={layout_type}."
        )
    parts_idx = _kd_tree(positions.T, int(recur_times), 0)
    ori_parts_idx, reo_parts_idx, reo_all_idx = _reorder_data(
        parts_idx,
        adj,
        int(spa_patchsize),
    )
    return (
        ori_parts_idx.tolist(),
        reo_parts_idx.tolist(),
        reo_all_idx.tolist(),
        len(parts_idx),
    )


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.cfg = build_patchstg_cfg(configs)
        model_cfg = self.cfg["model"]
        self.full_num_nodes = int(self.cfg["node_num"])
        self.known_nodes = list(self.cfg["known_nodes"])
        self.output_dim = configs.c_out

        if bool(model_cfg.get("use_revin", False)):
            raise ValueError("ALFTemporal no longer includes the failed PatchRevIN variant.")
        full_adj = load_project_adjacency(
            {"data_path": configs.data_path}
        ).cpu().numpy()
        positions = load_project_positions({"data_path": configs.data_path})
        ori_parts_idx, reo_parts_idx, reo_all_idx, spa_patchnum = (
            _build_spatial_layout(
                positions,
                full_adj,
                int(model_cfg["recur_times"]),
                int(model_cfg["spa_patchsize"]),
                model_cfg.get("spatial_layout", "kd_tree"),
            )
        )
        self.ori_parts_idx = ori_parts_idx
        self.reo_parts_idx = reo_parts_idx
        self.reo_all_idx = reo_all_idx

        tem_patchsize = int(model_cfg["tem_patchsize"])
        tem_patchnum = int(model_cfg["tem_patchnum"])
        if tem_patchsize * tem_patchnum != configs.pred_len:
            raise ValueError(
                "PatchSTG requires tem_patchsize * tem_patchnum == pred_len "
                "under the current adapter"
            )
        if str(model_cfg.get("attention_type", "")).lower() not in {
            "crossformer_tsa",
            "crossformer",
            "tsa",
        }:
            raise ValueError(
                "ALFTemporal requires attention_type=crossformer_tsa after "
                "final-model cleanup."
            )
        if not bool(model_cfg.get("use_timefilter_patch_mixer", False)):
            raise ValueError(
                "ALFTemporal requires use_timefilter_patch_mixer=true after "
                "final-model cleanup."
            )

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
            crossformer_router_factor=int(
                model_cfg.get("crossformer_router_factor", 4)
            ),
            crossformer_dropout=float(model_cfg.get("crossformer_dropout", 0.1)),
            timefilter_n_heads=int(model_cfg.get("timefilter_n_heads", 4)),
            timefilter_layers=int(model_cfg.get("timefilter_layers", 1)),
            timefilter_d_ff=int(model_cfg.get("timefilter_d_ff", 0) or 0),
            timefilter_top_p=float(model_cfg.get("timefilter_top_p", 0.5)),
            timefilter_alpha=float(model_cfg.get("timefilter_alpha", 0.1)),
            timefilter_dropout=float(model_cfg.get("timefilter_dropout", 0.1)),
            timefilter_residual_scale=float(
                model_cfg.get("timefilter_residual_scale", 0.10)
            ),
        )

    def _resolve_current_node_ids(self, current_num_nodes):
        if current_num_nodes == len(self.known_nodes):
            return self.known_nodes
        if current_num_nodes == self.full_num_nodes:
            return list(range(self.full_num_nodes))
        return list(range(current_num_nodes))

    def _build_temporal_features(self, x_enc, x_t_mark):
        if x_t_mark is None or x_t_mark.shape[-1] < 3:
            raise ValueError(
                "PatchSTG requires x_t_mark with minute/hour/day-of-week features"
            )
        minute = x_t_mark[..., 0].to(x_enc.device, dtype=x_enc.dtype)
        hour = x_t_mark[..., 1].to(x_enc.device, dtype=x_enc.dtype)
        dow = x_t_mark[..., 2].to(x_enc.device, dtype=x_enc.dtype)
        tod = hour * 12.0 + torch.floor(minute / 5.0)
        return torch.stack([tod, dow], dim=-1).unsqueeze(2).expand(
            -1, -1, self.full_num_nodes, -1
        )

    def _scatter_to_full_nodes(self, x_enc, current_node_ids):
        if x_enc.shape[2] == self.full_num_nodes:
            return x_enc[..., :self.output_dim]
        full_x = torch.zeros(
            x_enc.shape[0],
            x_enc.shape[1],
            self.full_num_nodes,
            self.output_dim,
            device=x_enc.device,
            dtype=x_enc.dtype,
        )
        full_x[..., current_node_ids, :] = x_enc[..., :self.output_dim]
        return full_x

    def forward(
        self,
        adj,
        x_enc,
        x_t_mark=None,
        pos_mark=None,
        x_dec=None,
        dec_t_mark=None,
        epoch=0,
    ):
        current_node_ids = self._resolve_current_node_ids(x_enc.shape[2])
        full_x = self._scatter_to_full_nodes(x_enc, current_node_ids)
        full_te = self._build_temporal_features(full_x, x_t_mark)
        pred = self.backbone(full_x, full_te).contiguous()
        if x_enc.shape[2] != self.full_num_nodes:
            pred = pred[..., current_node_ids, :]
        return {"pred": pred}
