from types import SimpleNamespace

import torch
import torch.nn as nn

from lib.utils import load_project_adjacency, load_project_positions
from model.comparison_model.family_utils import build_slpf_cfg
from .SLPF import SLPF as SLPFBackbone


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.cfg = build_slpf_cfg(configs)
        self.full_num_nodes = int(self.cfg["num_locs"])
        self.known_nodes = list(self.cfg["known_nodes"])
        self.output_dim = configs.c_out

        model_cfg = self.cfg["model"]
        backbone_args = SimpleNamespace(
            num_locs=self.full_num_nodes,
            num_unsensed_locs=0,
            lag=int(self.cfg["lag"]),
            input_dim=int(self.cfg["input_dim"]),
            embed_dim=int(model_cfg["embed_dim"]),
            horizon=int(self.cfg["horizon"]),
            num_layer=int(model_cfg["num_layer"]),
            device=self.cfg["device"],
        )
        full_adj = load_project_adjacency({"data_path": configs.data_path})
        self.backbone = SLPFBackbone(
            sensed_locations=list(range(self.full_num_nodes)),
            unsensed_locations=[],
            A=full_adj,
            args=backbone_args,
            positions=load_project_positions({"data_path": configs.data_path}),
        )
        self.backbone.stage = "forecast"

    def _resolve_current_node_ids(self, current_num_nodes):
        if current_num_nodes == len(self.known_nodes):
            return self.known_nodes
        if current_num_nodes == self.full_num_nodes:
            return list(range(self.full_num_nodes))
        return list(range(current_num_nodes))

    def _scatter_signal(self, x_enc, current_node_ids):
        if x_enc.shape[2] == self.full_num_nodes:
            return x_enc[..., : self.output_dim]
        full_x = torch.zeros(x_enc.shape[0], x_enc.shape[1], self.full_num_nodes, self.output_dim, device=x_enc.device, dtype=x_enc.dtype)
        full_x[..., current_node_ids, :] = x_enc[..., : self.output_dim]
        return full_x

    def _build_rank_feature(self, values, current_node_ids):
        rank = torch.zeros(values.shape[0], values.shape[1], self.full_num_nodes, 1, device=values.device, dtype=values.dtype)
        local_values = values[..., current_node_ids, 0]
        local_rank = torch.argsort(local_values, dim=-1).to(dtype=values.dtype).unsqueeze(-1)
        rank[..., current_node_ids, :] = local_rank
        return rank

    def _build_temporal_features(self, x_enc, x_t_mark):
        if x_t_mark is None or x_t_mark.shape[-1] < 3:
            raise ValueError("SLPF requires x_t_mark with minute/hour/day-of-week features")
        minute = x_t_mark[..., 0].to(x_enc.device, dtype=x_enc.dtype)
        hour = x_t_mark[..., 1].to(x_enc.device, dtype=x_enc.dtype)
        dow = x_t_mark[..., 2].to(x_enc.device, dtype=x_enc.dtype)
        tod = hour * 12.0 + torch.floor(minute / 5.0)
        tod = tod.unsqueeze(2).expand(-1, -1, self.full_num_nodes).unsqueeze(-1)
        dow = dow.unsqueeze(2).expand(-1, -1, self.full_num_nodes).unsqueeze(-1)
        return tod, dow

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        current_node_ids = self._resolve_current_node_ids(x_enc.shape[2])
        full_x = self._scatter_signal(x_enc, current_node_ids)
        tod, dow = self._build_temporal_features(full_x, x_t_mark)
        rank = self._build_rank_feature(full_x, current_node_ids)
        full_features = torch.cat([full_x, tod, dow, rank], dim=-1)
        pred = self.backbone(full_features).contiguous()
        if x_enc.shape[2] != self.full_num_nodes:
            pred = pred[..., current_node_ids, :]
        return {"pred": pred}
