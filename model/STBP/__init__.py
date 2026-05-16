from types import SimpleNamespace

import torch
import torch.nn as nn

from lib.utils import load_project_adjacency
from model.family_utils import build_stbp_cfg
from .stbp import STBP_Model


class _SilentLogger:

    def info(self, *args, **kwargs):
        return None


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.cfg = build_stbp_cfg(configs)
        self.full_num_nodes = int(self.cfg["base_node_size"])
        self.known_nodes = list(self.cfg["known_nodes"])
        self.in_channel = int(self.cfg["model"]["in_channel"])
        self.y_len = int(self.cfg["y_len"])
        self.output_dim = configs.c_out

        self.model_args = SimpleNamespace(
            dropout=float(self.cfg["dropout"]),
            y_len=self.y_len,
            base_node_size=self.full_num_nodes,
            year=int(self.cfg["year"]),
            model=self.cfg["model"],
            logger=_SilentLogger(),
        )
        self.backbone = STBP_Model(self.model_args)
        self.register_buffer("full_adj", load_project_adjacency({"data_path": configs.data_path}), persistent=False)

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

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        current_node_ids = self._resolve_current_node_ids(x_enc.shape[2])
        full_x = self._scatter_signal(x_enc, current_node_ids)
        batch_data = SimpleNamespace(x=full_x[..., 0].permute(0, 2, 1).reshape(-1, self.in_channel))
        pred = self.backbone(batch_data, self.full_adj.to(x_enc.device))
        pred = pred.reshape(x_enc.shape[0], self.full_num_nodes, self.y_len).permute(0, 2, 1).unsqueeze(-1).contiguous()
        if x_enc.shape[2] != self.full_num_nodes:
            pred = pred[..., current_node_ids, :]
        return {"pred": pred}
