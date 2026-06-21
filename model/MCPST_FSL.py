import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from lib.utils import load_project_adjacency
from model.family_utils import build_mcpst_fsl_cfg


def _load_reference_model_class():
    model_path = Path(__file__).resolve().parent / "MCPST-FSL" / "model.py"
    spec = importlib.util.spec_from_file_location("mcpst_fsl_reference_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MCPST-FSL model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MCPST_FSL


MCPST_FSL_Backbone = _load_reference_model_class()


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.cfg = build_mcpst_fsl_cfg(configs)
        self.full_num_nodes = int(self.cfg["num_nodes"])
        self.known_nodes = list(self.cfg["known_nodes"])
        self.output_dim = configs.c_out
        self.pred_len = configs.pred_len
        self.backbone = MCPST_FSL_Backbone(self.cfg)
        self.register_buffer("full_adj", load_project_adjacency({"data_path": configs.data_path}), persistent=False)

    def _resolve_current_node_ids(self, current_num_nodes):
        if current_num_nodes == len(self.known_nodes):
            return self.known_nodes
        if current_num_nodes == self.full_num_nodes:
            return list(range(self.full_num_nodes))
        return list(range(current_num_nodes))

    def _scatter_to_full_nodes(self, x_enc, current_node_ids):
        raw_x = x_enc[..., : self.output_dim]
        if raw_x.shape[2] == self.full_num_nodes:
            return raw_x
        full_x = torch.zeros(
            raw_x.shape[0],
            raw_x.shape[1],
            self.full_num_nodes,
            self.output_dim,
            device=raw_x.device,
            dtype=raw_x.dtype,
        )
        full_x[..., current_node_ids, :] = raw_x
        return full_x

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        current_node_ids = self._resolve_current_node_ids(x_enc.shape[2])
        full_x = self._scatter_to_full_nodes(x_enc, current_node_ids)
        outputs = self.backbone(full_x, self.full_adj.to(device=x_enc.device, dtype=x_enc.dtype))
        pred = outputs["flow_predictions"].permute(0, 2, 1).unsqueeze(-1).contiguous()
        if x_enc.shape[2] != self.full_num_nodes:
            pred = pred[..., current_node_ids, :]
        return {"pred": pred}
