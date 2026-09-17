import torch
import torch.nn as nn

from model.comparison_model.family_utils import (
    SourceNativeV4Mixin, build_gwnet_cfg, normalize_adj_mx,
    source_native_input)
from .source_adapter import GWNET as GWNETBackbone


class Model(SourceNativeV4Mixin, nn.Module):

    def __init__(self, configs, runtime_context=None):
        super().__init__()
        self.cfg = build_gwnet_cfg(configs, runtime_context)
        self.raw_enc_in = int(getattr(configs, "raw_enc_in", configs.enc_in))
        self.pred_len = configs.pred_len
        self.output_dim = configs.c_out
        self.full_num_nodes = self.cfg["node_num"]
        self.known_nodes = self.cfg["known_nodes"]
        model_cfg = self.cfg["model"]

        full_adj = runtime_context.adjacency.detach().cpu().numpy()
        supports = normalize_adj_mx(full_adj, model_cfg["adj_type"])
        supports = [torch.tensor(item, dtype=torch.float32) for item in supports]

        self.backbone = GWNETBackbone(
            node_num=self.full_num_nodes,
            input_dim=self.cfg["input_dim"],
            output_dim=self.cfg["output_dim"],
            seq_len=self.cfg["seq_len"],
            horizon=self.cfg["horizon"],
            supports=supports,
            adp_adj=model_cfg["adp_adj"],
            adjacency=runtime_context.adjacency,
            positions=runtime_context.positions,
            dropout=model_cfg["dropout"],
            residual_channels=model_cfg["residual_channels"],
            dilation_channels=model_cfg["dilation_channels"],
            skip_channels=model_cfg["skip_channels"],
            end_channels=model_cfg["end_channels"],
            kernel_size=model_cfg["kernel_size"],
            blocks=model_cfg["blocks"],
            layers=model_cfg["layers"],
        )

    def _resolve_current_node_ids(self, current_num_nodes):
        if current_num_nodes == len(self.known_nodes):
            return self.known_nodes
        if current_num_nodes == self.full_num_nodes:
            return list(range(self.full_num_nodes))
        return list(range(current_num_nodes))

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        current_node_ids = self._resolve_current_node_ids(x_enc.shape[2])
        self.backbone.current_node_ids = current_node_ids
        # Source-native: Graph-WaveNet takes the traffic series.
        pred = self.backbone(source_native_input(x_enc, self.raw_enc_in))
        if pred.shape[-1] != 1:
            pred = pred[..., -1:]
        pred = pred.squeeze(-1).reshape(pred.shape[0], self.output_dim, self.pred_len, pred.shape[2]).permute(0, 2, 3, 1).contiguous()
        return {"pred": pred, "source_raw_mae": True}
