import torch
import torch.nn as nn

from model.comparison_model.family_utils import SourceNativeV4Mixin, build_mcpst_fsl_cfg
from .model import MCPST_FSL as MCPST_FSL_Backbone


class Model(SourceNativeV4Mixin, nn.Module):

    def __init__(self, configs, runtime_context=None):
        super().__init__()
        self.cfg = build_mcpst_fsl_cfg(configs, runtime_context)
        self.raw_enc_in = int(getattr(configs, "raw_enc_in", configs.enc_in))
        self.full_num_nodes = int(self.cfg["num_nodes"])
        self.known_nodes = list(self.cfg["known_nodes"])
        self.output_dim = configs.c_out
        self.pred_len = configs.pred_len
        self.backbone = MCPST_FSL_Backbone(self.cfg)
        self.register_buffer("full_adj", runtime_context.adjacency.detach().clone(), persistent=False)

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
        # Keep the source physics regularisers that do not require inventing
        # unmasked labels outside the lifecycle cohort. Flow supervision itself
        # remains the runner's masked source-declared MSE.
        physics = (outputs["order_parameter"] - 0.5).square().mean()
        physics = physics + 0.1 * outputs["spectral_gap"].clamp(0, 1).square().mean()
        consistency = (outputs["physics_weights"].sum() - 1.0).square()
        return {"pred": pred, "aux_losses": {
            "mcpst_physics": 0.03 * physics,
            "mcpst_consistency": 0.015 * consistency,
        }}
