import torch
import torch.nn as nn

from model.comparison_model.family_utils import build_stfit_like_cfg
from model.comparison_model.stgcn_n.stgcn import STGCN_n


STFIT_FULL_DEFAULTS = {
    "graph_learning": True,
    "dynamic_graph": True,
    "dynamic_mix": True,
    "aug_node": True,
    "sim_loss": True,
    "fst_loss": True,
    "is_vae": False,
    "anchor_lambda": 0.5,
    "sparse_threshold": 0.9,
    "gumbel_softmax": True,
    "gumbel_tau": 1.0,
    "vae_loss_weight": 2.0,
}


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.raw_enc_in = getattr(configs, "raw_enc_in", configs.enc_in)
        self.cfg = build_stfit_like_cfg(configs, STFIT_FULL_DEFAULTS)
        self.cfg["model"]["in_channels"] = self.raw_enc_in
        self.cfg["input_size"] = self.raw_enc_in
        self.cfg["vae"]["input_dim"] = self.raw_enc_in
        self.pred_len = configs.pred_len
        self.output_dim = configs.c_out
        self.backbone = STGCN_n(self.cfg)

    def on_train_epoch_start(self, epoch):
        if not self.cfg.get("aug_node", False):
            for param in self.backbone.parameters():
                param.requires_grad_(True)
            return

        aug_node_epoch = epoch % 2 == 0
        for name, param in self.backbone.named_parameters():
            if aug_node_epoch:
                param.requires_grad_("vae" in name)
            else:
                param.requires_grad_("vae" not in name)

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        history_data = x_enc[..., :self.raw_enc_in].permute(0, 2, 3, 1).contiguous()
        if self.training and self.cfg.get("aug_node", False):
            self.backbone.train_num_nodes = history_data.shape[1]
        if x_dec is None:
            future_data = torch.zeros(
                history_data.shape[0],
                history_data.shape[1],
                history_data.shape[2],
                self.pred_len,
                device=history_data.device,
                dtype=history_data.dtype,
            )
        else:
            future_data = x_dec.permute(0, 2, 3, 1).contiguous()
            if future_data.shape[2] < history_data.shape[2]:
                pad = torch.zeros(
                    future_data.shape[0],
                    future_data.shape[1],
                    history_data.shape[2] - future_data.shape[2],
                    future_data.shape[3],
                    device=future_data.device,
                    dtype=future_data.dtype,
                )
                future_data = torch.cat([future_data, pad], dim=2)

        pred, future_data_out, embedding, vae_loss = self.backbone(history_data, future_data)
        pred = pred.permute(0, 2, 1).unsqueeze(-1)

        original_num_nodes = x_enc.shape[2]
        aux_losses = {}
        target_override = None

        aug_node_epoch = self.training and self.cfg.get("aug_node", False) and epoch % 2 == 0
        if aug_node_epoch and pred.shape[2] > original_num_nodes and self.cfg.get("fst_loss", False):
            pred = pred[:, :, original_num_nodes:, :]
            target_override = future_data_out[:, original_num_nodes:, : self.output_dim, :].permute(0, 3, 1, 2).contiguous()
            if torch.is_tensor(vae_loss):
                aux_losses["stfit_vae"] = 0.01 * vae_loss
        else:
            pred = pred[:, :, :original_num_nodes, :]

        output = {"pred": pred}
        if target_override is not None:
            output["target"] = target_override
        if aux_losses:
            output["aux_losses"] = aux_losses
        return output
