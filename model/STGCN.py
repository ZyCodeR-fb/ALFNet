import torch
import torch.nn as nn

from model.family_utils import build_stfit_like_cfg
from model.stgcn_n.stgcn import STGCN_n


STGCN_BASELINE_DEFAULTS = {
    "graph_learning": False,
    "dynamic_graph": False,
    "dynamic_mix": False,
    "aug_node": False,
    "sim_loss": False,
    "fst_loss": False,
    "is_vae": False,
}
class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.cfg = build_stfit_like_cfg(configs, STGCN_BASELINE_DEFAULTS)
        self.pred_len = configs.pred_len
        self.output_dim = configs.c_out
        self.backbone = STGCN_n(self.cfg)

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        history_data = x_enc.permute(0, 2, 3, 1).contiguous()
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
        pred, _, _, _ = self.backbone(history_data, future_data)
        pred = pred.permute(0, 2, 1).unsqueeze(-1)
        return {"pred": pred}
