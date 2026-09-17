import torch
import torch.nn as nn

from model.comparison_model.family_utils import (
    SourceNativeV4Mixin, build_stfit_like_cfg, source_native_input)
from model.comparison_model.stgcn_n.stgcn import STGCN_n


STGCN_BASELINE_DEFAULTS = {
    "graph_learning": False,
    "dynamic_graph": False,
    "dynamic_mix": False,
    "aug_node": False,
    "sim_loss": False,
    "fst_loss": False,
    "is_vae": False,
}
class Model(SourceNativeV4Mixin, nn.Module):

    def __init__(self, configs, runtime_context=None):
        super().__init__()
        self.cfg = build_stfit_like_cfg(
            configs, STGCN_BASELINE_DEFAULTS, runtime_context)
        self.raw_enc_in = int(getattr(configs, "raw_enc_in", configs.enc_in))
        self.pred_len = configs.pred_len
        self.output_dim = configs.c_out
        self.backbone = STGCN_n(self.cfg)

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        # Source-native input: the traffic series only. Lifecycle channels are
        # ALFNet's method inputs, not this baseline's.
        history_data = source_native_input(x_enc, self.raw_enc_in).permute(0, 2, 3, 1).contiguous()
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
        return {"pred": pred, "source_raw_mae": True}
