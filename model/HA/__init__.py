import torch.nn as nn

from model.family_utils import build_ha_cfg
from .ha import HA as HABackbone


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.cfg = build_ha_cfg(configs)
        self.output_dim = configs.c_out
        self.backbone = HABackbone(self.cfg)

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        history_data = x_enc[..., : self.output_dim].permute(0, 2, 3, 1).contiguous()
        pred, _, _, _ = self.backbone(history_data, None)
        pred = pred.permute(0, 2, 1).unsqueeze(-1)
        return {"pred": pred}
