import torch
import torch.nn as nn

from model.comparison_model.family_utils import build_stid_cfg
from .stid import STID as STIDBackbone


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.cfg = build_stid_cfg(configs)
        self.full_num_nodes = self.cfg["node_num"]
        self.known_nodes = self.cfg["known_nodes"]
        model_cfg = self.cfg["model"]
        self.use_time_of_day = bool(model_cfg["if_T_i_D"])
        self.use_day_of_week = bool(model_cfg["if_D_i_W"])
        self.time_of_day_size = int(model_cfg["time_of_day_size"])
        self.day_of_week_size = int(model_cfg["day_of_week_size"])
        self.backbone = STIDBackbone(
            node_num=self.cfg["node_num"],
            input_dim=self.cfg["model"]["input_dim"],
            output_dim=self.cfg["output_dim"],
            seq_len=self.cfg["seq_len"],
            horizon=self.cfg["horizon"],
            node_dim=model_cfg["node_dim"],
            input_len=model_cfg["input_len"],
            embed_dim=model_cfg["embed_dim"],
            output_len=model_cfg["output_len"],
            num_layer=model_cfg["num_layer"],
            temp_dim_tid=model_cfg["temp_dim_tid"],
            temp_dim_diw=model_cfg["temp_dim_diw"],
            time_of_day_size=model_cfg["time_of_day_size"],
            day_of_week_size=model_cfg["day_of_week_size"],
            if_T_i_D=model_cfg["if_T_i_D"],
            if_D_i_W=model_cfg["if_D_i_W"],
            if_node=model_cfg["if_node"],
        )

    def _resolve_current_node_ids(self, current_num_nodes):
        if current_num_nodes == len(self.known_nodes):
            return self.known_nodes
        if current_num_nodes == self.full_num_nodes:
            return list(range(self.full_num_nodes))
        return list(range(current_num_nodes))

    def _build_stid_input(self, x_enc, x_t_mark):
        features = [x_enc]
        if self.use_time_of_day:
            if x_t_mark is None or x_t_mark.shape[-1] < 2:
                raise ValueError("STID requires x_t_mark with at least minute/hour features when if_T_i_D=True")
            minute = x_t_mark[..., 0].to(x_enc.device, dtype=x_enc.dtype)
            hour = x_t_mark[..., 1].to(x_enc.device, dtype=x_enc.dtype)
            tod = ((hour * 60.0) + minute) / float(self.time_of_day_size)
            tod = tod.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_enc.shape[2], -1)
            features.append(tod)
        if self.use_day_of_week:
            if x_t_mark is None or x_t_mark.shape[-1] < 3:
                raise ValueError("STID requires x_t_mark with day-of-week feature when if_D_i_W=True")
            dow = x_t_mark[..., 2].to(x_enc.device, dtype=x_enc.dtype) / float(self.day_of_week_size)
            dow = dow.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_enc.shape[2], -1)
            features.append(dow)
        return torch.cat(features, dim=-1)

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        model_input = self._build_stid_input(x_enc, x_t_mark)
        self.backbone.current_node_ids = self._resolve_current_node_ids(model_input.shape[2])
        pred = self.backbone(model_input)
        return {"pred": pred.contiguous()}
