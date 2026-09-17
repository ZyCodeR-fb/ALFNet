"""Official full-graph UIGNN quantile/reconstruction branch, multi-horizon repair."""
import torch
from torch import nn
from lib.v4_batch import read_common,bind_static_assets
from .structure_full import IGNNK


class Model(nn.Module):
    @classmethod
    def from_runtime_context(cls,configs,context):return cls(configs,context)

    def __init__(self,configs,runtime_context):
        super().__init__()
        bind_static_assets(self,configs,runtime_context,'UIGNN')
        self.raw_enc_in=int(configs.raw_enc_in)
        self.full_num_nodes=runtime_context.num_nodes
        self.known_nodes=list(runtime_context.known_nodes)
        self.horizon=int(configs.pred_len)
        self.core=IGNNK(int(configs.input_len),100,1,forecasting=True,EDL=False,mc_dropout=False)
        # Independent quantiles for every horizon; no repeated one-step prediction.
        self.core.lin_out=nn.Linear(int(configs.input_len),3*self.horizon)
        a=runtime_context.adjacency.float()
        self.register_buffer('forward_walk',(a/a.sum(1,keepdim=True).clamp_min(1e-12)).T)
        at=a.T
        self.register_buffer('backward_walk',(at/at.sum(1,keepdim=True).clamp_min(1e-12)).T)

    def forward_v4(self,batch,runtime_context,epoch=0,teacher_forcing_mask=None):
        common=read_common(batch,runtime_context,self.static_asset_hash,self.raw_enc_in,
            'UIGNN',self._v4_adjacency_cache)
        raw=common['base_visible_input'][...,0];mask=batch['input_available_mask']
        x=torch.where(mask & torch.isfinite(raw),raw,torch.zeros_like(raw))
        forecasts,_,reconstruction=self.core(x,self.forward_walk,self.backward_walk)
        q=forecasts.reshape(x.shape[0],self.horizon,3,x.shape[2]).permute(0,1,3,2)
        rec=((reconstruction-x).square()[mask].mean() if mask.any() else reconstruction.sum()*0.)
        return {'pred':q[...,1:2],'source_quantiles':q,'source_objective':'uignn_quantile',
            'aux_losses':{'source_reconstruction':rec}}
