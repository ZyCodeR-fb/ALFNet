"""model1 GinAR core with shared structural parameterization for inductive nodes."""
import math
import torch
from torch import nn
from torch.nn.utils import parametrize
from layers.structural_node_encoder import StructuralNodeEncoder
from lib.v4_batch import read_common, bind_static_assets
from model.comparison_model.family_utils import normalize_adj_mx
from .ginar_arch import GinAR


class StructuralParameter(nn.Module):
    def __init__(self, context, width, kind='embedding'):
        super().__init__()
        self.kind=kind
        self.left=StructuralNodeEncoder(context.adjacency,context.positions,out_dim=width)
        if kind=='matrix':
            self.right=StructuralNodeEncoder(context.adjacency,context.positions,out_dim=width)

    def forward(self, original):
        x=self.left()
        if self.kind=='matrix':return x@self.right().T/math.sqrt(x.shape[0])
        if self.kind=='bias':return x[:,0]
        return x


class SharedAffine(nn.Module):
    def __init__(self, shape, value):
        super().__init__()
        self.value=nn.Parameter(torch.full((shape[0],1),value))

    def forward(self, original):return self.value.expand_as(original)


class Model(nn.Module):
    @classmethod
    def from_runtime_context(cls, configs, context):return cls(configs,context)

    def __init__(self, configs, runtime_context):
        super().__init__()
        bind_static_assets(self,configs,runtime_context,'GinAR')
        self.raw_enc_in=int(configs.raw_enc_in)
        self.full_num_nodes=runtime_context.num_nodes
        self.known_nodes=list(runtime_context.known_nodes)
        supports=normalize_adj_mx(runtime_context.adjacency.cpu().numpy(),'doubletransition')
        self.core=GinAR(configs.input_len,self.full_num_nodes,configs.pred_len,
            self.raw_enc_in,16,8,2,.15,[torch.tensor(x,dtype=torch.float32) for x in supports])
        for cell in (self.core.ginar_first,self.core.ginar_other):
            for module,name,kind,width in ((cell,'GL','embedding',8),(cell.att,'GL','embedding',8),
                    (cell.emb2,'weight','matrix',16),(cell.emb2,'bias','bias',1)):
                getattr(module,name).requires_grad_(False)
                parametrize.register_parametrization(module,name,StructuralParameter(runtime_context,width,kind))
            for name,value in (('weight',1.),('bias',0.)):
                getattr(cell.layernorm,name).requires_grad_(False)
                parametrize.register_parametrization(cell.layernorm,name,SharedAffine(cell.layernorm.normalized_shape,value))
            # Declared but never used by the official forward.
            cell.att.GL2.requires_grad_(False)
        self.core.lay_norm.requires_grad_(False)  # also unused in model1

    def forward_v4(self,batch,runtime_context,epoch=0,teacher_forcing_mask=None):
        common=read_common(batch,runtime_context,self.static_asset_hash,self.raw_enc_in,
            'GinAR',self._v4_adjacency_cache)
        x=common['base_visible_input']
        mask=batch['input_available_mask'].unsqueeze(-1)
        x=torch.where(mask & torch.isfinite(x),x,torch.zeros_like(x))
        return {'pred':self.core(x).unsqueeze(-1)}
