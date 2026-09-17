"""One source-native GWNET adapter for both datasets; no experimental hooks."""
import torch
from torch import nn
from layers.structural_node_encoder import StructuralNodeEncoder
from .source_model import gwnet

class GWNET(gwnet):
    def __init__(self, supports, adp_adj, adjacency, positions, node_num,
                 input_dim, output_dim, horizon, seq_len=12, **kwargs):
        super().__init__('cpu', node_num, supports=supports, gcn_bool=True,
                         addaptadj=adp_adj, in_dim=input_dim,
                         out_dim=output_dim*horizon, **kwargs)
        for bank in (self.gate_convs, self.residual_convs, self.skip_convs):
            for i, old in enumerate(bank):
                new=nn.Conv2d(old.in_channels,old.out_channels,old.kernel_size,
                              dilation=(old.dilation[0],old.dilation[0]))
                new.load_state_dict(old.state_dict(),strict=True);bank[i]=new
        if adp_adj:
            del self.nodevec1,self.nodevec2
            self.nodevec_left=StructuralNodeEncoder(adjacency,positions,10)
            self.nodevec_right=StructuralNodeEncoder(adjacency,positions,10)

    def forward(self, input):
        ids=getattr(self,'current_node_ids',None)
        if ids is not None and list(ids)!=list(range(input.shape[2])):
            raise ValueError('Source GWNET requires the full canonical node axis')
        self.supports=[s.to(input.device) for s in self.supports]
        if self.addaptadj:
            self.nodevec1=self.nodevec_left();self.nodevec2=self.nodevec_right().t()
        return super().forward(input.transpose(1,3))
