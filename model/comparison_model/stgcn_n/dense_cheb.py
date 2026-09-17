"""Exact ChebConv polynomial via dense batch matrices, retaining its norm/weights."""
import torch

def dense_cheb(conv, x, edge_index, edge_weight, dynamic):
    # x is [B,T,N,C]. PyG propagates messages from edge row to edge col.
    b,t,n,c=x.shape
    edge,norm=conv.__norm__(edge_index,b*n if dynamic else n,edge_weight,
                           conv.normalization,dtype=x.dtype)
    row,col=edge
    if dynamic:
        if not torch.equal(row//n,col//n):
            raise ValueError('Cross-sample graph edge in dense ChebConv')
        matrix=x.new_zeros(b,n,n).index_put((row//n,col%n,row%n),norm,accumulate=True)
    else:
        matrix=x.new_zeros(n,n).index_put((col,row),norm,accumulate=True).expand(b,-1,-1)
    def propagate(value):
        flat=value.permute(0,2,1,3).reshape(b,n,t*c)
        return torch.bmm(matrix,flat).reshape(b,n,t,c).permute(0,2,1,3)
    tx0=x;out=conv.lins[0](tx0)
    if len(conv.lins)>1:
        tx1=propagate(tx0);out=out+conv.lins[1](tx1)
    for lin in conv.lins[2:]:
        tx2=2*propagate(tx1)-tx0;out=out+lin(tx2);tx0,tx1=tx1,tx2
    return out if conv.bias is None else out+conv.bias
