import copy
import pytest
import torch
from torch_geometric.nn import ChebConv
from model.comparison_model.stgcn_n.dense_cheb import dense_cheb

@pytest.mark.parametrize('dynamic',[False,True])
@pytest.mark.parametrize('normalization',['sym','rw',None])
def test_matches_pyg_features_edge_weights_and_parameter_gradients(dynamic,normalization):
    torch.manual_seed(15);b,t,n,c=2,3,7,4
    layer=ChebConv(c,5,K=3,normalization=normalization).double();other=copy.deepcopy(layer)
    nodes=b*n if dynamic else n
    graph=torch.rand(b if dynamic else 1,n,n,dtype=torch.double)
    graph[graph<.3]=0
    if dynamic:
        batch,row,col=graph.nonzero(as_tuple=True);edge=torch.stack([row+batch*n,col+batch*n]);weights=graph[batch,row,col].requires_grad_()
    else:
        row,col=graph[0].nonzero(as_tuple=True);edge=torch.stack([row,col]);weights=graph[0,row,col].requires_grad_()
    w2=weights.detach().clone().requires_grad_();x=torch.randn(b,t,n,c,dtype=torch.double,requires_grad=True);x2=x.detach().clone().requires_grad_()
    packed=x.permute(1,0,2,3).reshape(t,b*n,c) if dynamic else x.reshape(b*t,n,c)
    expected=layer(packed,edge,weights)
    expected=expected.reshape(t,b,n,5).permute(1,0,2,3) if dynamic else expected.reshape(b,t,n,5)
    actual=dense_cheb(other,x2,edge,w2,dynamic)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    grad=torch.randn_like(actual);(expected*grad).sum().backward();(actual*grad).sum().backward()
    torch.testing.assert_close(x.grad,x2.grad,rtol=1e-9,atol=1e-9)
    torch.testing.assert_close(weights.grad,w2.grad,rtol=1e-9,atol=1e-9)
    for p,q in zip(layer.parameters(),other.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-9,atol=1e-9)
