import torch
from experiments.reconstruction_gradient_20261009 import detach_raw
def test_forward_identity_but_selective_backward():
    x=torch.randn(2,12,3,7,requires_grad=True);y=detach_raw(x,1)
    assert torch.equal(x,y);y.sum().backward()
    assert torch.count_nonzero(x.grad[...,:1])==0 and torch.all(x.grad[...,1:]==1)
