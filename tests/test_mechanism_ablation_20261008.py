import torch
from experiments.mechanism_ablation_20261008 import expert_context,remove_gain,remove_shift,self_only

def test_context_preserves_raw_mask_and_caller_input():
    x=torch.randn(2,12,3,7);old=x.clone();y=expert_context(None,(None,x))[1]
    assert torch.equal(old,x) and torch.equal(y[...,:1],x[...,:1]) and torch.equal(y[...,-1],x[...,-1])
    assert torch.equal(y[...,-6:-1],x[...,-1:].expand_as(y[...,-6:-1]))

def test_film_component_removal_keeps_other_component_and_gradients():
    x=torch.randn(2,3,8,requires_grad=True)
    a=remove_gain(None,None,x);b=remove_shift(None,None,x)
    assert torch.count_nonzero(a[...,:4])==0 and torch.equal(a[...,4:],x[...,4:])
    assert torch.count_nonzero(b[...,4:])==0 and torch.equal(b[...,:4],x[...,:4])
    a.sum().backward();assert torch.count_nonzero(x.grad[...,:4])==0 and torch.all(x.grad[...,4:]==1)

def test_refiner_only_graph_substitution_preserves_predictions_roles():
    y=torch.randn(2,12,3,1);a=torch.randn(3,3);roles=torch.randn(2,3,4)
    z=self_only(None,(y,a,roles));assert z[0] is y and z[2] is roles and torch.equal(z[1],torch.eye(3))
