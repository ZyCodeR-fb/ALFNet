from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.flow_last_round import reconstruction_loss,degree_supports,relative_elements


def test_reconstruction_only_hidden_past_K():
    data=SimpleNamespace(context=SimpleNamespace(phase='train'),boundaries={'train':(0,20)},
                         normalised=np.ones((40,5),dtype=np.float32),natural=np.ones((40,5),dtype=bool))
    b=dict(origin=torch.tensor([0,3]),input_available_mask=torch.zeros(2,12,5,dtype=torch.bool))
    b['input_available_mask'][:,:,0]=True
    filled=torch.zeros(2,12,5,1,requires_grad=True)
    loss=reconstruction_loss(filled,b,data,[0,2]);assert loss.item()==1.
    data.normalised[:,[1,3,4]]=np.nan;data.normalised[20:]=99999
    assert reconstruction_loss(filled,b,data,[0,2]).item()==1.
    loss.backward();assert torch.count_nonzero(filled.grad[:,:,2])>0
    assert torch.count_nonzero(filled.grad[:,:,[0,1,3,4]])==0
    b['origin']=torch.tensor([9,10])
    with pytest.raises(AssertionError):reconstruction_loss(filled,b,data,[0,2])
    data.context.phase='validation'
    with pytest.raises(AssertionError):reconstruction_loss(filled,b,data,[0,2])


def test_degree_slots_preserve_directional_and_identity_and_isolates():
    a=torch.tensor([[0.,1,1,1],[1,0,0,0],[1,0,0,0],[0,0,0,0]])
    p=a/a.sum(-1,keepdim=True).clamp_min(1)
    original=torch.stack([torch.eye(4),p,p,p,p.T])
    result=degree_supports(original,a)
    assert torch.equal(result[[0,3,4]],original[[0,3,4]])
    assert not torch.equal(result[1],result[2])
    assert torch.isfinite(result).all() and result[1:,3,:3].abs().sum()==original[1:,3,:3].abs().sum()
    permutation=torch.tensor([2,0,3,1])
    expected=result[:,permutation][:,:,permutation]
    torch.testing.assert_close(degree_supports(original[:,permutation][:,:,permutation],a[permutation][:,permutation]),expected)


def test_relative_loss_finite_at_zero_and_negative_prediction():
    p=torch.tensor([-2.,-1.,0.,2.],requires_grad=True)
    y=torch.tensor([-1.,-.99,0.,1.])
    loss=relative_elements(p,y,dict(mean=100.,std=100.))
    assert torch.isfinite(loss).all() and (loss>=0).all()
    loss.sum().backward();assert torch.isfinite(p.grad).all()
