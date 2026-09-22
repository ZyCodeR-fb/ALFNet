from types import SimpleNamespace
import torch
from experiments.alfnet_ablation_adapter import ObservedOnly,ZeroTemporal,ZeroFiLM,IdentityRefiner,remove_age
from layers.lifecycle_moge import LifecycleRouter


def test_observed_only_does_not_use_hidden_values():
    x=torch.randn(2,12,4,7);x[...,6]=torch.randint(0,2,x[...,6].shape).float()
    changed=x.clone();changed[...,0]=torch.where(x[...,6]>.5,x[...,0],torch.full_like(x[...,0],12345.))
    m=ObservedOnly(1,6);out=m(None,x)
    assert torch.equal(out,m(None,changed))
    assert torch.equal(out[...,1:],x[...,1:])
    assert torch.equal(out[...,0][x[...,6]>.5],x[...,0][x[...,6]>.5])
    assert torch.count_nonzero(out[...,0][x[...,6]<.5])==0


def test_no_age_preserves_role_mask_signal_and_original_input():
    x=torch.randn(2,12,4,7);original=x.clone();args=remove_age(SimpleNamespace(age_index=5,mask_index=6),(None,x,'marks'))
    assert torch.equal(x,original) and args[2]=='marks'
    assert torch.equal(args[1][...,5],x[...,6])
    assert torch.equal(args[1][...,:5],x[...,:5]) and torch.equal(args[1][...,6],x[...,6])


def test_removed_condition_branch_is_parameter_free_identity_film():
    patch=ZeroTemporal(64);film=ZeroFiLM(32);x=torch.randn(2,12,4,7)
    gamma,beta=film(patch.forward_v4(x,None)['features']).chunk(2,dim=-1)
    h=torch.randn_like(gamma);assert torch.equal((1+gamma)*h+beta,h)
    assert sum(p.numel() for p in patch.parameters())+sum(p.numel() for p in film.parameters())==0


def test_refiner_is_exact_identity():
    x=torch.randn(2,12,4,1);assert IdentityRefiner()(x,None,None) is x


def test_soft_router_topk_is_numerically_inactive():
    a=LifecycleRouter(6,5,k=1,mode='soft');b=LifecycleRouter(6,5,k=5,mode='soft')
    with torch.no_grad():a.w_gate.normal_()
    b.load_state_dict(a.state_dict());x=torch.randn(2,4,6)
    for training in (True,False):
        a.train(training);b.train(training);ga,la=a(x);gb,lb=b(x)
        assert torch.equal(ga,gb) and torch.equal(la,lb)
