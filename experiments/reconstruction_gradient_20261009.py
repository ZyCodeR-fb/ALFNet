"""Gradient-path ablations: identical forward values and parameter count."""
from types import MethodType
import torch
def detach_raw(x,n):return torch.cat((x[...,:n].detach(),x[...,n:]),dim=-1)
def apply(session):
    key=session.args.model_family_cfg.get('recipe',{}).get('study_ablation','reference')
    if key=='reference':return session
    assert key in ('stop_graph_recon_gradient','stop_temporal_recon_gradient')
    m=session.model;assert not hasattr(m,'study_ablation');m.study_ablation=key;n=m.raw_enc_in
    def hook(module,args):
        a=list(args);a[1]=detach_raw(a[1],n);return tuple(a)
    if key=='stop_graph_recon_gradient':m.graph_expert.register_forward_pre_hook(hook)
    else:
        original=m.patch_expert.forward_v4
        def forward_v4(module,x,*args,**kwargs):return original(detach_raw(x,n),*args,**kwargs)
        m.patch_expert.forward_v4=MethodType(forward_v4,m.patch_expert)
        m.patch_expert.register_forward_pre_hook(hook)
    return session
