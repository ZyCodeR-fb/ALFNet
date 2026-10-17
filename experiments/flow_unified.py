"""Unified visible-support rules, identical on every dataset; study-only adapter."""
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
from types import MethodType
import torch
from experiments.flow_support_transfer import observed_statistics

VARIANTS=('triple','triple_qdelta','triple_halfcenter','triple_spatial','triple_spatial_qdelta')

def reference_and_scale(x,mask,adjacency,variant):
    assert variant in VARIANTS
    bridge,last,available=observed_statistics(x,mask)
    q=mask.to(x.dtype).mean(1)[...,None]
    center=last
    if 'spatial' in variant:
        a=adjacency.to(x).clamp_min(0)
        assert a.ndim==2 and a.shape==(x.size(2),x.size(2))
        a=a+a.T
        # Weight only actually observed donors by current-window coverage.
        numerator=torch.einsum('nm,bmc->bnc',a,last[:,0]*q)
        denominator=torch.einsum('nm,bmc->bnc',a,q)
        global_reference=(last[:,0]*q).sum(1,keepdim=True)/q.sum(1,keepdim=True).clamp_min(1e-6)
        neighbor=torch.where(denominator>0,numerator/denominator.clamp_min(1e-6),global_reference)
        center=torch.where(available,last,neighbor[:,None]).detach()
    if 'halfcenter' in variant:center=.5*center
    scale=.3+.7*q if 'qdelta' in variant else 1.
    return bridge,center,available,scale

def run_experts(model,adjacency,x_enc,x_t_mark,pos_mark,x_dec,dec_t_mark,epoch,
                full_node_axis=False,teacher_mask=None):
    mask=x_enc[...,model.mask_index]>.5
    bridge,center,available,scale=reference_and_scale(x_enc[...,:model.raw_enc_in],mask,
                                                   adjacency,model.unified_variant)
    decoder=x_dec if x_dec is None else x_dec-center[...,:model.output_dim]
    def hook(module,args,filled):
        raw=torch.where((~mask[...,None]) & available,bridge,filled[...,:model.raw_enc_in])
        return torch.cat([raw-center,filled[...,model.raw_enc_in:]],dim=-1)
    handle=model.shared_input_imputer.register_forward_hook(hook)
    old_scale=model.graph_expert.decoder.scale
    model.graph_expert.decoder.scale=scale
    try:
        pred,features,aux=type(model)._run_experts(model,adjacency,x_enc,x_t_mark,pos_mark,
            decoder,dec_t_mark,epoch,full_node_axis,teacher_mask)
    finally:
        handle.remove();model.graph_expert.decoder.scale=old_scale
    return pred+center[...,:model.output_dim],features,aux

@contextmanager
def adapter():
    from experiments import v4_runner
    from scripts import train_v4
    base=v4_runner.build_v4_session;contract=train_v4.model_method_contract
    def build(*args,**kwargs):
        s=base(*args,**kwargs)
        v=s.args.model_family_cfg.get('recipe',{}).get('unified_variant')
        if v is None:return s
        assert s.model_name=='ALFNet' and v in VARIANTS
        s.model.unified_variant=v
        s.model._run_experts=MethodType(run_experts,s.model)
        return s
    def describe(s):
        result=contract(s);v=getattr(s.model,'unified_variant',None)
        if v:
            result['flow_unified']=dict(variant=v,bridge=True,center_strength=.5 if 'halfcenter' in v else 1.,
                unseen_center='observed-neighbor weighted mean, observed-global fallback' if 'spatial' in v else 'zero',
                decoder_scale='.3+.7*visible_fraction' if 'qdelta' in v else '1',
                scope='current visible input and public adjacency only; no node ID or dataset routing',extra_labels=False)
        return result
    with ExitStack() as stack:
        for obj,key,val in [(v4_runner,'build_v4_session',build),(train_v4,'build_v4_session',build),
                            (train_v4,'model_method_contract',describe)]:
            stack.enter_context(patch.object(obj,key,val))
        yield
