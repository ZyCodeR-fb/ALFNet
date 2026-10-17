"""Study-only observed-history bridge and reversible centering; no new labels."""
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
from types import MethodType
import torch

VARIANTS=('bridge','center','delta1','bridge_delta1','bridge_center')

def observed_statistics(x,mask):
    """Only visible positions participate, including when hidden values are NaN."""
    mask=mask.bool()
    assert x.ndim==4 and mask.shape==x.shape[:-1]
    safe=torch.where(mask[...,None],x,torch.zeros_like(x))
    assert torch.isfinite(safe).all()
    length=x.size(1)
    index=torch.arange(length,device=x.device).view(1,length,1).expand_as(mask)
    left=torch.where(mask,index,-1).cummax(1).values
    right=torch.where(mask,index,length).flip(1).cummin(1).values.flip(1)
    li=torch.where(left>=0,left,right).clamp(0,length-1)
    ri=torch.where(right<length,right,left).clamp(0,length-1)
    lv=safe.gather(1,li[...,None].expand_as(x))
    rv=safe.gather(1,ri[...,None].expand_as(x))
    fraction=((index-li)/(ri-li).clamp_min(1)).clamp(0,1)[...,None]
    available=mask.any(1,keepdim=True)[...,None]
    bridge=torch.where(available,lv+(rv-lv)*fraction,torch.zeros_like(x))
    last_index=left[:,-1:].clamp_min(0)
    last=safe.gather(1,last_index[...,None].expand(-1,-1,-1,x.size(-1)))
    last=torch.where(available,last,torch.zeros_like(last)).detach()
    return bridge,last,available

def transform_filled(filled,x,mask_index,raw_dim,bridge=False,center=False):
    mask=x[...,mask_index]>.5
    interpolated,last,available=observed_statistics(x[...,:raw_dim],mask)
    raw=filled[...,:raw_dim]
    if bridge:
        raw=torch.where((~mask[...,None]) & available,interpolated,raw)
    if center:raw=raw-last
    return torch.cat([raw,filled[...,raw_dim:]],dim=-1),last

def run_experts(model,adjacency,x_enc,x_t_mark,pos_mark,x_dec,dec_t_mark,epoch,
                full_node_axis=False,teacher_mask=None):
    variant=model.support_transfer_variant
    center='center' in variant
    _,last,_=observed_statistics(x_enc[...,:model.raw_enc_in],x_enc[...,model.mask_index]>.5)
    decoder=x_dec if x_dec is None or not center else x_dec-last[...,:model.output_dim]
    def hook(module,args,filled):
        return transform_filled(filled,x_enc,model.mask_index,model.raw_enc_in,
                                bridge='bridge' in variant,center=center)[0]
    handle=model.shared_input_imputer.register_forward_hook(hook)
    try:
        pred,features,aux=type(model)._run_experts(model,adjacency,x_enc,x_t_mark,pos_mark,
            decoder,dec_t_mark,epoch,full_node_axis,teacher_mask)
    finally:handle.remove()
    if center:pred=pred+last[...,:model.output_dim]
    return pred,features,aux

@contextmanager
def adapter():
    from experiments import v4_runner
    from scripts import train_v4
    base=v4_runner.build_v4_session;contract=train_v4.model_method_contract
    def build(*args,**kwargs):
        s=base(*args,**kwargs)
        variant=s.args.model_family_cfg.get('recipe',{}).get('support_transfer_variant')
        if variant is None:return s
        assert s.model_name=='ALFNet' and variant in VARIANTS
        s.model.support_transfer_variant=variant
        s.model._run_experts=MethodType(run_experts,s.model)
        if 'delta1' in variant:s.model.graph_expert.decoder.scale=1.
        return s
    def describe(s):
        result=contract(s);v=getattr(s.model,'support_transfer_variant',None)
        if v:
            result['support_transfer']=dict(variant=v,bridge='bridge' in v,center='center' in v,
                decoder_scale=s.model.graph_expert.decoder.scale,statistics='current visible 12-step history only; zero center and GRIN fallback for q=0',
                teacher='centered consistently before unchanged teacher mask',extra_labels=False)
        return result
    with ExitStack() as stack:
        for obj,key,val in [(v4_runner,'build_v4_session',build),(train_v4,'build_v4_session',build),
                            (train_v4,'model_method_contract',describe)]:
            stack.enter_context(patch.object(obj,key,val))
        yield
