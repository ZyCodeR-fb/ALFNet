"""Isolated flow candidates: hidden-history supervision, degree scalers, relative loss."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import numpy as np
import torch
import weakref
from experiments.focal_loss import focal_elements as raw_focal

RELATIVE_NAME='mae_focal_relative20'
VARIANTS=('reconstruction','degree_supports','reconstruction_degree','relative20')


def relative_elements(prediction,target,scaler):
    raw=raw_focal(prediction,target,scaler)
    safe=torch.nan_to_num(target,nan=0.,posinf=0.,neginf=0.)
    y=safe*float(scaler['std'])+float(scaler['mean'])
    # Original objective retained; floor belongs to training loss, never metrics.
    rel=(prediction-safe).abs()*float(scaler['std'])/y.abs().clamp_min(.1*float(scaler['std']))
    return .8*raw+.2*rel


def degree_supports(supports,adjacency):
    result=supports.clone()
    degree=(adjacency>0).sum(-1).to(supports.dtype)
    logd=torch.log1p(degree)
    nonzero=degree>0
    delta=logd[nonzero].mean() if nonzero.any() else logd.new_tensor(1.)
    rho=(logd/delta.clamp_min(1e-6)).clamp(.5,2.)
    # Same five slots/Theta; replace duplicate mean/weighted-mean slots only.
    result[1]=supports[1]*rho[:,None]
    result[2]=supports[2]/rho[:,None]
    return result


def reconstruction_loss(filled,batch,dataset,known_nodes):
    assert dataset.context.phase=='train'
    origins=batch['origin'].detach().cpu().numpy().astype(np.int64)
    length=filled.shape[1]
    start,end=dataset.boundaries['train']
    assert (origins>=start).all() and (origins+length<=end).all()
    known=np.asarray(known_nodes,dtype=np.int64)
    time=origins[:,None]+np.arange(length)[None,:]
    # Select K and past time BEFORE any target arithmetic. No V/T labels enter.
    target=dataset.normalised[time[:,:,None],known[None,None,:]]
    natural=dataset.natural[time[:,:,None],known[None,None,:]]
    nodes=torch.as_tensor(known,device=filled.device)
    hidden=~batch['input_available_mask'].index_select(2,nodes).bool()
    valid=hidden & torch.as_tensor(natural,device=filled.device)
    if not valid.any():return filled.sum()*0.
    truth=torch.as_tensor(np.nan_to_num(target),device=filled.device,dtype=filled.dtype)
    prediction=filled.index_select(2,nodes)[...,0]
    return (prediction-truth).abs()[valid].mean()


@contextmanager
def adapter():
    from experiments import v4_runner as v4
    from scripts import train_v4 as training
    from model.ALFGraph import ALFGraph as graph_module
    original_build=v4.build_v4_session
    original_forward=v4._forward
    original_compute=v4.compute_losses
    original_contract=training.model_method_contract
    original_supports=graph_module.build_lifecycle_supports

    def build(*args,**kwargs):
        s=original_build(*args,**kwargs)
        variant=s.args.model_family_cfg.get('recipe',{}).get('flow_last_variant')
        if variant is None:return s
        assert s.model_name=='ALFNet' and s.dataset in ('PEMS03','PEMS04','PEMS08') and variant in VARIANTS
        s.model.flow_last_variant=variant
        if 'reconstruction' in variant:
            model_ref=weakref.ref(s.model)
            def capture(module,args,value):
                model=model_ref()
                if model is not None and model.training:model._study_imputed=value
            s.model.shared_input_imputer.register_forward_hook(capture)
        if 'degree' in variant:
            graph_ref=weakref.ref(s.model.graph_expert)
            implementation=type(s.model.graph_expert).forward
            def supports(adjacency,*a,**kw):
                return degree_supports(original_supports(adjacency,*a,**kw),adjacency)
            def forward(*a,**kw):
                with patch.object(graph_module,'build_lifecycle_supports',supports):
                    return implementation(graph_ref(),*a,**kw)
            s.model.graph_expert.forward=forward
        return s

    def forward(session,batch,context,epoch,teacher_forcing_mask):
        result=original_forward(session,batch,context,epoch,teacher_forcing_mask)
        variant=getattr(session.model,'flow_last_variant','')
        if 'reconstruction' in variant and session.model.training:
            assert context.phase=='train'
            filled=session.model.__dict__.pop('_study_imputed')
            extra=.25*reconstruction_loss(filled,batch,session.train_dataset,session.train_context.known_nodes)
            result=dict(result)
            result['aux_losses']=dict(result.get('aux_losses') or {},hidden_history_reconstruction=extra)
        return result

    def compute(model_output,batch,loss_name='mse',scaler=None):
        if loss_name!=RELATIVE_NAME:return original_compute(model_output,batch,loss_name,scaler)
        with patch.object(v4,'FOCAL_NAME',RELATIVE_NAME),patch.object(v4,'focal_elements',relative_elements):
            return original_compute(model_output,batch,loss_name,scaler)

    def contract(session):
        d=original_contract(session)
        variant=getattr(session.model,'flow_last_variant',None)
        if variant:
            d['flow_last_round']=dict(variant=variant,reconstruction_weight=.25 if 'reconstruction' in variant else 0.,
                hidden_history_scope='train-time K only, training loss labels only' if 'reconstruction' in variant else None,
                support_slots=['identity','degree_amplified_mean','degree_attenuated_mean','forward_diffusion','backward_diffusion'] if 'degree' in variant else None,
                loss=session.args.loss,relative_weight=.2 if variant=='relative20' else 0.)
            if 'degree' in variant:d['graph_support_names']=d['flow_last_round']['support_slots']
        return d

    with ExitStack() as stack:
        for obj,name,value in [(v4,'build_v4_session',build),(training,'build_v4_session',build),
                              (v4,'_forward',forward),(v4,'compute_losses',compute),
                              (training,'model_method_contract',contract)]:
            stack.enter_context(patch.object(obj,name,value))
        yield
