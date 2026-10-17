"""Prespecified, parameter-matched mechanism interventions; no hidden labels."""
import torch

KEYS=('no_expert_lifecycle','no_film_gain','no_film_shift','pointwise_refiner')
def expert_context(module,args):
    args=list(args);x=args[1].clone()
    # Filled feature tensor has raw channels followed by four roles, age, mask.
    x[...,-6:-1]=x[...,-1:]
    args[1]=x;return tuple(args)
def remove_gain(module,args,out):
    gamma,beta=out.chunk(2,dim=-1);return torch.cat((torch.zeros_like(gamma),beta),dim=-1)
def remove_shift(module,args,out):
    gamma,beta=out.chunk(2,dim=-1);return torch.cat((gamma,torch.zeros_like(beta)),dim=-1)
def self_only(module,args):
    args=list(args);a=args[1];args[1]=torch.eye(a.size(0),device=a.device,dtype=a.dtype);return tuple(args)
def apply(session):
    key=session.args.model_family_cfg.get('recipe',{}).get('study_ablation','reference')
    if key=='reference':return session
    assert key in KEYS and session.model_name=='ALFNet'
    m=session.model;assert not hasattr(m,'study_ablation');m.study_ablation=key
    if key=='no_expert_lifecycle':
        # Change role/age before shared reconstruction and expert processing only;
        # ALFNet.forward retains original role-dependent refinement and targets.
        m.shared_input_imputer.register_forward_pre_hook(expert_context)
    elif key=='no_film_gain':m.film_head.register_forward_hook(remove_gain)
    elif key=='no_film_shift':m.film_head.register_forward_hook(remove_shift)
    else:m.graph_refiner.register_forward_pre_hook(self_only)
    return session
