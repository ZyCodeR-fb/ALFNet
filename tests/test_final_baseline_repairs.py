from types import SimpleNamespace
import torch
import pytest
from pathlib import Path
from experiments.v4_runner import compute_losses
from model.comparison_model.STID import Model as STID
from scripts.train_v4 import build_optimizer,build_scheduler
from scripts import train_v4
from experiments.v4_runner import RunnerRefused

def test_stid_clock_covers_all_day_without_early_clamping():
    model=SimpleNamespace(raw_enc_in=1,use_time_of_day=True,use_day_of_week=True,time_of_day_size=288,day_of_week_size=7)
    x=torch.zeros(1,3,2,1);marks=torch.tensor([[[0.,0.,1.],[0.,12.,1.],[55.,23.,1.]]])
    features=STID._build_stid_input(model,x,marks)
    assert ((features[0,:,0,1]*288).long()==torch.tensor([0,144,287])).all()

def test_uignn_independent_quantiles_respect_legal_mask():
    q=torch.full((4,1,1,3),1.,requires_grad=True)
    batch={'target':torch.tensor([0.,0.,0.,10.]).reshape(4,1,1,1),'loss_mask':torch.ones(4,1,1,dtype=torch.bool)}
    output={'pred':q[...,1:2],'source_quantiles':q,'source_objective':'uignn_quantile'}
    losses=compute_losses(output,batch,'uignn_quantile');losses['optimize'].backward()
    g=q.grad.sum((0,1,2));assert g[0]>0 and g[2]<0
    q.grad=None;batch['loss_mask'][3]=False;compute_losses(output,batch,'uignn_quantile')['optimize'].backward();assert (q.grad[3]==0).all()

def test_ginar_schedule_steps_epochs_and_resumes_on_current_optimizer():
    s=SimpleNamespace(model_name='GinAR',model=torch.nn.Linear(2,1),args=SimpleNamespace(lr=.006))
    opt=build_optimizer(s);scheduler=build_scheduler(s,opt);opt.step();scheduler.step()
    assert opt.param_groups[0]['lr']==.003
    restored=build_optimizer(s);restored.load_state_dict(opt.state_dict());resumed=build_scheduler(s,restored);resumed.load_state_dict(scheduler.state_dict())
    for _ in range(14):restored.step();resumed.step()
    assert restored.param_groups[0]['lr']==.0015 and opt.param_groups[0]['lr']==.003


def test_stid_source_adam_scheduler_and_raw_mae():
    s=SimpleNamespace(model_name='STID',model=torch.nn.Linear(2,1),
                      args=SimpleNamespace(lr=.002,weight_decay=.0001))
    opt=build_optimizer(s)
    assert type(opt) is torch.optim.Adam
    scheduler=build_scheduler(s,opt);opt.step();scheduler.step()
    assert opt.param_groups[0]['lr']==.001
    p=torch.ones(1,12,2,1,requires_grad=True)
    b={'target':torch.zeros_like(p),'loss_mask':torch.ones(1,12,2,dtype=torch.bool)}
    a=compute_losses({'pred':p,'source_raw_mae':True},b,'mae',{'std':13.})
    assert a['base']==13. and a['role_mix'] is None

def test_final_slpf_is_the_accepted_single_stage_adapter():
    from model.comparison_model.SLPF import Model
    assert Model.method_variant == 'source_forecast_single_stage_v1'
