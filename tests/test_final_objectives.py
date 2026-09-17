import pytest
import torch
from experiments.focal_loss import focal_elements
from experiments.v4_runner import compute_losses, RunnerRefused

def test_focal_mask_and_exact_role_mixture():
    p=torch.tensor([.1,2.,5.],requires_grad=True).reshape(1,1,3,1);p.retain_grad()
    target=torch.zeros_like(p);target[0,0,2]=float('nan')
    roles=torch.zeros(1,1,3,6);roles[0,0,0,0]=1;roles[0,0,1:,1]=1
    b=dict(target=target,loss_mask=torch.ones(1,1,3,dtype=torch.bool),lifecycle_context=roles)
    out=dict(pred=p,forecast_loss_mode='role_balanced_mix',forecast_role_balanced_beta=.1)
    loss=compute_losses(out,b,'mae_focal_raw02',{'std':13.})['optimize']
    expected=(p[0,0,:2,0]*(2*torch.sigmoid(.2*13*p[0,0,:2,0])-1)).mean()
    torch.testing.assert_close(loss,expected);loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad[0,0,2]==0
    with pytest.raises(RunnerRefused):compute_losses({'pred':p},b,'mae_focal_raw02',{'std':13.})
    with pytest.raises(ValueError):focal_elements(p,target,{'std':0})

@pytest.mark.parametrize('tod,dow', [(True,False),(False,True),(True,True)])
def test_stid_unequal_calendar_embedding_widths(tod,dow):
    from model.comparison_model.STID.stid import STID
    core=STID(node_num=8,input_dim=1+int(tod)+int(dow),output_dim=1,seq_len=12,
        horizon=12,node_dim=4,input_len=12,embed_dim=4,output_len=12,num_layer=1,
        temp_dim_tid=3,temp_dim_diw=5,time_of_day_size=288,day_of_week_size=7,
        if_T_i_D=tod,if_D_i_W=dow,if_node=True,adjacency=torch.eye(8),
        tid_index=1,diw_index=1+int(tod))
    x=torch.zeros(2,12,8,1+int(tod)+int(dow))
    output=core(x)
    assert output.shape==(2,12,8,1)
    output.square().mean().backward()
