import torch
from experiments.flow_support_transfer import observed_statistics,transform_filled

def test_expert_wrapper_centers_teacher_restores_output_and_removes_hook():
    from experiments.flow_support_transfer import run_experts
    class Fill(torch.nn.Module):
        def forward(self,x):return x
    class Fake:
        raw_enc_in=output_dim=1
        mask_index=6
        support_transfer_variant='center'
        def __init__(self):self.shared_input_imputer=Fill()
        def _run_experts(self,a,x,tm,pm,decoder,dt,epoch,full,teacher):
            filled=self.shared_input_imputer(x)
            assert torch.equal(teacher,expected_mask)
            torch.testing.assert_close(decoder,expected_decoder)
            assert filled[:,-1:,:,:1].count_nonzero()==0
            return filled[:,-1:,:,:1],filled,None
    x=torch.zeros(1,12,2,7);x[...,:1]=3.;x[...,6]=1
    decoder=torch.full((1,1,2,1),8.)
    expected_decoder=torch.full_like(decoder,5.)
    expected_mask=torch.tensor([[[[True],[False]]]])
    model=Fake();pred,_,_=run_experts(model,None,x,None,None,decoder,None,0,True,expected_mask)
    torch.testing.assert_close(pred,torch.full_like(pred,3.))
    assert len(model.shared_input_imputer._forward_hooks)==0

def test_interpolation_edges_hidden_poison_and_empty():
    x=torch.tensor([float('nan'),2.,float('nan'),6.,float('nan')]).reshape(1,5,1,1)
    mask=torch.tensor([0,1,0,1,0]).reshape(1,5,1).bool()
    bridge,last,has=observed_statistics(x,mask)
    torch.testing.assert_close(bridge.flatten(),torch.tensor([2.,2.,4.,6.,6.]))
    assert last.item()==6 and has.item()
    poisoned=torch.where(mask[...,None],x,torch.full_like(x,99999.))
    torch.testing.assert_close(observed_statistics(poisoned,mask)[0],bridge)
    b,c,h=observed_statistics(x,torch.zeros_like(mask))
    assert b.count_nonzero()==c.count_nonzero()==0 and not h.any()

def test_bridge_preserves_visible_and_unseen_and_context():
    x=torch.zeros(1,5,3,7);x[:,1,0,0]=2;x[:,3,0,0]=6;x[:,[1,3],0,6]=1
    x[:,:,2,0]=7;x[:,:,2,6]=1
    filled=torch.randn_like(x,requires_grad=True)
    y,_=transform_filled(filled,x,6,1,bridge=True)
    torch.testing.assert_close(y[:,:,1],filled[:,:,1])
    torch.testing.assert_close(y[:,:,2],filled[:,:,2])
    torch.testing.assert_close(y[...,1:],filled[...,1:])
    torch.testing.assert_close(y[:,[1,3],0],filled[:,[1,3],0])
    y.sum().backward();assert torch.isfinite(filled.grad).all()
    assert filled.grad[:,[0,2,4],0,0].count_nonzero()==0
    assert filled.grad[:,:,1,0].count_nonzero()==5

def test_center_translation_and_permutation():
    torch.manual_seed(1)
    x=torch.randn(2,12,4,7);x[...,6]=(torch.rand(2,12,4)>.5).float();x[:,:,0,6]=0
    filled=torch.randn_like(x)
    y,last=transform_filled(filled,x,6,1,center=True)
    torch.testing.assert_close(y[...,:1]+last,filled[...,:1])
    torch.testing.assert_close(y[:,:,0],filled[:,:,0])
    shift=torch.randn(2,1,4,1);shift[:,:,0]=0
    x2=x.clone();x2[...,:1]+=shift
    f2=filled.clone();f2[...,:1]+=shift
    torch.testing.assert_close(transform_filled(f2,x2,6,1,center=True)[0],y)
    order=torch.tensor([2,0,3,1])
    torch.testing.assert_close(transform_filled(filled[:,:,order],x[:,:,order],6,1,True,True)[0],
                               transform_filled(filled,x,6,1,True,True)[0][:,:,order])
