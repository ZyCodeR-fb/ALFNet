from types import SimpleNamespace
import pytest
import torch
from experiments.flow_unified import reference_and_scale,run_experts,VARIANTS

def test_spatial_reference_uses_visible_donors_only_and_handles_empty():
    x=torch.zeros(1,12,4,1);mask=torch.zeros(1,12,4,dtype=torch.bool)
    x[:,:,0]=2.;mask[:,:,0]=True
    x[:,-6:,1]=8.;mask[:,-6:,1]=True
    x[:,:,2:]=float('nan')
    a=torch.tensor([[0.,0.,1.,0.],[0.,0.,1.,0.],[1.,1.,0.,0.],[0.,0.,0.,0.]])
    _,c,_,s=reference_and_scale(x,mask,a,'triple_spatial_qdelta')
    torch.testing.assert_close(c.flatten(),torch.tensor([2.,8.,4.,4.]))
    torch.testing.assert_close(s.flatten(),torch.tensor([1.,.65,.3,.3]))
    poisoned=torch.where(mask[...,None],x,torch.full_like(x,99999.))
    torch.testing.assert_close(reference_and_scale(poisoned,mask,a,'triple_spatial_qdelta')[1],c)
    for v in VARIANTS:
        _,z,_,_=reference_and_scale(x,torch.zeros_like(mask),a,v)
        assert z.count_nonzero()==0 and torch.isfinite(z).all()

def test_spatial_reference_permutation_translation_and_no_extra_parameters():
    torch.manual_seed(32)
    x=torch.randn(2,12,4,1);mask=torch.rand(2,12,4)>.4;mask[:,:,1]=False
    a=torch.rand(4,4);p=torch.tensor([2,0,3,1])
    for v in VARIANTS:
        b,c,h,s=reference_and_scale(x,mask,a,v)
        bp,cp,hp,sp=reference_and_scale(x[:,:,p],mask[:,:,p],a[p][:,p],v)
        torch.testing.assert_close(cp,c[:,:,p]);torch.testing.assert_close(bp,b[:,:,p])
        if 'spatial' in v:
            c2=reference_and_scale(x+5,mask,a,v)[1]
            torch.testing.assert_close(c2,c+5)
        assert not c.requires_grad

def test_dynamic_scale_teacher_and_cleanup_on_exception():
    class Fill(torch.nn.Module):
        def forward(self,x):return x
    class Fake:
        raw_enc_in=output_dim=1;mask_index=6;unified_variant='triple_qdelta'
        def __init__(self):
            self.shared_input_imputer=Fill();self.graph_expert=SimpleNamespace(decoder=SimpleNamespace(scale=.3));self.fail=False
        def _run_experts(self,a,x,tm,pm,teacher,dt,epoch,full,teacher_mask):
            torch.testing.assert_close(self.graph_expert.decoder.scale,torch.tensor([[[1.],[.3]]]))
            torch.testing.assert_close(teacher,torch.tensor([[[[5.],[8.]]]]))
            assert teacher_mask is expected
            if self.fail:raise RuntimeError('probe')
            f=self.shared_input_imputer(x)
            return f[:,-1:,:,:1],f,None
    x=torch.zeros(1,12,2,7);x[:,:,0,0]=3.;x[:,:,0,6]=1
    teacher=torch.full((1,1,2,1),8.);expected=torch.zeros_like(teacher,dtype=torch.bool)
    m=Fake();a=torch.eye(2)
    pred,_,_=run_experts(m,a,x,None,None,teacher,None,0,True,expected)
    torch.testing.assert_close(pred,x[:,-1:,:,:1]);assert m.graph_expert.decoder.scale==.3
    m.fail=True
    with pytest.raises(RuntimeError):run_experts(m,a,x,None,None,teacher,None,0,True,expected)
    assert m.graph_expert.decoder.scale==.3 and not m.shared_input_imputer._forward_hooks
