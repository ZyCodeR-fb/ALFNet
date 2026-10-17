import copy
import pytest
from scripts.run_hparam_completion import candidate,specs

@pytest.mark.parametrize('kind,value',[('dropout',.2),('weight_decay',1e-3),('capacity',20),('lr_factor',.5)])
def test_one_axis_changes_only_declared_fields_and_all_lr_groups(kind,value):
    base={'training':{'lr':5e-5,'weight_decay':1e-4},'family':{'experts':{'patch':{'config_path':'old'}},
        'recipe':{'graph_expert_lr':5e-4,'patch_expert_lr':1e-4,'gate_lr':5e-5}}}
    nested={'family':{'model':{'spa_patchsize':12,'crossformer_dropout':.1,'crossformer_router_factor':4}}}
    original=copy.deepcopy((base,nested));a,b=candidate(base,nested,kind,value,'new')
    assert (base,nested)==original
    assert b['family']['model']['crossformer_router_factor']==4
    assert a['family']['experts']['patch']['config_path']=='new'
    a['family']['experts']['patch']['config_path']='old'
    if kind=='dropout':assert b['family']['model'].pop('crossformer_dropout')==value;b['family']['model']['crossformer_dropout']=.1
    elif kind=='capacity':assert b['family']['model'].pop('spa_patchsize')==value;b['family']['model']['spa_patchsize']=12
    elif kind=='weight_decay':assert a['training']['weight_decay']==value;a['training']['weight_decay']=1e-4
    else:
        assert a['training']['lr']==base['training']['lr']*value
        a['training']['lr']/=value
        for k in base['family']['recipe']:
            assert a['family']['recipe'][k]==base['family']['recipe'][k]*value
            a['family']['recipe'][k]/=value
    assert a==base and b==nested

def test_only_missing_current_anchor_points_are_scheduled():
    assert sum(len(specs(ds)) for ds in ('PEMS03','PEMS04','PEMS08','PEMSBAY'))==24
    assert specs('PEMS03')[-2:]==[('lr_half','lr_factor',.5),('lr_double','lr_factor',2.)]
    assert specs('PEMS04')[-2:]==[('capacity_mid','capacity',14),('capacity_high','capacity',18)]
