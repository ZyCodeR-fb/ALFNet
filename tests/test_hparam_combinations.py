import copy
from scripts.run_hparam_combinations import candidate,SPECS
from scripts.collect_hparam_combinations import unambiguous


def test_absolute_lr_not_multiplied_twice():
    a={'training':{'lr':1e-4,'weight_decay':1e-4},'family':{'recipe':{'graph_expert_lr':.001,'patch_expert_lr':.0002,'gate_lr':.0001},'experts':{'patch':{'config_path':'old'}}}}
    b={'family':{'model':{'spa_patchsize':8,'crossformer_dropout':.1}}}
    old=copy.deepcopy((a,b));x,y=candidate(a,b,(12,2.,.1,.001),'new')
    assert x['family']['recipe']['graph_expert_lr']==.001
    assert x['training']['lr']==.0001 and (a,b)==old
    assert y['family']['model']['spa_patchsize']==12 and x['training']['weight_decay']==.001


def test_tradeoffs_not_silently_promoted():
    def row(w,a,b,c):return dict(first_cells=w,table=dict(main=dict(LifeAvg=dict(MAE=a,RMSE=b,MAPE=c))))
    base=row(10,2,3,4)
    assert unambiguous(row(11,2,3,4),base)
    assert unambiguous(row(10,1.9,3,4),base)
    assert not unambiguous(row(11,2.1,3,4),base)
    assert not unambiguous(row(9,1,2,3),base)
    assert not unambiguous(base,base)


def test_bounded_unique_scope():
    assert sum(map(len,SPECS.values()))==12
    assert all(2<=len(v)<=3 and len(set(v))==len(v) for v in SPECS.values())
