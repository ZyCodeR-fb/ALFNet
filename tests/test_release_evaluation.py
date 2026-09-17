import copy
import pytest
from scripts.eval_release import validation_agrees, digest

def tables():
    return {'main':{'LifeAvg':{'MAE':1.,'RMSE':2.,'MAPE':3.,'sample_count':4}},
            'phase':{'Cold':{'MAE':1.,'RMSE':2.,'MAPE':3.,'sample_count':4}}}

def test_validation_gate_rejects_drift_and_count_changes():
    expected=tables();actual=copy.deepcopy(expected)
    assert validation_agrees(actual,expected)==(True,0.)
    actual['phase']['Cold']['RMSE']+=1e-3
    assert not validation_agrees(actual,expected)[0]
    actual=tables();actual['main']['LifeAvg']['sample_count']=3
    with pytest.raises(AssertionError):validation_agrees(actual,expected)

def test_validation_gate_rejects_nonfinite_result():
    expected=tables();actual=tables();actual['main']['LifeAvg']['MAE']=float('nan')
    with pytest.raises(AssertionError):validation_agrees(actual,expected)

def test_plan_identity_includes_checkpoint_and_authorization():
    p={'runs':[{'checkpoint_sha256':'a'}],'test_authorized':True}
    h=digest(p);p['runs'][0]['checkpoint_sha256']='b';assert digest(p)!=h
    h=digest(p);p['test_authorized']=False;assert digest(p)!=h
