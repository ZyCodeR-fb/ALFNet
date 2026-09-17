import numpy as np
import pytest
import torch
from scripts.analyze_observation_support import (
    layout_mask, intervene, select_pairs, INPUT_FIELDS, LAYOUTS, summarize)
from data_provider.lifecycle_roles import ROLE_VS, ROLE_NAS, ROLE_FS, ROLE_AAS


@pytest.fixture(scope='module')
def data():
    from experiments.v4_runner import build_v4_session
    torch.set_num_threads(2)
    s=build_v4_session('configs/paper/metrla_seed32_nine_models/METRLA_ALFNet.yaml',
                       device='cpu',batch_size=1,num_workers=0)
    return s.validation_loader.dataset


def test_layout_counts_and_temporal_positions():
    for q in (0,3,6,9,12):
        for layout in LAYOUTS:
            assert layout_mask(q,layout).sum()==q
    assert np.where(layout_mask(3,'recent'))[0].tolist()==[9,10,11]
    assert np.where(layout_mask(3,'early'))[0].tolist()==[0,1,2]
    assert np.where(~layout_mask(6,'middle_gap'))[0].tolist()==list(range(3,9))


def test_paired_scope_and_canonical_state(data):
    p=select_pairs(data,1)[0]; i=p['index']; n=p['node']; base=data[i]
    join=intervene(data,i,n,'join',3)
    assert join['lifecycle_role_ids'][:9,n].eq(ROLE_VS).all()
    assert join['lifecycle_role_ids'][9:,n].eq(ROLE_NAS).all()
    torch.testing.assert_close(join['lifecycle_state_age'][9:,n],torch.tensor([1,2,3])/12)
    fault=intervene(data,i,n,'fault',6)
    assert fault['lifecycle_role_ids'][6:,n].eq(ROLE_FS).all()
    other=torch.arange(data.num_nodes)!=n
    for key in INPUT_FIELDS:
        assert torch.equal(join[key][:,other],base[key][:,other])
    assert torch.equal(join['target'],fault['target'])
    full=intervene(data,i,n)
    q12=intervene(data,i,n,keep=layout_mask(12,'recent'))
    for key in INPUT_FIELDS:
        assert torch.equal(full[key],q12[key])
    q3=intervene(data,i,n,keep=layout_mask(3,'recent'))
    assert q3['lifecycle_role_ids'][:,n].eq(ROLE_AAS).all()
    assert q3['input_available_mask'][:,n].sum()==3
    assert not q3['base_visible_input'][:9,n].any()


def test_natural_missingness_never_revealed(data):
    p=select_pairs(data,1)[0]; n=p['node']; i=p['index']; t=p['origin']
    old=data.natural[t,n]; data.natural[t,n]=False
    try:
        x=intervene(data,i,n,keep=np.ones(12,dtype=bool))
        assert not x['input_available_mask'][0,n]
        assert x['base_visible_input'][0,n,0]==0
        # Natural packet loss does not relabel a mature sensor.
        assert x['lifecycle_role_ids'][0,n]==ROLE_AAS
    finally:
        data.natural[t,n]=old
