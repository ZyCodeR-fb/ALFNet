"""Verify completed paired inference artifacts without executing models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.v4_utils import sha256_file


def verify(path):
    path=Path(path)
    assert (path/'COMPLETE').exists()
    df=pd.read_csv(path/'paired_predictions.csv',dtype={'mask':str})
    manifest=json.loads((path/'manifest.json').read_text())
    sources=json.loads((path/'sources.json').read_text())
    n=2*manifest['per_track']
    assert df.groupby(['model','condition']).size().eq(n).all()
    assert len(df)==3*20*n
    assert not df.duplicated(['model','condition','pair_id']).any()
    assert df.groupby('pair_id').target.nunique().eq(1).all()
    assert df.groupby('pair_id').origin.nunique().eq(1).all()
    assert df.groupby('pair_id').node.nunique().eq(1).all()
    assert df.groupby(['pair_id','condition'])['mask'].nunique().eq(1).all()
    assert (df['mask'].str.count('1')==df.q).all()
    assert df['mask'].str.len().eq(12).all()
    for key,value in sources.items():
        assert sha256_file(value['checkpoint'])==value['checkpoint_sha256'],key
        assert sha256_file(value['config'])==value['config_sha256'],key
    summary=pd.read_csv(path/'metrics.csv')
    for _,r in summary.iterrows():
        subset=df[(df.model==r.model)&(df.condition==r.condition)]
        if r.track!='all':subset=subset[subset.track==r.track]
        pred=np.array([json.loads(x) for x in subset.prediction])
        target=np.array([json.loads(x) for x in subset.target])
        e=pred-target
        expected=[np.abs(e).mean(),np.sqrt((e**2).mean()),100*(np.abs(e)/np.maximum(np.abs(target),1e-6)).mean()]
        np.testing.assert_allclose(expected,[r.MAE,r.RMSE,r.MAPE],rtol=2e-6,atol=2e-6)
    for model in sources:
        a=df[(df.model==model)&(df.condition=='full')].sort_values('pair_id')
        b=df[(df.model==model)&(df.condition=='q12_recent')].sort_values('pair_id')
        # Independently forwarded identical inputs must give identical outputs.
        assert a.prediction.tolist()==b.prediction.tolist(),model
    equal_mask_arms=[('join_3','q3_recent'),('join_6','q6_recent'),
                     ('fault_3','q9_early'),('fault_6','q6_early'),('outage','q0_recent')]
    for model in ('MoGERNN','Mask-only'):
        for left,right in equal_mask_arms:
            a=df[(df.model==model)&(df.condition==left)].sort_values('pair_id')
            b=df[(df.model==model)&(df.condition==right)].sort_values('pair_id')
            assert a['mask'].tolist()==b['mask'].tolist()
            assert a.prediction.tolist()==b.prediction.tolist(),(model,left,right)
    result=dict(passed=True,models=len(sources),conditions=20,pairs=n,predictions=len(df),
                paired_targets_fixed=True,masks_equal_across_models=True,
                all_three_metrics_recomputed=True,full_anchor_reproducible=True,
                metadata_blind_controls_invariant=True,
                existing_checkpoints_unchanged=True)
    (path/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');a=p.parse_args();verify(a.path)
