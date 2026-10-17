"""Isolated retry of the one OOM-aborted ablation; unchanged config and seed."""
import argparse,copy,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_final_ablations_20261006 as original
from scripts.run_metrla_hparams import read,write,sha,digest
NAME='ablation_recovery_20261008';KEY='09_PEMSBAY_local_context'
PLAN=ROOT/'configs/paper'/NAME/'plan.json'
def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']
def freeze():
    old=original.check();root=original.output(old);s=read(root/'queue_status.json')
    assert [j['key'] for j in s['jobs'] if j['status']!='complete']==[KEY]
    assert 'torch.OutOfMemoryError' in (root/(KEY+'.log')).read_text()
    assert not PLAN.exists()
    p=dict(kind=NAME,parent_plan_sha256=old['plan_sha256'],configs={KEY:old['configs'][KEY]},anchors=old['anchors'],max_concurrent=1,test_authorized=False,
        reason='Only unfinished authorized ablation failed from concurrent GPU OOM; identical config/seed, new output root preserves failure',files=dict(old['files']))
    p['files'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    p['plan_sha256']=digest(p);write(PLAN,p)
    gate=read(root/'PREFLIGHT_COMPLETE.json');gate['reports']=[r for r in gate['reports'] if r['key']==KEY]
    gate.update(plan_sha256=p['plan_sha256'],inherited_from=old['plan_sha256']);write(output(p)/'PREFLIGHT_COMPLETE.json',gate)
    print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256']
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def run():
    p=check();assert not (output(p)/(KEY+'_complete.json')).exists()
    with patch.object(original,'check',check),patch.object(original,'output',output):original.worker(KEY)
    write(output(p)/'COMPLETE.json',dict(passed=True,plan_sha256=p['plan_sha256'],key=KEY,test_accessed=False))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['freeze','run']);globals()[a.parse_args().mode]()
