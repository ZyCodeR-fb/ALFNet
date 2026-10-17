"""User-authorized evaluation of the fixed final-model WD=.001 candidate."""
import argparse
import copy
import sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import read,write,sha,digest
from scripts import evaluate_flow_lr_cross_20261004 as evaluator
NAME='pems08_wd_test_20261012'
PLAN=ROOT/'configs/paper'/NAME/'plan.json'
KEY='13_PEMS08_weight_decay_0.001'
def output(p):return ROOT/'artifacts/evaluation'/NAME/p['plan_sha256']
def freeze():
    import torch,yaml
    from scripts.run_final_ablations_20261006 import canonical
    assert not PLAN.exists()
    source=ROOT/'paper/flow_ofat_20261006/results.json'
    r=copy.deepcopy(read(source)['rows'][KEY]);cfg=yaml.safe_load((ROOT/r['config']).read_text())
    old=yaml.safe_load((ROOT/'configs/paper/flow_lr_cross_20261004/05_PEMS08_triple_halfcenter_lr1.yaml').read_text())
    a,b=canonical(cfg),canonical(old)
    assert a['training']['weight_decay']==.001 and b['training']['weight_decay']==.0001
    b['training']['weight_decay']=.001;assert a==b
    cp=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
    report=read(ROOT/r['report']);parent=read(ROOT/'configs/paper/flow_ofat_20261006/plan.json')
    assert cp['plan_sha256']==parent['plan_sha256'] and cp['config_sha256']==sha(ROOT/r['config'])
    assert cp['seed']==32 and sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
    assert cp['epoch']==report['best_epoch']==min((e['validation']['checkpoint_life_avg_mae'],e['epoch']) for e in report['history'])[1]
    assert all(e[s]['skipped']==0 for e in report['history'] for s in ('train','validation'))
    assert all(torch.isfinite(v).all() for v in cp['model_state'].values() if v.is_floating_point())
    for f,h in parent['files'].items():
        if f.startswith(('model/','layers/','lib/','data_provider/','experiments/')):assert sha(ROOT/f)==h,f
    r.update(model='ALFNet',epoch=cp['epoch'],training_config_sha256=cp['config_sha256'],training_plan_sha256=cp['plan_sha256'],table=report['development_validation_table'])
    files=[source,Path(__file__),ROOT/'configs/paper/flow_ofat_20261006/plan.json']
    files.extend(ROOT/r[f] for f in ('config','checkpoint','report'))
    files.extend(ROOT/cfg['family']['experts'][e]['config_path'] for e in ('graph','patch'))
    files.extend(ROOT/cfg['dataset'][f] for f in ('data_path','unknown_nodes_path'))
    for folder in ('model','layers','lib','experiments','data_provider','utils'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.extend(ROOT/f for f in ('scripts/test_all_final_20261001.py','scripts/evaluate_flow_lr_cross_20261004.py','scripts/train_v4.py','scripts/eval_release.py','args.py'))
    p=dict(kind=NAME,status='authorized_test',test_authorized=True,authorization='2026-10-12 user explicitly requests WD=.001 test and conditional promotion followed by five paired ablations.',entries={KEY:r},max_concurrent=1,selection='Compare all three LifeAvg errors and strict first-place count among eight retained baselines. Clear promotion if all three improve and first count is at least current 30/33; report any tradeoff explicitly. Test-informed retention, not independent confirmation.',files={str(f.relative_to(ROOT)):sha(f) for f in set(files)},audit=dict(only_change='weight_decay .0001 -> .001',original_model_source_hashes_match=True,finite_weights=True,zero_skips=True))
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and p['test_authorized']
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def worker():
    with patch.object(evaluator,'check',check),patch.object(evaluator,'output',output):evaluator.worker(KEY)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','worker']);a=p.parse_args();globals()[a.mode]()
