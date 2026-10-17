"""Sixteen transparent mechanism controls, no outcome-based inclusion filter."""
import argparse,copy,time,sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_final_ablations_20261006 as base
from scripts.run_metrla_hparams import read,write,sha,digest
from experiments.mechanism_ablation_20261008 import apply
NAME='mechanism_ablations_20261008';DIR=ROOT/'configs/paper'/NAME;PLAN=DIR/'plan.json';OUT=ROOT/'paper'/NAME
LABELS={'no_expert_lifecycle':'w/o Expert Lifecycle Context','no_film_gain':'w/o FiLM Multiplicative Modulation','no_film_shift':'w/o FiLM Additive Modulation','pointwise_refiner':'Pointwise Refinement'}
def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']
def scope():
    s=ExitStack()
    for n,v in [('NAME',NAME),('DIR',DIR),('PLAN',PLAN),('OUT',OUT),('LABELS',LABELS),('output',output),('apply_intervention',apply)]:s.enter_context(patch.object(base,n,v))
    return s
def freeze():
    with scope():base.freeze()
    p=read(PLAN);p['authority']='2026-10-08 user authorizes additional mechanistic ablations on four datasets, max5; all prespecified results retained, no guarantee of degradation'
    p['reporting']='All16 controls and earlier24 ablations retained; no filter requiring Full to win12 LifeAvg metrics; no newtest'
    p['wait_recovery']='configs/paper/ablation_recovery_20261008/plan.json'
    p['files']['experiments/mechanism_ablation_20261008.py']=sha(ROOT/'experiments/mechanism_ablation_20261008.py')
    p['plan_sha256']=digest(p);write(PLAN,p);print('FINAL_PLAN',p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and len(p['configs'])==16 and not p['test_authorized']
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def preflight():
    with scope(),patch.object(base,'check',check):base.preflight()
def worker(key):
    with scope(),patch.object(base,'check',check):base.worker(key)
def collect():
    from scripts.package_supplements_20261008 import book_from_cells
    p=check();root=output(p);rows={d+'_Full':dict(r,label='Full') for d,r in p['anchors'].items()}
    for k,e in p['configs'].items():
        rp=Path(read(root/(k+'_complete.json'))['report_path']);r=read(rp)
        assert r['plan_sha256']==p['plan_sha256'] and r['model_method_contract']['paired_ablation']==e['ablation']
        rows[k]=dict(dataset=e['dataset'],label=e['label'],config=e['path'],table=r['development_validation_table'],report=str(rp.relative_to(ROOT)),checkpoint=str(rp.with_name('best.pt').relative_to(ROOT)))
    cells=[]
    for k,r in rows.items():
        for g in ('main','phase'):
            for c in r['table'][g]:
                for m in ('MAE','RMSE','MAPE'):cells.append(dict(dataset=r['dataset'],label=r['label'],run=k,section=g,condition=c,metric=m,value=r['table'][g][c][m]))
    n=book_from_cells(OUT/'validation_results.xlsx',cells,['seed32 validation; all prespecified interventions retained; exploratory follow-up; no test'],['dataset','section'],['label'])
    assert n==660;write(OUT/'results.json',dict(rows=rows,plan_sha256=p['plan_sha256'],test_accessed=False));write(root/'COMPLETE.json',dict(passed=True,runs=16,cells=n))
def queue():
    p=check();r=read(ROOT/p['wait_recovery']);root=ROOT/'artifacts/experiments/ablation_recovery_20261008'/r['plan_sha256']
    while not (root/'COMPLETE.json').exists():
        write(output(p)/'waiting.json',dict(status='waiting_for_single_ablation_recovery',updated=time.time()));time.sleep(30)
    assert read(root/'COMPLETE.json')['passed']
    if not (output(p)/'PREFLIGHT_COMPLETE.json').exists():preflight()
    from scripts import recover_baseline_extension as runner
    with ExitStack() as stack:
        for n,v in [('check',check),('output',output),('collect',collect),('__file__',__file__),('isolate_failed_families',lambda jobs,configs:None)]:stack.enter_context(patch.object(runner,n,v))
        runner.queue()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','check','preflight','worker','queue','collect']);p.add_argument('--key');a=p.parse_args()
    if a.mode=='worker':worker(a.key)
    elif a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
