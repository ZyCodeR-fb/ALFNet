"""Eight gradient-path controls queued after the sixteen mechanism runs."""
import argparse,time,sys
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_mechanism_ablations_20261008 as parent
from scripts.run_metrla_hparams import read,write,sha,digest
from experiments.reconstruction_gradient_20261009 import apply
NAME='reconstruction_gradient_20261009';DIR=ROOT/'configs/paper'/NAME;PLAN=DIR/'plan.json';OUT=ROOT/'paper'/NAME
LABELS={'stop_graph_recon_gradient':'Stop Graph-to-Reconstruction Gradient','stop_temporal_recon_gradient':'Stop Temporal-to-Reconstruction Gradient'}
def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']
def scope():
    stack=ExitStack()
    for n,v in [('NAME',NAME),('DIR',DIR),('PLAN',PLAN),('OUT',OUT),('LABELS',LABELS),('output',output),('apply',apply)]:stack.enter_context(patch.object(parent,n,v))
    return stack
def freeze():
    prior=parent.check()
    with scope():parent.freeze()
    p=read(PLAN);p['authority']='2026-10-09 user requests inspect outcomes and additional scientifically motivated ablations if few generalize; max5, all outcomes retained'
    p['reporting']='All 8 gradient-path controls plus earlier40 ablations retained; do not select reporting by whether Full wins'
    p['wait_mechanism']=dict(plan='configs/paper/mechanism_ablations_20261008/plan.json',sha256=prior['plan_sha256'])
    p['plan_sha256']=digest(p);write(PLAN,p);print('FINAL_PLAN',p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and len(p['configs'])==8 and p['status']=='authorized_validation_only'
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def preflight():
    with scope(),patch.object(parent,'check',check):parent.preflight()
def worker(key):
    with scope(),patch.object(parent,'check',check):parent.worker(key)
def collect():
    # Existing collector's serialization is independent of the matrix size;
    # export directly with the same complete table layout.
    from scripts.package_supplements_20261008 import book_from_cells
    p=check();root=output(p);rows={d+'_Full':dict(r,label='Full') for d,r in p['anchors'].items()}
    for k,e in p['configs'].items():
        rp=Path(read(root/(k+'_complete.json'))['report_path']);r=read(rp)
        assert r['plan_sha256']==p['plan_sha256'] and r['model_method_contract']['paired_ablation']==e['ablation']
        rows[k]=dict(dataset=e['dataset'],label=e['label'],config=e['path'],report=str(rp.relative_to(ROOT)),checkpoint=str(rp.with_name('best.pt').relative_to(ROOT)),table=r['development_validation_table'])
    cells=[dict(dataset=r['dataset'],label=r['label'],section=g,condition=c,metric=m,value=r['table'][g][c][m]) for r in rows.values() for g in ('main','phase') for c in r['table'][g] for m in ('MAE','RMSE','MAPE')]
    n=book_from_cells(OUT/'validation_results.xlsx',cells,['seed32 validation; gradient-path controls; all results retained'],['dataset','section'],['label']);assert n==396
    write(OUT/'results.json',dict(rows=rows,plan_sha256=p['plan_sha256'],test_accessed=False));write(root/'COMPLETE.json',dict(passed=True,runs=8,cells=n))
def queue():
    p=check();previous=read(ROOT/p['wait_mechanism']['plan']);assert previous['plan_sha256']==p['wait_mechanism']['sha256']
    root=ROOT/'artifacts/experiments/mechanism_ablations_20261008'/previous['plan_sha256']
    while not (root/'COMPLETE.json').exists():
        if (root/'ATTENTION_REQUIRED.json').exists():raise RuntimeError('predecessor failed; do not start new studies')
        write(output(p)/'waiting.json',dict(status='waiting_for_mechanism16',updated=time.time()));time.sleep(30)
    assert read(root/'COMPLETE.json')['passed']
    if not (output(p)/'PREFLIGHT_COMPLETE.json').exists():preflight()
    from scripts import recover_baseline_extension as runner
    with ExitStack() as s:
        for n,v in [('check',check),('output',output),('collect',collect),('__file__',__file__),('isolate_failed_families',lambda jobs,configs:None)]:s.enter_context(patch.object(runner,n,v))
        runner.queue()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','check','preflight','worker','queue','collect']);p.add_argument('--key');a=p.parse_args()
    if a.mode=='worker':worker(a.key)
    elif a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
