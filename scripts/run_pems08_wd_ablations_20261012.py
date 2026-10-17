"""Five paired ablations at the newly confirmed PEMS08 WD=.001 anchor."""
import argparse,copy,sys
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import read,write,sha,digest
from scripts import run_final_ablations_20261006 as original
from scripts import test_pems08_wd_20261012 as test
NAME='pems08_wd_ablations_20261012';DIR=ROOT/'configs/paper'/NAME;PLAN=DIR/'plan.json';OUT=ROOT/'paper'/NAME
KINDS=('no_age','static_router','local_context','no_temporal_film','no_refiner')
def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']
def freeze():
    assert not PLAN.exists();p0=test.check();anchor=p0['entries'][test.KEY]
    decision=ROOT/'paper/pems08_wd_test_20261012/decision.json';assert read(decision)['promoted']
    old=read(ROOT/'configs/paper/final_ablations_20261006/plan.json');configs={};files=[decision,test.PLAN,Path(__file__)]
    DIR.mkdir(parents=True,exist_ok=True)
    for i,kind in enumerate(KINDS):
        row=next(v for v in old['configs'].values() if v['dataset']=='PEMS08' and v['ablation']==kind)
        c=yaml.safe_load((ROOT/row['path']).read_text());c['training']['weight_decay']=.001
        key=f'{i:02d}_PEMS08_{kind}';c['runtime']['model_variant']=NAME+'_'+key
        path=DIR/(key+'.yaml');path.write_text(yaml.safe_dump(c,sort_keys=False))
        a=original.canonical(yaml.safe_load((ROOT/anchor['config']).read_text()));a['family']['recipe']['study_ablation']=kind
        if kind=='static_router':a['family']['experts']['graph']['config_path']['family']['recipe']['moge_router_mode']='static'
        if kind=='local_context':a['family']['experts']['patch']['config_path']['family']['model']['local_context_only']=True
        assert original.canonical(c)==a,key
        configs[key]=dict(row,path=str(path.relative_to(ROOT)),sha256=sha(path));files.append(path)
        files.extend(ROOT/c['family']['experts'][e]['config_path'] for e in ('graph','patch'))
    files.extend(ROOT/f for f in p0['files'])
    files.extend(ROOT/f for f in ('scripts/run_final_ablations_20261006.py','scripts/run_hparam_completion.py','scripts/run_metrla_hparams.py','scripts/recover_baseline_extension.py'))
    p=dict(kind=NAME,status='authorized_validation_only',configs=configs,anchors={'PEMS08':anchor},seed=32,max_concurrent=5,test_authorized=False,authority='2026-10-12 user requests five paired ablations after candidate improves test.',scope='One fresh training per ablation at WD=.001; original definitions, max_epoch100/patience10/start40; retain every outcome; no automatic retries or ablation test.',files={str(f.relative_to(ROOT)):sha(f) for f in set(files)})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and len(p['configs'])==5 and not p['test_authorized']
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def delegated(fn,*args):
    with patch.object(original,'check',check),patch.object(original,'output',output):return fn(*args)
def worker(key):delegated(original.worker,key)
def collect():
    from openpyxl import Workbook
    p=check();root=output(p);rows={'PEMS08_Full':dict(p['anchors']['PEMS08'],label='Full')}
    for key,e in p['configs'].items():
        done=read(root/(key+'_complete.json'));assert done['passed'];rp=Path(done['report_path']);r=read(rp)
        assert r['plan_sha256']==p['plan_sha256'] and r['config_sha256']==e['sha256']
        assert r['model_method_contract']['paired_ablation']==e['ablation']
        assert all(h[s]['skipped']==0 for h in r['history'] for s in ('train','validation'))
        assert min((h['validation']['checkpoint_life_avg_mae'],h['epoch']) for h in r['history'])[1]==r['best_epoch']
        rows[key]=dict(dataset='PEMS08',label=e['label'],config=e['path'],report=str(rp.relative_to(ROOT)),checkpoint=str(rp.with_name('best.pt').relative_to(ROOT)),checkpoint_sha256=sha(rp.with_name('best.pt')),table=r['development_validation_table'])
    write(OUT/'results.json',dict(plan_sha256=p['plan_sha256'],rows=rows,split='validation',test_accessed=False))
    wb=Workbook();wb.remove(wb.active)
    for group in ('main','phase'):
        ws=wb.create_sheet(group);cols=[(c,m) for c in rows['PEMS08_Full']['table'][group] for m in ('MAE','RMSE','MAPE')]
        ws.append(['Variant']+[c+'/'+m for c,m in cols])
        for r in rows.values():ws.append([r['label']]+[r['table'][group][c][m] for c,m in cols])
    wb.save(OUT/'validation_results.xlsx');write(root/'COMPLETE.json',dict(passed=True,runs=5,test_accessed=False))
def queue():
    p=check()
    if not (output(p)/'PREFLIGHT_COMPLETE.json').exists():delegated(original.preflight)
    from scripts import recover_baseline_extension as runner
    with ExitStack() as s:
        for n,v in [('check',check),('output',output),('collect',collect),('__file__',__file__),('isolate_failed_families',lambda jobs,configs:None)]:s.enter_context(patch.object(runner,n,v))
        runner.queue()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','queue','worker','collect']);p.add_argument('--key');a=p.parse_args()
    if a.mode=='worker':worker(a.key)
    else:globals()[a.mode]()
