"""Frozen evaluation of five validation-selected final flow candidates."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import ExitStack
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_flow_unified as train
from scripts import evaluate_flow_repair as prior
from scripts import test_all_final_20261001 as evaluator
from scripts.run_metrla_hparams import read,write,sha,digest
from scripts.reselect_eight_baselines import audit,EXCLUDED,SELECTED
from experiments.flow_unified import adapter
NAME='flow_unified_test_20261004'
OUT=ROOT/'paper'/NAME
PLAN=ROOT/'configs/paper'/NAME/'plan.json'
SOURCE=train.OUT/'results.json'
KEYS=('00_PEMS04_triple','01_PEMS08_triple','06_PEMS04_triple_spatial',
      '07_PEMS08_triple_spatial','05_PEMS08_triple_halfcenter')
def output(p):return ROOT/'artifacts/evaluation'/NAME/p['plan_sha256']
def analyze():
    with ExitStack() as stack:
        for name,value in [('train',train),('OUT',OUT),('SOURCE',SOURCE),('KEYS',KEYS)]:
            stack.enter_context(patch.object(prior,name,value))
        prior.analyze()
def freeze():
    import torch,yaml
    assert not PLAN.exists();train.check()
    assert read(OUT/'validation_audit.json')['passed']
    rows=read(SOURCE)['rows'];entries={}
    files=[SOURCE,OUT/'validation_audit.json',OUT/'TEST_PLAN.md',train.PLAN,ROOT/'paper/flow_support_test_20261003/test_results.json']
    for k in KEYS:
        r=copy.deepcopy(rows[k]);cp=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
        assert cp['seed']==32 and sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        r.update(model='ALFNet',epoch=cp['epoch'],training_config_sha256=cp['config_sha256'],training_plan_sha256=cp['plan_sha256'])
        assert sha(ROOT/r['config'])==r['training_config_sha256']
        entries[k]=r;cfg=yaml.safe_load((ROOT/r['config']).read_text())
        files.extend(ROOT/r[f] for f in ('config','checkpoint','report'))
        files.extend(ROOT/cfg['family']['experts'][e]['config_path'] for e in ('graph','patch'))
        files.extend(ROOT/cfg['dataset'][f] for f in ('data_path','unknown_nodes_path'))
        files.append(ROOT/f"configs/paper/v4_data_manifests/{r['dataset']}_run_manifest.json")
    files.append(ROOT/'paper/all_final_test_20261001/all_results_and_provenance.json')
    for folder in ('model','layers','experiments','lib','data_provider','utils','scripts'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.append(ROOT/'args.py')
    p=dict(kind=NAME,test_authorized=True,authorization='2026-10-04 user: analyze completed support-transfer runs and test qualifying or worthwhile candidates; same task preserves earlier explicit exploratory-test approval',entries=entries,max_concurrent=3,
           training_adapter='experiments.flow_unified.adapter',scope='Five fixed validation finalists, no training; reuse all baseline and original tests',
           files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and p['test_authorized']
    assert set(p['entries'])==set(KEYS) and p['max_concurrent']==3
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def worker(key):
    import torch
    from experiments import v4_runner
    p=check();cp=torch.load(ROOT/p['entries'][key]['checkpoint'],map_location='cpu',weights_only=False)
    with ExitStack() as stack:
        stack.enter_context(adapter())
        adapted_build=v4_runner.build_v4_session
        def verified_build(*args,**kwargs):
            from scripts.train_v4 import model_method_contract
            s=adapted_build(*args,**kwargs)
            assert model_method_contract(s)==cp['model_method_contract']
            return s
        stack.enter_context(patch.object(v4_runner,'build_v4_session',verified_build))
        stack.enter_context(patch.object(evaluator,'check',check))
        stack.enter_context(patch.object(evaluator,'output',output))
        evaluator.worker(key)
    f=output(p)/(key+'.json');r=read(f)
    r['method_contract']=cp['model_method_contract']
    r['table']['note']='Five finalists fixed from validation; historical test informed repair design, not an untouched confirmation set.'
    write(f,r)
def collect():
    p=check();rows={k:read(output(p)/(k+'.json')) for k in KEYS}
    prior_tests=read(ROOT/'paper/flow_support_test_20261003/test_results.json')['rows']
    for k,r in prior_tests.items():
        if k not in rows:rows[k]=r
    old=read(ROOT/'paper/all_final_test_20261001/all_results_and_provenance.json');ranking={}
    for ds in train.DATASETS:
        rows[SELECTED[ds]]=old['rows'][SELECTED[ds]]
        bases={e['model']:old['rows'][k] for k,e in old['entries'].items() if e['dataset']==ds and e['model'] not in EXCLUDED and e['model']!='ALFNet'}
        assert len(bases)==8
        ks=[k for k,r in rows.items() if r['dataset']==ds]
        for k in ks:
            r=rows[k];assert r['passed']
            if k in KEYS:assert r['checkpoint_sha256']==p['entries'][k]['checkpoint_sha256']
            for baseline in bases.values():
                for g,cs in prior.GROUPS.items():
                    for c in cs:
                        for f,v in r['table'][g][c].items():
                            if f.endswith('_count'):assert baseline['table'][g][c][f]==v
        ranking[ds]=audit(rows,ks,bases)
    cells=prior.workbook(rows,ranking,OUT/'test_results.xlsx')
    write(OUT/'test_results.json',dict(rows=rows,ranking=ranking,plan_sha256=p['plan_sha256'],verified_cells=cells))
    write(output(p)/'COMPLETE.json',dict(passed=True,new_tests=5,reused_controls=6))
    print(ranking)
def queue():
    import fcntl
    p=check();root=output(p);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'queue_status.json').exists()
    jobs=[dict(key=k,status='queued') for k in KEYS];active={};last=0
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    while any(j['status'] in ('queued','running') for j in jobs):
        for k,(proc,log) in list(active.items()):
            if proc.poll() is not None:
                j=next(j for j in jobs if j['key']==k)
                j.update(status='complete' if proc.returncode==0 and (root/(k+'.json')).exists() else 'failed',exit_code=proc.returncode)
                log.close();del active[k]
        j=next((j for j in jobs if j['status']=='queued'),None)
        if j and len(active)<p['max_concurrent'] and time.monotonic()-last>20:
            free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())
            if free>20000:
                k=j['key'];log=(root/(k+'.log')).open('x')
                proc=subprocess.Popen([sys.executable,'-u',__file__,'worker','--key',k],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[k]=(proc,log);j.update(status='running',pid=proc.pid);last=time.monotonic()
        write(root/'queue_status.json',dict(jobs=jobs,plan_sha256=p['plan_sha256'],updated=time.time()))
        if active:time.sleep(10)
    assert all(j['status']=='complete' for j in jobs),jobs
    collect()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['analyze','freeze','check','worker','queue','collect']);p.add_argument('--key');a=p.parse_args()
    if a.mode=='worker':worker(a.key)
    elif a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
