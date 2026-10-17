"""Exactly two authorized seed32 repeats; keep original outcomes and no test."""
import argparse,csv,sys
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_final_ablations_20261006 as original
from scripts.run_metrla_hparams import read,write,sha,digest
NAME='ablation_repeat_20261012'
PLAN=ROOT/'configs/paper'/NAME/'plan.json'
OUT=ROOT/'paper'/NAME
KEYS=('02_PEMS04_no_age','07_PEMS08_static_router')
def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']
def freeze():
    assert not PLAN.exists()
    source=ROOT/'configs/paper/final_ablations_20261006/plan.json';old=read(source)
    assert digest(old)==old['plan_sha256']
    prefixes=('model/','layers/','lib/','experiments/','utils/','data_provider/')
    core=['scripts/train_v4.py','scripts/run_hparam_completion.py','scripts/run_final_ablations_20261006.py','scripts/run_metrla_hparams.py']
    files={f:h for f,h in old['files'].items() if f.startswith(prefixes) or f in core}
    for f,h in files.items():assert sha(ROOT/f)==h,('historical training code changed',f)
    # Configs/data/nested experts and historical anchors are unchanged.
    for f,h in old['files'].items():
        if f.startswith(('configs/','data_v4/')):
            assert sha(ROOT/f)==h,f
            files[f]=h
    prior=ROOT/'paper/ablations_complete_20261010/results_and_audit.json'
    evidence=read(prior);rows=evidence['rows'];configs={k:old['configs'][k] for k in KEYS}
    for k in KEYS:rows[k]['checkpoint_sha256']=evidence['audits'][k]['checkpoint_sha256']
    anchors={d:old['anchors'][d] for d in ('PEMS04','PEMS08')}
    for r in list(anchors.values())+[rows[k] for k in KEYS]:
        assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        for name in ['checkpoint','report','config']:
            if name in r:files[r[name]]=sha(ROOT/r[name])
    for f in [source,prior,Path(__file__),ROOT/'scripts/recover_baseline_extension.py',OUT/'PLAN.md']:
        files[str(f.relative_to(ROOT))]=sha(f)
    p=dict(kind=NAME,status='authorized_validation_only',configs=configs,anchors=anchors,originals={k:rows[k] for k in KEYS},
        parent_plan_sha256=old['plan_sha256'],seed=32,max_concurrent=2,test_authorized=False,
        authorization='User 2026-10-12 explicitly requests PEMS04 noAge and PEMS08 staticRouting retraining.',
        scope='One fresh start per existing config; same seed/split/hyperparameters/definitions, isolated output; no result selection or automatic additional retry.',
        files=files)
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and set(p['configs'])==set(KEYS)
    assert p['seed']==32 and not p['test_authorized']
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def delegated(fn,*args):
    with patch.object(original,'check',check),patch.object(original,'output',output):return fn(*args)
def worker(key):
    assert key in KEYS
    delegated(original.worker,key)
def collect():
    p=check();root=output(p);rows={};cells=[];lines=['# 两项原配置重跑（validation）','','全部新旧结果保留；同seed重复不是独立种子证据，不按误差方向筛选。','',
        '| 数据集 / 消融 | 记录 | MAE | RMSE | MAPE |','|---|---|---:|---:|---:|']
    for k in KEYS:
        done=read(root/(k+'_complete.json'));assert done['passed'];rp=Path(done['report_path']);r=read(rp)
        assert r['plan_sha256']==p['plan_sha256'] and r['config_sha256']==p['configs'][k]['sha256']
        assert r['model_method_contract']['paired_ablation']==p['configs'][k]['ablation']
        assert all(e[s]['skipped']==0 for e in r['history'] for s in ('train','validation'))
        assert min((e['validation']['checkpoint_life_avg_mae'],e['epoch']) for e in r['history'])[1]==r['best_epoch']
        ds=p['configs'][k]['dataset'];tables={'Full':p['anchors'][ds]['table'],'original_ablation':p['originals'][k]['table'],'repeat_ablation':r['development_validation_table']}
        rows[k]=dict(dataset=ds,report=str(rp.relative_to(ROOT)),checkpoint=str(rp.with_name('best.pt').relative_to(ROOT)),checkpoint_sha256=sha(rp.with_name('best.pt')),tables=tables)
        for label,t in tables.items():
            lines.append('| '+k+' | '+label+' | '+' | '.join(f"{t['main']['LifeAvg'][m]:.6f}" for m in ('MAE','RMSE','MAPE'))+' |')
            for group in ('main','phase'):
                for c,v in t[group].items():
                    for m in ('MAE','RMSE','MAPE'):cells.append([ds,k,label,group,c,m,v[m]])
    write(OUT/'results.json',dict(plan_sha256=p['plan_sha256'],rows=rows,test_accessed=False,automatic_replacement=False))
    with (OUT/'validation_cells.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['dataset','run','record','section','condition','metric','value']);w.writerows(cells)
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(root/'COMPLETE.json',dict(passed=True,new_runs=2,test_accessed=False,retained_both_outcomes=True))
def queue():
    p=check()
    if not (output(p)/'PREFLIGHT_COMPLETE.json').exists():delegated(original.preflight)
    from scripts import recover_baseline_extension as runner
    with ExitStack() as stack:
        for n,v in [('check',check),('output',output),('collect',collect),('__file__',__file__),('isolate_failed_families',lambda jobs,configs:None)]:stack.enter_context(patch.object(runner,n,v))
        runner.queue() # Only two entries: actual concurrency <=2, even with shared max5 dispatcher.
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['freeze','check','queue','worker','collect']);a.add_argument('--key');args=a.parse_args()
    if args.mode=='worker':worker(args.key)
    elif args.mode=='check':print(check()['plan_sha256'])
    else:globals()[args.mode]()
