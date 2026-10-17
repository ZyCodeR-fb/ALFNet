"""Complete the two-method, two-learning-rate validation factorial without new modules."""
import argparse
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_flow_unified as base
from scripts import run_hparam_completion as previous
from scripts.run_metrla_hparams import read,write,sha,digest
from scripts.collect_hparams_final import params
from experiments.flow_unified import adapter as unified_adapter
from experiments.flow_support_transfer import adapter as support_adapter
NAME='flow_lr_cross_20261004'
DIR=ROOT/'configs/paper'/NAME;PLAN=DIR/'plan.json';OUT=ROOT/'paper'/NAME
BASE_TABLE=OUT/'fixed_baselines.json'
SOURCES=('06_PEMS04_bridge_delta1','07_PEMS08_bridge_delta1','04_PEMS04_triple_halfcenter','05_PEMS08_triple_halfcenter')
def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']
def freeze():
    base.check();assert not PLAN.exists()
    old=read(base.OUT/'results.json')['rows']
    write(BASE_TABLE,read(base.BASE_TABLE));DIR.mkdir(parents=True,exist_ok=True)
    files=[BASE_TABLE,OUT/'PLAN.md',OUT/'DIAGNOSIS_AND_LITERATURE.md',OUT/'cohort_diagnostics.json',OUT/'baseline_pressure.json']
    entries={};reused={};anchors={}
    for source in SOURCES:
        r=old[source];reused[source]=r;ds=r['dataset'];hp=params(r)
        anchors.setdefault(ds,dict(config=r['config'],checkpoint=r['checkpoint'],table=r['table']))
        cfg=copy.deepcopy(yaml.safe_load((ROOT/r['config']).read_text()))
        multiplier=2. if ds=='PEMS04' else 1.
        ratio=multiplier/hp['lr_multiplier']
        key=source+'_lr'+str(int(multiplier))
        cfg['runtime']['model_variant']=NAME+'_'+key
        cfg['training']['lr']*=ratio
        recipe=cfg['family']['recipe']
        for name in ('graph_expert_lr','patch_expert_lr','gate_lr'):recipe[name]*=ratio
        path=DIR/(key+'.yaml');assert not path.exists();path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        entries[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),model='ALFNet',dataset=ds,seed=32,
            axis='matched_lr',value=multiplier,capacity=hp['capacity'],dropout=hp['dropout'],weight_decay=hp['weight_decay'],
            lrs=[recipe[k] for k in ('graph_expert_lr','patch_expert_lr','gate_lr')])
        files.extend([path,ROOT/r['config'],ROOT/r['checkpoint'],ROOT/r['report']])
        files.extend(ROOT/cfg['family']['experts'][e]['config_path'] for e in ('graph','patch'))
        files.extend(ROOT/cfg['dataset'][f] for f in ('data_path','unknown_nodes_path'))
        files.append(ROOT/f'configs/paper/v4_data_manifests/{ds}_run_manifest.json')
    for folder in ('model','layers','lib','experiments','utils','data_provider','scripts'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.append(ROOT/'args.py')
    p=dict(kind=NAME,status='authorized_validation_only',configs=entries,anchors=anchors,reused=reused,
        max_concurrent=5,test_authorized=False,seed=32,
        authority='2026-10-04 user authorizes targeted PEMS04/08 optimization; choose direction2, max5; no baseline replacement/retraining',
        selection='Fixed checkpoint minimum validation LifeAvg MAE; rank all33 validation cells and all3LifeAvg; prefer a shared mechanism; no new test',
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and len(p['configs'])==4
    assert not p['test_authorized'] and p['max_concurrent']==5
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def context():
    stack=ExitStack()
    for obj in (base,previous):
        for name,value in [('check',check),('output',output),('NAME',NAME),('BASE_TABLE',BASE_TABLE)]:
            stack.enter_context(patch.object(obj,name,value))
    stack.enter_context(support_adapter());stack.enter_context(unified_adapter())
    return stack
def preflight():
    with context():previous.preflight()
def worker(key):
    with context():previous.worker(key)
def collect():
    with context():previous.collect()
    data=read(OUT/'results.json')
    from scripts.evaluate_flow_repair import workbook
    from scripts.reselect_eight_baselines import audit
    baselines=read(BASE_TABLE)['records'];ranking={}
    for ds in ('PEMS04','PEMS08'):
        keys=[k for k,r in data['rows'].items() if r['dataset']==ds]
        ranking[ds]=audit(data['rows'],keys,{n:r for n,r in baselines[ds].items() if n!='ALFNet'})
    cells=workbook(data['rows'],ranking,OUT/'validation_results.xlsx')
    write(OUT/'ranking_eight_baselines.json',dict(ranking=ranking,verified_cells=cells,test_accessed=False))
    summary=OUT/'results_summary.md'
    summary.write_text(summary.read_text().replace('# 五数据集超参数结果（validation）','# 两机制×两学习率匹配对照（validation）').replace('复用31项＋新24项；容量/学习率旧锚点系列单独保留，不把不同锚点误作单变量曲线。','复用4项＋新4项；两数据集各包含两机制×两学习率，全部seed32。'))
def queue():
    p=check()
    if not (output(p)/'PREFLIGHT_COMPLETE.json').exists():preflight()
    from scripts import recover_baseline_extension as runner
    with ExitStack() as stack:
        for name,value in [('check',check),('output',output),('collect',collect),('__file__',__file__),('isolate_failed_families',lambda jobs,configs:None)]:
            stack.enter_context(patch.object(runner,name,value))
        runner.queue()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','check','preflight','worker','queue','collect']);p.add_argument('--key');a=p.parse_args()
    if a.mode=='worker':worker(a.key)
    elif a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
