"""Six PEMS03 configurations of the accepted support-conditioned scheme."""
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
NAME='pems03_support_20261004'
DIR=ROOT/'configs/paper'/NAME;PLAN=DIR/'plan.json';OUT=ROOT/'paper'/NAME
BASE_TABLE=OUT/'fixed_baselines.json'
def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']
def freeze():
    from scripts.run_flow_repair import check as original_check
    from scripts.reselect_eight_baselines import EXCLUDED
    original_check();assert not PLAN.exists()
    source=ROOT/'paper/hparams_combinations_20260930/results.json'
    full=ROOT/'paper/hparams_combinations_20260930/results_and_provenance.json'
    source_key='PEMS03_capacity_mid';r=read(source)['rows'][source_key];hp=params(r)
    records=read(full)['records']['PEMS03']
    write(BASE_TABLE,dict(records={'PEMS03':{n:v for n,v in records.items() if n not in EXCLUDED or n=='ALFNet'}}))
    DIR.mkdir(parents=True,exist_ok=True)
    files=[BASE_TABLE,OUT/'PLAN.md',source,full]
    entries={};reused={source_key:r}
    anchors={'PEMS03':dict(config=r['config'],checkpoint=r['checkpoint'],table=r['table'])}
    original=yaml.safe_load((ROOT/r['config']).read_text())
    files.extend(ROOT/r[f] for f in ('config','checkpoint','report'))
    files.extend(ROOT/original['family']['experts'][e]['config_path'] for e in ('graph','patch'))
    files.extend(ROOT/original['dataset'][f] for f in ('data_path','unknown_nodes_path'))
    files.append(ROOT/'configs/paper/v4_data_manifests/PEMS03_run_manifest.json')
    for strength,label in [(0.,'zero'),(.5,'half'),(1.,'full')]:
        for multiplier in (1.,2.):
            key=f'PEMS03_center_{label}_lr{int(multiplier)}'
            cfg=copy.deepcopy(original);cfg['runtime']['model_variant']=NAME+'_'+key
            cfg['training']['lr']*=multiplier/hp['lr_multiplier']
            recipe=cfg['family']['recipe']
            if strength==0:recipe['support_transfer_variant']='bridge_delta1'
            else:recipe['unified_variant']='triple_halfcenter' if strength==.5 else 'triple'
            for n in ('graph_expert_lr','patch_expert_lr','gate_lr'):recipe[n]*=multiplier/hp['lr_multiplier']
            path=DIR/(key+'.yaml');assert not path.exists();path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            entries[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),model='ALFNet',dataset='PEMS03',seed=32,
                axis='center_lr',value=dict(center=strength,lr_multiplier=multiplier),capacity=hp['capacity'],dropout=hp['dropout'],weight_decay=hp['weight_decay'],
                lrs=[recipe[k] for k in ('graph_expert_lr','patch_expert_lr','gate_lr')])
            files.append(path)
    for folder in ('model','layers','lib','experiments','utils','data_provider','scripts'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.append(ROOT/'args.py')
    p=dict(kind=NAME,status='authorized_validation_only',configs=entries,anchors=anchors,reused=reused,
        max_concurrent=5,test_authorized=False,seed=32,
        authority='2026-10-04 user accepts interpolation+decoder scale+centering as one parameterized scheme; authorizes PEMS03 multiple configurations, retain if qualified',
        selection='checkpoint minimum full validation LifeAvg MAE; qualifying >=27/33 vs eight baselines and all three LifeAvg leading; retain all results; no automatic test',
        parameterization='same visible interpolation, decoder scale1, center alpha*last_visible, alpha=0/.5/1, empty history center0; same teacher/output centering',
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and len(p['configs'])==6
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
    for ds in ('PEMS03',):
        keys=[k for k,r in data['rows'].items() if r['dataset']==ds]
        ranking[ds]=audit(data['rows'],keys,{n:r for n,r in baselines[ds].items() if n!='ALFNet'})
    cells=workbook(data['rows'],ranking,OUT/'validation_results.xlsx')
    write(OUT/'ranking_eight_baselines.json',dict(ranking=ranking,verified_cells=cells,test_accessed=False))
    summary=OUT/'results_summary.md'
    summary.write_text(summary.read_text().replace('# 五数据集超参数结果（validation）','# PEMS03中心化强度×学习率（validation）').replace('复用31项＋新24项；容量/学习率旧锚点系列单独保留，不把不同锚点误作单变量曲线。','复用1项原版＋新6项，同方案中心强度0/.5/1×学习率1/2，全部seed32。'))
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

