"""Sixteen bounded speed-transfer configurations; existing flow adapters unchanged."""
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
NAME='speed_support_20261004'
DIR=ROOT/'configs/paper'/NAME;PLAN=DIR/'plan.json';OUT=ROOT/'paper'/NAME
BASE_TABLE=OUT/'fixed_baselines.json'
DATASETS=('METRLA','PEMSBAY')
def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']
def freeze():
    from scripts.run_flow_repair import check as original_check
    from scripts.reselect_eight_baselines import EXCLUDED
    original_check();assert not PLAN.exists()
    source=ROOT/'paper/hparams_combinations_20260930/results.json'
    full=ROOT/'paper/hparams_combinations_20260930/results_and_provenance.json'
    manifest=ROOT/'paper/final_four_tables_20261001/selected_configuration_manifest.json'
    selected=read(manifest)['selected'];old=read(source)['rows'];records=read(full)['records']
    write(BASE_TABLE,dict(records={ds:{n:v for n,v in records[ds].items() if n not in EXCLUDED or n=='ALFNet'} for ds in DATASETS}))
    DIR.mkdir(parents=True,exist_ok=True)
    files=[BASE_TABLE,OUT/'PLAN.md',source,full,manifest];entries={};reused={};anchors={}
    for ds in DATASETS:
        source_key=selected[ds];r=old[source_key];hp=params(r);reused[source_key]=r
        anchors[ds]=dict(config=r['config'],checkpoint=r['checkpoint'],table=r['table'],key=source_key)
        original=yaml.safe_load((ROOT/r['config']).read_text())
        files.extend(ROOT/r[f] for f in ('config','checkpoint','report'))
        files.extend(ROOT/original['family']['experts'][e]['config_path'] for e in ('graph','patch'))
        files.extend(ROOT/original['dataset'][f] for f in ('data_path','unknown_nodes_path'))
        files.append(ROOT/f'configs/paper/v4_data_manifests/{ds}_run_manifest.json')
        matrix=[(a,lr,hp['weight_decay']) for a in (0.,.5) for lr in (.5,1.,2.)]+[(1.,1.,hp['weight_decay'])]
        if ds=='PEMSBAY':matrix += [(a,2.,1e-4) for a in (0.,.5)]
        for strength,multiplier,wd in matrix:
            label={0.:'zero',.5:'half',1.:'full'}[strength]
            key=f'{ds}_center_{label}_lr{multiplier:g}_wd{wd:g}'
            cfg=copy.deepcopy(original);cfg['runtime']['model_variant']=NAME+'_'+key
            cfg['training']['lr']*=multiplier/hp['lr_multiplier'];cfg['training']['weight_decay']=wd
            recipe=cfg['family']['recipe']
            if strength==0:recipe['support_transfer_variant']='bridge_delta1'
            else:recipe['unified_variant']='triple_halfcenter' if strength==.5 else 'triple'
            for n in ('graph_expert_lr','patch_expert_lr','gate_lr'):recipe[n]*=multiplier/hp['lr_multiplier']
            path=DIR/(key+'.yaml');assert not path.exists();path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            entries[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),model='ALFNet',dataset=ds,seed=32,
                axis='support_transfer',value=dict(center=strength,lr_multiplier=multiplier,weight_decay=wd),capacity=hp['capacity'],dropout=hp['dropout'],weight_decay=wd,
                lrs=[recipe[k] for k in ('graph_expert_lr','patch_expert_lr','gate_lr')])
            files.append(path)
    for folder in ('model','layers','lib','experiments','utils','data_provider','scripts'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.append(ROOT/'args.py')
    p=dict(kind=NAME,status='authorized_validation_only',configs=entries,anchors=anchors,reused=reused,
        max_concurrent=5,test_authorized=False,seed=32,
        authority='2026-10-04 user authorizes speed transfer on METRLA/PEMSBAY with multiple center/LR/regularization settings,max5; preserve original if new scheme not superior',
        selection='Checkpoint minimum full validation LifeAvg MAE; compare33cells vs8baselines and selected original, all3LifeAvg; prefer clear gains, report tradeoffs without arbitrary regression thresholds; no automatictest',
        parameterization='visible interpolation; residual decoder scale1; center alpha lastvisible 0/.5/1, empty0; same teacher/output transforms',
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and len(p['configs'])==16
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
    for ds in DATASETS:
        keys=[k for k,r in data['rows'].items() if r['dataset']==ds]
        ranking[ds]=audit(data['rows'],keys,{n:r for n,r in baselines[ds].items() if n!='ALFNet'})
    cells=workbook(data['rows'],ranking,OUT/'validation_results.xlsx')
    write(OUT/'ranking_eight_baselines.json',dict(ranking=ranking,verified_cells=cells,test_accessed=False))
    summary=OUT/'results_summary.md'
    summary.write_text(summary.read_text().replace('# 五数据集超参数结果（validation）','# Speed支持方案迁移（validation）').replace('复用31项＋新24项；容量/学习率旧锚点系列单独保留，不把不同锚点误作单变量曲线。','复用2项当前选定配置＋新16项，METR-LA7项、PEMS-BAY9项，全部seed32。'))
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


