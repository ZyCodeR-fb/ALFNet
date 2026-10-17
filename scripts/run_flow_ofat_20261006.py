"""Final-scheme flow OFAT: fourteen missing runs, four verified reuses."""
import argparse
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_hparam_completion as base
from scripts.run_metrla_hparams import read,write,sha,digest
from experiments.flow_support_transfer import adapter as support_adapter
from experiments.flow_unified import adapter as unified_adapter
NAME='flow_ofat_20261006';DIR=ROOT/'configs/paper'/NAME;PLAN=DIR/'plan.json'
OUT=ROOT/'paper'/NAME;BASE_TABLE=OUT/'fixed_baselines.json'
DATASETS=('PEMS04','PEMS08')
ANCHORS={'PEMS04':'06_PEMS04_bridge_delta1','PEMS08':'05_PEMS08_triple_halfcenter_lr1'}
DOUBLE={'PEMS04':'06_PEMS04_bridge_delta1_lr2','PEMS08':'05_PEMS08_triple_halfcenter'}
def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']
def normalized(cfg):
    c=copy.deepcopy(cfg);c['runtime'].pop('model_variant',None)
    for e in ('graph','patch'):
        slot=c['family']['experts'][e];slot['config_path']=yaml.safe_load((ROOT/slot['config_path']).read_text())
    return c
def specs(ds):
    caps=(14,18) if ds=='PEMS04' else (8,10)
    return [('capacity',v) for v in caps]+[('lr_factor',.5),('dropout',.05),('dropout',.2),('weight_decay',1e-5),('weight_decay',1e-3)]
def freeze():
    assert not PLAN.exists()
    from scripts.run_flow_lr_cross_20261004 import check as prior_check
    from scripts.reselect_eight_baselines import EXCLUDED
    prior=prior_check();source=ROOT/'paper/flow_lr_cross_20261004/results.json';rows=read(source)['rows']
    records=read(ROOT/'paper/hparams_combinations_20260930/results_and_provenance.json')['records']
    write(BASE_TABLE,dict(records={d:{n:r for n,r in records[d].items() if n not in EXCLUDED or n=='ALFNet'} for d in DATASETS}))
    DIR.mkdir(parents=True,exist_ok=True)
    files=[source,BASE_TABLE,OUT/'PLAN.md'];entries={};reused={};anchors={};audit=[]
    for ds in DATASETS:
        anchor=rows[ANCHORS[ds]];cfg=yaml.safe_load((ROOT/anchor['config']).read_text())
        nested=yaml.safe_load((ROOT/cfg['family']['experts']['patch']['config_path']).read_text())
        anchors[ds]=dict(config=anchor['config'],checkpoint=anchor['checkpoint'],table=anchor['table'])
        for key in (ANCHORS[ds],DOUBLE[ds]):
            r=copy.deepcopy(rows[key]);report=read(ROOT/r['report'])
            assert report['development_validation_table']==r['table']
            assert report['config_sha256']==sha(ROOT/r['config'])
            assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
            assert all(e[s]['skipped']==0 for e in report['history'] for s in ('train','validation'))
            assert min((e['validation']['checkpoint_life_avg_mae'],e['epoch']) for e in report['history'])[1]==report['best_epoch']
            actual=normalized(yaml.safe_load((ROOT/r['config']).read_text()));expected=normalized(cfg)
            if key==DOUBLE[ds]:
                expected['training']['lr']*=2
                for n in ('graph_expert_lr','patch_expert_lr','gate_lr'):expected['family']['recipe'][n]*=2
            assert actual==expected,(ds,key,'not a paired LR reuse')
            r.update(axis='anchor' if key==ANCHORS[ds] else 'lr_factor',value=1 if key==ANCHORS[ds] else 2)
            reused[key]=r;files.extend(ROOT/r[f] for f in ('config','checkpoint','report'))
            audit.append(dict(key=key,paired_semantics_verified=True,checkpoint_sha256=r['checkpoint_sha256']))
        for i,(axis,value) in enumerate(specs(ds)):
            key=f'{i*2+DATASETS.index(ds):02d}_{ds}_{axis}_{value:g}'
            pp=DIR/(key+'_patch.yaml');path=DIR/(key+'.yaml')
            c,n=base.candidate(cfg,nested,axis,value,str(pp.relative_to(ROOT)))
            c['runtime']['model_variant']=NAME+'_'+key
            pp.write_text(yaml.safe_dump(n,sort_keys=False));path.write_text(yaml.safe_dump(c,sort_keys=False))
            # Removing the intended single factor must restore every semantic field.
            got=normalized(c);expected=normalized(cfg)
            if axis=='capacity':got['family']['experts']['patch']['config_path']['family']['model']['spa_patchsize']=nested['family']['model']['spa_patchsize']
            elif axis=='dropout':got['family']['experts']['patch']['config_path']['family']['model']['crossformer_dropout']=nested['family']['model']['crossformer_dropout']
            elif axis=='weight_decay':got['training']['weight_decay']=cfg['training']['weight_decay']
            else:
                got['training']['lr']/=value
                for name in ('graph_expert_lr','patch_expert_lr','gate_lr'):got['family']['recipe'][name]/=value
            assert got==expected,key
            entries[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),model='ALFNet',dataset=ds,seed=32,axis=axis,value=value,
                capacity=n['family']['model']['spa_patchsize'],dropout=n['family']['model']['crossformer_dropout'],weight_decay=c['training']['weight_decay'],
                lrs=[c['family']['recipe'][n] for n in ('graph_expert_lr','patch_expert_lr','gate_lr')])
            files.extend([path,pp])
        files.extend(ROOT/cfg['family']['experts'][e]['config_path'] for e in ('graph','patch'))
        files.extend(ROOT/cfg['dataset'][f] for f in ('data_path','unknown_nodes_path'))
        files.append(ROOT/f'configs/paper/v4_data_manifests/{ds}_run_manifest.json')
    # Historical enhanced-scheme configs contain no other matches to the 14 targets.
    matches={}
    for folder in ('flow_support_transfer_20261003','flow_unified_20261003','flow_lr_cross_20261004','hparams_completion_20260928','hparams_combinations_20260930'):
        pth=ROOT/'paper'/folder/'results.json';files.append(pth)
        for key,r in read(pth)['rows'].items():
            if r['dataset'] not in DATASETS:continue
            c=normalized(yaml.safe_load((ROOT/r['config']).read_text()))
            for new,e in entries.items():
                if c==normalized(yaml.safe_load((ROOT/e['path']).read_text())):matches[new]=key
    assert not matches,('duplicate training',matches)
    for folder in ('model','layers','lib','experiments','utils','data_provider','scripts'):files.extend((ROOT/folder).rglob('*.py'))
    files.append(ROOT/'args.py')
    p=dict(kind=NAME,status='authorized_validation_only',configs=entries,reused=reused,anchors=anchors,semantic_reuse_audit=audit,
        parent_plan_sha256=prior['plan_sha256'],max_concurrent=5,test_authorized=False,seed=32,
        authority='2026-10-06 user authorizes only PEMS04/08 fixed-anchor OFAT completion, reuse existing equivalent results, max5 concurrent',
        selection='Fixed final anchors; minimum validation LifeAvg MAE checkpoint; report all33 cells; no model reselection or test',
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and len(p['configs'])==14 and len(p['reused'])==4
    assert not p['test_authorized'] and p['max_concurrent']==5
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def context():
    stack=ExitStack()
    for name,value in [('check',check),('output',output),('NAME',NAME),('BASE_TABLE',BASE_TABLE)]:stack.enter_context(patch.object(base,name,value))
    stack.enter_context(support_adapter());stack.enter_context(unified_adapter());return stack
def preflight():
    with context():base.preflight()
def worker(key):
    with context():base.worker(key)
def collect():
    with context():base.collect()
    from scripts.evaluate_flow_repair import workbook
    from scripts.reselect_eight_baselines import audit
    data=read(OUT/'results.json');records=read(BASE_TABLE)['records'];ranking={}
    for ds in DATASETS:ranking[ds]=audit(data['rows'],[k for k,r in data['rows'].items() if r['dataset']==ds],{n:r for n,r in records[ds].items() if n!='ALFNet'})
    cells=workbook(data['rows'],ranking,OUT/'validation_results.xlsx')
    write(OUT/'completion_audit.json',dict(verified_cells=cells,total_configurations=18,new_runs=14,reused=4,test_accessed=False))
    path=OUT/'results_summary.md';path.write_text(path.read_text().replace('# 五数据集超参数结果（validation）','# PEMS04/08 最终方案单因素实验（validation）').replace('复用31项＋新24项；容量/学习率旧锚点系列单独保留，不把不同锚点误作单变量曲线。','复用4项＋新14项；每个数据集固定最终锚点，只改变一个因素，完整保留33格。不自动更换最终配置。'))
def queue():
    p=check()
    if not (output(p)/'PREFLIGHT_COMPLETE.json').exists():preflight()
    from scripts import recover_baseline_extension as runner
    with ExitStack() as stack:
        for name,value in [('check',check),('output',output),('collect',collect),('__file__',__file__),('isolate_failed_families',lambda jobs,configs:None)]:stack.enter_context(patch.object(runner,name,value))
        runner.queue()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','check','preflight','worker','queue','collect']);p.add_argument('--key');a=p.parse_args()
    if a.mode=='worker':worker(a.key)
    elif a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
