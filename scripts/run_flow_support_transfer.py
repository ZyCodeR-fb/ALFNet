"""Ten bounded support-transfer runs, no baseline/speed training or test opening."""
import argparse
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_hparam_completion as previous
from scripts.run_metrla_hparams import read,write,sha,digest
from scripts.collect_hparams_final import params
from scripts.reselect_eight_baselines import SELECTED,EXCLUDED
from experiments.flow_support_transfer import adapter,VARIANTS

NAME='flow_support_transfer_20261003'
DIR=ROOT/'configs/paper'/NAME
PLAN=DIR/'plan.json'
OUT=ROOT/'paper'/NAME
BASE_TABLE=OUT/'fixed_baselines.json'
DATASETS=('PEMS04','PEMS08')


def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']


def freeze():
    from scripts.run_flow_repair import check as prior_check
    prior_check();assert not PLAN.exists()
    prior=read(ROOT/'paper/hparams_combinations_20260930/results.json')['rows']
    full=ROOT/'paper/hparams_combinations_20260930/results_and_provenance.json'
    records=read(full)['records']
    write(BASE_TABLE,dict(records={ds:{n:r for n,r in records[ds].items() if n not in EXCLUDED or n=='ALFNet'} for ds in DATASETS}))
    DIR.mkdir(parents=True,exist_ok=True)
    entries={};anchors={};reused={}
    files=[full,BASE_TABLE,OUT/'PLAN.md',OUT/'LITERATURE_AND_DIAGNOSIS.md',OUT/'historical_diagnosis.json',ROOT/'tests/test_flow_support_transfer.py']
    for ds in DATASETS:
        k=SELECTED[ds];r=prior[k];anchors[ds]=dict(config=r['config'],checkpoint=r['checkpoint'],table=r['table']);reused[k]=r
        base=yaml.safe_load((ROOT/r['config']).read_text());hp=params(r)
        files.extend(ROOT/r[k] for k in ('config','checkpoint','report'))
        files.extend(ROOT/base['family']['experts'][e]['config_path'] for e in ('graph','patch'))
        files.extend(ROOT/base['dataset'][f] for f in ('data_path','unknown_nodes_path'))
        files.append(ROOT/f'configs/paper/v4_data_manifests/{ds}_run_manifest.json')
        for i,v in enumerate(VARIANTS):
            key=f'{i*2+DATASETS.index(ds):02d}_{ds}_{v}'
            cfg=copy.deepcopy(base);cfg['runtime']['model_variant']=NAME+'_'+key
            cfg['family']['recipe']['support_transfer_variant']=v
            path=DIR/(key+'.yaml');assert not path.exists();path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            entries[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),model='ALFNet',dataset=ds,seed=32,
                axis='method',value=v,capacity=hp['capacity'],dropout=hp['dropout'],weight_decay=hp['weight_decay'],
                lrs=[hp[k] for k in ('graph_lr','temporal_lr','gate_lr')])
            files.append(path)
    for folder in ('model','layers','lib','experiments','utils','data_provider','scripts'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.append(ROOT/'args.py')
    p=dict(kind=NAME,status='authorized_validation_only',configs=entries,anchors=anchors,reused=reused,
        max_concurrent=5,test_authorized=False,seed=32,
        authority='2026-10-03 user: reopen four-dataset optimization, first PEMS04/08 multiple candidates max5; speed transfer later if promising, never retrain baselines',
        auxiliary_privilege='No extra labels; interpolation and centering use current visible input only, same teacher mask',
        selection='checkpoint complete validation LifeAvg MAE; compare all33 cells and all3LifeAvg; no hard regression thresholds; no new test without frozen selection',
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])


def check():
    p=read(PLAN)
    assert digest(p)==p['plan_sha256'] and len(p['configs'])==10
    assert p['max_concurrent']==5 and not p['test_authorized']
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    for r in p['configs'].values():
        assert r['dataset'] in DATASETS and r['model']=='ALFNet' and r['value'] in VARIANTS
    return p


def context():
    stack=ExitStack()
    for name,value in [('check',check),('output',output),('NAME',NAME),('BASE_TABLE',BASE_TABLE)]:
        stack.enter_context(patch.object(previous,name,value))
    stack.enter_context(adapter());return stack


def preflight():
    with context():previous.preflight()


def worker(key):
    with context():previous.worker(key)


def collect():
    p=check();root=output(p)
    for key,r in p['configs'].items():
        done=read(root/(key+'_complete.json'));report=read(done['report_path'])
        assert report['model_method_contract']['support_transfer']['variant']==r['value']
        assert all(h[s]['skipped']==0 for h in report['history'] for s in ('train','validation'))
    with context():previous.collect()
    summary=OUT/'results_summary.md'
    summary.write_text(summary.read_text().replace('# 五数据集超参数结果（validation）','# PEMS04/08观测历史迁移实验（validation）').replace('复用31项＋新24项；容量/学习率旧锚点系列单独保留，不把不同锚点误作单变量曲线。','复用2项原版＋新10项方法候选，全部seed32，共396格；无新test。'))
    data=read(OUT/'results.json')
    from scripts.evaluate_flow_repair import workbook
    from scripts.reselect_eight_baselines import audit
    bases=read(BASE_TABLE)['records'];ranking={}
    for ds in DATASETS:
        rows=data['rows'];keys=[k for k,r in rows.items() if r['dataset']==ds]
        ranking[ds]=audit(rows,keys,{n:r for n,r in bases[ds].items() if n!='ALFNet'})
    cells=workbook(data['rows'],ranking,OUT/'validation_results.xlsx')
    write(OUT/'ranking_eight_baselines.json',dict(ranking=ranking,verified_cells=cells,test_accessed=False))


def queue():
    p=check()
    if not (output(p)/'PREFLIGHT_COMPLETE.json').exists():preflight()
    from scripts import recover_baseline_extension as runner
    with ExitStack() as stack:
        for name,value in [('check',check),('output',output),('collect',collect),('__file__',__file__),
                            ('isolate_failed_families',lambda jobs,configs:None)]:
            stack.enter_context(patch.object(runner,name,value))
        runner.queue()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','check','preflight','worker','queue','collect']);parser.add_argument('--key');a=parser.parse_args()
    if a.mode=='worker':worker(a.key)
    elif a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
