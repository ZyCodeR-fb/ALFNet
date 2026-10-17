"""Final 12 validation-only combinations, reusing the verified common trainer."""
import argparse
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_hparam_completion as previous
from scripts.run_metrla_hparams import read,write,sha,digest
from scripts.collect_hparams_final import params
from scripts.organize_hparam_ofat import expanded,ALLOWED

NAME='hparams_combinations_20260930'
DIR=ROOT/'configs/paper'/NAME
PLAN=DIR/'plan.json'
PRIOR=ROOT/'paper/validation_seed32_hparams_final_20260930'
DATASETS=('PEMSBAY','PEMS03','PEMS04','PEMS08','METRLA')
# Absolute multipliers relative to graph5e-4/temporal1e-4/gate5e-5.
SPECS={
 'METRLA':[(12,2.,.1,1e-4),(8,2.,.1,1e-3),(12,2.,.1,1e-3)],
 'PEMSBAY':[(16,2.,.05,1e-4),(16,2.,.1,1e-5)],
 'PEMS03':[(20,.5,.2,1e-4),(20,.5,.1,1e-3)],
 'PEMS04':[(10,.5,.05,1e-5),(14,.5,.05,1e-4)],
 'PEMS08':[(6,.5,.05,1e-5),(6,.5,.2,1e-5),(6,.5,.05,1e-3)],
}


def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']


def tuple_of(r):
    q=params(r)
    return tuple(q[k] for k in ('capacity','lr_multiplier','dropout','weight_decay'))


def candidate(base,nested,spec,path):
    a,b=copy.deepcopy(base),copy.deepcopy(nested)
    cap,lr,drop,wd=spec
    b['family']['model']['spa_patchsize']=cap
    b['family']['model']['crossformer_dropout']=drop
    a['training']['weight_decay']=wd
    a['training']['lr']=5e-5*lr
    for k,v in zip(('graph_expert_lr','patch_expert_lr','gate_lr'),(5e-4,1e-4,5e-5)):
        a['family']['recipe'][k]=v*lr
    a['family']['experts']['patch']['config_path']=path
    return a,b


def freeze():
    previous.check();assert not PLAN.exists()
    DIR.mkdir(parents=True,exist_ok=True)
    old=read(ROOT/'paper/hparams_completion_20260928/results.json')['rows']
    records=read(PRIOR/'results_and_provenance.json')['records']
    files=[PRIOR/'results_and_provenance.json',PRIOR/'final_tables.xlsx',PRIOR/'paper_hparams_registry.json',
           ROOT/'paper/hparams_completion_20260928/results.json',ROOT/'paper'/NAME/'PLAN.md']
    entries={};anchors={};seen={(r['dataset'],tuple_of(r)) for r in old.values()}
    for ds in DATASETS:
        r=records[ds]['ALFNet'];anchors[ds]=dict(config=r['config'],checkpoint=r['checkpoint'],table=r['table'])
        base=yaml.safe_load((ROOT/r['config']).read_text())
        nested_path=ROOT/base['family']['experts']['patch']['config_path'];nested=yaml.safe_load(nested_path.read_text())
        reference=expanded(r)
        files.extend([ROOT/r['config'],ROOT/r['checkpoint'],nested_path,ROOT/base['family']['experts']['graph']['config_path']])
        files.extend(ROOT/base['dataset'][k] for k in ('data_path','unknown_nodes_path'))
        files.append(ROOT/f'configs/paper/v4_data_manifests/{ds}_run_manifest.json')
        for i,spec in enumerate(SPECS[ds]):
            assert (ds,spec) not in seen,(ds,spec)
            seen.add((ds,spec));key=f'{i*5+DATASETS.index(ds):02d}_{ds}_combo{i+1}'
            pp=DIR/(key+'_patch.yaml');cfg=DIR/(key+'.yaml')
            a,b=candidate(base,nested,spec,str(pp.relative_to(ROOT)));a['runtime']['model_variant']=NAME+'_'+key
            for f,x in ((pp,b),(cfg,a)):
                assert not f.exists();f.write_text(yaml.safe_dump(x,sort_keys=False))
            row=dict(path=str(cfg.relative_to(ROOT)),sha256=sha(cfg),dataset=ds,model='ALFNet',seed=32,
                     axis='combination',value=list(spec),capacity=spec[0],lr_multiplier=spec[1],dropout=spec[2],weight_decay=spec[3],
                     lrs=[a['family']['recipe'][k] for k in ('graph_expert_lr','patch_expert_lr','gate_lr')])
            assert tuple_of(dict(config=row['path']))==spec
            new=expanded(dict(config=row['path']))
            diff={k for k in new.keys()|reference.keys() if new.get(k)!=reference.get(k)}
            assert diff<=ALLOWED,(key,diff-ALLOWED)
            entries[key]=row;files.extend([cfg,pp])
    for r in old.values():files.extend([ROOT/r['config'],ROOT/r['report'],ROOT/r['checkpoint']])
    for folder in ('model','layers','lib','experiments','utils','data_provider','scripts'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.append(ROOT/'args.py')
    p=dict(kind=NAME,status='authorized_validation_only',configs=entries,reused=old,anchors=anchors,
           max_concurrent=5,test_authorized=False,seed=32,
           authority='2026-09-30 user authorizes final 2-3 combinations/dataset, max5 training; update validation tables if improved',
           selection='checkpoint: full validation33 LifeAvg MAE; compare wins and all LifeAvg metrics; no hard regression thresholds',
           final_selection='automatic replacement only for unambiguous nonworse wins+LifeAvg triple; tradeoffs remain review candidates, not rejected',
           files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])


def check():
    p=read(PLAN)
    assert digest(p)==p['plan_sha256'] and len(p['configs'])==12 and len(p['reused'])==55
    assert p['max_concurrent']==5 and not p['test_authorized']
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    seen={(r['dataset'],tuple_of(r)) for r in p['reused'].values()}
    for row in p['configs'].values():
        key=(row['dataset'],tuple_of(dict(config=row['path'])))
        assert key not in seen,key
        seen.add(key)
    return p


def context():
    stack=ExitStack()
    for name,value in [('check',check),('output',output),('NAME',NAME),('BASE_TABLE',PRIOR/'results_and_provenance.json')]:
        stack.enter_context(patch.object(previous,name,value))
    return stack


def preflight():
    with context():previous.preflight()


def worker(key):
    with context():previous.worker(key)


def collect():
    with context():previous.collect()
    from scripts.collect_hparam_combinations import publish
    publish()


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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['freeze','check','preflight','worker','queue','collect']);parser.add_argument('--key')
    args=parser.parse_args()
    if args.mode=='worker':worker(args.key)
    elif args.mode=='check':print(check()['plan_sha256'])
    else:globals()[args.mode]()
