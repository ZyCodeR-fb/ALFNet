"""Four datasets, six paired ablations, gated behind the current OFAT queue."""
import argparse,copy,csv,time,sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import read,write,sha,digest,identity
from scripts import run_hparam_completion as training
from experiments.flow_support_transfer import adapter as support
from experiments.flow_unified import adapter as unified
from experiments.alfnet_ablation_adapter import apply_intervention
NAME='final_ablations_20261006';DIR=ROOT/'configs/paper'/NAME;PLAN=DIR/'plan.json';OUT=ROOT/'paper'/NAME
LABELS={'no_age':'w/o State Age','static_router':'w/o Adaptive Routing','local_context':'w/o Spatial Context','no_imputer':'w/o ALFRecon','no_temporal_film':'w/o ALFTemporal–FiLM','no_refiner':'w/o Graph Refinement'}
DATASETS=('METRLA','PEMSBAY','PEMS04','PEMS08')
def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']
def canonical(c):
    c=copy.deepcopy(c);c['runtime'].pop('model_variant',None)
    for e in ('graph','patch'):
        c['family']['experts'][e]['config_path']=yaml.safe_load((ROOT/c['family']['experts'][e]['config_path']).read_text())
    return c
def freeze():
    from scripts.run_flow_ofat_20261006 import check as predecessor_check
    prior=predecessor_check();assert not PLAN.exists();DIR.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'paper/final_scheme_20261006/selected_configuration_manifest.json'
    selected=read(manifest)['entries'];sources=[ROOT/'paper'/s/'results.json' for s in ('hparams_combinations_20260930','flow_lr_cross_20261004','pems03_support_20261004')]
    rows={k:r for f in sources for k,r in read(f)['rows'].items()}
    keys={d:selected[d]['key'] for d in selected}
    anchors={d:rows[keys[d]] for d in DATASETS};configs={};files=[manifest,OUT/'PLAN.md',*sources];reuse_audit=[]
    for ds,r in anchors.items():
        report=read(ROOT/r['report']);assert report['development_validation_table']==r['table']
        assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        assert sha(ROOT/r['config'])==report['config_sha256']
        files.extend(ROOT/r[k] for k in ('config','checkpoint','report'))
        c=yaml.safe_load((ROOT/r['config']).read_text());original=canonical(c)
        files.extend(ROOT/c['family']['experts'][e]['config_path'] for e in ('graph','patch'))
        files.extend(ROOT/c['dataset'][f] for f in ('data_path','unknown_nodes_path'))
        files.append(ROOT/f'configs/paper/v4_data_manifests/{ds}_run_manifest.json')
        for i,kind in enumerate(LABELS):
            key=f'{i*4+DATASETS.index(ds):02d}_{ds}_{kind}';cfg=copy.deepcopy(c)
            cfg['runtime']['model_variant']=NAME+'_'+key;cfg['family']['recipe']['study_ablation']=kind
            for e in ('graph','patch'):
                nested=yaml.safe_load((ROOT/cfg['family']['experts'][e]['config_path']).read_text())
                if e=='graph' and kind=='static_router':nested['family']['recipe']['moge_router_mode']='static'
                if e=='patch' and kind=='local_context':nested['family']['model']['local_context_only']=True
                path=DIR/(key+'_'+e+'.yaml');path.write_text(yaml.safe_dump(nested,sort_keys=False));files.append(path)
                cfg['family']['experts'][e]['config_path']=str(path.relative_to(ROOT))
            path=DIR/(key+'.yaml');path.write_text(yaml.safe_dump(cfg,sort_keys=False));files.append(path)
            expected=copy.deepcopy(original);expected['family']['recipe']['study_ablation']=kind
            if kind=='static_router':expected['family']['experts']['graph']['config_path']['family']['recipe']['moge_router_mode']='static'
            if kind=='local_context':expected['family']['experts']['patch']['config_path']['family']['model']['local_context_only']=True
            assert canonical(cfg)==expected,key
            configs[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),dataset=ds,model='ALFNet',seed=32,ablation=kind,label=LABELS[kind])
    # Discover historical study-ablation configs; only exact semantic matches can be reused.
    for path in (ROOT/'configs/paper').rglob('*.yaml'):
        if path.parent==DIR:continue
        c=yaml.safe_load(path.read_text())
        if not isinstance(c,dict) or not c.get('family',{}).get('recipe',{}).get('study_ablation'):continue
        if 'experts' not in c['family']:continue
        old=canonical(c);matches=[k for k,r in configs.items() if canonical(yaml.safe_load((ROOT/r['path']).read_text()))==old]
        reuse_audit.append(dict(config=str(path.relative_to(ROOT)),matches=matches))
    assert not any(r['matches'] for r in reuse_audit),'exact existing ablation found; inspect completion before scheduling'
    for folder in ('model','layers','lib','experiments','utils','data_provider','scripts'):files.extend((ROOT/folder).rglob('*.py'))
    files.append(ROOT/'args.py')
    p=dict(kind=NAME,status='authorized_validation_only',anchors=anchors,configs=configs,prior_ablation_audit=reuse_audit,
        predecessor=dict(plan='configs/paper/flow_ofat_20261006/plan.json',sha256=prior['plan_sha256']),
        max_concurrent=5,test_authorized=False,seed=32,protected_external_pid_note=2546461,
        authority='2026-10-06 user corrected scope to four datasets, excluding PEMS03; six named ablations after OFAT finishes, max5',
        pems03_scope='excluded by explicit user correction; superseded five-dataset plan never launched',
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and len(p['configs'])==24 and not p['test_authorized']
    assert all(r['dataset'] in DATASETS for r in p['configs'].values())
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def context():
    from experiments import v4_runner
    from scripts import train_v4
    stack=ExitStack();stack.enter_context(support());stack.enter_context(unified())
    build=v4_runner.build_v4_session;contract=train_v4.model_method_contract
    def wrapped(*args,**kwargs):return apply_intervention(build(*args,**kwargs))
    def describe(s):
        r=contract(s);r['paired_ablation']=getattr(s.model,'study_ablation','reference');return r
    for obj,name,value in [(v4_runner,'build_v4_session',wrapped),(train_v4,'build_v4_session',wrapped),(train_v4,'model_method_contract',describe)]:stack.enter_context(patch.object(obj,name,value))
    return stack
def preflight():
    import torch
    from experiments import v4_runner
    from scripts.train_v4 import build_optimizer
    torch.set_num_threads(4);p=check();root=output(p);reports=[]
    with context():
        for ds,a in p['anchors'].items():
            s=v4_runner.build_v4_session(ROOT/a['config'],device='cpu',run_seed=32,num_workers=0)
            ids=identity(s);initial={n:t.detach().clone() for n,t in s.model.named_parameters()};lrs=[g['lr'] for g in build_optimizer(s).param_groups]
            cp=torch.load(ROOT/a['checkpoint'],map_location='cpu',weights_only=False)
            assert all(cp[k]==v for k,v in ids.items());s.model.load_state_dict(cp['model_state'],strict=True);del s,cp
            for key,row in p['configs'].items():
                if row['dataset']!=ds:continue
                print('PREFLIGHT_START',key,flush=True);s=v4_runner.build_v4_session(ROOT/row['path'],device='cpu',run_seed=32,num_workers=0)
                assert identity(s)==ids and [g['lr'] for g in build_optimizer(s).param_groups]==lrs
                assert (s.args.max_epoch,s.args.patience,s.args.early_stopping_start_epoch)==(100,10,40)
                for n,t in s.model.named_parameters():
                    if n in initial:assert torch.equal(t,initial[n]),(key,n)
                    else:assert row['ablation']=='static_router' and n.endswith('w_static') and torch.count_nonzero(t)==0
                count=sum(t.numel() for t in s.model.parameters());del s
                torch.cuda.reset_peak_memory_stats();start=time.time()
                training.safe_train(ROOT/row['path'],None,p,a['table'],smoke=True)
                item=dict(key=key,passed=True,parameter_count=count,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,seconds=time.time()-start)
                write(root/'preflight'/(key+'.json'),item);reports.append(item);torch.cuda.empty_cache();print('PREFLIGHT_OK',key,flush=True)
    write(root/'PREFLIGHT_COMPLETE.json',dict(passed=True,plan_sha256=p['plan_sha256'],reports=reports))
def worker(key):
    import torch
    torch.set_num_threads(4);p=check();root=output(p);row=p['configs'][key]
    assert read(root/'PREFLIGHT_COMPLETE.json')['plan_sha256']==p['plan_sha256']
    with context():r=training.safe_train(ROOT/row['path'],root/'runs'/key,p,p['anchors'][row['dataset']]['table'])
    write(root/(key+'_complete.json'),dict(passed=True,plan_sha256=p['plan_sha256'],report_path=str(Path(r['run_directory'])/'run_report.json')))
def collect():
    from openpyxl import Workbook
    p=check();root=output(p);rows={ds+'_Full':dict(r,label='Full') for ds,r in p['anchors'].items()}
    for key,row in p['configs'].items():
        done=read(root/(key+'_complete.json'));rp=Path(done['report_path']);r=read(rp)
        assert r['plan_sha256']==p['plan_sha256'] and r['model_method_contract']['paired_ablation']==row['ablation']
        assert min((e['validation']['checkpoint_life_avg_mae'],e['epoch']) for e in r['history'])[1]==r['best_epoch']
        rows[key]=dict(dataset=row['dataset'],label=row['label'],config=row['path'],table=r['development_validation_table'],report=str(rp.relative_to(ROOT)),checkpoint=str(rp.with_name('best.pt').relative_to(ROOT)),checkpoint_sha256=sha(rp.with_name('best.pt')))
    wb=Workbook();wb.active.title='说明';wb.active.append(['seed32 validation; four final paired anchors; PEMS03 excluded; no test'])
    cells=[]
    for ds in DATASETS:
        for group in ('main','phase'):
            ref=p['anchors'][ds]['table'][group];cols=[(c,m) for c in ref for m in ('MAE','RMSE','MAPE')]
            ws=wb.create_sheet(ds+'_'+group);ws.append(['Variant']+[c+'/'+m for c,m in cols])
            for key,r in rows.items():
                if r['dataset']!=ds:continue
                for c in ref:
                    for f,v in ref[c].items():
                        if f.endswith('_count'):assert r['table'][group][c][f]==v
                ws.append([r['label']]+[r['table'][group][c][m] for c,m in cols])
                for c,m in cols:cells.append([ds,key,r['label'],group,c,m,r['table'][group][c][m]])
    assert len(cells)==924;wb.save(OUT/'validation_results.xlsx')
    with (OUT/'validation_cells.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['dataset','run','label','section','condition','metric','value']);w.writerows(cells)
    write(OUT/'results.json',dict(plan_sha256=p['plan_sha256'],split='validation',rows=rows,test_accessed=False))
    write(root/'COMPLETE.json',dict(passed=True,runs=24,reused_full=4,verified_cells=len(cells),test_accessed=False))
def wait_previous():
    p=check();old=read(ROOT/p['predecessor']['plan']);assert old['plan_sha256']==p['predecessor']['sha256']
    root=ROOT/'artifacts/experiments/flow_ofat_20261006'/old['plan_sha256']
    while True:
        if (root/'ATTENTION_REQUIRED.json').exists():raise RuntimeError('predecessor failed; do not start ablations')
        if (root/'COMPLETE.json').exists():
            assert read(root/'COMPLETE.json')['passed']
            s=read(root/'queue_status.json');assert s['plan_sha256']==old['plan_sha256'] and all(j['status']=='complete' and j['exit_code']==0 for j in s['jobs'])
            print('PREDECESSOR_COMPLETE',flush=True);return
        write(output(p)/'waiting.json',dict(status='waiting_for_ofat',predecessor=old['plan_sha256'],updated=time.time()))
        time.sleep(30)
def queue():
    wait_previous();p=check()
    if not (output(p)/'PREFLIGHT_COMPLETE.json').exists():preflight()
    from scripts import recover_baseline_extension as runner
    with ExitStack() as stack:
        for n,v in [('check',check),('output',output),('collect',collect),('__file__',__file__),('isolate_failed_families',lambda jobs,configs:None)]:stack.enter_context(patch.object(runner,n,v))
        runner.queue()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','check','preflight','worker','collect','queue']);p.add_argument('--key');a=p.parse_args()
    if a.mode=='worker':worker(a.key)
    elif a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
