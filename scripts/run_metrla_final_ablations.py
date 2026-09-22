"""Nine paired METRLA ablations of the selected final configuration; no test."""
import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_pemsbay_patch16 import read,write,sha,digest
from experiments.alfnet_ablation_adapter import KEYS,build_study_session,install_training_adapter
CONFIGS=ROOT/'configs/paper/metrla_final_ablations_20260920'
PLAN=CONFIGS/'plan.json'


def selected_reference():
    from scripts.evaluate_metrla_two_candidates import check,output
    p=check();root=output(p);assert read(root/'COMPLETE.json')['tests']==2
    selection=read(root/'selection.json');key=selection['selected']
    if key=='original':
        base=ROOT/'artifacts/models/main/METRLA/ALFNet/seed32'
        cfg=base/'config.yaml';checkpoint=base/'best.pt';table=base/'development_validation_table.json'
    else:
        row=p['candidates'][key];cfg=ROOT/row['config'];checkpoint=ROOT/row['checkpoint'];table=ROOT/row['validation_table']
    return {'key':key,'config':str(cfg.relative_to(ROOT)),'checkpoint':str(checkpoint.relative_to(ROOT)),
            'validation_table':str(table.relative_to(ROOT)),'selection':str((root/'selection.json').relative_to(ROOT))}


def configs_for(ref,key):
    outer=yaml.safe_load((ROOT/ref['config']).read_text())
    graph=yaml.safe_load((ROOT/outer['family']['experts']['graph']['config_path']).read_text())
    patch=yaml.safe_load((ROOT/outer['family']['experts']['patch']['config_path']).read_text())
    outer['runtime']['model_variant']='final_ablation_'+key
    outer['family']['recipe']['study_ablation']=key
    if key=='mask_only':outer['family']['recipe']['observation_state_mode']='mask_only'
    if key=='static_router':graph['family']['recipe']['moge_router_mode']='static'
    if key=='local_context':patch['family']['model']['local_context_only']=True
    if key=='plain_mae':outer['training']['loss']='mae'
    if key=='no_role_balance':outer['family']['recipe']['forecast_role_balanced_beta']=0.
    for branch in ('graph','patch'):
        outer['family']['experts'][branch]['config_path']=str((CONFIGS/f'{key}_{branch}.yaml').relative_to(ROOT))
    return outer,graph,patch


def freeze():
    assert not PLAN.exists();ref=selected_reference();CONFIGS.mkdir(parents=True,exist_ok=True)
    entries={}
    for k in KEYS:
        for name,data in zip((k,k+'_graph',k+'_patch'),configs_for(ref,k)):
            path=CONFIGS/(name+'.yaml');assert not path.exists();path.write_text(yaml.safe_dump(data,sort_keys=False))
        path=CONFIGS/(k+'.yaml');entries[k]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'dataset':'METRLA','seed':32,'ablation':k}
    files=[ROOT/f for f in (ref['config'],ref['checkpoint'],ref['validation_table'],ref['selection'],'args.py','paper/ablations_metrla_20260920/PLAN.md','configs/paper/v4_data_manifests/METRLA_run_manifest.json','data_v4/METR-LA/raw_data.pkl','data_v4/METR-LA/unknown_nodes.npy')]
    for folder in ('model','layers','lib','experiments','utils','data_provider','scripts'):
        files.extend((ROOT/folder).rglob('*.py'))
    for folder in ('configs/paper/final_seed32','configs/paper/metrla_hparams_20260920',str(CONFIGS.relative_to(ROOT))):
        files.extend((ROOT/folder).glob('*.yaml'))
    p={'kind':'metrla_final_ablation_v1','status':'authorized_validation_only','authorization':'User2026-09-20 requests expanded ablations after final two-candidate test choice, METRLA display, max5 parallel training.','reference':ref,'configs':entries,'max_concurrent':5,'test_authorized':False,'max_epochs':100,'early_stopping_start_epoch':40,'patience':10,'files':{str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))},'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])


def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and not p['test_authorized'] and set(p['configs'])==set(KEYS)
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    for k in KEYS:
        for name,data in zip((k,k+'_graph',k+'_patch'),configs_for(p['reference'],k)):
            assert yaml.safe_load((CONFIGS/(name+'.yaml')).read_text())==data
    return p


def output(p):return ROOT/'artifacts/experiments/metrla_final_ablations_20260920'/p['plan_sha256']


def preflight():
    import torch
    from experiments.v4_runner import build_v4_session
    from scripts.train_v4 import train,build_optimizer
    from scripts.run_metrla_hparams import identity
    torch.set_num_threads(4);p=check();root=output(p);assert not (root/'PREFLIGHT_COMPLETE.json').exists()
    base=build_v4_session(ROOT/p['reference']['config'],device='cpu',run_seed=32,num_workers=0)
    ids=identity(base);initial={n:t.detach().clone() for n,t in base.model.named_parameters()};lrs=[g['lr'] for g in build_optimizer(base).param_groups]
    checkpoint=torch.load(ROOT/p['reference']['checkpoint'],map_location='cpu',weights_only=False)
    for k,v in ids.items():assert checkpoint[k]==v,k
    base.model.load_state_dict(checkpoint['model_state'],strict=True);del base,checkpoint
    install_training_adapter();reports=[]
    for k,row in p['configs'].items():
        print('PREFLIGHT_START',k,flush=True);s=build_study_session(ROOT/row['path'],device='cpu',run_seed=32,num_workers=0)
        assert identity(s)==ids
        retained=0;added=[]
        for name,t in s.model.named_parameters():
            if name in initial:assert torch.equal(t,initial[name]),(k,name);retained+=t.numel()
            else:added.append(name);assert k=='static_router' and name.endswith('w_static') and torch.count_nonzero(t)==0
        assert [g['lr'] for g in build_optimizer(s).param_groups]==lrs
        count=sum(t.numel() for t in s.model.parameters());del s
        torch.cuda.reset_peak_memory_stats();started=time.time()
        report=train(ROOT/row['path'],output_root=None,device='cuda:0',seed=32,max_epochs=1,max_train_batches=2,max_validation_batches=2,batch_size=64,num_workers=0,save=False,plan=p,collect_development_table=False)
        assert all(report['history'][0][phase]['skipped']==0 for phase in ('train','validation'))
        item={'key':k,'passed':True,'parameter_count':count,'identical_retained_initial_parameters':retained,'new_parameter_names':added,'actual_optimizer_lrs':lrs,'exposure_scaler_seed_identical':True,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'seconds':time.time()-started}
        write(root/'preflight'/(k+'.json'),item);reports.append(item);print('PREFLIGHT_OK',k,flush=True);torch.cuda.empty_cache()
    write(root/'PREFLIGHT_COMPLETE.json',{'plan_sha256':p['plan_sha256'],'passed':True,'reports':reports})


def worker(k):
    import torch
    from scripts.train_v4 import train
    torch.set_num_threads(4);p=check();root=output(p);assert read(root/'PREFLIGHT_COMPLETE.json')['plan_sha256']==p['plan_sha256']
    install_training_adapter();started=time.time()
    r=train(ROOT/p['configs'][k]['path'],output_root=root/'runs'/k,device='cuda:0',seed=32,num_workers=0,plan=p,collect_development_table=True)
    assert all(e[phase]['skipped']==0 for e in r['history'] for phase in ('train','validation'))
    write(root/(k+'_complete.json'),{'plan_sha256':p['plan_sha256'],'passed':True,'report_path':str(Path(r['run_directory'])/'run_report.json'),'seconds':time.time()-started,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3})


def collect():
    p=check();root=output(p);out=ROOT/'paper/ablations_metrla_20260920';tables={'Full':read(ROOT/p['reference']['validation_table'])};meta={}
    for k in KEYS:
        done=read(root/(k+'_complete.json'));assert done['passed'] and done['plan_sha256']==p['plan_sha256'];r=read(done['report_path']);tables[k]=r['development_validation_table'];meta[k]={**done,'best_epoch':r['best_epoch'],'epochs_run':r['epochs_run'],'parameter_count':r['parameter_count']}
    cells=[]
    for k,t in tables.items():
        for s in ('main','phase'):
            for c,values in tables['Full'][s].items():
                for f,v in values.items():
                    if f.endswith('_count'):assert t[s][c][f]==v
                for m in ('MAE','RMSE','MAPE'):cells.append([k,s,c,m,t[s][c][m],100*(t[s][c][m]/values[m]-1)])
    write(out/'results.json',{'plan_sha256':p['plan_sha256'],'reference':p['reference'],'split':'validation','tables':tables,'provenance':meta,'no_test':True})
    with (out/'ablation_cells.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['variant','section','condition','metric','value','relative_change_percent']);w.writerows(cells)
    lines=['# METRLA final-configuration ablations (validation, seed32)','', '| Variant | MAE | RMSE | MAPE (%) | Parameters |','|---|---:|---:|---:|---:|']
    for k,t in tables.items():lines.append('| '+k+' | '+' | '.join(f"{t['main']['LifeAvg'][m]:.6f}" for m in ('MAE','RMSE','MAPE'))+f" | {meta.get(k,{}).get('parameter_count',1037529)} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n');write(root/'COMPLETE.json',{'plan_sha256':p['plan_sha256'],'runs':len(KEYS),'test_accessed':False})


def queue():
    p=check();root=output(p);root.mkdir(parents=True,exist_ok=True);lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'queue_status.json').exists(),'inspect existing queue; no duplicates'
    gate=read(root/'PREFLIGHT_COMPLETE.json');assert gate['passed'];required=max(12.,max(r['peak_reserved_gib'] for r in gate['reports'])*1.3+4)
    jobs=[{'key':k,'status':'queued'} for k in KEYS];active={};last=0
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    while any(j['status'] in ('queued','running') for j in jobs):
        for k,(proc,log) in list(active.items()):
            j=next(j for j in jobs if j['key']==k)
            if proc.poll() is not None:
                j.update(status='complete' if proc.returncode==0 and (root/(k+'_complete.json')).exists() else 'failed',exit_code=proc.returncode,finished=time.time());log.close();del active[k]
        if any(j['status']=='failed' for j in jobs):
            for j in jobs:
                if j['status']=='queued':j['status']='blocked'
        pending=next((j for j in jobs if j['status']=='queued'),None)
        if pending and len(active)<5 and time.monotonic()-last>45:
            free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())/1024
            if free>required:
                k=pending['key'];log=(root/(k+'.log')).open('x');proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--key',k],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);active[k]=proc,log;pending.update(status='running',pid=proc.pid,started=time.time());last=time.monotonic()
        write(root/'queue_status.json',{'plan_sha256':p['plan_sha256'],'jobs':jobs,'updated':time.time(),'required_free_gib':required});time.sleep(15)
    assert all(j['status']=='complete' for j in jobs),'failure; inspect, no auto retry'
    collect()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('freeze','check','preflight','worker','queue','collect'));ap.add_argument('--key',choices=KEYS);a=ap.parse_args()
    if a.mode=='worker':worker(a.key)
    elif a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
