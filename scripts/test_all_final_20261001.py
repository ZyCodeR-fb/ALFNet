"""Authorized full seed32 test inventory:67 ALFNet configurations+50 baselines."""
import argparse,copy,csv,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import read,write,sha,digest
from scripts.collect_hparams_final import GROUPS,METRICS,params
NAME='all_final_test_20261001';PLAN=ROOT/'configs/paper'/NAME/'plan.json'
OUT=ROOT/'paper'/NAME
VALID_SELECTED=dict(METRLA='METRLA_lr_double',PEMSBAY='05_PEMSBAY_combo2',PEMS03='16_PEMS03_lr_half',PEMS04='PEMS04_lr_half',PEMS08='03_PEMS08_combo1')

def output(p):return ROOT/'artifacts/evaluation'/NAME/p['plan_sha256']

def freeze():
    import torch
    from scripts.run_hparam_combinations import check as oldcheck,output as oldoutput
    p0=oldcheck();assert all(j['status']=='complete' and j['exit_code']==0 for j in read(oldoutput(p0)/'queue_status.json')['jobs'])
    assert not PLAN.exists()
    source=ROOT/'paper/hparams_combinations_20260930/results.json';data=read(source)
    baseline=ROOT/'paper/hparams_combinations_20260930/results_and_provenance.json';records=read(baseline)['records']
    entries={};files=[source,baseline,ROOT/'paper'/NAME/'PLAN.md'];audit=[]
    for key,r in data['rows'].items():entries[key]=dict(r,model='ALFNet')
    for ds,ms in records.items():
        for name,r in ms.items():
            if name=='ALFNet':continue
            r=copy.deepcopy(r)
            if 'source' in r:
                s=ROOT/r['source'];r.update(config=str((s/('config.yaml' if (s/'config.yaml').exists() else 'config.original.yaml')).relative_to(ROOT)),checkpoint=str((s/'best.pt').relative_to(ROOT)))
            entries[ds+'_'+name]=dict(r,dataset=ds,model=name)
    assert len(entries)==117
    # Prior successful test results: reuse only exact checkpoint hashes with current model/data code parity.
    reuse={}
    for f in (ROOT/'artifacts/evaluation').rglob('*.json'):
        if f.stat().st_size>200000:continue
        try:r=read(f)
        except Exception:continue
        if isinstance(r,dict) and r.get('split')=='test' and r.get('passed') and 'table' in r and 'checkpoint_sha256' in r:
            reuse.setdefault(r['checkpoint_sha256'],(f,r))
    for key,r in entries.items():
        if not (ROOT/r['checkpoint']).exists():
            directory=ROOT/f"artifacts/models/main/{r['dataset']}/{r['model']}/seed32"
            provenance=read(directory/'provenance.json')
            assert provenance['original_checkpoint']==r['checkpoint']
            assert sha(directory/'best.pt')==provenance['checkpoint_sha256']
            r.update(checkpoint=str((directory/'best.pt').relative_to(ROOT)),config=str((directory/'config.yaml').relative_to(ROOT)))
        cp=ROOT/r['checkpoint'];cfg=ROOT/r['config'];state=torch.load(cp,map_location='cpu',weights_only=False)
        assert state['seed']==32 and state['dataset']==r['dataset'] and state['model']==r['model']
        assert all(torch.isfinite(v).all() for v in state['model_state'].values() if v.is_floating_point())
        r.update(checkpoint_sha256=sha(cp),config_sha256=sha(cfg),training_config_sha256=state['config_sha256'],training_plan_sha256=state['plan_sha256'],epoch=int(state['epoch']),seed_contract=state['seed_contract'])
        if r['config_sha256']!=state['config_sha256']:
            assert sha(cp.with_name('config.original.yaml'))==state['config_sha256']
            a=yaml.safe_load(cfg.read_text());b=yaml.safe_load(cp.with_name('config.original.yaml').read_text())
            if r['model']=='ALFNet':
                for e in ('graph','patch'):
                    assert yaml.safe_load((ROOT/a['family']['experts'][e]['config_path']).read_text())==yaml.safe_load(cp.with_name(f'expert.{e}.original.yaml').read_text())
                    a['family']['experts'][e]['config_path']=b['family']['experts'][e]['config_path']
            assert a==b,(key,'relocated config semantics')
        if r.get('report'):
            report=read(ROOT/r['report']);assert report['development_validation_table']==r['table']
            assert min((e['validation']['checkpoint_life_avg_mae'],e['epoch']) for e in report['history'])[1]==report['best_epoch']
            assert all(e[t]['skipped']==0 for e in report['history'] for t in ('train','validation'))
        config=yaml.safe_load(cfg.read_text());files.extend([cp,cfg])
        for e in config.get('family',{}).get('experts',{}).values():
            if isinstance(e,dict) and e.get('config_path'):files.append(ROOT/e['config_path'])
        for fld in ('data_path','unknown_nodes_path'):files.append(ROOT/config['dataset'][fld])
        files.append(ROOT/f"configs/paper/v4_data_manifests/{r['dataset']}_run_manifest.json")
        # All current source checks below are stronger than checkpoint identity alone.
        prior=reuse.get(r['checkpoint_sha256'])
        if prior:
            f,t=prior;plans=list((ROOT/'configs/paper').glob('**/*.json'));matched=None
            for pf in plans:
                try:op=read(pf)
                except Exception:continue
                if isinstance(op,dict) and op.get('plan_sha256')==t.get('plan_sha256'):matched=op;break
            if matched:
                parity=True
                for n,h in matched.get('files',{}).items():
                    if n.startswith(('model/','layers/','lib/','data_provider/','experiments/')):
                        if n=='experiments/v4_runner.py':
                            import hashlib
                            txt=(ROOT/n).read_text().replace('    "Merlin": "model.comparison_model.Merlin",\n','')
                            parity &= sha(ROOT/n)==h or hashlib.sha256(txt.encode()).hexdigest()==h
                        else:parity &= (ROOT/n).exists() and sha(ROOT/n)==h
                if parity and matched.get('files'):
                    r['reuse_test']=str(f.relative_to(ROOT));files.append(f)
        audit.append(dict(key=key,checkpoint_verified=True,reused_test=r.get('reuse_test')))
        del state
    for folder in ('model','layers','lib','data_provider','experiments','utils','scripts'):files.extend((ROOT/folder).rglob('*.py'))
    files.append(ROOT/'args.py')
    p=dict(kind=NAME,test_authorized=True,authorization='2026-10-01 user explicitly authorizes all baselines and all hyperparameter tests, including exploratory test-based choice',
           entries=entries,validation_selected=VALID_SELECTED,max_concurrent=5,files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))},
           exploratory_selection='Max first cells among candidates with all3LifeAvg better than each strongest baseline; tie mean relative LifeAvg then MAE. If none meets triple lead, use LifeAvg Pareto candidates, then same ordering. This is descriptive test-based selection, not independent confirmation.')
    p['plan_sha256']=digest(p);write(PLAN,p);write(OUT/'freeze_audit.json',audit);print(p['plan_sha256']);print('reuse',sum('reuse_test' in r for r in entries.values()))

def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and p['test_authorized'] and len(p['entries'])==117
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p

def worker(key):
    import torch
    from experiments.v4_runner import build_v4_session,to_device
    from experiments.v4_validation_collector import collect_validation_table,ValidationCellCollector,cell_count
    from scripts.eval_release import validation_agrees
    from lib.v4_seeds import seed_process
    from lib.runtime_context import build_runtime_context
    from data_provider.v4_lifecycle_loader import V4LifecycleDataset
    from torch.utils.data import DataLoader
    from utils.model_output import unpack_model_output
    torch.set_num_threads(4);p=check();r=p['entries'][key];root=output(p);dest=root/(key+'.json')
    assert not dest.exists()
    if r.get('reuse_test'):
        t=read(ROOT/r['reuse_test']);assert t['checkpoint_sha256']==r['checkpoint_sha256'] and t['passed']
        write(dest,dict(t,evaluation_plan_sha256=p['plan_sha256'],reused_from=r['reuse_test']));return
    cp=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
    s=build_v4_session(ROOT/r['config'],device='cuda:0',run_seed=32,num_workers=0)
    assert cp['config_sha256']==r['training_config_sha256'] and cp['plan_sha256']==r['training_plan_sha256']
    assert cp['scaler']==s.scaler and cp['seed_contract']==s.seed_contract
    assert cp['static_asset_hash']==s.train_context.static_asset_hash()
    assert cp['train_exposure_hash']==s.train_context.exposure_hash() and cp['validation_exposure_hash']==s.validation_context.exposure_hash()
    s.model.load_state_dict(cp['model_state'],strict=True);s.model.eval();seed=s.seed_contract['derived']['training_runtime'];seed_process(seed)
    vt=collect_validation_table(s,epoch=r['epoch']);ok,delta=validation_agrees(vt,r['table'])
    info=dict(passed=ok,max_abs_difference=delta,table=vt)
    if s.model_name=='STFiT':
        # Source eval samples Gumbel; historical checkpoint has no eval RNG. Verify fixed-seed replay, disclose historical difference.
        seed_process(seed);again=collect_validation_table(s,epoch=r['epoch']);replay,rd=validation_agrees(vt,again)
        info.update(passed=replay,historical_agreement=ok,gate='fixed_seed_replay_stochastic_source',replay_delta=rd)
    write(root/'validation'/(key+'.json'),info);assert info['passed'],(key,delta)
    write(root/'opened'/(key+'.json'),dict(plan_sha256=p['plan_sha256'],authorization=p['authorization'],time=time.time()))
    seed_process(seed);ctx=build_runtime_context(r['dataset'],'test',seed=32);assert ctx.static_asset_hash()==cp['static_asset_hash']
    dataset=V4LifecycleDataset(ctx,ROOT/s.args.data_path,input_len=s.args.input_len,pred_len=s.args.pred_len,slide_step=s.args.slide_step,freq=s.args.freq,seed=s.seed_contract['derived']['train_schedule'],epoch=0,scaler=dict(cp['scaler']),validation_seed=s.args.lcbt_validation_seed)
    loader=DataLoader(dataset,batch_size=s.train_loader.batch_size,shuffle=False,num_workers=0)
    collector=ValidationCellCollector(ctx,cp['scaler'],allow_phase='test')
    with torch.no_grad():
        for i,b in enumerate(loader):
            b=to_device(b,s.device);pred,_=unpack_model_output(s.model.forward_v4(b,ctx,epoch=r['epoch'],teacher_forcing_mask=None));assert torch.isfinite(pred).all();collector.update(b,pred)
            if i%100==0:print('TEST_BATCH',key,i,len(loader),flush=True)
    table=collector.table();assert cell_count(table)==33;table.pop('development_validation_only',None);table.update(split='test',note='Authorized exhaustive exploration; not independent model selection confirmation')
    write(dest,dict(passed=True,plan_sha256=p['plan_sha256'],checkpoint_sha256=r['checkpoint_sha256'],dataset=r['dataset'],model=r['model'],seed=32,split='test',table=table,scaler=cp['scaler'],test_exposure_hash=ctx.exposure_hash(),test_samples=len(dataset),evaluation_runtime_seed=seed,teacher_forcing_positions=0))

def queue():
    p=check();root=output(p);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'queue_status.json').exists()
    jobs=[dict(key=k,status='queued') for k in sorted(p['entries'],key=lambda k:('reuse_test' not in p['entries'][k],p['entries'][k]['model']=='ALFNet',k))];active={};last=0
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    while any(j['status'] in ('queued','running') for j in jobs):
        for k,(proc,log) in list(active.items()):
            code=proc.poll()
            if code is not None:
                j=next(j for j in jobs if j['key']==k);j.update(status='complete' if code==0 and (root/(k+'.json')).exists() else 'failed',exit_code=code);log.close();del active[k]
        j=next((j for j in jobs if j['status']=='queued'),None)
        if j and len(active)<5 and time.monotonic()-last>20:
            free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())
            if free>20000:
                k=j['key'];log=(root/(k+'.log')).open('x');proc=subprocess.Popen([sys.executable,'-u',__file__,'worker','--key',k],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);active[k]=(proc,log);j.update(status='running',pid=proc.pid);last=time.monotonic()
        write(root/'queue_status.json',dict(jobs=jobs,plan_sha256=p['plan_sha256']));time.sleep(10)
    if all(j['status']=='complete' for j in jobs):
        from scripts.collect_all_final_test import collect
        collect()
    else:write(root/'ATTENTION_REQUIRED.json',dict(failed=[j for j in jobs if j['status']=='failed']))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['freeze','check','queue','worker']);a.add_argument('--key');a=a.parse_args()
    if a.mode=='worker':worker(a.key)
    elif a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
