"""Frozen 20-checkpoint release: full validation agreement, then one test pass."""
import argparse,fcntl,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PLAN=ROOT/'configs/paper/release_test_seed32.json'
TOLERANCE=1e-5
RELATIVE_TOLERANCE=1e-5
ARCHIVE='archive/research_history_20260917.tar.gz'
STFIT_SOURCE='workspaces/final_completion_frozen_459749c0482f937d/'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256']
    for name,value in p['files'].items():assert sha(ROOT/name)==value,name
    return p
def output(p):return ROOT/'artifacts/evaluation/seed32'/p['plan_sha256']


def freeze():
    assert not PLAN.exists(),'release test plan already frozen'
    release=read(ROOT/'artifacts/final_seed32_manifest.json')
    assert len(release['runs'])==20 and release['test_authorized']
    files=[ROOT/'args.py',ROOT/'artifacts/final_seed32_manifest.json',ROOT/ARCHIVE]
    for folder in ('model','layers','lib','experiments','utils','data_provider','scripts'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.extend((ROOT/'configs/paper/final_seed32').glob('*.yaml'))
    for ds,name in (('METRLA','METR-LA'),('PEMSBAY','PEMS-BAY')):
        files.extend([ROOT/f'configs/paper/v4_data_manifests/{ds}_run_manifest.json',
            ROOT/f'data_v4/{name}/raw_data.pkl',ROOT/f'data_v4/{name}/unknown_nodes.npy'])
    for row in release['runs']:
        files.extend([ROOT/row['checkpoint'],ROOT/row['validation_table']])
    p=dict(kind='final_locked_seed32_test',status='authorized',runs=sorted(release['runs'],key=lambda r:r['model']!='STFiT'),
        test_authorized=True,authorization=release['authorization'],max_concurrent=4,
        validation_tolerance=TOLERANCE,validation_relative_tolerance=RELATIVE_TOLERANCE,
        selector='frozen best checkpoints; no test selection',
        stochastic_validation='STFiT: same fixed training_runtime seed, archived-source/current-source paired full validation; historical best RNG unavailable',
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])


def assert_session(row,cp,s):
    assert s.run_seed==32 and cp['seed']==32 and cp['dataset']==row['dataset'] and cp['model']==row['model']
    assert cp['config_sha256']==row['original_config_sha256']
    assert cp['plan_sha256']==row['original_plan_sha256']
    assert cp['model_variant']==s.args.model_variant==row['original_model_variant']
    assert cp['seed_contract']==s.seed_contract==row['seed_contract']
    assert cp['scaler']==s.scaler==row['identity']['scaler']
    assert s.train_context.static_asset_hash()==cp['static_asset_hash']
    assert s.train_context.exposure_hash()==cp['train_exposure_hash']
    assert s.validation_context.exposure_hash()==cp['validation_exposure_hash']
    for key,value in row['identity'].items():assert cp[key]==value,key
    s.model.load_state_dict(cp['model_state'],strict=True)


def validation_agrees(actual,expected,tolerance=TOLERANCE):
    differences=[];agreements=[]
    for group in ('main','phase'):
        assert actual[group].keys()==expected[group].keys()
        for label in expected[group]:
            for key,value in expected[group][label].items():
                if key in ('MAE','RMSE','MAPE'):
                    assert math.isfinite(actual[group][label][key]),(group,label,key)
                    delta=abs(actual[group][label][key]-value)
                    differences.append(delta)
                    agreements.append(delta<=tolerance+RELATIVE_TOLERANCE*abs(value))
                elif key.endswith('_count'):
                    assert actual[group][label][key]==value,(group,label,key)
    maximum=max(differences);return all(agreements),maximum


def worker(key,phase):
    import torch
    from experiments.v4_runner import build_v4_session,to_device
    from experiments.v4_validation_collector import collect_validation_table,ValidationCellCollector,cell_count
    from data_provider.v4_lifecycle_loader import V4LifecycleDataset
    from lib.runtime_context import build_runtime_context
    from lib.lcbt_v4 import build_v4_teacher_forcing_mask
    from torch.utils.data import DataLoader
    from utils.model_output import unpack_model_output
    from lib.v4_seeds import seed_process
    torch.set_num_threads(4);p=check();root=output(p)
    row=next(r for r in p['runs'] if r['dataset']+'_'+r['model']==key)
    path=root/phase/(key+'.json')
    if path.exists():
        old=read(path);assert old['plan_sha256']==p['plan_sha256'] and old['checkpoint_sha256']==row['checkpoint_sha256']
        assert old['passed'];print('ALREADY COMPLETE',key,phase);return
    started=time.time()
    cp=torch.load(ROOT/row['checkpoint'],map_location='cpu',weights_only=False)
    session=build_v4_session(ROOT/row['config'],device='cuda:0',run_seed=32,num_workers=0)
    assert_session(row,cp,session)
    epoch=int(cp['epoch']);session.model.eval()
    evaluation_seed=session.seed_contract['derived']['training_runtime']
    seed_process(evaluation_seed)
    info=dict(plan_sha256=p['plan_sha256'],checkpoint_sha256=row['checkpoint_sha256'],
        dataset=row['dataset'],model=row['model'],seed=32,split=phase,best_epoch=epoch,
        scaler=dict(cp['scaler']),scaler_source='checkpoint; no test refit',teacher_forcing_positions=0,
        evaluation_runtime_seed=evaluation_seed)
    if phase=='validation':
        if row['model']!='STFiT':
            table=collect_validation_table(session,epoch=epoch)
            passed,delta=validation_agrees(table,read(ROOT/row['validation_table']))
        else:
            # Source Gumbel sampling remains stochastic in eval. Best checkpoints
            # lack the historical runtime RNG; check cleanup parity against source.
            import tarfile,types
            current=session.model
            dependencies=list((ROOT/'model/comparison_model/stgcn_n').rglob('*.py'))
            dependencies += [ROOT/'model/comparison_model/family_utils.py',ROOT/'lib/v4_batch.py']
            with tarfile.open(ROOT/ARCHIVE) as archive:
                for dependency in dependencies:
                    assert archive.extractfile(STFIT_SOURCE+str(dependency.relative_to(ROOT))).read()==dependency.read_bytes()
                source=archive.extractfile(STFIT_SOURCE+'model/comparison_model/STFiT/STFiT.py').read()
            module=types.ModuleType('release_archived_stfit')
            exec(compile(source,STFIT_SOURCE+'STFiT.py','exec'),module.__dict__)
            reference=module.Model.from_runtime_context(session.args,session.train_context).to(session.device)
            reference.load_state_dict(cp['model_state'],strict=True)
            generator=session.validation_loader.generator
            generator_state=generator.get_state() if generator is not None else None
            seed_process(evaluation_seed)
            table=collect_validation_table(session,epoch=epoch)
            session.model=reference
            if generator is not None:generator.set_state(generator_state)
            seed_process(evaluation_seed)
            reference_table=collect_validation_table(session,epoch=epoch)
            session.model=current
            historical_passed,delta=validation_agrees(table,read(ROOT/row['validation_table']))
            passed,paired_delta=validation_agrees(table,reference_table)
            info.update(validation_gate='archived_source_fixed_rng_pair',historical_agreement=historical_passed,
                historical_rng_available=False,paired_max_abs_difference=paired_delta,
                reference_source_sha256=hashlib.sha256(source).hexdigest(),reference_table=reference_table)
        info.update(passed=passed,max_abs_difference=delta,table=table)
    else:
        assert read(root/'VALIDATION_COMPLETE.json')['plan_sha256']==p['plan_sha256']
        assert read(root/'TEST_OPENED.json')['plan_sha256']==p['plan_sha256']
        # The first construction of a test context happens only beyond both gates.
        context=build_runtime_context(row['dataset'],'test',seed=32)
        assert context.static_asset_hash()==cp['static_asset_hash']
        scaler=dict(cp['scaler'])
        dataset=V4LifecycleDataset(context,ROOT/session.args.data_path,
            input_len=session.args.input_len,pred_len=session.args.pred_len,
            slide_step=session.args.slide_step,freq=session.args.freq,
            seed=session.seed_contract['derived']['train_schedule'],epoch=0,
            scaler=scaler,validation_seed=session.args.lcbt_validation_seed)
        assert dataset.scaler is scaler
        loader=DataLoader(dataset,batch_size=session.train_loader.batch_size,shuffle=False,num_workers=0)
        collector=ValidationCellCollector(context,scaler,allow_phase='test')
        with torch.no_grad():
            for index,batch in enumerate(loader):
                batch=to_device(batch,session.device)
                forced=build_v4_teacher_forcing_mask(batch,session.seed_contract['derived']['teacher_forcing'],epoch,False)
                assert not forced.any()
                pred,_=unpack_model_output(session.model.forward_v4(batch,context,epoch=epoch,teacher_forcing_mask=None))
                assert torch.isfinite(pred).all(),'nonfinite predictions must not silently drop from metrics'
                collector.update(batch,pred)
                if index%100==0:print('TEST_BATCH',key,index,len(loader),flush=True)
        table=collector.table();table.pop('development_validation_only',None)
        table.update(split='test',note='Locked seed32 checkpoint, authorized confirmation test; no model selection.')
        info.update(passed=True,table=table,test_exposure_hash=context.exposure_hash(),test_samples=len(dataset))
    assert cell_count(info['table'])==33
    info.update(seconds=time.time()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    write(path,info)
    assert info['passed'],f'validation mismatch {info.get("max_abs_difference")}'
    print('COMPLETE',key,phase,info['seconds'],flush=True)


def queue():
    p=check();root=output(p);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'queue_status.json').exists(),'existing queue: inspect, never silently duplicate'
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/mogernn-mpl')
    for phase in ('validation','test'):
        if phase=='test':
            records=[read(root/'validation'/(r['dataset']+'_'+r['model']+'.json')) for r in p['runs']]
            assert all(r['passed'] and r['plan_sha256']==p['plan_sha256'] for r in records)
            check();write(root/'VALIDATION_COMPLETE.json',dict(plan_sha256=p['plan_sha256'],count=20,passed=True))
            assert not (root/'TEST_OPENED.json').exists(),'test already opened; explicit identical-identity resume required'
            write(root/'TEST_OPENED.json',dict(plan_sha256=p['plan_sha256'],opened_at=time.time(),authorization=p['authorization']))
        jobs=[dict(key=r['dataset']+'_'+r['model'],status='queued') for r in p['runs']]
        active={};last=0
        while any(j['status'] in ('queued','running') for j in jobs):
            for key,(proc,log) in list(active.items()):
                code=proc.poll()
                if code is not None:
                    job=next(j for j in jobs if j['key']==key)
                    job.update(status='complete' if code==0 else 'failed',exit_code=code,finished=time.time())
                    log.close();del active[key]
            if any(j['status']=='failed' for j in jobs):
                for j in jobs:
                    if j['status']=='queued':j['status']='blocked'
            pending=next((j for j in jobs if j['status']=='queued'),None)
            if pending and len(active)<p['max_concurrent'] and time.monotonic()-last>15:
                free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())/1024
                if free>24:
                    logpath=root/f"{phase}_{pending['key']}.log";log=logpath.open('x')
                    command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--key',pending['key'],'--phase',phase]
                    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    pending.update(status='running',pid=proc.pid,started=time.time(),log=str(logpath.relative_to(ROOT)))
                    active[pending['key']]=(proc,log);last=time.monotonic()
            write(root/'queue_status.json',dict(plan_sha256=p['plan_sha256'],phase=phase,jobs=jobs,updated=time.time()))
            time.sleep(3)
        write(root/(phase+'_status.json'),dict(plan_sha256=p['plan_sha256'],jobs=jobs))
        assert all(j['status']=='complete' for j in jobs),f'{phase} failed; no automatic retry'
    write(root/'TEST_COMPLETE.json',dict(plan_sha256=p['plan_sha256'],completed_at=time.time(),count=20))
    subprocess.run([sys.executable,'scripts/collect_release.py'],cwd=ROOT,check=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('freeze','check','worker','queue'))
    ap.add_argument('--phase',choices=('validation','test'));ap.add_argument('--key');a=ap.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='check':print(check()['plan_sha256'])
    elif a.mode=='worker':worker(a.key,a.phase)
    else:queue()
