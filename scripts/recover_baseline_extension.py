"""Explicit recovery: preserve five completed Merlin runs; finish 24 flow baselines."""
import argparse,copy,fcntl,os,subprocess,sys,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_baseline_extension as base
from scripts.run_metrla_hparams import read,write,sha,digest

NAME='baseline_recovery_20260927'
PLAN=ROOT/'configs/paper'/NAME/'plan.json'
OLD=ROOT/'configs/paper/baseline_extension_20260927/plan.json'

def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']

def check():
    p=read(PLAN)
    assert digest(p)==p['plan_sha256'] and len(p['configs'])==24
    assert not p['test_authorized'] and p['max_concurrent']==5
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p

def freeze():
    assert not PLAN.exists()
    old=base.check();oldroot=base.output(old)
    assert len(read(oldroot/'PREFLIGHT_COMPLETE.json')['reports'])==29
    records=copy.deepcopy(read(ROOT/old['reuse']))
    sources=[OLD,ROOT/old['reuse'],oldroot/'PREFLIGHT_COMPLETE.json',oldroot/'queue_status.json',Path(__file__),
        ROOT/'paper/baseline_extension_20260927/RECOVERY.md',
        ROOT/'paper/baseline_extension_20260927/diagnosis_PEMS03_0.006.json',oldroot/'PEMS03_GinAR.log']
    for ds in base.DATASETS:
        done=read(oldroot/(ds+'_Merlin_complete.json'));rp=Path(done['report_path']);r=read(rp)
        assert done['passed'] and r['plan_sha256']==old['plan_sha256']
        assert min((e['validation']['checkpoint_life_avg_mae'],e['epoch']) for e in r['history'])[1]==r['best_epoch']
        assert all(e[x]['skipped']==0 for e in r['history'] for x in ('train','validation'))
        teacher=read(r['teacher_report'])
        assert teacher['epochs_run']<=100 and r['epochs_run']<=100
        assert all(e[x]['skipped']==0 for e in teacher['history'] for x in ('train','validation'))
        cp=rp.with_name('best.pt')
        records['records'][ds]['Merlin']=dict(table=r['development_validation_table'],report=str(rp.relative_to(ROOT)),
            report_sha256=sha(rp),checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=sha(cp),
            config=r['config'],config_sha256=r['config_sha256'],best_epoch=r['best_epoch'],source_plan_sha256=old['plan_sha256'])
        sources.extend([rp,cp,Path(r['teacher_report'])])
    PLAN.parent.mkdir(parents=True,exist_ok=True)
    reuse=PLAN.parent/'reused_31_records.json';write(reuse,records);sources.append(reuse)
    configs={k:copy.deepcopy(v) for k,v in old['configs'].items() if v['model']!='Merlin'}
    for key,row in configs.items():
        if row['model']=='GinAR':
            diag=ROOT/f'paper/baseline_extension_20260927/diagnosis_{row["dataset"]}_0.001.json'
            evidence=read(diag)
            assert evidence.get('epoch_complete') and evidence['validation']['skipped']==0 and 'failed_at' not in evidence
            assert evidence['lr']==.001 and not evidence['test_accessed']
            for group in ('main','phase'):
                for condition,values in records['records'][row['dataset']]['ALFNet']['table'][group].items():
                    for field,count in values.items():
                        if field.endswith('_count'):assert evidence['table'][group][condition][field]==count
            c=yaml.safe_load((ROOT/row['path']).read_text());c['training']['lr']=.001
            c['runtime']['model_variant']=NAME+'_'+key
            path=PLAN.parent/(key+'.yaml');path.write_text(yaml.safe_dump(c,sort_keys=False))
            row.update(path=str(path.relative_to(ROOT)),sha256=sha(path))
            sources.extend([path,diag,ROOT/'paper/baseline_extension_20260927/diagnose_ginar.py'])
    files=dict(old['files']);files.update({str(f.relative_to(ROOT)):sha(f) for f in sources})
    p=dict(kind=NAME,status='authorized_validation_only',configs=configs,max_concurrent=5,test_authorized=False,
        seed=32,reuse=str(reuse.relative_to(ROOT)),files=files,parent_plan_sha256=old['plan_sha256'],
        reason='PEMS03 GinAR lr.006 gradient norm overflow at batch4; flow GinAR lr.001 full-epoch finite checks required; architecture unchanged',
        failure_policy='isolate failed model family, continue unrelated models, preserve failure evidence; no automatic retry')
    p['plan_sha256']=digest(p);write(PLAN,p)
    gate=read(oldroot/'PREFLIGHT_COMPLETE.json')
    # The 21 unchanged configs/code inherit their exact prior GPU probes.
    # GinAR has stronger, new full-epoch train+full-validation evidence.
    gate.update(plan_sha256=p['plan_sha256'],inherited_from=old['plan_sha256'])
    gate['reports']=[dict(r,plan_sha256=p['plan_sha256'],evidence='new full-epoch diagnostic' if r['key'].endswith('_GinAR') else 'unchanged source/config prior preflight') for r in gate['reports'] if r['key'] in configs]
    write(output(p)/'PREFLIGHT_COMPLETE.json',gate)
    print(p['plan_sha256'])

def configure():
    base.check=check;base.output=output

def worker(key):
    configure();base.worker(key)

def collect():
    configure();base.collect()

def isolate_failed_families(jobs,configs):
    failed_models={configs[j['key']]['model'] for j in jobs if j['status']=='failed'}
    for j in jobs:
        if j['status']=='queued' and configs[j['key']]['model'] in failed_models:
            j.update(status='blocked_same_model',reason='inspect same-family failure before launching')

def queue():
    p=check();root=output(p);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'queue_status.json').exists()
    peaks={r['key']:r['peak_reserved_gib'] for r in read(root/'PREFLIGHT_COMPLETE.json')['reports']}
    # Prioritize independent model families, rather than one failure blocking all jobs.
    order=sorted(p['configs'],key=lambda k:(p['configs'][k]['model']=='GinAR',k))
    jobs=[dict(key=k,status='queued') for k in order];active={};last=0.
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/mogernn-mpl')
    while any(j['status'] in ('queued','running') for j in jobs):
        for key,(proc,log) in list(active.items()):
            code=proc.poll()
            if code is not None:
                j=next(j for j in jobs if j['key']==key)
                j.update(status='complete' if code==0 and (root/(key+'_complete.json')).exists() else 'failed',exit_code=code,finished=time.time())
                log.close();del active[key]
        isolate_failed_families(jobs,p['configs'])
        pending=next((j for j in jobs if j['status']=='queued'),None)
        if pending and len(active)<5 and time.monotonic()-last>30:
            free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())/1024
            if free>max(6.,peaks[pending['key']]*1.5+3.):
                key=pending['key'];log=(root/(key+'.log')).open('x')
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--key',key],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                pending.update(status='running',pid=proc.pid,started=time.time());active[key]=proc,log;last=time.monotonic()
        write(root/'queue_status.json',dict(plan_sha256=p['plan_sha256'],jobs=jobs,updated=time.time(),max_concurrent=5))
        time.sleep(15)
    if all(j['status']=='complete' for j in jobs):collect()
    else:
        write(root/'ATTENTION_REQUIRED.json',dict(failed=[j for j in jobs if j['status']!='complete'],no_test_access=True))
        raise RuntimeError('some jobs failed; unaffected jobs finished, preserve evidence for explicit repair')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['freeze','check','queue','worker','collect']);parser.add_argument('--key');args=parser.parse_args()
    if args.mode=='worker':worker(args.key)
    elif args.mode=='check':print(check()['plan_sha256'])
    else:globals()[args.mode]()
