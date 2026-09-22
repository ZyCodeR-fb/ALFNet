"""Authorized fixed METRLA ablation evaluation; all validation gates before test."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_pemsbay_patch16 import read,write,sha,digest,agree,GROUPS,METRICS
from experiments.alfnet_ablation_adapter import KEYS
PLAN=ROOT/'configs/paper/metrla_ablation_test_20260921.json'
SOURCE=ROOT/'paper/test_seed32_hparam_patch16_20260918/results_and_provenance.json'
KEY=None


def freeze():
    from scripts.run_metrla_final_ablations import check as training_check
    tp=training_check();assert not PLAN.exists()
    r=read(ROOT/'paper/ablations_metrla_20260920/results.json');entries={}
    files=[SOURCE,ROOT/'paper/ablations_metrla_20260920/results.json',ROOT/tp['reference']['checkpoint']]
    for key in KEYS:
        report_path=Path(r['provenance'][key]['report_path']);rep=read(report_path);run=report_path.parent;cfg=(ROOT/rep['config']).resolve()
        entries[key]={'candidate':key,'checkpoint':str((run/'best.pt').relative_to(ROOT)),'checkpoint_sha256':sha(run/'best.pt'),'config':str(cfg.relative_to(ROOT)),'config_sha256':sha(cfg),'validation_table':str((run/'development_validation_table.json').relative_to(ROOT)),'training_plan_sha256':tp['plan_sha256'],'selected_epoch':rep['best_epoch']}
        files.extend([run/'best.pt',report_path,run/'development_validation_table.json',cfg])
    files.extend(ROOT/f for f in tp['files'])
    files.append(Path(__file__).resolve())
    p={'kind':'metrla_fixed_nine_ablation_test','test_authorized':True,'authorization':'2026-09-21 user explicitly authorizes test the completed nine ablations and update paper/experiments_draft_cn.md. Full reused from existing same-checkpoint test.','candidates':entries,'max_concurrent':5,'no_model_selection':True,'reference_checkpoint':tp['reference']['checkpoint'],'reference_checkpoint_sha256':sha(ROOT/tp['reference']['checkpoint']),'files':{str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))}}
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])


def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and p['test_authorized'] and set(p['candidates'])==set(KEYS)
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p


def output(p):return ROOT/'artifacts/evaluation/metrla_ablation_test_20260921'/p['plan_sha256']
def candidate_plan(key):
    p=check();return dict(p,**p['candidates'][key])
def candidate_output(p):return output(p)/p['candidate']

def session_and_checkpoint(plan):
    import torch
    from experiments.alfnet_ablation_adapter import build_study_session as build_v4_session
    checkpoint = torch.load(ROOT / plan['checkpoint'], map_location='cpu', weights_only=False)
    session = build_v4_session(ROOT / plan['config'], device='cuda:0', run_seed=32, num_workers=0)
    assert checkpoint['dataset'] == session.dataset == 'METRLA'
    assert checkpoint['model'] == session.model_name == 'ALFNet'
    assert checkpoint['seed'] == session.run_seed == 32
    assert checkpoint['config_sha256'] == plan['config_sha256']
    assert checkpoint['plan_sha256'] == plan['training_plan_sha256']
    assert checkpoint['epoch'] == plan['selected_epoch']
    assert checkpoint['seed_contract'] == session.seed_contract
    assert checkpoint['scaler'] == session.scaler
    assert checkpoint['static_asset_hash'] == session.train_context.static_asset_hash()
    assert checkpoint['train_exposure_hash'] == session.train_context.exposure_hash()
    assert checkpoint['validation_exposure_hash'] == session.validation_context.exposure_hash()
    session.model.load_state_dict(checkpoint['model_state'], strict=True)
    session.model.eval()
    return session, checkpoint


def validation():
    import torch
    from experiments.v4_validation_collector import collect_validation_table, cell_count
    from lib.v4_seeds import seed_process
    plan = candidate_plan(KEY)
    root = candidate_output(plan)
    path = root / 'validation.json'
    if path.exists():
        old = read(path)
        assert old['plan_sha256'] == plan['plan_sha256'] and old['passed']
        print('ALREADY COMPLETE validation')
        return
    torch.set_num_threads(4)
    started = time.time()
    session, checkpoint = session_and_checkpoint(plan)
    seed_process(session.seed_contract['derived']['training_runtime'])
    table = collect_validation_table(session, epoch=int(checkpoint['epoch']))
    delta = agree(table, read(ROOT / plan['validation_table']))
    assert cell_count(table) == 33
    write(path, dict(plan_sha256=plan['plan_sha256'], checkpoint_sha256=plan['checkpoint_sha256'],
                     passed=True, split='validation', best_epoch=checkpoint['epoch'], max_abs_difference=delta,
                     table=table, seconds=time.time()-started,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))


def test():
    import torch
    from data_provider.v4_lifecycle_loader import V4LifecycleDataset
    from experiments.v4_runner import to_device
    from experiments.v4_validation_collector import ValidationCellCollector, cell_count
    from lib.lcbt_v4 import build_v4_teacher_forcing_mask
    from lib.runtime_context import build_runtime_context
    from lib.v4_seeds import seed_process
    from torch.utils.data import DataLoader
    from utils.model_output import unpack_model_output
    plan = candidate_plan(KEY)
    root = candidate_output(plan)
    assert read(root / 'VALIDATION_COMPLETE.json')['plan_sha256'] == plan['plan_sha256']
    assert read(root / 'TEST_OPENED.json')['plan_sha256'] == plan['plan_sha256']
    path = root / 'test.json'
    if path.exists():
        old = read(path)
        assert old['plan_sha256'] == plan['plan_sha256'] and old['passed']
        print('ALREADY COMPLETE test')
        return
    torch.set_num_threads(4)
    started = time.time()
    session, checkpoint = session_and_checkpoint(plan)
    seed_process(session.seed_contract['derived']['training_runtime'])
    context = build_runtime_context('METRLA', 'test', seed=32)
    assert context.static_asset_hash() == checkpoint['static_asset_hash']
    scaler = dict(checkpoint['scaler'])
    dataset = V4LifecycleDataset(context, ROOT / session.args.data_path, input_len=session.args.input_len,
                                 pred_len=session.args.pred_len, slide_step=session.args.slide_step, freq=session.args.freq,
                                 seed=session.seed_contract['derived']['train_schedule'], epoch=0, scaler=scaler,
                                 validation_seed=session.args.lcbt_validation_seed)
    loader = DataLoader(dataset, batch_size=session.train_loader.batch_size, shuffle=False, num_workers=0)
    collector = ValidationCellCollector(context, scaler, allow_phase='test')
    with torch.no_grad():
        for index, batch in enumerate(loader):
            batch = to_device(batch, session.device)
            forced = build_v4_teacher_forcing_mask(batch, session.seed_contract['derived']['teacher_forcing'],
                                                    int(checkpoint['epoch']), False)
            assert not forced.any()
            prediction, _ = unpack_model_output(session.model.forward_v4(batch, context, epoch=int(checkpoint['epoch']),
                                                                          teacher_forcing_mask=None))
            assert torch.isfinite(prediction).all()
            collector.update(batch, prediction)
            if index % 100 == 0:
                print('TEST_BATCH', index, len(loader), flush=True)
    table = collector.table()
    table.pop('development_validation_only', None)
    table.update(split='test', note='Fixed nine-variant ablation test; validation selected checkpoints; no architecture selection from these results.')
    assert cell_count(table) == 33
    previous = read(ROOT / 'paper/test_seed32_hparam_patch16_20260918/results_and_provenance.json')['records']['METRLA']['ALFNet']
    assert context.exposure_hash() == previous['test_exposure_hash']
    assert scaler == previous['scaler']
    for group, rows in GROUPS.items():
        for condition in rows:
            for key, value in previous['table'][group][condition].items():
                if key.endswith('_count'):
                    assert table[group][condition][key] == value
    write(path, dict(plan_sha256=plan['plan_sha256'], checkpoint_sha256=plan['checkpoint_sha256'], passed=True,
                     dataset='METRLA', model='ALFNet', seed=32, split='test', best_epoch=checkpoint['epoch'],
                     scaler=scaler, teacher_forcing_positions=0, test_exposure_hash=context.exposure_hash(),
                     test_samples=len(dataset), table=table, seconds=time.time()-started,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))


def collect():
    p=check();root=output(p);old=read(SOURCE)['records']['METRLA']['ALFNet']
    assert old['checkpoint_sha256']==p['reference_checkpoint_sha256'],'Full is not same checkpoint'
    records={'Full':old}
    for k in KEYS:
        record=read(root/k/'test.json');assert record['passed'] and record['plan_sha256']==p['plan_sha256'];records[k]=record
    out=ROOT/'paper/ablations_metrla_test_20260921';out.mkdir(parents=True,exist_ok=True)
    write(out/'results.json',{'plan_sha256':p['plan_sha256'],'split':'test','records':records,'full_reused_from':str(SOURCE.relative_to(ROOT)),'final_model_unchanged':True})
    cells=[];lines=['# METR-LA fixed ablations, test, seed32','', 'Full reuses the same checkpoint and exact test record as the main table. All nine variants reported; no model selection.','']
    for section,conds in GROUPS.items():
        columns=[(c,m) for c in conds for m in METRICS];lines+=['## '+section,'','| Variant | '+' | '.join(c+'/'+m for c,m in columns)+' |','|---|'+'---:|'*len(columns)]
        for key,record in records.items():
            vals=[]
            for c,m in columns:
                v=record['table'][section][c][m];rank=sorted(set(x['table'][section][c][m] for x in records.values())).index(v)+1
                cell=f'{v:.6f}';vals.append('**'+cell+'**' if rank==1 else '<u>'+cell+'</u>' if rank==2 else cell);cells.append([key,section,c,m,v,rank,100*(v/old['table'][section][c][m]-1)])
            lines.append('| '+key+' | '+' | '.join(vals)+' |')
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    with (out/'cells.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['variant','section','condition','metric','value','rank','relative_change_percent']);w.writerows(cells)
    write(root/'COMPLETE.json',{'plan_sha256':p['plan_sha256'],'tests':9,'full_reused':True,'cells':len(cells)})


def queue():
    p=check();root=output(p);root.mkdir(parents=True,exist_ok=True);lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'queue_status.json').exists(),'inspect existing queue before continuing'
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    for phase in ('validation','test'):
        if phase=='test':
            for k in KEYS:
                assert read(root/k/'validation.json')['passed']
                write(root/k/'VALIDATION_COMPLETE.json',{'plan_sha256':p['plan_sha256']})
                write(root/k/'TEST_OPENED.json',{'plan_sha256':p['plan_sha256'],'authorization':p['authorization'],'opened_at':time.time()})
            write(root/'VALIDATION_COMPLETE.json',{'plan_sha256':p['plan_sha256'],'passed':True,'count':9})
        jobs=[{'key':k,'status':'queued'} for k in KEYS];active={}
        while any(j['status'] in ('queued','running') for j in jobs):
            for k,(proc,log) in list(active.items()):
                if proc.poll() is not None:
                    j=next(j for j in jobs if j['key']==k);j.update(status='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode);log.close();del active[k]
            if any(j['status']=='failed' for j in jobs):
                for j in jobs:
                    if j['status']=='queued':j['status']='blocked'
            pending=next((j for j in jobs if j['status']=='queued'),None)
            if pending and len(active)<5:
                free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())/1024
                if free>20:
                    k=pending['key'];log=(root/(k+'_'+phase+'.log')).open('x');proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),phase,'--key',k],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);active[k]=proc,log;pending.update(status='running',pid=proc.pid)
            write(root/'queue_status.json',{'plan_sha256':p['plan_sha256'],'phase':phase,'jobs':jobs,'updated':time.time()});time.sleep(10)
        assert all(j['status']=='complete' for j in jobs),'failure; no automatic retry'
    collect()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('freeze','check','validation','test','queue','collect'));ap.add_argument('--key',choices=KEYS);a=ap.parse_args();KEY=a.key
    if a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
