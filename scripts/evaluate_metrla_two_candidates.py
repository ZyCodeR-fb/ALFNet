"""Bounded two-candidate METRLA test comparison; explicit exploratory selection."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_pemsbay_patch16 import read,write,sha,digest,agree,GROUPS,METRICS
PLAN=ROOT/'configs/paper/metrla_two_candidate_test_20260920.json'
KEYS=('lr_double','patch12')
KEY=None
SOURCE=ROOT/'paper/test_seed32_hparam_patch16_20260918/results_and_provenance.json'


def freeze():
    assert not PLAN.exists()
    hp=read(ROOT/'paper/hparams_metrla_20260920/results.json')
    candidates={};files=[SOURCE,ROOT/'paper/validation_seed32_hparam_patch16_20260918/results_and_provenance.json',ROOT/'paper/hparams_metrla_20260920/results.json',ROOT/'args.py',ROOT/'configs/paper/v4_data_manifests/METRLA_run_manifest.json',ROOT/'data_v4/METR-LA/raw_data.pkl',ROOT/'data_v4/METR-LA/unknown_nodes.npy']
    for k in KEYS:
        report_path=Path(hp['provenance'][k]['report_path']);r=read(report_path);run=report_path.parent;cfg=ROOT/r['config']
        candidates[k]={'candidate':k,'checkpoint':str((run/'best.pt').relative_to(ROOT)),'checkpoint_sha256':sha(run/'best.pt'),'config':str(cfg.relative_to(ROOT)),'config_sha256':sha(cfg),'validation_table':str((run/'development_validation_table.json').relative_to(ROOT)),'training_plan_sha256':r['plan_sha256'],'selected_epoch':r['best_epoch']}
        files.extend([cfg,run/'best.pt',run/'development_validation_table.json',report_path])
    for folder in ('model','layers','lib','experiments','utils','data_provider','scripts'):
        files.extend((ROOT/folder).rglob('*.py'))
    for folder in ('configs/paper/final_seed32','configs/paper/metrla_hparams_20260920'):
        files.extend((ROOT/folder).glob('*.yaml'))
    p={'kind':'metrla_two_candidate_test_exploratory','test_authorized':True,'post_test_exploratory':True,'authorization':'2026-09-20 user explicitly requests test lr_double and patch12 and replacement if superior test and best winning-cell count; supersedes earlier validation-only selection restriction for these two only.','candidates':candidates,'selection_rule':'All three test LifeAvg metrics <= original Focal and at least one strictly lower; first-place cell count across33 vs fixed nine baselines >= original. Among eligible choose most first places, then most seconds, then smallest LifeAvg MAE; tie keep original. Record failed original validation constraints; no claim of independent test confirmation.','files':{str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))}}
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])


def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and p['test_authorized'];assert set(p['candidates'])==set(KEYS)
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p


def output(p):return ROOT/'artifacts/evaluation/metrla_two_candidate_20260920'/p['plan_sha256']
def candidate_plan(key):
    p=check();return dict(p,**p['candidates'][key])
def candidate_output(p):return output(p)/p['candidate']

def session_and_checkpoint(plan):
    import torch
    from experiments.v4_runner import build_v4_session
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
    table.update(split='test', note='Explicitly authorized two-candidate test-based selection; exploratory test reuse, not independent confirmation.')
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


def worker(key):
    global KEY
    KEY=key;p=candidate_plan(key);root=candidate_output(p)
    validation()
    assert read(root/'validation.json')['passed']
    write(root/'VALIDATION_COMPLETE.json',{'plan_sha256':p['plan_sha256'],'passed':True})
    write(root/'TEST_OPENED.json',{'plan_sha256':p['plan_sha256'],'authorization':p['authorization'],'opened_at':time.time()})
    test()
    write(root/'COMPLETE.json',{'plan_sha256':p['plan_sha256'],'passed':True})


def select():
    p=check();records=read(SOURCE)['records']['METRLA'];base=records['ALFNet']['table'];tables={'original':base}
    for k in KEYS:tables[k]=read(output(p)/k/'test.json')['table']
    summaries={}
    for k,t in tables.items():
        places=[]
        for s,conds in GROUPS.items():
            for c in conds:
                for m in METRICS:
                    vals=[v['table'][s][c][m] for n,v in records.items() if n!='ALFNet']+[t[s][c][m]]
                    places.append(sorted(set(vals)).index(t[s][c][m])+1)
        la=t['main']['LifeAvg'];d={m:la[m]/base['main']['LifeAvg'][m]-1 for m in METRICS}
        summaries[k]={'first':places.count(1),'second':places.count(2),'LifeAvg':la,'relative_changes':d}
    eligible=[k for k in KEYS if all(v<=0 for v in summaries[k]['relative_changes'].values()) and any(v<0 for v in summaries[k]['relative_changes'].values()) and summaries[k]['first']>=summaries['original']['first']]
    winner=min(eligible,key=lambda k:(-summaries[k]['first'],-summaries[k]['second'],summaries[k]['LifeAvg']['MAE'])) if eligible else 'original'
    write(output(p)/'selection.json',{'plan_sha256':p['plan_sha256'],'rule':p['selection_rule'],'selected':winner,'summaries':summaries,'test_based_selection':True,'validation_constraints_waived_by_new_authorization':True})
    print(json.dumps({'selected':winner,'summaries':summaries},ensure_ascii=False),flush=True)


def queue():
    p=check();root=output(p);root.mkdir(parents=True,exist_ok=True)
    assert not (root/'queue_status.json').exists(),'inspect existing queue; no duplicate'
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    jobs=[]
    for k in KEYS:
        log=(root/(k+'.log')).open('x');proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--key',k],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((k,proc,log))
    while True:
        states=[{'key':k,'pid':proc.pid,'exit_code':proc.poll(),'status':'running' if proc.poll() is None else 'complete' if proc.returncode==0 else 'failed'} for k,proc,log in jobs]
        write(root/'queue_status.json',{'plan_sha256':p['plan_sha256'],'jobs':states,'updated':time.time()})
        if all(proc.poll() is not None for k,proc,log in jobs):break
        time.sleep(10)
    for k,proc,log in jobs:log.close();assert proc.returncode==0,k
    select();write(root/'COMPLETE.json',{'plan_sha256':p['plan_sha256'],'tests':2})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('freeze','check','worker','queue','select'));ap.add_argument('--key',choices=KEYS);a=ap.parse_args()
    if a.mode=='worker':worker(a.key)
    elif a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
