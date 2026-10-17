"""Complete test companions for the existing fixed-anchor OFAT figure."""
import sys,copy,argparse
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import read,write,sha,digest
from scripts import evaluate_flow_lr_cross_20261004 as base
from experiments.flow_support_transfer import adapter
NAME='figure_hparams_test_20261017';OUT=ROOT/'paper'/NAME;PLAN=ROOT/'configs/paper'/NAME/'plan.json'
def output(p):return ROOT/'artifacts/evaluation'/NAME/p['plan_sha256']
def freeze():
    import pandas as pd,torch,yaml
    assert not PLAN.exists()
    source=ROOT/'paper/final_supplements_20261008/03_hyperparameters_validation.csv';df=pd.read_csv(source);df=df[(df.use=='正文单变量')&df.dataset.isin(['PEMS04','PEMS08'])];keys=set(df.run)
    rows={}
    for f in [ROOT/'paper/flow_ofat_20261006/results.json',ROOT/'paper/flow_support_transfer_20261003/results.json',ROOT/'paper/flow_lr_cross_20261004/results.json']:
        rows.update(read(f)['rows'])
    reuse={}
    for f in (ROOT/'artifacts/evaluation').rglob('*.json'):
        if f.stat().st_size>200000:continue
        try:r=read(f)
        except Exception:continue
        if isinstance(r,dict) and r.get('passed') and r.get('split')=='test' and 'checkpoint_sha256' in r and 'table' in r:reuse[r['checkpoint_sha256']]=f
    files=[source,Path(__file__)];entries={}
    for k in sorted(keys):
        r=copy.deepcopy(rows[k]);cp=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
        assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256'] and sha(ROOT/r['config'])==cp['config_sha256'] and cp['seed']==32
        report=read(ROOT/r['report']);assert report['development_validation_table']==r['table'] and report['best_epoch']==cp['epoch']
        r.update(model='ALFNet',epoch=cp['epoch'],training_config_sha256=cp['config_sha256'],training_plan_sha256=cp['plan_sha256'])
        if r['checkpoint_sha256'] in reuse:
            f=reuse[r['checkpoint_sha256']];r['reuse_test']=str(f.relative_to(ROOT));files.append(f)
        entries[k]=r;files.extend(ROOT/r[f] for f in ('checkpoint','report','config'))
        c=yaml.safe_load((ROOT/r['config']).read_text());files.extend(ROOT/c['family']['experts'][e]['config_path'] for e in ('graph','patch'));files.extend(ROOT/c['dataset'][f] for f in ('data_path','unknown_nodes_path'))
    for folder in ('model','layers','lib','experiments','data_provider','utils'):files.extend((ROOT/folder).rglob('*.py'))
    files.extend(ROOT/f for f in ('scripts/evaluate_flow_lr_cross_20261004.py','scripts/test_all_final_20261001.py','scripts/train_v4.py','scripts/eval_release.py','args.py'))
    p=dict(kind=NAME,test_authorized=True,status='authorized_test',authorization='2026-10-17 user requests validation and test companions for experiment figures.',entries=entries,max_concurrent=1,scope='Same flow OFAT keys and anchors; keep every result; speed tests reused from existing completed tables; no new training or configuration selection.',files={str(f.relative_to(ROOT)):sha(f) for f in set(files)})
    p['plan_sha256']=digest(p);write(PLAN,p);print('entries',len(entries),'new',sum('reuse_test' not in r for r in entries.values()),p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256']
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def worker(k):
    with adapter(),patch.object(base,'check',check),patch.object(base,'output',output):base.worker(k)
def run():
    import subprocess
    p=check();root=output(p);root.mkdir(parents=True,exist_ok=True)
    for k in p['entries']:
        if (root/(k+'.json')).exists():continue
        print('START',k,flush=True)
        with (root/(k+'.log')).open('x') as log:subprocess.run([sys.executable,'-u',__file__,'worker','--key',k],stdout=log,stderr=subprocess.STDOUT,check=True)
    write(OUT/'results.json',dict(rows={k:read(root/(k+'.json')) for k in p['entries']},plan_sha256=p['plan_sha256']));write(root/'COMPLETE.json',dict(passed=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run','worker']);p.add_argument('--key');a=p.parse_args()
    worker(a.key) if a.mode=='worker' else globals()[a.mode]()
