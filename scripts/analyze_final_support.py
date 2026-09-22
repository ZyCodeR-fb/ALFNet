"""Fixed final checkpoints, paired validation interventions and error-blind cases."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_release import read, write, sha, digest, assert_session
from scripts.analyze_observation_support import (
    select_pairs, arms, intervene, summarize, write_csv, INPUT_FIELDS)

PLAN = ROOT / 'configs/paper/final_support_figures_20260922.json'
OUT = ROOT / 'artifacts/evaluation/final_support_figures_20260922'
MODELS = ('ALFNet', 'UIGNN', 'MoGERNN', 'STBP')


def freeze():
    assert not PLAN.exists()
    release = read(ROOT / 'artifacts/final_seed32_manifest.json')
    runs = {r['model']: r for r in release['runs']
            if r['dataset'] == 'METRLA' and r['model'] in MODELS}
    validation = ROOT / 'paper/validation_seed32_hparam_patch16_20260918/results_and_provenance.json'
    records = read(validation)['records']['METRLA']
    classes = [('UIGNN', 'GinAR'), ('MoGERNN', 'STFiT', 'SLPF'),
               ('STGCN', 'STID', 'PatchSTG', 'STBP')]
    selected = [min(c, key=lambda m: records[m]['table']['main']['LifeAvg']['MAE']) for c in classes]
    assert selected == list(MODELS[1:])
    files = [validation, ROOT/'artifacts/final_seed32_manifest.json', ROOT/'args.py']
    for folder in ('model', 'layers', 'lib', 'experiments', 'utils', 'data_provider'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.extend([Path(__file__), ROOT/'scripts/analyze_observation_support.py', ROOT/'scripts/eval_release.py'])
    files.extend((ROOT/'configs/paper/final_seed32').glob('*.yaml'))
    files.extend([ROOT/'data_v4/METR-LA/raw_data.pkl', ROOT/'data_v4/METR-LA/unknown_nodes.npy'])
    files.extend(ROOT/r['checkpoint'] for r in runs.values())
    p = dict(kind='final_support_figures_validation', split='validation', test_authorized=False,
             authorization='User 2026-09-22 requests updated final-model Figures3/4 and case study; fixed-weight mechanism inference only.',
             models=list(MODELS), runs=runs, representative_rule='minimum existing validation LifeAvg MAE within each related-work class',
             per_track=64, selection_seed=20260908, batch_size=16, max_concurrent=1,
             case_rule='first four selected held-out pairs in fixed random order, assigned full/join_3/fault_6/outage; no prediction-based selection',
             files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    p['plan_sha256'] = digest(p)
    write(PLAN,p)
    print(p['plan_sha256'])


def check():
    p=read(PLAN)
    assert digest(p)==p['plan_sha256'] and p['split']=='validation'
    for f,h in p['files'].items(): assert sha(ROOT/f)==h,f
    return p


def run(device):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from experiments.v4_runner import build_v4_session, _forward, to_device, build_model_teacher_mask
    from lib.v4_seeds import seed_process
    p=check(); out=OUT/p['plan_sha256']
    assert not out.exists(), 'Never overwrite an existing evaluation'
    out.mkdir(parents=True)
    write(out/'plan.json',p)
    torch.set_num_threads(4)
    pairs=None; reference=None; all_rows=[]; cases=[]
    for name in MODELS:
        r=p['runs'][name]
        s=build_v4_session(ROOT/r['config'], device=device, run_seed=32, batch_size=16, num_workers=0)
        cp=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
        assert_session(r,cp,s)
        s.model.eval(); seed_process(s.seed_contract['derived']['training_runtime'])
        ds=s.validation_loader.dataset
        assert ds.context.phase=='validation'
        if pairs is None:
            pairs=select_pairs(ds,p['per_track'],p['selection_seed'])
            reference=[ds[x['index']] for x in pairs]
            write(out/'pairs.json',pairs)
            case_specs=[dict(pair_id=i,condition=c) for i,c in enumerate(('full','join_3','fault_6','outage'))]
            write(out/'case_selection.json',case_specs)
        for x,ref in zip(pairs,reference):
            other=ds[x['index']]
            for k in INPUT_FIELDS+('target','target_available_mask'):
                assert torch.equal(ref[k],other[k]),(name,k)
        rows=[]
        for arm in arms():
            for start in range(0,len(pairs),p['batch_size']):
                part=pairs[start:start+p['batch_size']]
                items=[intervene(ds,x['index'],x['node'],arm['scenario'],arm['evidence'],arm['keep']) for x in part]
                batch=to_device(default_collate(items),device)
                teacher=build_model_teacher_mask(s,batch,cp['epoch'],False)
                assert not teacher.any()
                with torch.inference_mode():
                    pred=_forward(s,batch,s.validation_context,cp['epoch'],teacher)['pred'].cpu().numpy()
                assert np.isfinite(pred).all()
                for j,(x,item) in enumerate(zip(part,items)):
                    n=x['node']; std=ds.scaler['std']; mean=ds.scaler['mean']
                    actual=item['target'][:,n,0].numpy()*std+mean
                    forecast=pred[j,:,n,0]*std+mean; diff=forecast-actual
                    q=int(item['input_available_mask'][:,n].sum())
                    if arm['q']>=0: assert q==arm['q']
                    row=dict(model=name,experiment=arm['experiment'],condition=arm['condition'],
                        layout=arm['layout'],q=q,**x,
                        mask=''.join(str(int(v)) for v in item['input_available_mask'][:,n]),
                        MAE=float(np.abs(diff).mean()),MSE=float(np.square(diff).mean()),
                        RMSE=float(np.sqrt(np.square(diff).mean())),
                        MAPE=float((np.abs(diff)/np.maximum(np.abs(actual),1e-6)).mean()*100),
                        prediction=json.dumps(forecast.tolist()),target=json.dumps(actual.tolist()))
                    rows.append(row)
                    if any(t['pair_id']==x['pair_id'] and t['condition']==arm['condition'] for t in case_specs):
                        cases.append(dict(row,history=(item['base_visible_input'][:,n,0].numpy()*std+mean).tolist()))
            print(name,arm['condition'],'complete',flush=True)
        write_csv(out/f'{name}_paired.csv',rows); all_rows.extend(rows)
        del s
        if device.startswith('cuda'): torch.cuda.empty_cache()
    assert len(all_rows)==4*128*20 and len(cases)==16
    write_csv(out/'paired_predictions.csv',all_rows)
    write_csv(out/'metrics.csv',summarize(all_rows))
    write(out/'cases.json',cases)
    write(out/'COMPLETE.json',dict(plan_sha256=p['plan_sha256'],models=4,pairs=128,arms=20,
          rows=len(all_rows),cases=4,test_accessed=False,teacher_forced_positions=0,
          aggregation='equal windows and horizons; pooled RMSE; not LifeAvg'))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','check','run']);a.add_argument('--device',default='cuda:0')
    args=a.parse_args()
    if args.action=='freeze': freeze()
    elif args.action=='check': print(check()['plan_sha256'])
    else: run(args.device)
