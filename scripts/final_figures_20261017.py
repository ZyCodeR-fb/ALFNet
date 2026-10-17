"""Final-weight paired interventions on two explicitly authorized splits."""
import sys,json,argparse
from pathlib import Path
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.eval_release import read,write,sha,digest
OUT=ROOT/'paper/final_figures_20261017';PLAN=OUT/'plan.json'
DATASETS=['METRLA','PEMSBAY','PEMS04','PEMS08']
def freeze():
    import copy,yaml
    assert not PLAN.exists()
    prior=read(ROOT/'paper/advisor_experiments_20261012/plan.json')
    final=read(ROOT/'paper/pems08_wd_test_20261012/selected_configuration_manifest.json')['entries']
    entries=copy.deepcopy(prior['entries']);files=[Path(__file__),ROOT/'scripts/analyze_observation_support.py']
    for d in DATASETS:
        entries[d]['ALFNet']=copy.deepcopy(final[d])
        for m,r in entries[d].items():
            assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
            r['evaluation_config_sha256']=sha(ROOT/r['config'])
            files.extend(ROOT/r[k] for k in ('checkpoint','config'))
            c=yaml.safe_load((ROOT/r['config']).read_text())
            for e in c.get('family',{}).get('experts',{}).values():
                if isinstance(e,dict) and e.get('config_path'):files.append(ROOT/e['config_path'])
            files.extend(ROOT/c['dataset'][k] for k in ('data_path','unknown_nodes_path'))
    for folder in ['model','layers','lib','experiments','data_provider','utils']:files.extend((ROOT/folder).rglob('*.py'))
    p=dict(entries=entries,training=False,test_authorized=True,seed=32,per_dataset=64,selection_seed=20260908,known_nodes_only=True,authorization='2026-10-17 user authorizes final-weight validation and test figure evaluation.',selection=prior['selection'],cases=prior['cases'],representative=prior['representative'],display='Keep both complete splits. Main mechanism figures use validation consistently; test companion figures disclose robustness and all reversals. No per-figure outcome-driven split selection.',files={str(f.relative_to(ROOT)):sha(f) for f in files})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and p['test_authorized']
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def run(ds,device,split):
    import numpy as np,torch
    from torch.utils.data import default_collate
    from experiments import v4_runner as vr
    from experiments.flow_support_transfer import adapter as support
    from experiments.flow_unified import adapter as unified
    from scripts.train_v4 import model_method_contract
    from scripts.analyze_observation_support import select_pairs,arms,intervene,INPUT_FIELDS,write_csv
    from lib.v4_seeds import seed_process
    p=check()
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    torch.set_num_threads(4)
    dest=OUT/split/ds;dest.mkdir(parents=True,exist_ok=True)
    pairs=None;reference=None
    for model,r in p['entries'][ds].items():
        path=dest/(model+'.csv')
        if path.exists():continue
        assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        assert sha(ROOT/r['config'])==r['evaluation_config_sha256']
        with ExitStack() as stack:
            stack.enter_context(support());stack.enter_context(unified())
            s=vr.build_v4_session(ROOT/r['config'],device=device,run_seed=32,batch_size=16,num_workers=0)
            cp=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
            assert cp['seed']==32 and cp['dataset']==ds and cp['model']==model
            assert cp['scaler']==s.scaler
            assert cp['static_asset_hash']==s.train_context.static_asset_hash()
            assert cp['validation_exposure_hash']==s.validation_context.exposure_hash()
            if model=='ALFNet':
                from scripts import train_v4
                assert cp['model_method_contract']==train_v4.model_method_contract(s)
            s.model.load_state_dict(cp['model_state'],strict=True);s.model.eval()
            seed_process(s.seed_contract['derived']['training_runtime'])
            ctx=s.validation_context
            data=s.validation_loader.dataset
            if split=='test':
                from lib.runtime_context import build_runtime_context
                from data_provider.v4_lifecycle_loader import V4LifecycleDataset
                ctx=build_runtime_context(ds,'test',seed=32)
                data=V4LifecycleDataset(ctx,ROOT/s.args.data_path,input_len=s.args.input_len,pred_len=s.args.pred_len,slide_step=s.args.slide_step,freq=s.args.freq,seed=s.seed_contract['derived']['train_schedule'],epoch=0,scaler=dict(cp['scaler']),validation_seed=s.args.lcbt_validation_seed)
            assert data.context.phase==split
            if pairs is None:
                pairs=[x for x in select_pairs(data,64,p['selection_seed']) if x['track']=='fault']
                reference=[data[x['index']] for x in pairs];write(dest/'pairs.json',pairs)
            for x,ref in zip(pairs,reference):
                other=data[x['index']]
                for k in INPUT_FIELDS+('target','target_available_mask'):
                    assert torch.equal(ref[k],other[k]),(ds,model,k)
            rows=[]
            for arm in arms():
                for start in range(0,len(pairs),16):
                    part=pairs[start:start+16]
                    items=[intervene(data,x['index'],x['node'],arm['scenario'],arm['evidence'],arm['keep']) for x in part]
                    batch=vr.to_device(default_collate(items),device)
                    teacher=vr.build_model_teacher_mask(s,batch,cp['epoch'],False);assert not teacher.any()
                    with torch.inference_mode():pred=vr._forward(s,batch,ctx,cp['epoch'],teacher)['pred'].cpu().numpy()
                    assert np.isfinite(pred).all()
                    for j,(x,item) in enumerate(zip(part,items)):
                        n=x['node'];std=data.scaler['std'];mean=data.scaler['mean']
                        truth=item['target'][:,n,0].numpy()*std+mean
                        forecast=pred[j,:,n,0]*std+mean;diff=forecast-truth
                        q=int(item['input_available_mask'][:,n].sum())
                        if arm['q']>=0:assert q==arm['q']
                        rows.append(dict(dataset=ds,model=model,condition=arm['condition'],experiment=arm['experiment'],layout=arm['layout'],q=q,**x,
                            MAE=float(abs(diff).mean()),MSE=float((diff**2).mean()),MAPE=float((abs(diff)/np.maximum(abs(truth),1e-6)).mean()*100),
                            prediction=json.dumps(forecast.tolist()),target=json.dumps(truth.tolist())))
                print(ds,model,arm['condition'],flush=True)
            assert len(rows)==1280
            write_csv(path.with_suffix('.tmp'),rows);path.with_suffix('.tmp').replace(path)
            write(dest/(model+'_audit.json'),dict(plan_sha256=p['plan_sha256'],checkpoint_sha256=sha(ROOT/r['checkpoint']),rows=len(rows),teacher_forcing=0,split=split))
            del s,cp
            if device.startswith('cuda'):torch.cuda.empty_cache()
    write(dest/'COMPLETE.json',dict(plan_sha256=p['plan_sha256'],models=4,windows=64,arms=20,test_accessed=split=='test'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);a.add_argument('--dataset',choices=DATASETS);a.add_argument('--split',choices=['validation','test']);a.add_argument('--device',default='cuda:0');x=a.parse_args()
    freeze() if x.action=='freeze' else run(x.dataset,x.device,x.split)
