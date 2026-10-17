"""Fixed-checkpoint, four-dataset paired validation analysis; no training/test."""
import sys, json, argparse
from pathlib import Path
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.eval_release import read,write,sha,digest
OUT=ROOT/'paper/advisor_experiments_20261012'
PLAN=OUT/'plan.json'
DATASETS=['METRLA','PEMSBAY','PEMS04','PEMS08']

def freeze():
    assert not PLAN.exists()
    final=read(ROOT/'paper/final_scheme_20261006/selected_configuration_manifest.json')['entries']
    old=read(ROOT/'paper/all_final_test_20261001/all_results_and_provenance.json')
    val=read(ROOT/'paper/hparams_combinations_20260930/results_and_provenance.json')['records']
    entries={}
    for ds in DATASETS:
        models=[min(group,key=lambda m:val[ds][m]['table']['main']['LifeAvg']['MAE']) for group in
                [('UIGNN','GinAR'),('MoGERNN','SLPF'),('STGCN','STID','PatchSTG','STBP')]]
        entries[ds]={'ALFNet':final[ds]}
        for m in models:
            entries[ds][m]=old['entries'][ds+'_'+m]
        for m,r in entries[ds].items():
            assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
            r['evaluation_config_sha256']=sha(ROOT/r['config'])
    files=[Path(__file__),ROOT/'scripts/analyze_observation_support.py']
    for folder in ['model','layers','lib','experiments','data_provider','utils']:
        files.extend((ROOT/folder).rglob('*.py'))
    p=dict(entries=entries,split='validation',test_accessed=False,training=False,seed=32,
           per_dataset=64,selection_seed=20260908,known_nodes_only=True,
           authorization='User requests four-dataset mechanism figures and redesigned case analysis with final models.',
           selection='First 64 eligible fault-track pairs in fixed random schedule order; naturally complete history and targets; no prediction-based filtering.',
           cases='For complete interruption, choose the ALFNet-winning window nearest the median positive MAE gain over the strongest aggregate representative; explicitly illustrative, all-window distribution shown alongside.',
           representative='Minimum full-validation LifeAvg MAE within each of three categories; excluded STFiT/Merlin.',
           files={str(f.relative_to(ROOT)):sha(f) for f in files})
    p['plan_sha256']=digest(p);write(PLAN,p)

def run(ds,device):
    import numpy as np,torch
    from torch.utils.data import default_collate
    from experiments import v4_runner as vr
    from experiments.flow_support_transfer import adapter as support
    from experiments.flow_unified import adapter as unified
    from scripts.train_v4 import model_method_contract
    from scripts.analyze_observation_support import select_pairs,arms,intervene,INPUT_FIELDS,write_csv
    from lib.v4_seeds import seed_process
    p=read(PLAN);assert digest(p)==p['plan_sha256']
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    torch.set_num_threads(4)
    dest=OUT/ds;dest.mkdir(parents=True,exist_ok=True)
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
            data=s.validation_loader.dataset;assert data.context.phase=='validation'
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
                    with torch.inference_mode():pred=vr._forward(s,batch,s.validation_context,cp['epoch'],teacher)['pred'].cpu().numpy()
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
            write(dest/(model+'_audit.json'),dict(plan_sha256=p['plan_sha256'],checkpoint_sha256=sha(ROOT/r['checkpoint']),rows=len(rows),teacher_forcing=0,split='validation'))
            del s,cp
            if device.startswith('cuda'):torch.cuda.empty_cache()
    write(dest/'COMPLETE.json',dict(plan_sha256=p['plan_sha256'],models=4,windows=64,arms=20,test_accessed=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);ap.add_argument('--dataset',choices=DATASETS);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    freeze() if a.action=='freeze' else run(a.dataset,a.device)
