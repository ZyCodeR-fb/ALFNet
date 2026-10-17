"""Complete four-axis ALFNet sensitivity with 31 verified reuses and 24 new runs."""
import argparse,copy,csv,json,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import read,write,sha,digest,identity,parameter_identity
NAME='hparams_completion_20260928'
DIR=ROOT/'configs/paper'/NAME
PLAN=DIR/'plan.json'
BASE_TABLE=ROOT/'paper/validation_seed32_baseline_extension_20260927/results_and_provenance.json'
DATASETS=['PEMS03','PEMS04','PEMS08','PEMSBAY']

def output(p):return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']

def specs(ds):
    shared=[('drop05','dropout',.05),('drop20','dropout',.2),('wd_low','weight_decay',1e-5),('wd_high','weight_decay',1e-3)]
    extra=([('lr_half','lr_factor',.5),('lr_double','lr_factor',2.)] if ds in ('PEMS03','PEMSBAY') else
           [('capacity_mid','capacity',14 if ds=='PEMS04' else 8),('capacity_high','capacity',18 if ds=='PEMS04' else 10)])
    return shared+extra

def candidate(base,nested,kind,value,patch_path):
    a,b=copy.deepcopy(base),copy.deepcopy(nested)
    if kind=='dropout':b['family']['model']['crossformer_dropout']=value
    elif kind=='weight_decay':a['training']['weight_decay']=value
    elif kind=='capacity':b['family']['model']['spa_patchsize']=value
    elif kind=='lr_factor':
        a['training']['lr']*=value
        for k in ('graph_expert_lr','patch_expert_lr','gate_lr'):a['family']['recipe'][k]*=value
    else:raise ValueError(kind)
    a['family']['experts']['patch']['config_path']=patch_path
    return a,b

def freeze():
    assert not PLAN.exists()
    DIR.mkdir(parents=True,exist_ok=True)
    records=read(BASE_TABLE)['records'];sources=[BASE_TABLE,ROOT/'paper'/NAME/'PLAN.md'];reused={};configs={};anchors={}
    old_plans=['configs/paper/metrla_hparams_20260920/plan.json','configs/paper/pemsbay_hparams_20260917/plan.json','configs/paper/flow_extension_20260926/plan.json']
    # Verify all executable model/data semantics; the sole historical runner delta is registration.
    import hashlib
    runner=(ROOT/'experiments/v4_runner.py').read_text().replace('    "Merlin": "model.comparison_model.Merlin",\n','')
    semantic=[]
    for old in old_plans:
        p=read(ROOT/old);sources.append(ROOT/old)
        for f,h in p['files'].items():
            if f.startswith(('model/','layers/','lib/','data_provider/','experiments/')):
                if f=='experiments/v4_runner.py':assert hashlib.sha256(runner.encode()).hexdigest()==h
                else:assert sha(ROOT/f)==h,f
        semantic.append(dict(source_plan=old,model_data_code_identical=True,runner_delta='Merlin registration only'))
    for ds,path,old in [('METRLA','paper/hparams_metrla_20260920/results.json',old_plans[0]),('PEMSBAY','paper/hparams_pemsbay_20260917/results.json',old_plans[1])]:
        r=read(ROOT/path);p=read(ROOT/old);sources.append(ROOT/path)
        for k,t in r['tables'].items():
            info=r['provenance'][k]
            if k=='baseline_reused':
                cp=ROOT/info['checkpoint'];cfg=cp.with_name('config.yaml');rp=cp.with_name('run_report.json')
            else:
                rp=Path(info['report_path']);cp=rp.with_name('best.pt');cfg=ROOT/p['configs'][k]['path']
            reused[ds+'_'+k]=dict(dataset=ds,table=t,checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=sha(cp),
                config=str(cfg.relative_to(ROOT)),config_sha256=sha(cfg),report=str(rp.relative_to(ROOT)))
            sources.extend([cp,cfg,rp])
    old=read(ROOT/old_plans[2]);oldroot=ROOT/'artifacts/experiments/flow_extension_20260926'/old['plan_sha256']
    for k,row in old['configs'].items():
        rp=Path(read(oldroot/(k+'_complete.json'))['report_path']);r=read(rp);cp=rp.with_name('best.pt');cfg=ROOT/row['path']
        reused[k]=dict(dataset=row['dataset'],table=r['development_validation_table'],checkpoint=str(cp.relative_to(ROOT)),
            checkpoint_sha256=sha(cp),config=str(cfg.relative_to(ROOT)),config_sha256=sha(cfg),report=str(rp.relative_to(ROOT)))
        sources.extend([cp,cfg,rp])
    assert len(reused)==31
    for k,r in reused.items():
        report=read(ROOT/r['report'])
        assert all(e[z]['skipped']==0 for e in report['history'] for z in ('train','validation'))
        assert report['early_stopping_metric']=='validation 33-cell main/LifeAvg/MAE'
        assert min((e['validation']['checkpoint_life_avg_mae'],e['epoch']) for e in report['history'])[1]==report['best_epoch']
        assert report['development_validation_table']==r['table']
    for ds in DATASETS:
        anchor=records[ds]['ALFNet'];cfg=ROOT/anchor['config'];base=yaml.safe_load(cfg.read_text())
        nested_path=ROOT/base['family']['experts']['patch']['config_path'];nested=yaml.safe_load(nested_path.read_text())
        graph=ROOT/base['family']['experts']['graph']['config_path']
        anchors[ds]=dict(config=str(cfg.relative_to(ROOT)),checkpoint=anchor['checkpoint'],table=anchor['table'])
        sources.extend([cfg,nested_path,graph,ROOT/anchor['checkpoint']])
        for i,(name,kind,value) in enumerate(specs(ds)):
            key=f'{i*4+DATASETS.index(ds):02d}_{ds}_{name}'; pp=DIR/(key+'_patch.yaml');path=DIR/(key+'.yaml')
            a,b=candidate(base,nested,kind,value,str(pp.relative_to(ROOT)));a['runtime']['model_variant']=NAME+'_'+key
            pp.write_text(yaml.safe_dump(b,sort_keys=False));path.write_text(yaml.safe_dump(a,sort_keys=False))
            configs[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),dataset=ds,model='ALFNet',seed=32,
                axis=kind,value=value,capacity=b['family']['model']['spa_patchsize'],dropout=b['family']['model'].get('crossformer_dropout',.1),
                weight_decay=a['training']['weight_decay'],lrs=[a['family']['recipe'][v] for v in ('graph_expert_lr','patch_expert_lr','gate_lr')])
            sources.extend([pp,path])
        for field in ('data_path','unknown_nodes_path'):sources.append(ROOT/base['dataset'][field])
        sources.append(ROOT/f'configs/paper/v4_data_manifests/{ds}_run_manifest.json')
    for folder in ('model','layers','lib','experiments','utils','data_provider','scripts'):sources.extend((ROOT/folder).rglob('*.py'))
    sources.append(ROOT/'args.py')
    p=dict(kind=NAME,status='authorized_validation_only',configs=configs,reused=reused,anchors=anchors,
        baseline_table=str(BASE_TABLE.relative_to(ROOT)),semantic_reuse_audit=semantic,max_concurrent=5,test_authorized=False,
        seed=32,selection='full validation33 LifeAvg MAE; report all metrics/wins/Pareto without hard regression gates',
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(sources))})
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])

def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and len(p['configs'])==24 and len(p['reused'])==31
    assert not p['test_authorized'] and p['max_concurrent']==5
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p

def safe_train(cfg,root,p,reference,smoke=False):
    from scripts import train_v4 as training
    original_train,original_validate=training.train_one_epoch,training.validate
    def epoch(*args,**kwargs):
        r=original_train(*args,**kwargs);assert r['skipped']==0,'training skipped invalid batch';return r
    def validate(*args,**kwargs):
        result=original_validate(*args,**kwargs);r=result[0] if isinstance(result,tuple) else result
        assert r['skipped']==0,'validation skipped invalid batch'
        if isinstance(result,tuple):
            for g in ('main','phase'):
                for c,values in reference[g].items():
                    for field,v in values.items():
                        if field.endswith('_count'):assert result[1][g][c][field]==v,(g,c,field)
        return result
    with patch.object(training,'train_one_epoch',epoch),patch.object(training,'validate',validate):
        return training.train(cfg,root,device='cuda:0',seed=32,num_workers=0,plan=p,save=not smoke,
            collect_development_table=not smoke,max_epochs=1 if smoke else None,
            max_train_batches=20 if smoke else None,max_validation_batches=4 if smoke else None)

def preflight():
    import torch
    from experiments.v4_runner import build_v4_session
    from scripts.train_v4 import build_optimizer
    torch.set_num_threads(4);p=check();root=output(p);initial={};identities={};reports=[]
    for ds,a in p['anchors'].items():
        s=build_v4_session(ROOT/a['config'],device='cpu',run_seed=32,num_workers=0)
        initial[ds]=parameter_identity(s.model);identities[ds]=identity(s)
        cp=torch.load(ROOT/a['checkpoint'],map_location='cpu',weights_only=False)
        assert all(cp[k]==v for k,v in identities[ds].items()),ds
        s.model.load_state_dict(cp['model_state'],strict=True)
        del cp,s
    for key,row in sorted(p['configs'].items()):
        print('PREFLIGHT_START',key,flush=True);cfg=ROOT/row['path'];ds=row['dataset']
        s=build_v4_session(cfg,device='cpu',run_seed=32,num_workers=0)
        assert identity(s)==identities[ds] and parameter_identity(s.model)==initial[ds]
        assert s.args.max_epoch==100 and s.args.patience==10 and s.args.early_stopping_start_epoch==40
        opt=build_optimizer(s);assert [g['lr'] for g in opt.param_groups]==row['lrs']
        assert all(g['weight_decay']==row['weight_decay'] for g in opt.param_groups)
        drops=[m.p for m in s.model.patch_expert.modules() if isinstance(m,torch.nn.Dropout)]
        assert drops and all(v==row['dropout'] for v in drops)
        assert len(s.train_context.layout_parts)==32 and len(s.model.patch_expert.reo_all_idx)==32*row['capacity']
        del s,opt
        torch.cuda.reset_peak_memory_stats();start=time.time()
        safe_train(cfg,None,p,p['anchors'][ds]['table'],smoke=True)
        r=dict(key=key,passed=True,plan_sha256=p['plan_sha256'],seconds=time.time()-start,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,train_batches=20,validation_batches=4)
        write(root/'preflight'/(key+'.json'),r);reports.append(r);torch.cuda.empty_cache()
        print('PREFLIGHT_OK',key,r,flush=True)
    write(root/'PREFLIGHT_COMPLETE.json',dict(passed=True,plan_sha256=p['plan_sha256'],reports=reports))

def worker(key):
    import torch
    torch.set_num_threads(4);p=check();root=output(p);row=p['configs'][key]
    assert read(root/'PREFLIGHT_COMPLETE.json')['plan_sha256']==p['plan_sha256']
    r=safe_train(ROOT/row['path'],root/'runs'/key,p,p['anchors'][row['dataset']]['table'])
    write(root/(key+'_complete.json'),dict(passed=True,plan_sha256=p['plan_sha256'],report_path=str(Path(r['run_directory'])/'run_report.json')))

def collect():
    p=check();root=output(p);rows=copy.deepcopy(p['reused']);baselines=read(BASE_TABLE)['records']
    for k,e in p['configs'].items():
        done=read(root/(k+'_complete.json'));assert done['passed'];rp=Path(done['report_path']);r=read(rp)
        assert r['plan_sha256']==p['plan_sha256'] and r['config_sha256']==e['sha256']
        assert min((v['validation']['checkpoint_life_avg_mae'],v['epoch']) for v in r['history'])[1]==r['best_epoch']
        rows[k]=dict(dataset=e['dataset'],config=e['path'],table=r['development_validation_table'],report=str(rp.relative_to(ROOT)),
            checkpoint=str(rp.with_name('best.pt').relative_to(ROOT)),checkpoint_sha256=sha(rp.with_name('best.pt')),axis=e['axis'],value=e['value'])
    summary={};metrics=('MAE','RMSE','MAPE')
    for ds in baselines:
        keys=[k for k,r in rows.items() if r['dataset']==ds]
        b=[r['table'] for n,r in baselines[ds].items() if n!='ALFNet']
        for k in keys:
            t=rows[k]['table'];rows[k]['first_cells']=sum(t[g][c][m]<min(x[g][c][m] for x in b) for g in ('main','phase') for c in t[g] for m in metrics)
        def vec(k):return [rows[k]['table']['main']['LifeAvg'][m] for m in metrics]
        pareto=[k for k in keys if not any(all(a<=b for a,b in zip(vec(j),vec(k))) and any(a<b for a,b in zip(vec(j),vec(k))) for j in keys if j!=k)]
        summary[ds]=dict(min_lifeavg_mae=min(keys,key=lambda k:vec(k)[0]),most_first_cells=max(keys,key=lambda k:rows[k]['first_cells']),lifeavg_pareto=pareto)
    out=ROOT/'paper'/NAME;write(out/'results.json',dict(plan_sha256=p['plan_sha256'],split='validation',rows=rows,summary=summary,test_accessed=False,automatic_final_replacement=False))
    with (out/'cells.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['run','dataset','section','condition','metric','value'])
        for k,r in rows.items():
            for g in ('main','phase'):
                for c,v in r['table'][g].items():
                    for m in metrics:w.writerow([k,r['dataset'],g,c,m,v[m]])
    md=['# 五数据集超参数结果（validation）','','复用31项＋新24项；容量/学习率旧锚点系列单独保留，不把不同锚点误作单变量曲线。','',
        '| Run | LifeAvg MAE | RMSE | MAPE (%) | 第一格/33 |','|---|---:|---:|---:|---:|']
    for k,r in rows.items():md.append('| '+k+' | '+' | '.join(f"{r['table']['main']['LifeAvg'][m]:.5f}" for m in metrics)+f" | {r['first_cells']} |")
    (out/'results_summary.md').write_text('\n'.join(md)+'\n');write(root/'COMPLETE.json',dict(passed=True,total_configs=len(rows),test_accessed=False))

def queue():
    from scripts import recover_baseline_extension as recovery
    p=check()
    if not (output(p)/'PREFLIGHT_COMPLETE.json').exists():preflight()
    # Reuse tested dispatch/memory/locking machinery with study-local authority.
    with ExitStack() as stack:
        for name,value in [('check',check),('output',output),('collect',collect),('__file__',__file__),
                           ('isolate_failed_families',lambda jobs,configs:None)]:
            stack.enter_context(patch.object(recovery,name,value))
        recovery.queue()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','check','preflight','queue','worker','collect']);p.add_argument('--key');a=p.parse_args()
    if a.mode=='worker':worker(a.key)
    elif a.mode=='check':print(check()['plan_sha256'])
    else:globals()[a.mode]()
