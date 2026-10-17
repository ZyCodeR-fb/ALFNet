"""Both fixed repeat checkpoints: full validation replay followed by test."""
import argparse,sys,csv
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import read,write,sha,digest
from scripts import repeat_flow_ablations_20261012 as train
from scripts import test_all_ablations_20261010 as base
from scripts import evaluate_speed_support_20261006 as queue_base
NAME='ablation_repeat_test_20261012';OUT=ROOT/'paper'/NAME;PLAN=ROOT/'configs/paper'/NAME/'plan.json'
def output(p):return ROOT/'artifacts/evaluation'/NAME/p['plan_sha256']
def freeze():
    import torch,yaml
    assert not PLAN.exists();old=train.check();root=train.output(old)
    assert read(root/'COMPLETE.json')['passed']
    assert all(j['status']=='complete' and j['exit_code']==0 for j in read(root/'queue_status.json')['jobs'])
    source=ROOT/'paper/ablation_repeat_20261012/results.json';src=read(source)['rows'];entries={};audit={};files=[source,train.PLAN]
    for k,r in src.items():
        rp=ROOT/r['report'];report=read(rp);cp=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
        config=old['configs'][k]['path'];cfg=yaml.safe_load((ROOT/config).read_text())
        assert cp['plan_sha256']==old['plan_sha256'] and cp['config_sha256']==sha(ROOT/config)
        assert cp['seed']==32 and sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        assert cp['epoch']==report['best_epoch']==min((e['validation']['checkpoint_life_avg_mae'],e['epoch']) for e in report['history'])[1]
        assert all(e[s]['skipped']==0 for e in report['history'] for s in ('train','validation'))
        assert all(torch.isfinite(v).all() for v in cp['model_state'].values() if v.is_floating_point())
        entries[k]=dict(dataset=r['dataset'],model='ALFNet',label=old['configs'][k]['label'],ablation=old['configs'][k]['ablation'],config=config,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],report=r['report'],epoch=cp['epoch'],training_config_sha256=cp['config_sha256'],training_plan_sha256=cp['plan_sha256'],table=report['development_validation_table'])
        audit[k]=dict(best_epoch=cp['epoch'],epochs_run=report['epochs_run'],zero_skips=True,finite_weights=True)
        files.extend([rp,ROOT/config,ROOT/r['checkpoint']])
        files.extend(ROOT/cfg['family']['experts'][e]['config_path'] for e in ('graph','patch'))
        files.extend(ROOT/cfg['dataset'][f] for f in ('data_path','unknown_nodes_path'))
    previous=ROOT/'paper/all_ablations_test_20261010/results_and_audit.json';files.append(previous)
    for folder in ('model','layers','lib','experiments','data_provider','utils'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.extend([Path(__file__),ROOT/'scripts/test_all_ablations_20261010.py',ROOT/'scripts/test_all_final_20261001.py',ROOT/'scripts/evaluate_speed_support_20261006.py',ROOT/'scripts/run_final_ablations_20261006.py',ROOT/'scripts/train_v4.py',ROOT/'scripts/eval_release.py',ROOT/'args.py'])
    p=dict(kind=NAME,status='authorized_test',test_authorized=True,authorization='2026-10-12 user requests checking whether completed repeats test LifeAvg worsens in all three metrics.',entries=entries,max_concurrent=2,training_audit=audit,previous=str(previous.relative_to(ROOT)),files={str(f.relative_to(ROOT)):sha(f) for f in set(files)},selection='Both fixed validation-best repeats evaluated regardless of outcome; keep original tests and no automatic replacement.')
    p['plan_sha256']=digest(p);write(PLAN,p);print(p['plan_sha256'])
def check():
    p=read(PLAN);assert digest(p)==p['plan_sha256'] and p['test_authorized'] and set(p['entries'])==set(train.KEYS)
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def worker(k):
    with patch.object(base,'check',check),patch.object(base,'output',output):base.worker(k)
def collect():
    p=check();root=output(p);old=read(ROOT/p['previous'])['rows'];rows={};cells=[];verdict={};replays={}
    lines=['# 两项消融重跑：test结果','','Full、原消融与重跑全部保留。变化相对Full，正数表示消融误差更高；不据此覆盖原论文结果。','', '| 数据集 / 消融 | 记录 | MAE | RMSE | MAPE | 相对Full变化%（MAE/RMSE/MAPE） | 全三项变差 |','|---|---|---:|---:|---:|---|---|']
    for k,e in p['entries'].items():
        ds=e['dataset'];new=read(root/(k+'.json'));v=read(root/'validation'/(k+'.json'))
        assert new['passed'] and v['passed'] and new['checkpoint_sha256']==e['checkpoint_sha256']
        replays[k]=dict(passed=True,max_abs_difference=v['max_abs_difference'])
        full=old[ds+'_Full']['table'];group={'Full':old[ds+'_Full'],'original_ablation':old[k],'repeat_ablation':new};rows[k]=group
        for label,r in group.items():
            for section in ('main','phase'):
                for c,x in r['table'][section].items():
                    for f,n in full[section][c].items():
                        if f.endswith('_count'):assert x[f]==n
                    for m in ('MAE','RMSE','MAPE'):cells.append([ds,k,label,section,c,m,x[m]])
            metrics=('MAE','RMSE','MAPE');a=r['table']['main']['LifeAvg'];d={m:100*(a[m]/full['main']['LifeAvg'][m]-1) for m in metrics}
            qualifies=all(x>0 for x in d.values())
            lines.append('| '+k+' | '+label+' | '+' | '.join(f'{a[m]:.6f}' for m in metrics)+' | '+' / '.join(f'{d[m]:+.2f}' for m in metrics)+' | '+('是' if qualifies else '否')+' |')
            if label=='repeat_ablation':verdict[k]=dict(delta_percent=d,qualifies=qualifies)
    write(OUT/'results.json',dict(plan_sha256=p['plan_sha256'],rows=rows,verdict=verdict,validation_replays=replays,test_accessed=True))
    with (OUT/'test_cells.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['dataset','run','record','section','condition','metric','value']);w.writerows(cells)
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    write(root/'COMPLETE.json',dict(passed=True,new_tests=2,metric_cells=len(cells),verdict=verdict))
    print(verdict)
def queue():
    p=check()
    with ExitStack() as s:
        for n,v in [('check',check),('output',output),('collect',collect),('KEYS',tuple(p['entries'])),('__file__',__file__)]:s.enter_context(patch.object(queue_base,n,v))
        queue_base.queue()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['freeze','check','worker','queue','collect']);a.add_argument('--key');x=a.parse_args()
    if x.mode=='worker':worker(x.key)
    elif x.mode=='check':print(check()['plan_sha256'])
    else:globals()[x.mode]()
