"""Record accepted four-dataset configurations and explain rank changes offline."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import read,write,sha
from scripts.reselect_eight_baselines import audit,EXCLUDED
from scripts.collect_hparams_final import params
from scripts.evaluate_flow_repair import GROUPS,METRICS
OUT=ROOT/'paper/final_scheme_20261006'
def main():
    h=read(ROOT/'paper/hparams_combinations_20260930/results.json')['rows']
    flow=read(ROOT/'paper/flow_lr_cross_20261004/results.json')['rows']
    vt=read(ROOT/'paper/hparams_combinations_20260930/results_and_provenance.json')['records']
    old=read(ROOT/'paper/all_final_test_20261001/all_results_and_provenance.json')
    ft=read(ROOT/'paper/flow_lr_cross_test_20261004/test_results.json')['rows']
    selected={'METRLA':('METRLA_patch7',0,.3,0),'PEMSBAY':('05_PEMSBAY_combo2',0,.3,0),'PEMS04':('06_PEMS04_bridge_delta1',1,1,0),'PEMS08':('05_PEMS08_triple_halfcenter_lr1',1,1,.5)}
    entries={};diagnostics={}
    for ds,(key,bridge,scale,alpha) in selected.items():
        r=(h if ds in ('METRLA','PEMSBAY') else flow)[key]
        test=(old['rows'] if ds in ('METRLA','PEMSBAY') else ft)[key]
        assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']==test['checkpoint_sha256']
        entries[ds]=dict(key=key,config=r['config'],config_sha256=sha(ROOT/r['config']),checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],interpolation=bool(bridge),decoder_scale=scale,center_alpha=alpha,hyperparameters=params(r))
        vb={n:x for n,x in vt[ds].items() if n not in EXCLUDED}
        tb={e['model']:old['rows'][k] for k,e in old['entries'].items() if e['dataset']==ds and e['model'] not in EXCLUDED};assert len(vb)==len(tb)==8
        valrank=audit({key:r},[key],vb)[key];testrank=audit({key:test},[key],tb)[key]
        changes=[]
        for g,cs in GROUPS.items():
            for c in cs:
                for m in METRICS:
                    a=r['table'][g][c][m];b=test['table'][g][c][m]
                    av,an=min((x['table'][g][c][m],n) for n,x in vb.items());bv,bn=min((x['table'][g][c][m],n) for n,x in tb.items())
                    if (a<av)!=(b<bv):changes.append(dict(condition=c,metric=m,validation=a,validation_best=av,validation_best_model=an,test=b,test_best=bv,test_best_model=bn,transition='gain' if b<bv else 'loss'))
        diagnostics[ds]=dict(validation_first=valrank['first_cells'],test_first=testrank['first_cells'],validation_lifeavg=valrank['lifeavg'],test_lifeavg=testrank['lifeavg'],changes=changes,
            lifeavg_change_pct={m:100*(testrank['lifeavg'][m]/valrank['lifeavg'][m]-1) for m in METRICS})
    write(OUT/'selected_configuration_manifest.json',dict(status='user accepted final parameterized scheme',seed=32,entries=entries,
        disclosure='same learnable backbone; dataset-specific input handling and decoder settings; historical test participated in configuration retention; no new inference'))
    write(OUT/'validation_test_rank_diagnosis.json',diagnostics)
    lines=['# 最终四数据集validation→test排名变化','','使用最终选定checkpoint已有结果；无新推理或test数据读取。每格第一定义为严格优于八基线，33格包含主表和阶段表，彼此并非独立样本。','',
        '| 数据集 | validation第一 | test第一 | LifeAvg MAE变化% | RMSE变化% | MAPE变化% |','|---|---:|---:|---:|---:|---:|']
    for ds,d in diagnostics.items():lines.append(f"| {ds} | {d['validation_first']} | {d['test_first']} | "+' | '.join(f"{d['lifeavg_change_pct'][m]:+.2f}" for m in METRICS)+' |')
    for ds,d in diagnostics.items():
        lines+=['',f'## {ds}','', '| 条件/指标 | 变化 | ALFNet val→test | 最强基线 val→test |','|---|---|---|---|']
        for x in d['changes']:lines.append(f"| {x['condition']}/{x['metric']} | {x['transition']} | {x['validation']:.5f}→{x['test']:.5f} | {x['validation_best']:.5f}({x['validation_best_model']})→{x['test_best']:.5f}({x['test_best_model']}) |")
    lines+=['','解释：speed并非多了大量新信息，METRLA只跨越4个比较阈值，PEMSBAY净增5（6增1失）。实际误差与基线也随时间/节点群体变化，排名取决于两者相对幅度。flow已在validation有27/30格，改善空间较小；PEMS08尽管LifeAvg变差，仍可保持相同总胜格。格数相同不保证同一组格子不变，更不代表分布相同。',
        '', 'validation与test按时间分段，并使用不同的未见节点V/T，因而时间/节点分布变化是合理来源；但这份已有指标分析没有对两个因素做因果隔离，不能把speed所有变化归结为某一种分布偏移。当前speed原配置的保留曾参考test，存在选择乐观性，不能把test优于validation都解释为天然泛化增强。需如实披露历史选择流程，不把test当作继续调参的独立验证。']
    (OUT/'RANK_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
