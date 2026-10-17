"""Export the authorized PEMS03 test and its unchanged eight baselines."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import read,write
from scripts.reselect_eight_baselines import audit,EXCLUDED
from scripts.evaluate_flow_repair import workbook,GROUPS,METRICS
OUT=ROOT/'paper/pems03_support_test_20261004'
def main():
    p=read(OUT/'test_results.json');key='PEMS03_center_half_lr1';r=p['rows'][key]
    old=read(ROOT/'paper/all_final_test_20261001/all_results_and_provenance.json')
    bases={e['model']:old['rows'][k] for k,e in old['entries'].items() if e['dataset']=='PEMS03' and e['model'] not in EXCLUDED};assert len(bases)==8
    rows={key:r,**bases};ranking=audit(rows,list(rows),bases)
    cells=workbook(rows,{'PEMS03':ranking},OUT/'test_comparison.xlsx');assert cells==297
    from openpyxl import load_workbook
    wb=load_workbook(OUT/'test_comparison.xlsx');wb['说明']['A1']='PEMS03九模型，第一加粗、第二下划线；胜格严格比较八基线最优值，基线自身并列不计第一。';wb.save(OUT/'test_comparison.xlsx')
    losses=[]
    for g,cs in GROUPS.items():
        for c in cs:
            for m in METRICS:
                n=min(bases,key=lambda n:bases[n]['table'][g][c][m]);v=bases[n]['table'][g][c][m]
                if r['table'][g][c][m]>=v:losses.append(dict(condition=c,metric=m,candidate=r['table'][g][c][m],baseline=n,best=v))
    best={m:min(b['table']['main']['LifeAvg'][m] for b in bases.values()) for m in METRICS}
    qualified=ranking[key]['first_cells']>=27 and all(r['table']['main']['LifeAvg'][m]<best[m] for m in METRICS)
    write(OUT/'selection.json',dict(candidate=key,validation_first_cells=26,test_first_cells=ranking[key]['first_cells'],
        qualified=qualified,lifeavg=r['table']['main']['LifeAvg'],baseline_best_lifeavg=best,losses=losses,
        test_plan=p['plan_sha256'],decision='Keep candidate and all PEMS03 evidence archived; do not claim80% or restore main dataset on performance grounds',formal_tables_overwritten=False))
    lines=['# PEMS03近门槛候选test结果','','用户明确授权在validation26/33未达门槛情况下测试最接近候选。固定半中心化/LR1、epoch57权重，完整33格validation复核最大差0，随后82个test batch完成，exit0。训练/评估数据划分、mask、指标和八基线不变。单新test，原版与旧reconstruction测试复用，共3配置99格。','',
        '| 配置 | test第一/33 | LifeAvg MAE | RMSE | MAPE% |','|---|---:|---:|---:|---:|']
    for k,x in p['rows'].items():lines.append('| '+k+' | '+str(p['ranking']['PEMS03'][k]['first_cells'])+' | '+' | '.join(f"{x['table']['main']['LifeAvg'][m]:.5f}" for m in METRICS)+' |')
    lines+=['','新候选20/33=60.61%，低于validation26/33，不满足27格目标。三项LifeAvg仍各自优于八基线最佳37.59699/50.62328/44.16910；相对原ALFNet改善1.32%/1.89%/21.01%，覆盖从8→20格。平均表现有实质改善，但没有满足用户同时要求的覆盖目标。','',
        '未胜13格：Unseen MAE/RMSE；Join MAE/RMSE；Cold MAE/MAPE；Warm MAE/RMSE；Ready MAE/RMSE；Outage三项。Unseen MAPE已领先，失败并非全部集中MAPE。','',
        '| 条件 | 指标 | 候选 | 最优基线 | 基线值 |','|---|---|---:|---|---:|']
    for x in losses:lines.append(f"| {x['condition']} | {x['metric']} | {x['candidate']:.5f} | {x['baseline']} | {x['best']:.5f} |")
    lines+=['','建议停止本轮PEMS03搜索，保持现有四数据集正文范围，将PEMS03完整结果保留为附录/泛化限制证据，不删除或隐瞒不利结果。不把按照test表现决定正文范围描述为事先固定数据集。没有自动启动其他候选test或训练。',
        '', '文件：test_comparison.xlsx/csv为候选与八基线主表/阶段表（第一加粗、第二下划线）；test_results.xlsx/csv保留新旧三候选；validation_results.xlsx保留七项validation；selection.json记录目标判定。历史正式四表与论文正文未覆盖。']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
