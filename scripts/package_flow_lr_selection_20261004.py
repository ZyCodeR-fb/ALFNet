"""Publish selected flow candidates versus unchanged eight baselines."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import read,write
from scripts.evaluate_flow_repair import workbook
from scripts.reselect_eight_baselines import audit,EXCLUDED
OUT=ROOT/'paper/flow_lr_cross_test_20261004'
SELECTED={'PEMS04':'06_PEMS04_bridge_delta1','PEMS08':'05_PEMS08_triple_halfcenter_lr1'}
def main():
    validation=read(ROOT/'paper/flow_lr_cross_20261004/results.json')['rows']
    tests=read(OUT/'test_results.json')['rows']
    valbase=read(ROOT/'paper/flow_lr_cross_20261004/fixed_baselines.json')['records']
    old=read(ROOT/'paper/all_final_test_20261001/all_results_and_provenance.json')
    evidence={}
    for split,source in [('validation',validation),('test',tests)]:
        rows={};rankings={};evidence[split]={}
        for ds,key in SELECTED.items():
            if split=='validation':bases={n:r for n,r in valbase[ds].items() if n not in EXCLUDED}
            else:bases={e['model']:old['rows'][k] for k,e in old['entries'].items() if e['dataset']==ds and e['model'] not in EXCLUDED}
            assert len(bases)==8
            group={key:source[key]}
            group.update({ds+'_'+n:dict(r,dataset=ds) for n,r in bases.items()})
            rows.update(group);rankings[ds]=audit(group,list(group),bases)
            v=source[key]['table']['main']['LifeAvg']
            qualified=rankings[ds][key]['first_cells']>=27 and all(v[m]<min(b['table']['main']['LifeAvg'][m] for b in bases.values()) for m in ('MAE','RMSE','MAPE'))
            evidence[split][ds]=dict(key=key,first_cells=rankings[ds][key]['first_cells'],lifeavg=v,qualified=qualified)
        path=OUT/(split+'_selected_comparison.xlsx');cells=workbook(rows,rankings,path)
        from openpyxl import load_workbook
        wb=load_workbook(path)
        wb['说明']['A1']='同一数据集九模型第一加粗、第二下划线；第一格数按八基线最优值严格比较，基线自身并列不计第一。'
        wb.save(path);assert cells==594
    write(OUT/'selection.json',dict(selected=SELECTED,evidence=evidence,formal_four_tables_overwritten=False,
        status='recommended flow configurations; no single unified center strength validated across all datasets',
        limitation='Exploratory single-seed results using historical test feedback; no new speed transfer'))
if __name__=='__main__':main()
