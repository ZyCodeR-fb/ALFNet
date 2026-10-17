"""Offline inventory of existing ALFNet candidates, no inference or selection."""
import json,csv,re,sys,copy
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.eval_release import read,write,sha
from scripts.collect_hparams_final import params
OUT=ROOT/'paper/full_vs_five_ablations_20261012'
DS=('PEMS04','PEMS08');MS=('MAE','RMSE','MAPE')
BASE=('UIGNN','GinAR','MoGERNN','SLPF','STGCN','STID','PatchSTG','STBP')
KINDS=('no_age','static_router','local_context','no_temporal_film','no_refiner')
def flatten(x,p=''):
    result={}
    for k,v in x.items():
        key=p+'.'+k if p else k
        if isinstance(v,dict):result.update(flatten(v,key))
        else:result[key]=v
    return result
def expanded(path):
    c=yaml.safe_load((ROOT/path).read_text());c=copy.deepcopy(c)
    c['runtime'].pop('model_variant',None)
    for e in ('graph','patch'):
        c['family']['experts'][e]['config_path']=yaml.safe_load((ROOT/c['family']['experts'][e]['config_path']).read_text())
    return flatten(c)
def compare(t,bases,abl):
    def checked(ref):
        for g in ('main','phase'):
            for c,v in ref[g].items():
                for k,n in v.items():
                    if k.endswith('_count'):assert t[g][c][k]==n,(g,c,k)
    for r in list(bases.values())+list(abl.values()):checked(r)
    first=sum(t[g][c][m]<min(r[g][c][m] for r in bases.values()) for g in ('main','phase') for c in t[g] for m in MS)
    per={k:sum(t['main']['LifeAvg'][m]<r['main']['LifeAvg'][m] for m in MS) for k,r in abl.items()}
    allbest=sum(t[g][c][m]<min(r[g][c][m] for r in abl.values()) for g in ('main','phase') for c in t[g] for m in MS)
    pairs=sum(t[g][c][m]<r[g][c][m] for r in abl.values() for g in ('main','phase') for c in t[g] for m in MS)
    return dict(lifeavg={m:t['main']['LifeAvg'][m] for m in MS},baseline_first=first,meets_80_percent=first>=27,
        lifeavg_pair_wins=sum(per.values()),lifeavg_wins_by_ablation=per,all_five_triples=sum(v==3 for v in per.values()),
        ablation_envelope_first=allbest,ablation_pair_wins=pairs,meets_both=first>=27 and sum(per.values())==15)
def main():
    OUT.mkdir(exist_ok=True)
    manuscript=(ROOT/'paper/experiments_draft_cn.md').read_text();section=manuscript.split('## 4.4')[1].split('## 4.5')[0]
    tab=ET.fromstring(re.search(r'<table>.*?</table>',section,re.S).group());names=[tr[0].text for tr in tab.find('tbody')]
    assert names==['ALFNet','w/o State Age','w/o Adaptive Routing','w/o Spatial Context','w/o ALFTemporal–FiLM','w/o Graph Refinement']
    old=read(ROOT/'paper/all_ablations_test_20261010/results_and_audit.json')['rows'];val=read(ROOT/'paper/ablations_complete_20261010/results_and_audit.json')['rows']
    repeated=read(ROOT/'paper/ablation_repeat_test_20261012/results.json')['rows'];rv=read(ROOT/'paper/ablation_repeat_20261012/results.json')['rows']
    ab={split:{d:{} for d in DS} for split in ('validation','test')}
    for ds in DS:
        for kind in KINDS:
            k=next(k for k in old if k[2:]=='_'+ds+'_'+kind)
            for split,source in [('validation',val),('test',old)]:ab[split][ds][kind]=source[k]['table']
            if ds=='PEMS04' and kind=='no_age':
                ab['test'][ds][kind]=repeated[k]['repeat_ablation']['table']
                ab['validation'][ds][kind]=rv[k]['tables']['repeat_ablation']
    discrepancy=[]
    for tr,kind in zip(list(tab.find('tbody'))[1:],KINDS):
        for ds,offset in [('PEMS04',7),('PEMS08',10)]:
            for i,m in enumerate(MS):
                shown=float(''.join(tr[offset+i].itertext()));actual=ab['test'][ds][kind]['main']['LifeAvg'][m]
                if shown!=float(f'{actual:.3f}'):discrepancy.append(dict(dataset=ds,ablation=kind,metric=m,shown=shown,consistent_record=actual))
    alltest=read(ROOT/'paper/all_final_test_20261001/all_results_and_provenance.json');baseval=read(ROOT/'paper/hparams_combinations_20260930/results_and_provenance.json')['records']
    baseline={'test':{d:{m:alltest['rows'][d+'_'+m]['table'] for m in BASE} for d in DS},'validation':{d:{m:baseval[d][m]['table'] for m in BASE} for d in DS}}
    tests={};candidates={};skipped=[]
    sources=sorted(set((ROOT/'paper').glob('*/results.json'))|set((ROOT/'paper').glob('*/test_results.json'))|{ROOT/'paper/all_final_test_20261001/all_results_and_provenance.json'})
    for f in sources:
        j=read(f)
        for k,r in j.get('rows',{}).items():
            if not isinstance(r,dict) or r.get('dataset') not in DS or 'table' not in r:continue
            if r.get('split')=='test' or r['table'].get('split')=='test':
                if r.get('passed') and r.get('model')=='ALFNet':
                    h=r['checkpoint_sha256']
                    if h in tests:
                        for g in ('main','phase'):
                            for c in r['table'][g]:
                                for m in MS:assert abs(tests[h]['table'][g][c][m]-r['table'][g][c][m])<1e-8
                    tests[h]=dict(table=r['table'],source=str(f.relative_to(ROOT)))
                continue
            if not all(x in r for x in ['config','checkpoint','report']):continue
            c=yaml.safe_load((ROOT/r['config']).read_text())
            if c.get('runtime',{}).get('model')!='ALFNet' or c.get('family',{}).get('recipe',{}).get('study_ablation'):continue
            if int(c['training']['seed'])!=32:continue
            h=r.get('checkpoint_sha256') or sha(ROOT/r['checkpoint'])
            if h in candidates:
                candidates[h]['aliases'].append(k);continue
            report=read(ROOT/r['report'])
            if report.get('development_validation_table')!=r['table']:
                skipped.append(dict(source=str(f.relative_to(ROOT)),key=k,reason='table/report mismatch'));continue
            assert sha(ROOT/r['checkpoint'])==h
            candidates[h]=dict(r,key=k,aliases=[k],source=str(f.relative_to(ROOT)),checkpoint_sha256=h)
    # Also inspect saved per-run test reports, including evaluations not yet packaged in paper/.
    artifact_reports=0
    for f in (ROOT/'artifacts/evaluation').rglob('*.json'):
        if f.stat().st_size>250000:continue
        r=read(f)
        if not isinstance(r,dict) or r.get('split')!='test' or not r.get('passed') or 'table' not in r:continue
        h=r.get('checkpoint_sha256')
        if h not in candidates:continue
        artifact_reports+=1
        if h in tests:
            for g in ('main','phase'):
                for c in r['table'][g]:
                    for m in MS:assert abs(tests[h]['table'][g][c][m]-r['table'][g][c][m])<1e-8
        else:tests[h]=dict(table=r['table'],source=str(f.relative_to(ROOT)))
    manifest=read(ROOT/'paper/final_scheme_20261006/selected_configuration_manifest.json')['entries'];final={d:expanded(manifest[d]['config']) for d in DS};results=[]
    hp={'lr','weight_decay','graph_expert_lr','patch_expert_lr','gate_lr','spa_patchsize','crossformer_dropout'}
    for h,r in candidates.items():
        ds=r['dataset'];flat=expanded(r['config']);diffs={k:[final[ds].get(k),flat.get(k)] for k in set(final[ds])|set(flat) if final[ds].get(k)!=flat.get(k)}
        structural={k:v for k,v in diffs.items() if k.split('.')[-1] not in hp}
        row=dict(key=r['key'],aliases=sorted(set(r['aliases'])),dataset=ds,checkpoint_sha256=h,config=r['config'],checkpoint=r['checkpoint'],source=r['source'],
            same_final_scheme=not structural,paired_with_current_ablations=not diffs,differences=diffs,other_than_hp_differences=structural,parameters=params(r),selected=h==manifest[ds]['checkpoint_sha256'])
        row['validation']=compare(r['table'],baseline['validation'][ds],ab['validation'][ds])
        row['test']=compare(tests[h]['table'],baseline['test'][ds],ab['test'][ds]) if h in tests else None
        row['test_source']=tests[h]['source'] if h in tests else None
        results.append(row)
    results.sort(key=lambda r:(r['dataset'],not r['same_final_scheme'],-(r['test'] or r['validation'])['lifeavg_pair_wins']))
    write(OUT/'inventory.json',dict(rows=results,manuscript_discrepancies=discrepancy,skipped=skipped,artifact_test_reports_checked=artifact_reports,comparison_rule='strict lower, >=27/33 vs eight baselines, 15/15 LifeAvg pairwise against five chosen ablations',ablation_reference=ab,new_test_access=False))
    export=[]
    for r in results:
        for split in ('validation','test'):
            if r[split] is None:continue
            s=r[split];export.append(dict(dataset=r['dataset'],key=r['key'],split=split,same_final_scheme=r['same_final_scheme'],paired=r['paired_with_current_ablations'],**{k:s[k] for k in ['baseline_first','lifeavg_pair_wins','all_five_triples','ablation_envelope_first','ablation_pair_wins','meets_both']},**s['lifeavg'],**r['parameters'],config=r['config']))
    with (OUT/'ranking.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
    for ds in DS:
        rr=[r for r in results if r['dataset']==ds];print(ds,'unique',len(rr),'same scheme',sum(r['same_final_scheme'] for r in rr),'tested',sum(r['test'] is not None for r in rr))
        for split in ('validation','test'):
            for family in [True,False]:
                ranked=sorted([r for r in rr if r[split] and r['same_final_scheme']==family],key=lambda r:(r[split]['meets_both'],r[split]['lifeavg_pair_wins'],r[split]['baseline_first'],r[split]['ablation_pair_wins']),reverse=True)
                print(split,'same scheme',family)
                for r in ranked[:5]:print(r['key'],r[split],r['parameters']['weight_decay'])
    print('manuscript mismatches',discrepancy)
if __name__=='__main__':main()
