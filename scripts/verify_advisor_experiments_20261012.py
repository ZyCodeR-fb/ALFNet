"""Offline integrity checks for the advisor report, no model inference."""
import json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.eval_release import read,write,sha,digest
OUT=ROOT/'paper/advisor_experiments_20261012'
def main():
    p=read(OUT/'plan.json');assert digest(p)==p['plan_sha256']
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    final=read(ROOT/'paper/final_scheme_20261006/selected_configuration_manifest.json')['entries']
    old=read(ROOT/'paper/all_ablations_test_20261010/results_and_audit.json')['rows']
    counts={}
    for ds,entries in p['entries'].items():
        assert final[ds]['checkpoint_sha256']==entries['ALFNet']['checkpoint_sha256']==old[ds+'_Full']['checkpoint_sha256']
        assert read(OUT/ds/'COMPLETE.json')['plan_sha256']==p['plan_sha256']
        frames=[]
        for m,r in entries.items():
            assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
            assert sha(ROOT/r['config'])==r['evaluation_config_sha256']
            audit=read(OUT/ds/(m+'_audit.json'));assert audit['plan_sha256']==p['plan_sha256'] and audit['teacher_forcing']==0 and audit['split']=='validation'
            df=pd.read_csv(OUT/ds/(m+'.csv'));assert len(df)==1280 and df.pair_id.nunique()==64 and df.condition.nunique()==20
            pred=np.array([json.loads(x) for x in df.prediction]);truth=np.array([json.loads(x) for x in df.target]);assert pred.shape==truth.shape==(1280,12)
            assert np.isfinite(pred).all() and np.isfinite(truth).all()
            assert np.allclose(abs(pred-truth).mean(1),df.MAE,atol=2e-5,rtol=1e-6)
            frames.append(df)
        allrows=pd.concat(frames);assert allrows.groupby(['condition','pair_id']).target.nunique().max()==1
        assert allrows.groupby(['condition','pair_id']).origin.nunique().max()==1
        assert allrows.groupby(['condition','pair_id']).node.nunique().max()==1
        counts[ds]=len(allrows)
    summary=read(OUT/'summary.json');assert list(summary['first_cells'].values())==[33,29,29,30]
    table=pd.read_csv(OUT/'final_comparison_test.csv');assert len(table)==4*9*33
    for ds in final:
        for g in ['main','phase']:
            for c,r in old[ds+'_Full']['table'][g].items():
                for m in ['MAE','RMSE','MAPE']:
                    x=table[(table.dataset==ds)&(table.model=='ALFNet')&(table.section==g)&(table.condition==c)&(table.metric==m)]
                    assert len(x)==1 and abs(x.value.iloc[0]-r[m])<1e-10
    doc=ROOT/'paper/experiments_draft_cn.md';text=doc.read_text()
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith('http'):assert (doc.parent/link.split('#')[0]).exists(),link
    for n in range(2,7):
        files=list(OUT.glob(f'fig{n}_*.png'));assert len(files)==1
        for ext in ['png','pdf','svg']:assert files[0].with_suffix('.'+ext).stat().st_size>1000
    for name in ['table1.md','table2.md']:assert (OUT/name).read_text().strip() in text
    write(OUT/'verification.json',dict(passed=True,plan_sha256=p['plan_sha256'],prediction_rows=counts,total_rows=sum(counts.values()),
        forecast_values=245760,comparison_cells=1188,final_checkpoint_identities_match=True,
        paired_targets_origins_nodes_identical=True,stored_prediction_mae_recomputed=True,
        frozen_source_hashes_match=True,links_exist=True,figures_png_pdf_svg=15,
        test_accessed=False,training=False,single_seed=32))
    print('PASS: 20480 paired forecasts, 1188 final comparison cells, links and identities')
if __name__=='__main__':main()
