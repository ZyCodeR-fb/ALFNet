"""Compare all reported metrics and plot all-window residuals; no inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODELS=['ALFNet','UIGNN','MoGERNN','STBP']


def main():
    plan=json.loads((ROOT/'configs/paper/final_support_figures_20260922.json').read_text())
    src=ROOT/'artifacts/evaluation/final_support_figures_20260922'/plan['plan_sha256']
    out=ROOT/'paper/support_metric_audit_20260922';out.mkdir(exist_ok=True)
    df=pd.read_csv(src/'paired_predictions.csv');summary=pd.read_csv(src/'metrics.csv')
    ranks=[]
    for metric in ['MAE','RMSE','MAPE']:
        for track in ['fault','join']:
            g=summary[(summary.track==track)&(summary.experiment=='support')].pivot(index='condition',columns='model',values=metric)
            ranks.append(dict(metric=metric,cohort='known' if track=='fault' else 'held_out',ALFNet_first=int((g.idxmin(axis=1)=='ALFNet').sum()),conditions=len(g)))
    pd.DataFrame(ranks).to_csv(out/'metric_rank_audit.csv',index=False)
    summary.to_csv(out/'all_metrics.csv',index=False)
    specs=[('fault','full','Known: full history'),('join','join_3','Held-out: Join-3'),
           ('fault','fault_6','Known: Fault-6'),('fault','outage','Known: Outage')]
    matrices={};orders={};stats=[]
    for track,c,title in specs:
        target=df[(df.track==track)&(df.condition==c)&(df.model=='ALFNet')].copy()
        target['truth_mean']=target.target.apply(lambda a: np.mean(json.loads(a)))
        ids=target.sort_values(['truth_mean','pair_id']).pair_id.tolist();assert len(ids)==64
        orders[c]=ids
        for m in MODELS:
            g=df[(df.track==track)&(df.condition==c)&(df.model==m)].set_index('pair_id').loc[ids]
            pred=np.array([json.loads(v) for v in g.prediction]);truth=np.array([json.loads(v) for v in g.target])
            assert pred.shape==truth.shape==(64,12)
            matrices[m,c]=pred-truth
            stats.append(dict(cohort=track,condition=c,model=m,windows=64,
                MAE=float(np.abs(pred-truth).mean()),bias=float((pred-truth).mean()),
                RMSE=float(np.sqrt(np.square(pred-truth).mean()))))
    maxabs=max(np.abs(a).max() for a in matrices.values())
    norm=SymLogNorm(linthresh=2,linscale=1,vmin=-maxabs,vmax=maxabs,base=10)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(4,4,figsize=(12,10),layout='constrained',sharex=True,sharey=True)
    for i,m in enumerate(MODELS):
        for j,(_,c,title) in enumerate(specs):
            ax=axes[i,j];im=ax.imshow(matrices[m,c],cmap='RdBu_r',norm=norm,aspect='auto',interpolation='nearest')
            if i==0:ax.set_title(title,fontsize=10)
            if j==0:ax.set_ylabel(m+'\nWindow order')
            ax.set_yticks([0,31,63],[1,32,64]);ax.set_xticks([0,5,11],[5,30,60])
            if i==3:ax.set_xlabel('Horizon (minutes)')
    cb=fig.colorbar(im,ax=axes,shrink=.65,pad=.025)
    cb.set_label('Prediction − truth (mph); symmetric-log scale')
    cb.set_ticks([-40,-20,-10,-2,0,2,10,20,40]);cb.set_ticklabels(['−40','−20','−10','−2','0','2','10','20','40'])
    for ext in ['png','pdf']:fig.savefig(out/f'residual_atlas.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(stats).to_csv(out/'residual_summary.csv',index=False)
    np.savez_compressed(out/'residual_matrices.npz',**{m+'__'+c:a for (m,c),a in matrices.items()})
    (out/'provenance.json').write_text(json.dumps(dict(plan_sha256=plan['plan_sha256'],source=str(src.relative_to(ROOT)),
        order_rule='within condition ascending true future mean; tie pair_id; shared across models',orders=orders,
        windows_per_condition=64,conditions=specs,signed_error='prediction minus truth in mph',
        color_scale='one shared symmetric logarithmic scale, linear inside [-2,2], no clipping',
        color_max_absolute=float(maxabs),new_inference=False,new_test=False),indent=2)+'\n')
    print(pd.DataFrame(ranks).to_string(index=False))
    print(pd.DataFrame(stats).round(4).to_string(index=False))


if __name__=='__main__':main()
