"""Offline cohort-aware presentation of frozen paired predictions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODELS=['ALFNet','UIGNN','MoGERNN','STBP']
COLORS=['#0072B2','#009E73','#D55E00','#CC79A7']
MARKERS=['o','s','^','D']


def main():
    plan=json.loads((ROOT/'configs/paper/final_support_figures_20260922.json').read_text())
    source=ROOT/'artifacts/evaluation/final_support_figures_20260922'/plan['plan_sha256']
    out=ROOT/'paper/support_cohorts_20260922';out.mkdir(exist_ok=True)
    df=pd.read_csv(source/'paired_predictions.csv',dtype={'mask':str})
    s=pd.read_csv(source/'metrics.csv');export=[]
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=240,bbox_inches='tight')
        plt.close(fig)
    def value(track,m,c,metric='MAE'):
        return float(s[(s.track==track)&(s.model==m)&(s.condition==c)][metric].iloc[0])
    fig,axes=plt.subplots(2,2,figsize=(10.3,6.2),layout='constrained',sharey='row')
    for col,(track,cs,x,labels,title) in enumerate([
        ('join',['join_3','join_6','full'],[3,6,12],['3','6','12 (full)'],'Held-out nodes: observation arrival'),
        ('fault',['full','fault_3','fault_6','outage'],[0,3,6,12],['0','3','6','12'],'Known nodes: interruption')]):
        for m,color,marker in zip(MODELS,COLORS,MARKERS):
            vals=np.array([value(track,m,c) for c in cs]);delta=vals-value(track,m,'full')
            for row,y in enumerate([vals,delta]):
                axes[row,col].plot(x,y,color=color,marker=marker,label=m,lw=2.4 if m=='ALFNet' else 1.5)
            export.extend(dict(figure=3,track=track,model=m,condition=c,MAE=v,change_from_full=d) for c,v,d in zip(cs,vals,delta))
        axes[0,col].set_title(title)
        for row in range(2):
            axes[row,col].set_xticks(x,labels);axes[row,col].grid(axis='y',alpha=.18)
            axes[row,col].set_xlabel('Recent observations' if col==0 else 'Missing recent steps')
        axes[1,col].axhline(0,color='.45',ls=':',lw=1)
    axes[0,0].set_ylabel('MAE (mph)');axes[1,0].set_ylabel('MAE change from full history (mph)')
    axes[0,0].legend(ncol=2,fontsize=8)
    save(fig,'trajectory_cohorts')
    fig,axes=plt.subplots(2,4,figsize=(13,6.1),sharex=True,sharey=True,layout='constrained')
    qs=[0,3,6,9,12]
    for row,track in enumerate(['fault','join']):
        for col,layout in enumerate(['recent','early','dispersed','middle_gap']):
            ax=axes[row,col]
            for m,color,marker in zip(MODELS,COLORS,MARKERS):
                cs=[f'q{q}_{"recent" if q in [0,12] else layout}' for q in qs]
                vals=[value(track,m,c) for c in cs]
                ax.plot(qs,vals,color=color,marker=marker,ms=4,label=m,lw=2.4 if m=='ALFNet' else 1.5)
                export.extend(dict(figure=4,track=track,model=m,condition=c,MAE=v) for c,v in zip(cs,vals))
            ax.set_title(layout.replace('_',' ').title());ax.set_xticks(qs);ax.grid(alpha=.18)
            if row==1:ax.set_xlabel('Available observations q')
        axes[row,0].set_ylabel(('Known nodes' if row==0 else 'Held-out nodes')+'\nMAE (mph)')
    axes[0,0].legend(ncol=2,fontsize=8)
    save(fig,'layout_cohorts')
    # Truth-only, post-hoc descriptive selection. All models see the same cases.
    # No errors or model predictions enter this selection rule.
    selection=[];case_rows=[]
    for track,condition in [('fault','full'),('join','join_3'),('fault','fault_6'),('fault','outage')]:
        base=df[(df.track==track)&(df.condition==condition)&(df.model=='ALFNet')].sort_values('pair_id')
        y=np.array([json.loads(v) for v in base.target])
        features=np.column_stack([y.mean(axis=1),y.std(axis=1),y[:,-1]-y[:,0]])
        center=np.median(features,axis=0);scale=np.percentile(features,75,axis=0)-np.percentile(features,25,axis=0)
        distance=np.abs((features-center)/np.maximum(scale,1e-6)).sum(axis=1)
        i=int(np.argmin(distance));pid=int(base.iloc[i].pair_id)
        selection.append(dict(track=track,condition=condition,pair_id=pid,node=int(base.iloc[i].node),
            feature_mean=float(features[i,0]),feature_std=float(features[i,1]),feature_change=float(features[i,2]),distance=float(distance[i])))
        case_rows.extend(df[(df.condition==condition)&(df.pair_id==pid)].to_dict('records'))
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.8),layout='constrained')
    for ax,sel in zip(axes.flat,selection):
        rows=[r for r in case_rows if r['pair_id']==sel['pair_id'] and r['condition']==sel['condition']]
        first=rows[0];x=np.arange(1,13)*5
        ax.plot(x,json.loads(first['target']),color='#222222',lw=2,label='Ground truth')
        for m,color,marker in zip(MODELS,COLORS,MARKERS):
            r=next(r for r in rows if r['model']==m)
            ax.plot(x,json.loads(r['prediction']),color=color,marker=marker,markevery=3,ms=3,
                lw=2 if m=='ALFNet' else 1.4,label=f'{m} ({r["MAE"]:.2f})')
        cohort='Held-out' if sel['track']=='join' else 'Known'
        ax.set_title(f'{cohort}: {sel["condition"].replace("_","-").title()} | node {sel["node"]}',pad=26)
        inset=ax.inset_axes([0,1.015,1,.045]);inset.imshow([[int(v) for v in first['mask']]],cmap='Blues',vmin=0,vmax=1,aspect='auto');inset.set_axis_off()
        ax.set_xlabel('Forecast horizon (minutes)');ax.set_ylabel('Speed (mph)');ax.grid(alpha=.15)
        ax.legend(fontsize=7,ncol=2,title='Model (MAE)',title_fontsize=7)
    save(fig,'cases_truth_center')
    pd.DataFrame(export).to_csv(out/'plot_values.csv',index=False)
    pd.DataFrame(case_rows).to_csv(out/'case_predictions.csv',index=False)
    (out/'selection.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),plan_sha256=plan['plan_sha256'],
        rule='post-hoc descriptive selection: nearest marginal-median truth mean/std/end-minus-start, L1 distance scaled by IQR; ties by pair_id; no prediction-error inputs',
        cases=selection),indent=2)+'\n')
    print(json.dumps(selection,indent=2))


if __name__=='__main__':main()
