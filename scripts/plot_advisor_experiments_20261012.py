"""Reproducible figures/tables from saved final metrics and paired predictions."""
import json,sys
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/advisor_experiments_20261012'
DS=['METRLA','PEMSBAY','PEMS04','PEMS08'];LABEL=['METR-LA','PEMS-BAY','PEMS04','PEMS08']
MS=['MAE','RMSE','MAPE'];BASE=['UIGNN','GinAR','MoGERNN','SLPF','STGCN','STID','PatchSTG','STBP']
COL={'ALFNet':'#0072B2','UIGNN':'#D55E00','MoGERNN':'#009E73','STBP':'#9467BD','PatchSTG':'#9467BD'}
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def read(p):return json.loads((ROOT/p).read_text())
def wide_table(values, names):
    """Dataset-grouped HTML header; rank using unrounded metrics."""
    lines=['<table>', '  <thead>', '    <tr>', '      <th rowspan="2" scope="col">模型</th>']
    lines += [f'      <th colspan="3" scope="colgroup">{label}</th>' for label in LABEL]
    lines += ['    </tr>', '    <tr>']
    lines += ['      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>' for _ in DS]
    lines += ['    </tr>', '  </thead>', '  <tbody>']
    for name in names:
        lines += ['    <tr>', f'      <td>{escape(name)}</td>']
        for d in DS:
            cells=[]
            for m in MS:
                ranks=sorted(set(values[d][n][m] for n in names))
                value=values[d][name][m];s=f'{value:.3f}'
                if value==ranks[0]:s=f'<b>{s}</b>'
                elif value==ranks[1]:s=f'<u>{s}</u>'
                cells.append(f'<td>{s}</td>')
            lines.append('      '+''.join(cells))
        lines += ['    </tr>']
    return '\n'.join(lines+['  </tbody>', '</table>'])+'\n'
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
def main():
    ab=read('paper/all_ablations_test_20261010/results_and_audit.json')['rows']
    old=read('paper/all_final_test_20261001/all_results_and_provenance.json')
    tables={d:{m:old['rows'][d+'_'+m]['table'] for m in BASE} for d in DS}
    for d in DS:tables[d]['ALFNet']=ab[d+'_Full']['table']
    cells=[]
    for d,models in tables.items():
        for model,t in models.items():
            for g in ['main','phase']:
                for c,r in t[g].items():
                    for m in MS:cells.append(dict(dataset=d,model=model,section=g,condition=c,metric=m,value=r[m]))
    pd.DataFrame(cells).to_csv(OUT/'final_comparison_test.csv',index=False)
    gains={};ranks={}
    for d,label in zip(DS,LABEL):
        t=tables[d]
        gains[d]={m:100*(1-t['ALFNet']['main']['LifeAvg'][m]/min(t[b]['main']['LifeAvg'][m] for b in BASE)) for m in MS}
        ranks[d]=sum(t['ALFNet'][g][c][m]<min(t[b][g][c][m] for b in BASE) for g in ['main','phase'] for c in t['ALFNet'][g] for m in MS)
    (OUT/'table1.md').write_text(wide_table({d:{n:t['main']['LifeAvg'] for n,t in rows.items()} for d,rows in tables.items()},BASE+['ALFNet']))
    conditions=['Unseen','Steady','Join','Fault','LifeAvg','Cold','Warm','Ready','Onset','Decay','Outage']
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained');heat=[]
    for ax,d,label in zip(axes.flat,DS,LABEL):
        a=np.array([[100*(1-tables[d]['ALFNet']['main' if c in conditions[:5] else 'phase'][c][m]/min(tables[d][b]['main' if c in conditions[:5] else 'phase'][c][m] for b in BASE)) for m in MS] for c in conditions])
        heat.extend(dict(dataset=d,condition=c,metric=m,gain=a[i,j]) for i,c in enumerate(conditions) for j,m in enumerate(MS))
        im=ax.imshow(a,cmap='RdBu',norm=TwoSlopeNorm(vmin=-40,vcenter=0,vmax=40),aspect='auto')
        for (i,j),v in np.ndenumerate(a):ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(v)>28 else '#17252f')
        ax.set(xticks=range(3),xticklabels=MS,yticks=range(11),yticklabels=conditions,title=f'{label}   |   {ranks[d]}/33 best')
        ax.axhline(4.5,color='white',lw=2)
    fig.colorbar(im,ax=axes,shrink=.8,label='Error reduction vs. best baseline in each cell (%)')
    save(fig,'fig2_condition_stage');pd.DataFrame(heat).to_csv(OUT/'fig2_values.csv',index=False)
    variants={'Full':'ALFNet','no_age':'w/o State Age','static_router':'w/o Adaptive Routing','local_context':'w/o Spatial Context','no_temporal_film':'w/o ALFTemporal–FiLM','no_refiner':'w/o Graph Refinement','pointwise_refiner':'Pointwise Refinement','stop_temporal_recon_gradient':'Stop Temporal→Recon Gradient'}
    selected={}
    for d,label in zip(DS,LABEL):
        selected[d]={name:next(r['table']['main']['LifeAvg'] for k,r in ab.items() if (k==d+'_Full' if key=='Full' else '_'+d+'_'+key==k[2:])) for key,name in variants.items()}
    (OUT/'table2.md').write_text(wide_table(selected,list(variants.values())))
    hp=pd.read_csv(ROOT/'paper/final_supplements_20261008/03_hyperparameters_validation.csv');hp=hp[(hp.use=='正文单变量')&(hp.condition=='LifeAvg')]
    fig,axes=plt.subplots(4,4,figsize=(14,11),layout='constrained');hv=[]
    for i,(d,label) in enumerate(zip(DS,LABEL)):
        for j,(axis,title) in enumerate([('capacity','Patch capacity'),('lr_multiplier','LR multiplier'),('dropout','Dropout'),('weight_decay','Weight decay')]):
            a=hp[(hp.dataset==d)&(hp.axis==axis)];anchor=a.anchor.iloc[0];ax=axes[i,j]
            for k,(m,color) in enumerate(zip(MS,['#0072B2','#D55E00','#009E73'])):
                b=a[a.metric==m].sort_values(axis);reference=float(b[b.run==anchor].value.iloc[0]);values=100*(b.value.to_numpy()/reference-1);y=np.arange(len(b))+(k-1)*.20
                ax.hlines(y,0,values,color=color,alpha=.4,lw=1.5);ax.scatter(values,y,color=color,s=23,label=m,zorder=3)
                hv.extend(dict(dataset=d,axis=axis,parameter=float(p),metric=m,change=float(v),anchor=anchor) for p,v in zip(b[axis],values))
            ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(len(b)),[f'{x:g}' for x in b[axis]]);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
            if i==0:ax.set_title(title)
            if j==0:ax.set_ylabel(label,fontweight='bold')
            if i==3:ax.set_xlabel('LifeAvg error change (%)')
    axes[0,0].legend(ncol=3,fontsize=7,loc='lower right');save(fig,'fig5_hyperparameters');pd.DataFrame(hv).to_csv(OUT/'fig5_values.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(dict(lifeavg_gain_vs_best=gains,first_cells=ranks),indent=2))
    if not all((OUT/d/'COMPLETE.json').exists() for d in DS):return
    frames={d:pd.concat([pd.read_csv(f) for f in (OUT/d).glob('*.csv')],ignore_index=True) for d in DS}
    stats=[];caseinfo=[]
    # Dot plots distinguish paired lifecycle interventions from continuous time.
    fig,axes=plt.subplots(4,2,figsize=(12,11),layout='constrained')
    for i,(d,label) in enumerate(zip(DS,LABEL)):
        df=frames[d];models=list(dict.fromkeys(df.model));models=sorted(models,key=lambda m:m!='ALFNet')
        for j,conds in enumerate([['join_3','join_6','full'],['full','fault_3','fault_6','outage']]):
            ax=axes[i,j]
            for k,m in enumerate(models):
                vals=[df[(df.model==m)&(df.condition==c)].MAE.mean() for c in conds]
                ax.scatter(vals,np.arange(len(conds))+(k-1.5)*.16,label=m,color=COL[m],s=43 if m=='ALFNet' else 24,marker='D' if m=='ALFNet' else 'o')
                stats.extend(dict(dataset=d,model=m,condition=c,MAE=float(v)) for c,v in zip(conds,vals))
            ax.set_yticks(range(len(conds)),['Recent 3','Recent 6','Full 12'] if j==0 else ['No interruption','Missing 3','Missing 6','Missing 12']);ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_xlabel('MAE (mph)' if i<2 else 'MAE (flow units)')
            ax.set_title(label+' | '+('Arrival' if j==0 else 'Interruption'));ax.legend(fontsize=8,ncol=2)
        lo=min(ax.get_xlim()[0] for ax in axes[i]);hi=max(ax.get_xlim()[1] for ax in axes[i])
        for ax in axes[i]:ax.set_xlim(lo,hi)
    save(fig,'fig3_trajectory')
    # Grouped bars: actual error differences at fixed q, each dataset its own scale.
    fig,axes=plt.subplots(4,4,figsize=(15,11),layout='constrained')
    layoutstats=[]
    for i,(d,label) in enumerate(zip(DS,LABEL)):
        df=frames[d];models=sorted(df.model.unique(),key=lambda m:m!='ALFNet')
        for j,layout in enumerate(['recent','early','dispersed','middle_gap']):
            ax=axes[i,j]
            for k,m in enumerate(models):
                vals=[df[(df.model==m)&(df.condition==f'q{q}_{layout}')].MAE.mean() for q in [3,6,9]]
                ax.bar(np.arange(3)+(k-1.5)*.19,vals,width=.18,color=COL[m],label=m,alpha=1 if m=='ALFNet' else .7)
                layoutstats.extend(dict(dataset=d,model=m,layout=layout,q=q,MAE=float(v)) for q,v in zip([3,6,9],vals))
            ax.set_xticks(range(3),[3,6,9]);ax.grid(axis='y',alpha=.15)
            if i==0:ax.set_title(layout.replace('_',' ').title())
            if j==0:ax.set_ylabel(label+'\nMAE'+(' (mph)' if i<2 else ' (flow)'))
            if i==3:ax.set_xlabel('Available observations q')
            if j==3:ax.legend(fontsize=7,ncol=2)
        ymax=max(ax.get_ylim()[1] for ax in axes[i])
        for ax in axes[i]:ax.set_ylim(0,ymax)
    save(fig,'fig4_layout')
    # Complete cohort ECDF + openly selected median positive-gain illustrative case.
    fig,axes=plt.subplots(4,2,figsize=(13,12),layout='constrained')
    for i,(d,label) in enumerate(zip(DS,LABEL)):
        df=frames[d];out=df[df.condition=='outage'];models=sorted(out.model.unique(),key=lambda m:m!='ALFNet')
        pivot=out.pivot(index='pair_id',columns='model',values='MAE');baseline=min([m for m in models if m!='ALFNet'],key=lambda m:pivot[m].mean())
        gain=pivot[baseline]-pivot.ALFNet;positive=gain[gain>0]
        pick=int((positive-positive.median()).abs().idxmin()) if len(positive) else int(gain.abs().idxmin())
        info=dict(dataset=d,baseline=baseline,pair_id=pick,win_fraction=float((gain>0).mean()),median_gain=float(gain.median()),case_gain=float(gain.loc[pick]),case_rule='nearest median positive MAE gain; illustrative, not random')
        for m in models:
            errors=np.sort(out[out.model==m].MAE.to_numpy());axes[i,0].step(errors,np.arange(1,65)/64,where='post',color=COL[m],label=m,lw=2.4 if m=='ALFNet' else 1.5)
            row=out[(out.model==m)&(out.pair_id==pick)].iloc[0];pred=np.array(json.loads(row.prediction));truth=np.array(json.loads(row.target))
            axes[i,1].plot(np.arange(1,13)*5,pred,color=COL[m],label=m,lw=2.4 if m=='ALFNet' else 1.3,alpha=1 if m=='ALFNet' else .8)
            if m=='ALFNet':
                axes[i,1].plot(np.arange(1,13)*5,truth,color='#222',label='Ground truth',lw=2,marker='o',ms=3)
                axes[i,1].fill_between(np.arange(1,13)*5,pred,truth,color=COL[m],alpha=.12)
                info.update(node=int(row.node),origin=int(row.origin))
        axes[i,0].set(title=f'{label} | all 64 windows',xlabel='Window MAE (mph)' if i<2 else 'Window MAE (flow)',ylabel='Cumulative fraction',ylim=(0,1.03),xlim=(0,None));axes[i,0].grid(alpha=.2);axes[i,0].legend(fontsize=8,loc='lower right')
        axes[i,1].set(title=f'{label} | illustrative positive-gain case',xlabel='Forecast horizon (minutes)',ylabel='Speed (mph)' if i<2 else 'Flow');axes[i,1].grid(alpha=.2);axes[i,1].legend(fontsize=7,ncol=3)
        caseinfo.append(info)
    save(fig,'fig6_cases')
    pd.DataFrame(stats).drop_duplicates().to_csv(OUT/'fig3_values.csv',index=False)
    pd.DataFrame(layoutstats).to_csv(OUT/'fig4_values.csv',index=False)
    (OUT/'case_selection.json').write_text(json.dumps(caseinfo,indent=2))
    print(json.dumps(caseinfo,indent=2))

if __name__=='__main__':main()
