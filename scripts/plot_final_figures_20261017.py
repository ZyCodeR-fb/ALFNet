"""Render consistent final-weight validation/test mechanism panels."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.plot_advisor_experiments_20261012 import np,pd,plt,DS,LABEL,MS,BASE,COL
OUT=None
def save(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(OUT/(name+'.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
def graph(split):
    global OUT
    OUT=ROOT/'paper/final_figures_20261017'/split
    assert all((OUT/d/'COMPLETE.json').exists() for d in DS)
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

if __name__=='__main__':
    for split in ('validation','test'):graph(split)
