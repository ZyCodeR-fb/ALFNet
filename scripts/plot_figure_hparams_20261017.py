"""Identical OFAT grids on validation and test; no new selection."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.plot_advisor_experiments_20261012 import np,pd,plt,DS,LABEL,MS
OUT=ROOT/'paper/final_figures_20261017'
def main():
    val=pd.read_csv(ROOT/'paper/final_supplements_20261008/03_hyperparameters_validation.csv');val=val[(val.use=='正文单变量')&val.dataset.isin(DS)].copy()
    speed=pd.read_csv(ROOT/'paper/final_four_tables_20261001/04_hyperparameters_test.csv')
    flow=json.loads((ROOT/'paper/figure_hparams_test_20261017/results.json').read_text())['rows']
    test=val.copy();test['split']='test'
    for idx,r in test.iterrows():
        if r.dataset in ['PEMS04','PEMS08']:value=flow[r.run]['table'][r.section][r.condition][r.metric]
        else:
            matches=speed[(speed.dataset==r.dataset)&(speed.run==r.run)&(speed.section==r.section)&(speed.condition==r.condition)&(speed.metric==r.metric)]
            assert len(matches)>0 and matches.value.nunique()==1;value=matches.value.iloc[0]
        test.at[idx,'value']=value
    for split,df in [('validation',val),('test',test)]:
        dest=OUT/split;dest.mkdir(exist_ok=True,parents=True);df.to_csv(dest/'hyperparameters_cells.csv',index=False)
        hp=df[df.condition=='LifeAvg'];fig,axes=plt.subplots(4,4,figsize=(14,11),layout='constrained');values=[]
        for i,(d,label) in enumerate(zip(DS,LABEL)):
            for j,(axis,title) in enumerate([('capacity','Patch capacity'),('lr_multiplier','LR multiplier'),('dropout','Dropout'),('weight_decay','Weight decay')]):
                a=hp[(hp.dataset==d)&(hp.axis==axis)];anchor=a.anchor.iloc[0];ax=axes[i,j]
                for k,(m,color) in enumerate(zip(MS,['#0072B2','#D55E00','#009E73'])):
                    b=a[a.metric==m].sort_values(axis);ref=float(b[b.run==anchor].value.iloc[0]);vs=100*(b.value.to_numpy()/ref-1);y=np.arange(len(b))+(k-1)*.2
                    ax.hlines(y,0,vs,color=color,alpha=.4);ax.scatter(vs,y,color=color,s=23,label=m)
                    values.extend(dict(dataset=d,axis=axis,parameter=float(x),metric=m,change=float(v),anchor=anchor) for x,v in zip(b[axis],vs))
                ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(len(b)),[f'{x:g}' for x in b[axis]]);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
                if i==0:ax.set_title(title)
                if j==0:ax.set_ylabel(label,fontweight='bold')
                if i==3:ax.set_xlabel('LifeAvg error change (%)')
        axes[0,0].legend(ncol=3,fontsize=7,loc='lower right');fig.suptitle(split.title()+' | same fixed OFAT anchors')
        for ext in ('png','pdf','svg'):fig.savefig(dest/('fig5_hyperparameters.'+ext),dpi=200,bbox_inches='tight')
        plt.close(fig);pd.DataFrame(values).to_csv(dest/'fig5_values.csv',index=False)
if __name__=='__main__':main()
