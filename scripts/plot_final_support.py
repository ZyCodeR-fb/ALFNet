"""Publication plots from the frozen four-model paired validation evaluation."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_final_support import PLAN, OUT, MODELS

COLORS={'ALFNet':'#0072B2','UIGNN':'#009E73','MoGERNN':'#D55E00','STBP':'#CC79A7'}
MARKERS={'ALFNet':'o','UIGNN':'s','MoGERNN':'^','STBP':'D'}
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})


def main():
    p=json.loads(PLAN.read_text());source=OUT/p['plan_sha256']
    assert (source/'COMPLETE.json').exists()
    dest=ROOT/'paper/final_support_20260922';dest.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(source/'metrics.csv')
    df.to_csv(dest/'metrics.csv',index=False)
    def save(fig,name):
        for ext in ('png','pdf'): fig.savefig(dest/f'{name}.{ext}',dpi=240,bbox_inches='tight')
        plt.close(fig)
    conditions=['full','join_3','join_6','fault_3','fault_6','outage']
    labels=['Full history','Join-3','Join-6','Fault-3','Fault-6','Outage']
    t=df[(df.track=='join')&(df.experiment=='trajectory')]
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    masks=np.array([[1]*12,[0]*9+[1]*3,[0]*6+[1]*6,[1]*9+[0]*3,[1]*6+[0]*6,[0]*12])
    axes[0].imshow(masks,cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
    axes[0].set_yticks(range(6),labels);axes[0].set_xticks([0,3,6,9,11],[-12,-9,-6,-3,-1])
    axes[0].set_xlabel('Historical step');axes[0].set_title('(a) Observed history')
    for ax,metric,title in zip(axes[1:],['MAE','RMSE'],['(b) Mean absolute error','(c) Root mean squared error']):
        for m in MODELS:
            vals=[t[(t.model==m)&(t.condition==c)][metric].iloc[0] for c in conditions]
            ax.plot(range(6),vals,color=COLORS[m],marker=MARKERS[m],label=m,lw=2 if m=='ALFNet' else 1.4)
        ax.set_xticks(range(6),labels,rotation=35,ha='right');ax.set_ylabel(metric+' (mph)');ax.set_title(title)
        ax.grid(axis='y',alpha=.18)
    axes[1].legend(fontsize=8,ncol=2)
    save(fig,'trajectory')
    s=df[(df.track=='all')&(df.experiment=='support')]
    fig,axes=plt.subplots(1,4,figsize=(13,3.4),sharey=True,layout='constrained')
    qs=[0,3,6,9,12]
    for i,(ax,layout) in enumerate(zip(axes,['recent','early','dispersed','middle_gap'])):
        for m in MODELS:
            vals=[s[(s.model==m)&(s.q==q)&(s.layout==('all' if q in (0,12) else layout))].MAE.iloc[0] for q in qs]
            ax.plot(qs,vals,color=COLORS[m],marker=MARKERS[m],label=m,lw=2 if m=='ALFNet' else 1.4)
        ax.set_title(f'({chr(97+i)}) '+layout.replace('_',' ').title());ax.set_xticks(qs)
        ax.set_xlabel('Available observations q');ax.grid(alpha=.18)
    axes[0].set_ylabel('MAE (mph)');axes[0].legend(fontsize=8)
    save(fig,'support_layout')
    cases=json.loads((source/'cases.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,6.6),layout='constrained')
    for i,ax in enumerate(axes.flat):
        group=[r for r in cases if r['pair_id']==i];r=group[0]
        mask=np.array([int(x) for x in r['mask']],bool);hist=np.array(r['history'])
        ax.plot(np.arange(-12,0)[mask],hist[mask],'ko',ms=3,label='Observed history')
        ax.plot(np.arange(12),json.loads(r['target']),color='black',lw=2,label='Ground truth')
        for m in MODELS:
            rr=next(x for x in group if x['model']==m)
            ax.plot(np.arange(12),json.loads(rr['prediction']),color=COLORS[m],label=m,
                    marker=MARKERS[m],markevery=3,ms=3,lw=1.6)
        for h in np.where(~mask)[0]: ax.axvspan(h-12-.5,h-12+.5,color='.8',alpha=.4,lw=0)
        ax.axvline(-.5,color='.4',ls=':',lw=.8)
        ax.set_title(f'({chr(97+i)}) {r["condition"].replace("_","-").title()}; node {r["node"]}; q={r["q"]}')
        ax.set_xlabel('Step relative to forecast start');ax.set_ylabel('Speed (mph)');ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(fontsize=7,ncol=2)
    save(fig,'cases')
    (dest/'provenance.json').write_text(json.dumps(dict(plan_sha256=p['plan_sha256'],source=str(source.relative_to(ROOT)),
        split='validation',models=list(MODELS),case_selection=p['case_rule'],representatives=p['representative_rule']),indent=2)+'\n')
    print(dest)


if __name__=='__main__': main()
