"""Publication figures from existing CSVs; never runs a forecasting model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})


def save(fig,directory,name):
    directory.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(directory/f'{name}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)


def condition_figure(figures):
    source=ROOT/'outputs_v4/reports/five_dataset_seed32_nine_models_validation/05df96f892b8ccb532f081c4ece8a389c035849af3b50a264de5669f220cd6d4/tables.json'
    tables=json.loads(source.read_text())['tables']['METRLA']
    groups=[['Unseen','Steady','Join','Fault'],
            ['Cold','Warm','Ready','Onset','Decay','Outage']]
    fig,axes=plt.subplots(1,2,figsize=(8.8,4.2),layout='constrained')
    exported=[]
    for ax,labels,title in zip(axes,groups,['(a) Main conditions','(b) Transition stages']):
        values=[]
        for c in labels:
            part='main' if c in groups[0] else 'phase'
            row=[]
            for m in ('MAE','RMSE','MAPE'):
                full=tables[part]['ALFNet'][c][m];base=tables[part]['MoGERNN'][c][m]
                gain=100*(base-full)/base;row.append(gain)
                exported.append(dict(condition=c,metric=m,ALFNet=full,MoGERNN=base,gain_percent=gain))
            values.append(row)
        a=np.array(values)
        im=ax.imshow(a,cmap='RdBu',vmin=-21,vmax=21,aspect='auto')
        ax.set_xticks(range(3),['MAE','RMSE','MAPE']);ax.set_yticks(range(len(labels)),labels)
        ax.set_title(title)
        for (i,j),x in np.ndenumerate(a):
            ax.text(j,i,f'{x:+.1f}%',ha='center',va='center',
                    color='white' if abs(x)>13 else 'black')
    fig.colorbar(im,ax=axes,label='Error reduction vs. MoGERNN (%)',shrink=.8)
    save(fig,figures,'METRLA_condition_gains')
    pd.DataFrame(exported).to_csv(figures/'METRLA_condition_gains_vs_MoGERNN.csv',index=False)


def trajectory_figure(data,figures,track='join'):
    summary=pd.read_csv(data/'metrics.csv'); rows=pd.read_csv(data/'paired_predictions.csv',dtype={'mask':str})
    conditions=['full','join_3','join_6','fault_3','fault_6','outage']
    labels=['Full','Join-3','Join-6','Fault-3','Fault-6','Outage']
    summary=summary[(summary.track==track)&(summary.experiment=='trajectory')]
    assert summary.pairs.eq(64).all()
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    masks=np.array([[int(x) for x in rows[rows.condition==c].iloc[0]['mask']] for c in conditions])
    axes[0].imshow(masks,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    axes[0].set_yticks(range(6),labels);axes[0].set_xticks([0,3,6,9,11],[-12,-9,-6,-3,-1])
    axes[0].set_xlabel('Historical step');axes[0].set_title('(a) Focal-node support')
    colors={'ALFNet':'#0072B2','MoGERNN':'#D55E00','Mask-only':'#009E73'}
    x=np.arange(6)
    for model in ('ALFNet','MoGERNN'):
        vals=[summary[(summary.model==model)&(summary.condition==c)].MAE.iloc[0] for c in conditions]
        axes[1].plot(x,vals,'o-',label=model,color=colors[model])
    axes[1].set_xticks(x,labels,rotation=35,ha='right');axes[1].set_ylabel('MAE')
    axes[1].set_title('(b) External baseline comparison');axes[1].legend(fontsize=9)
    gains=[]
    for c in conditions:
        control=summary[(summary.model=='Mask-only')&(summary.condition==c)].MAE.iloc[0]
        full=summary[(summary.model=='ALFNet')&(summary.condition==c)].MAE.iloc[0]
        gains.append(100*(control-full)/control)
    axes[2].bar(x,gains,width=.6,color=[colors['ALFNet'] if g>=0 else '#999999' for g in gains])
    axes[2].axhline(0,color='black',lw=.8)
    axes[2].set_xticks(x,labels,rotation=35,ha='right');axes[2].set_ylabel('Full MAE reduction vs. Mask-only (%)')
    axes[2].set_title('(c) Internal mechanism control')
    cohort='Held-out nodes' if track=='join' else 'Training-known nodes'
    fig.suptitle(cohort+'; 64 matched windows',fontsize=11)
    name='METRLA_trajectory_intervention' if track=='join' else 'METRLA_trajectory_known_appendix'
    save(fig,figures,name)


def support_figure(data,figures):
    df=pd.read_csv(data/'metrics.csv');df=df[(df.track=='all')&(df.experiment=='support')]
    layouts=['recent','early','dispersed','middle_gap'];qs=[0,3,6,9,12]
    def matrix(model):
        return np.array([[df[(df.model==model)&(df.q==q)&(df.layout==('all' if q in (0,12) else layout))].MAE.iloc[0]
                          for q in qs] for layout in layouts])
    a,b=matrix('ALFNet'),matrix('MoGERNN');gain=100*(b-a)/b
    fig,axes=plt.subplots(1,3,figsize=(12,3.2),layout='constrained')
    lo=min(a.min(),b.min());hi=max(a.max(),b.max())
    for ax,values,title in zip(axes[:2],[a,b],['(a) ALFNet MAE','(b) MoGERNN MAE']):
        im=ax.imshow(values,cmap='YlOrRd',vmin=lo,vmax=hi,aspect='auto')
        ax.set_title(title)
        for (i,j),v in np.ndenumerate(values):
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v>lo+.65*(hi-lo) else 'black')
    fig.colorbar(im,ax=axes[:2],shrink=.85,label='MAE')
    lim=max(abs(gain.min()),abs(gain.max()))
    im=axes[2].imshow(gain,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
    for (i,j),v in np.ndenumerate(gain):
        axes[2].text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=9,
                     color='white' if abs(v)>.65*lim else 'black')
    axes[2].set_title('(c) ALFNet relative gain')
    fig.colorbar(im,ax=axes[2],shrink=.85,label='MAE reduction (%)')
    for ax in axes:
        ax.set_xticks(range(5),qs);ax.set_xlabel('Available historical observations q')
        ax.set_yticks(range(4),['Recent','Early','Dispersed','Middle gap'])
    save(fig,figures,'METRLA_q_temporal_layout')


def export_gains(data):
    df=pd.read_csv(data/'metrics.csv');rows=[]
    for _,r in df[df.model!='MoGERNN'].iterrows():
        b=df[(df.model=='MoGERNN')&(df.condition==r.condition)&(df.track==r.track)].iloc[0]
        row=r.to_dict()
        for m in ['MAE','RMSE','MAPE']:
            row[m+'_baseline']=b[m];row[m+'_gain_percent']=100*(b[m]-r[m])/b[m]
        rows.append(row)
    pd.DataFrame(rows).to_csv(data/'relative_gains.csv',index=False)


def case_figure(figures):
    records=json.loads((ROOT/'outputs_v4/reports/support_analysis_v1/case_records.json').read_text())
    records=[r for r in records if r['dataset']=='METRLA']
    fig,axes=plt.subplots(4,2,figsize=(10,10),layout='constrained')
    for ax,r in zip(axes.flat,records):
        mask=np.array(r['mask'],dtype=bool);history=np.array(r['history'])
        ax.plot(np.arange(-12,0)[mask],history[mask],'ko',ms=3,label='Observed history')
        ax.plot(np.arange(12),r['ground_truth'],'k-',label='Ground truth')
        for name,color in [('ALFNet','#0072B2'),('MoGERNN','#D55E00')]:
            ax.plot(np.arange(12),r['predictions'][name],label=name,color=color)
        for h in np.where(~mask)[0]:
            ax.axvspan(h-12-.45,h-12+.45,color='gray',alpha=.15,lw=0)
        ax.axvline(-.5,color='gray',ls=':',lw=.8)
        label={'zero_history':'Unseen','steady':'Steady'}.get(r['condition'],r['condition'])
        ax.set_title(f'{label}; node {r["node"]}; q={mask.sum()}')
        ax.set_xlabel('Steps relative to forecast start');ax.set_ylabel('Traffic speed')
    axes.flat[0].legend(fontsize=8,ncol=2)
    save(fig,figures,'METRLA_cases_appendix')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',default=str(ROOT/'outputs_v4/reports/observation_support_v1'))
    parser.add_argument('--figures',default=str(ROOT/'paper/figures'));parser.add_argument('--conditions-only',action='store_true')
    parser.add_argument('--revision-figures-only',action='store_true',help='Redraw only Figures 2/3 and known-node appendix')
    args=parser.parse_args();data=Path(args.data);figures=Path(args.figures)
    condition_figure(figures)
    if not args.revision_figures_only:
        case_figure(figures)
    if not args.conditions_only:
        assert (data/'COMPLETE').exists()
        trajectory_figure(data,figures);trajectory_figure(data,figures,track='fault')
        if not args.revision_figures_only:
            support_figure(data,figures);export_gains(data)
    print(figures)
