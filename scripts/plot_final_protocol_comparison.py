"""Four-model comparison using existing full-test tables; no inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'paper/test_seed32_hparam_patch16_20260918/results_and_provenance.json'
    records=json.loads(source.read_text())['records']['METRLA']
    models=['ALFNet','UIGNN','MoGERNN','STBP']
    conditions=['Unseen','Steady','Join','Fault','Cold','Warm','Ready','Onset','Decay','Outage']
    colors=['#0072B2','#009E73','#D55E00','#CC79A7'];markers=['o','s','^','D']
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.1),layout='constrained');rows=[]
    for ax,metric in zip(axes,['MAE','RMSE','MAPE']):
        for model,color,marker in zip(models,colors,markers):
            values=[records[model]['table']['main' if i<4 else 'phase'][c][metric] for i,c in enumerate(conditions)]
            ax.plot(range(10),values,color=color,marker=marker,ms=4,lw=2 if model=='ALFNet' else 1.3,label=model)
            rows.extend(dict(model=model,condition=c,metric=metric,value=v) for c,v in zip(conditions,values))
        ax.axvline(3.5,color='.6',ls=':',lw=1)
        ax.set_xticks(range(10),conditions,rotation=50,ha='right')
        ax.set_ylabel(metric+(' (%)' if metric=='MAPE' else ' (mph)'))
        ax.set_title(metric);ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=2,fontsize=8)
    out=ROOT/'paper/final_support_20260922'
    for ext in ['png','pdf']:fig.savefig(out/f'canonical_test_comparison.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'canonical_test_comparison.csv',index=False)
    print(out/'canonical_test_comparison.png')


if __name__=='__main__':main()
