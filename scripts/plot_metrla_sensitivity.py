"""Render all completed validation sensitivity points, with no test inference."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import check,output,read,SPECS


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=check(); assert read(output(p)/'COMPLETE.json')['runs']==9
    out=ROOT/'paper/hparams_metrla_20260920'; r=read(out/'results.json')
    assert r['plan_sha256']==p['plan_sha256'] and len(r['tables'])==10
    axes_specs=[('Patch capacity',0,8,['patch7','patch10','patch12']),
                ('Learning-rate multiplier',1,1.,['lr_half','lr_double']),
                ('Attention dropout',2,.1,['drop05','drop20']),
                ('Weight decay',3,1e-4,['wd_low','wd_high'])]
    fig,axes=plt.subplots(1,4,figsize=(16,3.8),constrained_layout=True)
    metrics=('MAE','RMSE','MAPE'); baseline=r['tables']['baseline_reused']['main']['LifeAvg']
    exported=[]
    for ax,(label,index,default,keys) in zip(axes,axes_specs):
        points=sorted([(default,'baseline_reused')]+[(SPECS[k][index],k) for k in keys])
        for metric in metrics:
            ys=[100*(r['tables'][k]['main']['LifeAvg'][metric]/baseline[metric]-1) for x,k in points]
            ax.plot([x for x,k in points],ys,marker='o',label=metric)
            exported.extend([{'factor':label,'value':x,'run':k,'metric':metric,'relative_change_percent':y} for (x,k),y in zip(points,ys)])
        if index in (1,3):ax.set_xscale('log')
        ax.axhline(0,color='gray',lw=.7); ax.axvline(default,color='gray',ls=':',lw=.7)
        ax.set_xticks([x for x,k in points],[str(x) for x,k in points]);ax.set_xlabel(label)
        ax.set_ylabel('Validation LifeAvg change (%)'); ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('METR-LA, seed32; one factor at a time; lower is better')
    for ext in ('png','pdf'):fig.savefig(out/f'sensitivity.{ext}',dpi=220)
    (out/'plot_data.json').write_text(json.dumps(exported,indent=2)+'\n')
    lines=['# METR-LA validation敏感性结果','',
           '固定最终Focal架构；单seed32，无误差条；本结果不自动替换最终模型。','',
           '| Run | MAE | RMSE | MAPE (%) | best epoch | epochs |','|---|---:|---:|---:|---:|---:|']
    for key,table in r['tables'].items():
        vals=table['main']['LifeAvg'];meta=r['provenance'][key]
        lines.append('| '+key+' | '+' | '.join(f'{vals[m]:.6f}' for m in metrics)+f" | {meta.get('best_epoch','reused')} | {meta.get('epochs_run','reused')} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
