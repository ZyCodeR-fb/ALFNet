"""Audit saved paired predictions and publish split-complete figure summaries."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_metrla_hparams import read,write,sha
from scripts.plot_advisor_experiments_20261012 import pd,np,plt,DS,LABEL,MS,BASE
OUT=ROOT/'paper/final_figures_20261017'
def main():
    from matplotlib.colors import TwoSlopeNorm
    from scripts.package_supplements_20261008 import book_from_cells
    plan=read(OUT/'plan.json');tables=read(ROOT/'paper/pems08_wd_test_20261012/selected_tables.json');abl=read(ROOT/'paper/pems08_wd_ablation_test_20261012/four_dataset_results.json')['tables']
    summary={};checks=[]
    for split in ('validation','test'):
        summary[split]={};dest=OUT/split;fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained');heat=[];cells=[]
        for ax,d,label in zip(axes.flat,DS,LABEL):
            t=tables[split][d];cols=[(g,c) for g in ('main','phase') for c in t['ALFNet'][g]]
            a=np.array([[100*(1-t['ALFNet'][g][c][m]/min(t[b][g][c][m] for b in BASE)) for m in MS] for g,c in cols]);im=ax.imshow(a,cmap='RdBu',norm=TwoSlopeNorm(vmin=-40,vcenter=0,vmax=40),aspect='auto')
            for (i,j),v in np.ndenumerate(a):ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(v)>28 else '#17252f')
            ax.set(xticks=range(3),xticklabels=MS,yticks=range(11),yticklabels=[c for g,c in cols],title=f'{label} | {(a>0).sum()}/33 best');ax.axhline(4.5,color='white',lw=2)
            heat.extend(dict(dataset=d,section=g,condition=c,metric=m,gain=float(a[i,j])) for i,(g,c) in enumerate(cols) for j,m in enumerate(MS))
            frames=[];ref=None
            for model,e in plan['entries'][d].items():
                rp=dest/d;au=read(rp/(model+'_audit.json'));assert au['checkpoint_sha256']==e['checkpoint_sha256'] and au['split']==split and au['teacher_forcing']==0
                df=pd.read_csv(rp/(model+'.csv'));assert len(df)==1280
                pairs=df[['pair_id','origin','node','condition','target']].sort_values(['pair_id','condition']).reset_index(drop=True)
                if ref is None:ref=pairs
                else:assert pairs.equals(ref),(split,d,model,'paired targets')
                truth=np.array([json.loads(x) for x in df.target]);pred=np.array([json.loads(x) for x in df.prediction]);err=pred-truth
                for m,recomputed in [('MAE',abs(err).mean(1)),('MSE',(err**2).mean(1)),('MAPE',(abs(err)/np.maximum(abs(truth),1e-6)).mean(1)*100)]:assert np.allclose(df[m],recomputed,rtol=2e-6,atol=2e-5),(d,model,m)
                frames.append(df);checks.append(dict(split=split,dataset=d,model=model,rows=1280,passed=True,checkpoint_sha256=e['checkpoint_sha256']))
                if split=='validation' and (d!='PEMS08' or model!='ALFNet'):
                    old=pd.read_csv(ROOT/'paper/advisor_experiments_20261012'/d/(model+'.csv'));assert np.allclose(old.MAE,df.MAE,rtol=0,atol=1e-10),'unchanged checkpoint replay'
            allf=pd.concat(frames);p=allf.groupby(['condition','model']).MAE.mean().unstack();best=p.drop(columns='ALFNet').min(axis=1);traj=['join_3','join_6','full','fault_3','fault_6','outage'];layouts=[c for c in p.index if c.startswith(('q3_','q6_','q9_'))]
            outage=allf[allf.condition=='outage'].pivot(index='pair_id',columns='model',values='MAE');base=min((m for m in outage if m!='ALFNet'),key=lambda m:outage[m].mean())
            summary[split][d]=dict(first33=int((a>0).sum()),trajectory_first=int((p.loc[traj,'ALFNet']<best[traj]).sum()),layout_first=int((p.loc[layouts,'ALFNet']<best[layouts]).sum()),outage_win_fraction=float((outage.ALFNet<outage[base]).mean()),outage_p90={m:float(outage[m].quantile(.9)) for m in outage},outage_means=p.loc['outage'].to_dict())
            for name,tb in abl[split][d].items():
                if name=='w/o Graph Refinement':continue
                for g in ('main','phase'):
                    for c,row in tb[g].items():
                        for m in MS:cells.append(dict(dataset=d,label=name,section=g,condition=c,metric=m,value=row[m]))
        fig.colorbar(im,ax=axes,shrink=.8,label='Error reduction vs. best baseline in each cell (%)');fig.suptitle(split.title())
        for ext in ('png','pdf','svg'):fig.savefig(dest/('fig2_condition_stage.'+ext),dpi=200,bbox_inches='tight')
        plt.close(fig);pd.DataFrame(heat).to_csv(dest/'fig2_values.csv',index=False)
        book_from_cells(dest/'four_ablations.xlsx',cells,['Four displayed ablations; omitted Graph Refinement remains in complete historical results.'],['dataset','section'],['label']);pd.DataFrame(cells).to_csv(dest/'four_ablations.csv',index=False)
    write(OUT/'verification.json',dict(passed=True,checks=checks,predictions=40960,unchanged_validation_replays=15));write(OUT/'split_summary.json',summary)
    lines=['# 最终图表核对与双划分结果（2026-10-17）','','全部模型使用当前最终权重。图2复用完整33格结果；图3/4/6各划分独立固定抽取64个已知节点窗口，保持选择规则与三类代表一致；图5保留原单因素锚点。没有训练或更换模型。','', '| 数据集 | validation 图3最优/6 | test 图3最优/6 | validation 图4最优/12 | test 图4最优/12 |','|---|---:|---:|---:|---:|']
    for d in DS:
        v=summary['validation'][d];t=summary['test'][d];lines.append(f'| {d} | {v["trajectory_first"]} | {t["trajectory_first"]} | {v["layout_first"]} | {t["layout_first"]} |')
    lines.extend(['','图3统计full端点只计一次。test图3更强，但PEMS-BAY图4由8/12降至5/12，因此不存在各图各数据集都更优的单一划分。正文依冻结计划统一保留validation机制分析，test作为配套稳健性展示；不逐图拼选更好的划分。test已被观察，不能描述为未受开发反馈影响的独立确认。','','正文表2删除Graph Refinement后按剩余Full+四项重排粗体/下划线。原始结果与反向证据保留在[五项完整记录](../pems08_wd_ablation_test_20261012/four_dataset_test.xlsx)，子集不能证明所有设计均必要。PEMS04 Full未变，此前修改的是去Age行；PEMS08 Full由WD1e-4更换为1e-3，现已更新相关推理。','','验证：32组模型×数据集×划分，共40960条预测；目标配对、三指标重算、权重SHA、teacher forcing为0已核验。15组未改权重validation复现一致。','','## 图像入口'])
    for n,title in [(2,'条件与阶段'),(3,'轨迹'),(4,'布局'),(5,'超参数'),(6,'案例')]:
        name={2:'fig2_condition_stage',3:'fig3_trajectory',4:'fig4_layout',5:'fig5_hyperparameters',6:'fig6_cases'}[n]
        lines.append(f'- 图{n} {title}：[validation](validation/{name}.png) / [test](test/{name}.png)')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n');print(summary)
if __name__=='__main__':main()
