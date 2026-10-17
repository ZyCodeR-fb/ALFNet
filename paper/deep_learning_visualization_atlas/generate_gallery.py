#!/usr/bin/env python3
"""Recreate an illustrative, synthetic-data visualization atlas (75 plots)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon, FancyBboxPatch, PathPatch, Arc
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, probplot
from scipy.special import expit
import seaborn as sns
import networkx as nx

ROOT=Path(__file__).resolve().parent
PNG=ROOT/'figures'; SVG=ROOT/'figures_svg'
PNG.mkdir(exist_ok=True);SVG.mkdir(exist_ok=True)
R=np.random.default_rng(20261012)
C={'blue':'#4179AE','teal':'#189D8F','gold':'#EDAC4A','red':'#D76670','purple':'#8C77B7','ink':'#243449','pale':'#E7ECF2','gray':'#8593A2','green':'#68A75C'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.4,'axes.titlesize':13,'axes.titleweight':'bold','axes.labelsize':9.2,'axes.edgecolor':'#B4BEC9','axes.labelcolor':C['ink'],'xtick.color':C['ink'],'ytick.color':C['ink'],'text.color':C['ink'],'grid.color':'#E4E8ED','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','legend.frameon':False,'axes.axisbelow':True})
meta=[]

def pretty(ax,grid='y'):
    if grid:ax.grid(axis=grid,alpha=.8,lw=.8)

def export(slug,title,draw,cat,caption):
    fig=plt.figure(figsize=(6.6,4.0),dpi=135)
    ax=fig.add_subplot(111)
    draw(fig,ax)
    fig.suptitle(title,x=.085,y=.975,ha='left',fontsize=13,color=C['ink'],weight='bold')
    fig.text(.91,.974,'SYNTHETIC DATA',ha='right',va='top',fontsize=6.8,color=C['gray'])
    fig.subplots_adjust(left=.115,right=.945,bottom=.18,top=.84)
    fig.savefig(PNG/(slug+'.png'),dpi=135,facecolor='white')
    fig.savefig(SVG/(slug+'.svg'),facecolor='white')
    plt.close(fig)
    meta.append((slug,title,cat,caption))

def draw01(fig,ax):
    names=['Model A','Model B','Model C','Proposed']; v=[3.38,3.21,3.10,2.86]; y=np.arange(4)
    ax.scatter(v,y,c=[C['gray']]*3+[C['teal']],s=[85,85,85,140],zorder=3)
    for a,b in zip(v,y):ax.text(a+.035,b,f'{a:.2f}',va='center',fontsize=9)
    ax.set(yticks=y,yticklabels=names,xlabel='MAE (lower is better)',xlim=(2.6,3.85));ax.invert_yaxis();pretty(ax,'x')

def draw02(fig,ax):
    names=['Base','+ block A','+ block B','Full'];val=np.array([2.2,2.55,2.85,3.1]); y=np.arange(4)
    ax.hlines(y,0,val,lw=2.5,colors=C['blue']);ax.scatter(val,y,s=95,c=[C['gray'],C['blue'],C['blue'],C['teal']]);ax.set(yticks=y,yticklabels=names,xlim=(0,3.5),xlabel='Accuracy gain score');ax.invert_yaxis();pretty(ax,'x')

def draw03(fig,ax):
    b=np.array([3.9,4.1,4.45,4.2]); p=np.array([3.3,3.52,3.56,3.5]);yy=np.arange(4)
    ax.hlines(yy,p,b,color='#C7D1DB',lw=3)
    ax.scatter(b,yy,c=C['gold'],s=80,label='Baseline',zorder=3);ax.scatter(p,yy,c=C['teal'],s=80,label='Proposed',zorder=3)
    ax.set(yticks=yy,yticklabels=['Set A','Set B','Set C','Set D'],xlabel='MAE (lower is better)');ax.invert_yaxis();ax.legend(loc='lower right');pretty(ax,'x')

def draw04(fig,ax):
    names=['A','B','C','D'];a=[.69,.78,.73,.81];b=[.76,.76,.84,.88]
    for i,n in enumerate(names):
        col=[C['blue'],C['gold'],C['purple'],C['teal']][i];ax.plot([0,1],[a[i],b[i]],marker='o',color=col,lw=2);ax.text(-.06,a[i],n,ha='right',va='center',color=col)
    ax.set(xticks=[0,1],xticklabels=['Before','After'],ylabel='F1 score',xlim=(-.17,1.12),ylim=(.61,.96));pretty(ax,'y')

def draw05(fig,ax):
    labels=['Dataset A','Dataset B','Dataset C','Dataset D','Overall'];m=np.array([.17,.08,.23,.13,.155]);err=np.array([.07,.085,.08,.09,.045]);yy=np.arange(len(labels))
    ax.axvline(0,c=C['gray'],ls='--',lw=1);ax.errorbar(m,yy,xerr=err,fmt='o',capsize=4,c=C['teal'],ecolor=C['blue'],lw=2)
    ax.set(yticks=yy,yticklabels=labels,xlabel='Relative improvement (95% CI)',xlim=(-.12,.37));ax.invert_yaxis();pretty(ax,'x')

def draw06(fig,ax):
    increments=[3.8,-.3,-.22,-.18,-.15]; names=['Base','Module A','Module B','Module C','Module D'];cur=0
    for i,v in enumerate(increments):
        if i==0:base=0;height=v
        else: base=cur+min(v,0);height=abs(v)
        ax.bar(i,height,bottom=base,color=C['gray'] if i==0 else C['teal'],width=.62)
        cur+=v
        if i<len(increments)-1:ax.plot([i+.31,i+.69],[cur,cur],color='#BCC8D1',lw=1,ls='--')
    ax.set(xticks=range(5),xticklabels=names,ylabel='MAE',ylim=(0,4.5));pretty(ax)

def draw07(fig,ax):
    fig.delaxes(ax);a=fig.add_subplot(111,polar=True);n=5;theta=np.linspace(0,2*np.pi,n,endpoint=False);angles=np.r_[theta,theta[0]]
    for vals,color,label in [([.76,.66,.69,.71,.7],C['blue'],'Baseline'),([.85,.82,.8,.84,.89],C['teal'],'Proposed')]:
        vals=np.r_[vals,vals[0]];a.plot(angles,vals,lw=2,color=color,label=label);a.fill(angles,vals,color=color,alpha=.1)
    a.set_xticks(theta);a.set_xticklabels(['Accuracy','Robustness','Efficiency','Stability','Transfer'],fontsize=8);a.set_ylim(0,1);a.set_yticks([.25,.5,.75,1]);a.set_yticklabels([]);a.grid(alpha=.35);a.spines['polar'].set_visible(False);a.legend(loc='upper right',bbox_to_anchor=(1.26,1.2),fontsize=7)

def draw08(fig,ax):
    ranks=[1.8,2.5,3.2,4.7,5.5];names=['Proposed','Model B','Model C','Model D','Model E']
    ax.hlines(0,1,6,lw=1.4,color=C['ink']);ax.hlines(.3,1.8,3.2,lw=5,color=C['blue'],alpha=.4)
    for i,(v,n) in enumerate(zip(ranks,names)):
        y=.7 if i%2==0 else -.65;ax.plot([v,v],[0,y],color=C['gray'],lw=1);ax.scatter([v],[0],c=C['teal'] if i==0 else C['blue'],s=65,zorder=3)
        ax.text(v,y+(.09 if y>0 else -.09),n,ha='center',va='bottom' if y>0 else 'top',fontsize=8)
    ax.set(xlim=(.7,6.4),ylim=(-1.12,1.08),xticks=[1,2,3,4,5,6],xlabel='Average rank (lower is better)');ax.set_yticks([])
    ax.text(1.1,.94,'Illustrative ranks; no significance claim',fontsize=7,color=C['gray'])

def draw09(fig,ax):
    ranks=np.array([[2,1,2,1,1],[1,2,1,2,2],[3,3,4,3,4],[4,4,3,4,3]]);x=np.arange(5)
    for i,name in enumerate(['Proposed','Model A','Model B','Model C']):ax.plot(x,ranks[i],lw=2,marker='o',label=name,color=[C['teal'],C['blue'],C['gold'],C['purple']][i])
    ax.set(xticks=x,xticklabels=['D1','D2','D3','D4','D5'],yticks=[1,2,3,4],ylabel='Rank');ax.invert_yaxis();ax.legend(fontsize=7,ncol=2,loc='lower left');pretty(ax)

# Training curves
E=np.arange(1,101)
def draw10(fig,ax):
    tr=.7*np.exp(-E/29)+.17+R.normal(0,.008,len(E));va=.62*np.exp(-E/24)+.24+np.clip((E-72),0,None)*.0015+R.normal(0,.015,len(E))
    ax.plot(E,tr,color=C['blue'],lw=2,label='Train');ax.plot(E,va,color=C['teal'],lw=2,label='Validation');ax.set(xlabel='Epoch',ylabel='Loss');ax.legend();pretty(ax)
def draw11(fig,ax):
    m=.72*np.exp(-E/31)+.23;sd=.1*np.exp(-E/44)+.025
    ax.fill_between(E,m-sd,m+sd,color=C['teal'],alpha=.25,label='±1 std');ax.plot(E,m,color=C['teal'],lw=2,label='Mean');ax.set(xlabel='Epoch',ylabel='Validation MAE');ax.legend();pretty(ax)
def draw12(fig,ax):
    x=np.arange(30);y=np.cumsum(np.maximum(0,R.normal(1.9,.5,30)));ax.fill_between(x,y,color=C['blue'],alpha=.3);ax.plot(x,y,lw=2,c=C['blue']);ax.set(xlabel='Training step (×100)',ylabel='Cumulative improvement');pretty(ax)
def draw13(fig,ax):
    x=np.arange(50);components=np.array([.8*np.exp(-x/20)+.09,.4*np.exp(-x/35)+.08,.25*np.exp(-x/25)+.06]);ax.stackplot(x,*components,colors=[C['blue'],C['teal'],C['gold']],alpha=.8,labels=['Prediction','Regularization','Auxiliary']);ax.legend(loc='upper right',fontsize=7);ax.set(xlabel='Epoch',ylabel='Loss components');pretty(ax)
def draw14(fig,ax):
    x=[0,20,40,65,90,100];v=[.001,.001,.0003,.0001,.00003,.00003];ax.step(x,v,where='post',lw=2.5,c=C['purple']);ax.scatter(x[:-1],v[:-1],c=C['purple']);ax.set(xlabel='Epoch',ylabel='Learning rate');pretty(ax)
def draw15(fig,ax):
    for i in range(10):ax.plot(E,.8*np.exp(-E/(24+R.normal(0,4)))+.26+R.normal(0,.02,100)+R.normal(0,.04),alpha=.38,lw=1.1,c=C['blue'])
    ax.set(xlabel='Epoch',ylabel='Validation loss');pretty(ax)
def draw16(fig,ax):
    x=np.array([5,10,20,40,60,80,100]);m=.87-.32*np.exp(-x/30)
    ax.plot(x,m,marker='o',lw=2,c=C['teal']);ax.set(xlabel='Training data (%)',ylabel='Accuracy',ylim=(.56,.9));pretty(ax)
def draw17(fig,ax):
    x=np.arange(-20,21);a=.2+.5*np.exp(-(x-2)**2/62);b=.2+.25*np.exp(-(x-2)**2/50)
    ax.axvline(0,ls='--',c=C['gray'],label='Event onset');ax.plot(x,a,lw=2,c=C['gold'],label='Baseline error');ax.plot(x,b,lw=2,c=C['teal'],label='Proposed error');ax.set(xlabel='Minutes relative to event',ylabel='Prediction error');ax.legend(fontsize=7);pretty(ax)

# Statistical distribution
DIST=[R.normal(3.4,.55,110),R.normal(3.13,.48,110),R.normal(2.89,.38,110)]
def draw18(fig,ax):
    b=ax.boxplot(DIST,labels=['Base','Model B','Proposed'],patch_artist=True,showfliers=True,medianprops={'color':C['ink'],'linewidth':2})
    for patch,color in zip(b['boxes'],[C['gray'],C['blue'],C['teal']]):patch.set_facecolor(color);patch.set_alpha(.55)
    ax.set(ylabel='Per-sample MAE');pretty(ax)
def draw19(fig,ax):
    v=ax.violinplot(DIST,showmedians=True,showextrema=False)
    for body,col in zip(v['bodies'],[C['gray'],C['blue'],C['teal']]):body.set_facecolor(col);body.set_edgecolor('white');body.set_alpha(.7)
    ax.set(xticks=[1,2,3],xticklabels=['Base','Model B','Proposed'],ylabel='Per-sample MAE');pretty(ax)
def draw20(fig,ax):
    for j,vals in enumerate(DIST):
        y=j+1;xx=np.linspace(min(vals)-.2,max(vals)+.2,300);den=gaussian_kde(vals)(xx);den=den/den.max()*.27
        ax.fill_between(xx,y,y+den,color=[C['gray'],C['blue'],C['teal']][j],alpha=.55)
        yy=y-R.uniform(.08,.22,len(vals));ax.scatter(vals,yy,s=4,alpha=.28,color=[C['gray'],C['blue'],C['teal']][j],rasterized=True)
        ax.plot([np.quantile(vals,.25),np.quantile(vals,.75)],[y,y],color=C['ink'],lw=4);ax.scatter([np.median(vals)],[y],c=C['ink'],s=18)
    ax.set(yticks=[1,2,3],yticklabels=['Base','Model B','Proposed'],xlabel='Per-sample MAE',ylim=(.7,3.7));pretty(ax,'x')
def draw21(fig,ax):
    sns.swarmplot(data=DIST,ax=ax,orient='v',palette=[C['gray'],C['blue'],C['teal']],size=3.5);ax.set(xticks=[0,1,2],xticklabels=['Base','Model B','Proposed'])
    ax.set(ylabel='Per-sample MAE');pretty(ax)
def draw22(fig,ax):
    for j,vals in enumerate(DIST):ax.scatter(np.full(len(vals),j+1)+R.uniform(-.17,.17,len(vals)),vals,alpha=.35,s=13,c=[C['gray'],C['blue'],C['teal']][j])
    ax.set(xticks=[1,2,3],xticklabels=['Base','Model B','Proposed'],ylabel='Error');pretty(ax)
def draw23(fig,ax):
    xx=np.linspace(1.5,5.5,320)
    for j,vals in enumerate([R.normal(3.55-.18*i,.38+.05*i,400) for i in range(5)]):
        yy=gaussian_kde(vals)(xx);yy/=max(yy);ax.fill_between(xx,j,j+yy*.85,alpha=.5,color=[C['teal'],C['blue'],C['gold'],C['purple'],C['red']][j]);ax.plot(xx,j+yy*.85,lw=1.5,color=C['ink'])
    ax.set(yticks=range(5),yticklabels=['H1','H3','H6','H9','H12'],xlabel='Absolute error');pretty(ax,'x')
def draw24(fig,ax):
    xx=np.linspace(1.2,5.6,300)
    for vals,col,label in zip(DIST,[C['gray'],C['blue'],C['teal']],['Base','B','Proposed']):ax.plot(xx,gaussian_kde(vals)(xx),lw=2,color=col,label=label)
    ax.legend();ax.set(xlabel='Absolute error',ylabel='Density');pretty(ax)
def draw25(fig,ax):
    for vals,col,label in zip(DIST,[C['gray'],C['blue'],C['teal']],['Base','B','Proposed']):x=np.sort(vals);ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=col,lw=2,label=label)
    ax.set(xlabel='Absolute error threshold',ylabel='Fraction ≤ threshold');ax.legend();pretty(ax)
def draw26(fig,ax):
    ax.hist(DIST[0],bins=16,alpha=.55,color=C['blue'],label='Base');ax.hist(DIST[2],bins=16,alpha=.55,color=C['teal'],label='Proposed');ax.set(xlabel='Error',ylabel='Count');ax.legend();pretty(ax)
def draw27(fig,ax):
    sample=R.normal(0,1,190)+.3*(R.random(190)<.05)*R.normal(3,1,190);osm,osr=probplot(sample,dist='norm',fit=False);ax.scatter(osm,osr,s=13,c=C['blue'],alpha=.65);ax.plot([-3,3],[-3,3],ls='--',c=C['red']);ax.set(xlabel='Theoretical quantiles',ylabel='Observed quantiles');pretty(ax)

# Hyperparameters
XX,YY=np.meshgrid(np.linspace(-3,3,60),np.linspace(-3,3,60));ZZ=.22*(XX-.7)**2+.15*(YY+.8)**2+.1*np.sin(XX*1.3)*np.cos(YY)
def draw28(fig,ax):
    co=ax.contourf(XX,YY,ZZ,levels=16,cmap='YlGnBu');ax.contour(XX,YY,ZZ,levels=8,colors='#16344A',alpha=.45,linewidths=.6);ax.scatter([.7],[-.8],c='white',edgecolor=C['red'],s=90,marker='*');fig.colorbar(co,ax=ax,label='Validation MAE',shrink=.85);ax.set(xlabel='log learning rate (scaled)',ylabel='log weight decay (scaled)')
def draw29(fig,ax):
    fig.delaxes(ax);a=fig.add_subplot(111,projection='3d');a.plot_surface(XX[::2,::2],YY[::2,::2],ZZ[::2,::2],cmap='viridis',alpha=.9,linewidth=0);a.set(xlabel='LR',ylabel='WD',zlabel='MAE');a.view_init(28,-58)
def draw30(fig,ax):
    x=R.uniform(.1,1,80);y=.55+.37*x+R.normal(0,.08,80);ax.scatter(x,y,c=C['teal'],alpha=.65);ax.set(xlabel='Regularization strength',ylabel='Accuracy');pretty(ax)
def draw31(fig,ax):
    x=R.uniform(8,128,40);y=.67+.002*x+R.normal(0,.045,40);sz=R.uniform(30,250,40);sc=ax.scatter(x,y,s=sz,c=sz,cmap='Blues',alpha=.65,edgecolor='white');fig.colorbar(sc,ax=ax,shrink=.8,label='FLOPs (scaled)');ax.set(xlabel='Hidden dimension',ylabel='Accuracy');pretty(ax)
def draw32(fig,ax):
    data=R.uniform(.1,.9,(24,5));data[:,4]=np.clip(.4+.35*data[:,0]+.2*data[:,2]- .25*data[:,1]+R.normal(0,.1,24),0,1)
    for r in data:ax.plot(range(5),r,color=plt.cm.viridis(r[-1]),alpha=.48,lw=1.4)
    ax.set(xticks=range(5),xticklabels=['LR','WD','Layers','Dim','Score'],ylabel='Normalized value',ylim=(0,1));pretty(ax)
def draw33(fig,ax):
    fig.delaxes(ax);x=R.normal(size=110);y=.5*x+R.normal(size=110)*.7;z=-.3*x+.6*y+R.normal(size=110)*.5
    from pandas.plotting import scatter_matrix
    import pandas as pd
    # custom 3x3 compact scatter matrix
    gs=fig.add_gridspec(3,3,left=.12,right=.93,top=.84,bottom=.16,wspace=.12,hspace=.12)
    v=[x,y,z];labels=['LR','WD','Score']
    for i in range(3):
        for j in range(3):
            a=fig.add_subplot(gs[i,j])
            if i==j:a.hist(v[i],bins=12,color=C['teal'],alpha=.65)
            else:a.scatter(v[j],v[i],s=7,color=C['blue'],alpha=.38)
            a.tick_params(labelsize=6,length=2);a.set_xticklabels([]) if i<2 else a.set_xlabel(labels[j],fontsize=7);a.set_yticklabels([]) if j>0 else a.set_ylabel(labels[i],fontsize=7)
def draw34(fig,ax):
    co=ax.contourf(XX,YY,ZZ+.16,cmap='viridis',levels=18);a=R.uniform(-2.7,2.7,(22,2));ax.scatter(a[:,0],a[:,1],c='white',edgecolor=C['ink'],s=26,lw=.7,label='Evaluated trials');fig.colorbar(co,ax=ax,shrink=.82,label='Fitted loss surface');ax.set(xlabel='Parameter A',ylabel='Parameter B');ax.legend(loc='lower left',fontsize=7)
def draw35(fig,ax):
    fig.delaxes(ax);gs=fig.add_gridspec(1,3,left=.09,right=.94,bottom=.2,top=.80,wspace=.26)
    for i in range(3):
        a=fig.add_subplot(gs[0,i]);x=np.linspace(0,1,25);y=.3*(x-.2-.2*i)**2+.19+.03*i;a.plot(x,y,c=[C['blue'],C['teal'],C['purple']][i],lw=2);a.set_title(['METR-LA','PEMS-BAY','Set C'][i],fontsize=9);a.set_xlabel('Parameter',fontsize=8);a.grid(alpha=.2);a.tick_params(labelsize=7)
def draw36(fig,ax):
    ax.axis('off');V=np.array([[.12,.15],[.88,.15],[.5,.83]]);ax.add_patch(Polygon(V,closed=True,fill=False,lw=2,edgecolor=C['ink']))
    points=R.dirichlet([2,3,2],size=65);xy=points@V;value=points[:,0]*.4+points[:,1]*.2+points[:,2]*.8
    ax.scatter(xy[:,0],xy[:,1],c=value,s=30,cmap='viridis',alpha=.8)
    for (x,y),name,dx,dy in zip(V,['Loss A','Loss B','Loss C'],[-.02,.02,0],[-.06,-.06,.05]):ax.text(x+dx,y+dy,name,ha='center',fontsize=9)
    ax.set(xlim=(0,1),ylim=(0,.98))

# Efficiency & compute
E_COST=np.array([1.2,1.6,2.7,3.2,4.4,5.3]);E_ERR=np.array([4.2,3.9,3.55,3.5,3.31,3.29]);
def draw37(fig,ax):
    xs=R.uniform(1,5.5,15);ys=np.interp(xs,E_COST,E_ERR)+R.uniform(.12,.55,15);ax.scatter(xs,ys,c=C['gray'],alpha=.55,s=35,label='Other configurations');ax.plot(E_COST,E_ERR,'-o',color=C['teal'],lw=2,label='Pareto frontier');ax.set(xlabel='Inference cost (a.u.)',ylabel='MAE (lower better)');ax.legend(fontsize=7);pretty(ax)
def draw38(fig,ax):
    x=np.array([10,19,24,31,38,50]);y=np.array([.77,.82,.84,.87,.88,.895]);ax.scatter(x,y,c=[C['blue']]*4+[C['teal']]*2,s=85);ax.set(xlabel='Latency (ms/sample)',ylabel='Accuracy',ylim=(.73,.93));pretty(ax)
def draw39(fig,ax):
    x=R.uniform(10,80,28);y=.77+.002*x+R.normal(0,.035,28);size=R.uniform(35,260,28);ax.scatter(x,y,s=size,c=R.uniform(1,10,28),cmap='viridis',alpha=.7);ax.set(xlabel='Latency (ms)',ylabel='Accuracy');pretty(ax)
def draw40(fig,ax):
    x=np.logspace(-1,2.3,200);roof=np.minimum(x*50,500);ax.plot(x,roof,c=C['blue'],lw=2.5,label='Hardware roof');ax.scatter([.8,3,10,40],[25,90,260,380],c=C['teal'],s=62,label='Kernels');ax.set_xscale('log');ax.set_yscale('log');ax.set(xlabel='Arithmetic intensity (ops/byte)',ylabel='Performance (GFLOP/s)');ax.legend(fontsize=7);pretty(ax)
def draw41(fig,ax):
    cats=['Base','Proposed'];comp=[np.array([25,32]),np.array([50,41]),np.array([15,10])];b=np.zeros(2)
    for vals,col,lab in zip(comp,[C['blue'],C['teal'],C['gold']],['Encoder','Predictor','Correction']):ax.bar(cats,vals,bottom=b,color=col,label=lab,width=.45);b+=vals
    ax.set(ylabel='Latency (ms)');ax.legend(ncol=3,fontsize=7);pretty(ax)
def draw42(fig,ax):
    ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    nodes=[(.07,.3,.22,.55,C['blue'],'Input'),(.44,.54,.17,.28,C['teal'],'Encoder'),(.44,.13,.17,.24,C['gold'],'Gate'),(.81,.28,.15,.45,C['purple'],'Output')]
    for x,y,w,h,c,lab in nodes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01',facecolor=c,alpha=.8,edgecolor='white'));ax.text(x+w/2,y+h/2,lab,color='white',ha='center',va='center',fontsize=9,weight='bold')
    for start,end,w,c in [((.29,.68),(.44,.68),13,C['blue']),((.29,.4),(.44,.25),8,C['gold']),((.61,.69),(.81,.58),13,C['teal']),((.61,.23),(.81,.4),8,C['gold'])]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=15,color=c,lw=w,alpha=.42,connectionstyle='arc3,rad=.1'))
def draw43(fig,ax):
    phases=[('Prepare',0,2),('Encode',2,4),('Predict',6,3),('Revise',9,2)];cols=[C['gray'],C['blue'],C['teal'],C['gold']]
    for i,(n,st,du) in enumerate(phases):ax.broken_barh([(st,du)],(i-.3,.6),facecolors=cols[i]);ax.text(st+du/2,i,n,ha='center',va='center',color='white',fontsize=9)
    ax.set(yticks=range(4),yticklabels=[p[0] for p in phases],xlabel='Runtime (ms)',ylim=(-.7,3.6));ax.invert_yaxis();pretty(ax,'x')
def draw44(fig,ax):
    b=np.array([1,2,4,8,16,32]);s=np.array([1,1.75,3.15,5.3,8.6,11.1]);ax.plot(b,b,'--',c=C['gray'],label='Ideal');ax.plot(b,s,'-o',c=C['teal'],label='Observed');ax.set(xlabel='Worker count',ylabel='Speedup',xlim=(0,33),ylim=(0,34));ax.legend();pretty(ax)
# Forecast diagnostics
PTRUE=R.normal(35,10,140); PPRED=PTRUE*.92+2.4+R.normal(0,4.2,140)
def draw45(fig,ax):
    ax.scatter(PTRUE,PPRED,s=16,alpha=.45,c=C['teal']);ax.plot([0,70],[0,70],'--',c=C['gray']);ax.set(xlabel='Ground truth',ylabel='Prediction',xlim=(5,67),ylim=(5,67));pretty(ax)
def draw46(fig,ax):
    resid=PPRED-PTRUE;ax.scatter(PPRED,resid,s=14,alpha=.5,c=C['blue']);ax.axhline(0,ls='--',c=C['red']);ax.set(xlabel='Predicted value',ylabel='Residual (pred - true)');pretty(ax)
def draw47(fig,ax):
    mean=(PTRUE+PPRED)/2;dif=PPRED-PTRUE;m=np.mean(dif);sd=np.std(dif)
    ax.scatter(mean,dif,s=15,c=C['blue'],alpha=.55);ax.axhline(m,c=C['teal'],lw=2,label='Mean difference');ax.axhline(m+1.96*sd,c=C['red'],ls='--',label='Limits of agreement');ax.axhline(m-1.96*sd,c=C['red'],ls='--');ax.set(xlabel='Mean of prediction and truth',ylabel='Difference');ax.legend(fontsize=7);pretty(ax)
def draw48(fig,ax):
    t=np.arange(30);mid=10+.09*t+1.5*np.sin(t/5);spread=.16*t+.3
    ax.fill_between(t,mid-2*spread,mid+2*spread,color=C['blue'],alpha=.13,label='95% interval');ax.fill_between(t,mid-spread,mid+spread,color=C['blue'],alpha=.26,label='68% interval');ax.plot(t,mid,c=C['blue'],lw=2,label='Median');ax.set(xlabel='Forecast horizon',ylabel='Forecast value');ax.legend(fontsize=7);pretty(ax)
def draw49(fig,ax):
    t=np.arange(36);m=10+np.sin(t/4)+.07*t;spread=.25+np.arange(36)*.07
    for k,alpha in [(1.65,.15),(.95,.24),(.5,.35)]:ax.fill_between(t,m-k*spread,m+k*spread,color=C['teal'],alpha=alpha)
    ax.plot(t,m,c=C['teal'],lw=2,label='Median');ax.plot(t,10+np.sin((t+1)/4)+.07*t+R.normal(0,.4,len(t)),c=C['gold'],lw=1.4,label='Truth');ax.set(xlabel='Time step',ylabel='Value');ax.legend(fontsize=7);pretty(ax)
def draw50(fig,ax):
    h=np.arange(1,13);a=2+.21*h+.04*np.sin(h);b=2+.12*h
    ax.plot(h,a,'-o',c=C['blue'],label='Baseline');ax.plot(h,b,'-s',c=C['teal'],label='Proposed');ax.set(xlabel='Forecast horizon',ylabel='MAE',xticks=[1,3,6,9,12]);ax.legend();pretty(ax)
def draw51(fig,ax):
    fig.delaxes(ax);a=fig.add_subplot(111,polar=True);theta=np.linspace(0,np.pi/2,150)
    for r in [.5,1,1.5]:a.plot(theta,np.full(len(theta),r),c='#D8E1E7',lw=.8)
    for corr,std,lab,col in [(.72,1.25,'Base',C['blue']),(.89,1.06,'Model B',C['gold']),(.95,1.02,'Proposed',C['teal'])]:
        a.scatter(np.arccos(corr),std,s=90,c=col,label=lab)
    a.scatter(0,1,s=95,c=C['red'],marker='*',label='Reference')
    a.set_thetamin(0);a.set_thetamax(90);a.set_rmax(1.7);a.set_xticks(np.arccos([1,.9,.7,.5,0]));a.set_xticklabels(['1.0','.9','.7','.5','0']);a.legend(loc='upper right',bbox_to_anchor=(1.4,1.25),fontsize=7)
def draw52(fig,ax):
    nominal=np.arange(.5,1,.05);obs=np.clip(nominal-.04+.015*np.sin(10*nominal),0,1)
    ax.plot([.45,1],[.45,1],'--',c=C['gray'],label='Ideal');ax.plot(nominal,obs,'-o',c=C['teal'],label='Observed');ax.set(xlabel='Nominal coverage',ylabel='Empirical coverage',xlim=(.48,1),ylim=(.45,1));ax.legend();pretty(ax)
def draw53(fig,ax):
    p=np.linspace(.05,.95,10);a=np.clip(p+.12*(p-.5)+R.normal(0,.025,10),0,1);ax.plot([0,1],[0,1],'--',c=C['gray'],label='Perfect');ax.plot(p,a,'o-',c=C['teal'],label='Model');ax.set(xlabel='Predicted confidence',ylabel='Observed frequency',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7);pretty(ax)
def draw54(fig,ax):
    fig.delaxes(ax);gs=fig.add_gridspec(1,2,left=.13,right=.96,bottom=.2,top=.78,wspace=.36)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);x=np.linspace(0,1,100);a.plot(x,np.sqrt(x),c=C['teal'],lw=2);a.plot(x,x,'--',c=C['gray']);a.set(xlabel='FPR',ylabel='TPR',title='ROC');a.grid(alpha=.2)
    rec=np.linspace(0,1,100);b.plot(rec,.9-.35*rec**2,c=C['blue'],lw=2);b.set(xlabel='Recall',ylabel='Precision',title='PR');b.set_ylim(0,1);b.grid(alpha=.2)
def draw55(fig,ax):
    v=np.array([[82,7,3],[8,69,5],[4,7,75]]);im=ax.imshow(v,cmap='Blues',vmin=0,vmax=90)
    for (i,j),value in np.ndenumerate(v):ax.text(j,i,str(value),ha='center',va='center',fontsize=12,color='white' if value>50 else C['ink'])
    ax.set(xticks=range(3),yticks=range(3),xticklabels=['A','B','C'],yticklabels=['A','B','C'],xlabel='Predicted label',ylabel='True label');fig.colorbar(im,ax=ax,shrink=.8)

# Interpretability & Representation
EMB=np.vstack([R.multivariate_normal(mu,[[.6,.16],[.16,.55]],size=55) for mu in [[-2,-.5],[2,-.6],[0,2.3]]]);LBL=np.repeat(np.arange(3),55)
def emb_scatter(ax,x,y):
    for i,col in enumerate([C['blue'],C['gold'],C['teal']]):ax.scatter(x[LBL==i],y[LBL==i],s=18,c=col,alpha=.62,label=f'Class {i+1}')
    ax.legend(fontsize=7);ax.set(xlabel='Dimension 1',ylabel='Dimension 2');pretty(ax)
def draw56(fig,ax):emb_scatter(ax,EMB[:,0]*1.5+R.normal(0,.16,len(EMB)),EMB[:,1]*1.4+R.normal(0,.16,len(EMB)))
def draw57(fig,ax):emb_scatter(ax,EMB[:,0]+.2*np.sin(EMB[:,1]),EMB[:,1]+.2*np.cos(EMB[:,0]))
def draw58(fig,ax):emb_scatter(ax,EMB[:,0]*.9+EMB[:,1]*.22,EMB[:,1]*.5-EMB[:,0]*.15)
def draw59(fig,ax):
    feat=['Speed','Time','Weather','Neighbor','Trend'];centers=[.7,.2,-.12,.4,-.3]
    for i,(name,mu) in enumerate(zip(feat,centers)):
        v=R.normal(mu,.22,55);y=np.full(len(v),i)+R.normal(0,.14,len(v));ax.scatter(v,y,c=v,cmap='coolwarm',vmin=-1,vmax=1,s=17,alpha=.7)
    ax.axvline(0,color=C['gray'],lw=1);ax.set(yticks=range(5),yticklabels=feat,xlabel='SHAP value');ax.invert_yaxis();pretty(ax,'x')
def draw60(fig,ax):
    labels=['Baseline','Speed','Weather','Neighbor','Trend','Prediction'];steps=[3.1,.7,-.25,.53,-.22];curr=steps[0]
    ax.barh(0,curr,color=C['gray']);
    for i,v in enumerate(steps[1:],1):
        left=curr if v>=0 else curr+v;ax.barh(i,abs(v),left=left,color=C['teal'] if v>=0 else C['red']);curr+=v
    ax.barh(5,curr,color=C['blue']);ax.set(yticks=range(6),yticklabels=labels,xlabel='Contribution to prediction');ax.invert_yaxis();pretty(ax,'x')
def draw61(fig,ax):
    g=nx.watts_strogatz_graph(17,4,.25,seed=7);pos=nx.spring_layout(g,seed=8);nx.draw_networkx_edges(g,pos,ax=ax,edge_color='#C3D2DC',width=1.7);nx.draw_networkx_nodes(g,pos,ax=ax,node_size=[200+R.integers(0,150) for _ in g.nodes],node_color=[C['blue'] if i%4 else C['teal'] for i in g.nodes]);ax.set_axis_off()
def draw62(fig,ax):
    ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    left=[(.05,.75,C['blue'],'Low'),(.05,.4,C['teal'],'Medium'),(.05,.12,C['gold'],'High')];right=[(.8,.75,C['gold'],'Update'),(.8,.4,C['blue'],'Keep'),(.8,.12,C['red'],'Skip')]
    for nodes in [left,right]:
        for x,y,col,name in nodes:ax.add_patch(Rectangle((x,y),.14,.13,color=col,alpha=.85));ax.text(x+.07,y+.065,name,ha='center',va='center',fontsize=8,color='white')
    for (i,j,weight) in [(0,0,5),(0,1,2),(1,0,2),(1,1,6),(2,0,1),(2,2,3)]:
        x0,y0,_,_=left[i];x1,y1,_,_=right[j];path=MPath([(x0+.14,y0+.065),(.42,y0+.065),(.56,y1+.065),(x1,y1+.065)],[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4]);ax.add_patch(PathPatch(path,edgecolor=[C['blue'],C['teal'],C['gold']][i],lw=weight*2,alpha=.28,facecolor='none'))
def draw63(fig,ax):
    mat=np.zeros((24,36));yy,xx=np.mgrid[:24,:36]
    for x,y,a,sx,sy in [(11,8,1,4,3),(26,16,.9,5,4),(17,19,.5,3,2)]:mat+=a*np.exp(-((xx-x)**2/(2*sx*sx)+(yy-y)**2/(2*sy*sy)))
    im=ax.imshow(mat,cmap='magma',aspect='auto');fig.colorbar(im,ax=ax,shrink=.8,label='Activation');ax.set(xlabel='Feature position',ylabel='Channel position')
def draw64(fig,ax):
    x,y=np.meshgrid(np.linspace(-3,3,120),np.linspace(-3,3,120));p=np.sin(x)+.65*y-.2*x*y;ax.contourf(x,y,p,levels=[-10,0,10],colors=['#d6e9f3','#d9eee8'],alpha=.9)
    a=R.uniform(-2.7,2.7,130);b=R.uniform(-2.7,2.7,130);group=(np.sin(a)+.65*b-.2*a*b)>0;ax.scatter(a[group],b[group],s=15,c=C['teal'],alpha=.65);ax.scatter(a[~group],b[~group],s=15,c=C['blue'],alpha=.65);ax.contour(x,y,p,levels=[0],colors=[C['ink']]);ax.set(xlabel='Feature 1',ylabel='Feature 2')
def draw65(fig,ax):
    x=np.arange(9);m=np.exp(-np.abs(x[:,None]-x[None,:])/1.5)+R.uniform(-.05,.05,(9,9));m=np.clip((m+m.T)/2,0,1)
    im=ax.imshow(m,cmap='YlGnBu',vmin=0,vmax=1);ax.set(xticks=x,yticks=x,xticklabels=[f'L{i+1}' for i in x],yticklabels=[f'L{i+1}' for i in x],xlabel='Model B layers',ylabel='Model A layers');fig.colorbar(im,ax=ax,shrink=.8,label='CKA')

# Spatial-temporal figures
POS=np.array([[.1,.25],[.3,.15],[.48,.30],[.64,.18],[.88,.28],[.2,.58],[.42,.65],[.63,.58],[.85,.74],[.37,.88],[.12,.84]])
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(1,6),(2,6),(3,7),(7,8)]
def plot_graph(ax,values,colors='viridis',show_edges=True):
    if show_edges:
        for i,j in EDGES:ax.plot(POS[[i,j],0],POS[[i,j],1],color='#AFBBC5',lw=2,zorder=0)
    p=ax.scatter(POS[:,0],POS[:,1],c=values,cmap=colors,s=150,edgecolor='white',lw=1.4,zorder=2);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');return p
def draw66(fig,ax):
    v=np.array([2.1,1.8,3.4,2.7,2.2,2.3,3.6,4.5,3.2,1.7,2.1]);p=plot_graph(ax,v,'YlOrRd');fig.colorbar(p,ax=ax,shrink=.72,label='Node MAE')
def draw67(fig,ax):
    values=np.array([.35,-.08,.23,.62,.31,-.1,.45,.53,.24,.12,-.03]);idx=np.argsort(values)
    ax.barh(np.arange(len(values)),values[idx],color=[C['teal'] if v>=0 else C['red'] for v in values[idx]]);ax.axvline(0,c=C['gray']);ax.set(yticks=np.arange(len(values)),yticklabels=[f'Node {i}' for i in idx],xlabel='MAE improvement (positive = better)');pretty(ax,'x')
def draw68(fig,ax):
    h=np.arange(1,13);b=2.4+.21*h+.03*np.sin(h);p=2.38+.12*h
    ax.fill_between(h,p-.1,p+.1,color=C['teal'],alpha=.18);ax.plot(h,b,'-o',c=C['blue'],label='Base');ax.plot(h,p,'-o',c=C['teal'],label='Revision');ax.set(xlabel='Horizon (steps)',ylabel='MAE',xticks=[1,3,6,9,12]);ax.legend();pretty(ax)
def draw69(fig,ax):
    fig.delaxes(ax);a=fig.add_subplot(111,projection='3d');x,y=np.meshgrid(np.arange(12),np.arange(11));z=1+.2*x+.6*np.sin(y/2)*np.cos(x/4);a.plot_surface(x,y,z,cmap='viridis',linewidth=0);a.view_init(28,-60);a.set(xlabel='Time',ylabel='Node ID',zlabel='Error')
def draw70(fig,ax):
    v=np.array([.1,.2,.3,.85,.7,.2,.3,.95,.65,.2,.2]);p=plot_graph(ax,v,'plasma');ax.scatter(POS[[3,7],0],POS[[3,7],1],facecolors='none',edgecolors=C['red'],s=320,lw=2);fig.colorbar(p,ax=ax,shrink=.72,label='Attention / priority')
def draw71(fig,ax):
    x=np.arange(-12,21);actual=1.1+expit((x-2)/2)*2+.06*np.sin(x/2);old=1.1+expit((x-5)/3)*1.8;new=1.1+expit((x-2.5)/2.2)*1.95
    ax.axvline(0,c=C['gray'],ls='--',label='Event');ax.plot(x,actual,c=C['ink'],lw=2,label='Ground truth');ax.plot(x,old,c=C['gold'],lw=2,label='Old forecast');ax.plot(x,new,c=C['teal'],lw=2,label='Revised');ax.set(xlabel='Time relative to event',ylabel='Traffic signal');ax.legend(fontsize=7);pretty(ax)
def draw72(fig,ax):
    times=np.arange(18);actions=R.random((5,18))>.7
    for i in range(5):
        for j in range(18):
            if actions[i,j]:ax.add_patch(Rectangle((j,i),.9,.7,facecolor=C['teal']))
    ax.set(xlim=(0,18),ylim=(0,5),xticks=[0,3,6,9,12,15,18],yticks=np.arange(5)+.35,yticklabels=[f'N{i}' for i in range(5)],xlabel='Decision time step');ax.invert_yaxis();pretty(ax,'x')
def draw73(fig,ax):
    gain=np.r_[R.normal(.3,.36,250),R.normal(-.26,.15,60)];ax.hist(gain,bins=24,color=C['blue'],alpha=.75);ax.axvline(0,c=C['red'],ls='--',lw=1.6,label='No improvement');ax.set(xlabel='Revision gain (old error - new error)',ylabel='Cell count');ax.legend(fontsize=7);pretty(ax)
def draw74(fig,ax):
    pct=np.array([0,5,10,20,40,60,80,100]);mae=np.array([4.0,3.48,3.29,3.08,2.96,2.92,2.915,2.91]);ax.plot(pct,mae,'-o',c=C['teal'],lw=2);ax.set(xlabel='Revision budget (%)',ylabel='MAE (lower better)');pretty(ax)
def draw75(fig,ax):
    g=nx.DiGraph();g.add_nodes_from(['Observe','Keep','Revise','Stop']);edges=[('Observe','Keep',.36),('Observe','Revise',.64),('Keep','Observe',.6),('Revise','Observe',.8),('Revise','Stop',.2)]
    for a,b,w in edges:g.add_edge(a,b,weight=w)
    pos={'Observe':(0,.55),'Keep':(.5,.85),'Revise':(.5,.23),'Stop':(.98,.23)}
    nx.draw_networkx_nodes(g,pos,ax=ax,node_size=2000,node_color=[C['blue'],C['gray'],C['teal'],C['gold']],alpha=.88,edgecolors='white')
    nx.draw_networkx_labels(g,pos,ax=ax,font_color='white',font_weight='bold',font_size=9)
    nx.draw_networkx_edges(g,pos,ax=ax,edgelist=[(a,b) for a,b,w in edges],width=[w*3 for a,b,w in edges],arrowsize=15,node_size=2000,connectionstyle='arc3,rad=.12',edge_color='#637A8A',arrows=True)
    ax.set(xlim=(-.17,1.2),ylim=(-.05,1.05));ax.axis('off')

CATEGORIES=[
('性能对比与算法排名',[
('dot_plot','Dot Plot','不同模型的单指标对比',draw01),('lollipop_plot','Lollipop Chart','消融或模型排名',draw02),('dumbbell_plot','Dumbbell Plot','同一数据集上改进前后的差值',draw03),('slope_chart','Slope Chart','两个条件下的性能变化',draw04),('forest_plot','Forest Plot','效应量及置信区间',draw05),('waterfall_chart','Waterfall Chart','模块累计增益与损失',draw06),('radar_chart','Radar Chart','归一化后的多目标概览',draw07),('critical_difference','Critical Difference Diagram','跨数据集平均排名；显著性需另行检验',draw08),('bump_chart','Bump Chart','跨任务算法名次变化',draw09)]),
('训练过程与动态趋势',[
('learning_curve','Learning Curve','训练损失和验证损失随 epoch 变化',draw10),('ribbon_plot','Ribbon Plot','均值与波动区间',draw11),('area_plot','Area Plot','累计指标随训练推进变化',draw12),('stacked_area','Stacked Area Plot','多个损失分量的组成',draw13),('step_plot','Step Plot','阶梯学习率或阈值策略',draw14),('spaghetti_plot','Spaghetti Plot','多个随机种子训练轨迹',draw15),('data_size_learning_curve','Learning Curve vs Data Size','样本规模与模型性能',draw16),('event_aligned_plot','Event-Aligned Plot','事件发生前后的指标演变',draw17)]),
('统计分布与误差分析',[
('box_plot','Box Plot','中位数、四分位数与离群点',draw18),('violin_plot','Violin Plot','不同方法的误差密度',draw19),('raincloud_plot','Raincloud Plot','密度、样本点与摘要统计',draw20),('beeswarm_plot','Beeswarm Plot','尽量避免点遮挡的个体观测分布',draw21),('strip_plot','Strip Plot','单个样本或重复实验结果',draw22),('ridge_plot','Ridge Plot','多条件下的误差分布',draw23),('kde_plot','KDE Plot','连续误差密度比较',draw24),('ecdf_plot','ECDF Plot','误差低于阈值的比例',draw25),('histogram','Histogram','误差频次或尾部分布',draw26),('qq_plot','Q-Q Plot','检验误差分布与参考分布的匹配程度',draw27)]),
('超参数敏感性与搜索',[
('contour_plot','Contour Plot','两个超参数的联合性能曲面',draw28),('surface_3d','3D Surface Plot','二维参数与损失的三维映射',draw29),('hyper_scatter','Scatter Plot','参数取值与性能关联',draw30),('hyper_bubble','Bubble Chart','参数、得分与资源量三重编码',draw31),('parallel_coordinates','Parallel Coordinates Plot','多超参数组合与最终得分',draw32),('scatter_matrix','Scatter Matrix / Pair Plot','参数之间的成对关系',draw33),('response_surface','Response Surface Plot','拟合的响应曲面与采样点',draw34),('small_multiples','Small Multiples / Facet Grid','跨数据集参数响应对比',draw35),('ternary_plot','Ternary Plot','总和固定的三个权重',draw36)]),
('计算成本与性能权衡',[
('pareto_frontier','Pareto Frontier Plot','性能与成本间的非支配关系（示意）',draw37),('accuracy_latency','Accuracy–Latency Scatter','精度与推理延迟的取舍',draw38),('efficiency_bubble','Cost Bubble Plot','延迟、性能与模型规模',draw39),('roofline_plot','Roofline Plot','硬件峰值与计算访存瓶颈',draw40),('cost_breakdown','Stacked Cost Breakdown','多个模块的推理耗时',draw41),('sankey_cost','Sankey / Flow Diagram','计算、数据或样本流向（示意）',draw42),('gantt_chart','Gantt Chart','推理流水线耗时和时序',draw43),('speedup_plot','Speedup Plot','扩展资源量下的加速比',draw44)]),
('预测质量与诊断',[
('prediction_scatter','Prediction vs Ground Truth Scatter','预测值与实际值的一致性',draw45),('residual_plot','Residual Plot','是否存在系统性残差模式',draw46),('bland_altman','Bland–Altman Plot','预测与真实值之间的一致性差异',draw47),('fan_chart','Fan Chart','不同置信/预测区间的不确定性扩张',draw48),('quantile_band','Quantile Band Plot','多个预测分位数及真值',draw49),('error_profile','Error Profile','预测步长与绝对误差',draw50),('taylor_diagram','Taylor Diagram','相关性、标准差与中心化 RMSE',draw51),('coverage_plot','Coverage Plot','名义覆盖率与经验覆盖率',draw52),('reliability_diagram','Reliability Diagram','分类概率校准程度',draw53),('roc_pr_curve','ROC / PR Curves','分类阈值下的精度—召回权衡',draw54),('confusion_matrix','Confusion Matrix','各分类之间的混淆情况',draw55)]),
('表示学习与可解释性',[
('tsne_projection','t-SNE Projection','高维表示的二维可视化示意',draw56),('umap_projection','UMAP Projection','嵌入样本的局部和全局结构展示',draw57),('pca_projection','PCA Projection','主成分空间的投影展示',draw58),('shap_beeswarm','SHAP Beeswarm Plot','特征归因的方向和分布',draw59),('shap_waterfall','SHAP Waterfall Plot','单样本预测的特征贡献',draw60),('network_graph','Network Graph','节点和边的拓扑结构',draw61),('sankey_alluvial','Sankey / Alluvial Plot','状态与路径转移',draw62),('activation_map','Activation Map','模型的空间激活强度',draw63),('decision_boundary','Decision Boundary Plot','二维特征空间中的分类边界',draw64),('cka_similarity','CKA Similarity Matrix','模型层间表示相似度',draw65)]),
('时空预测与动态决策',[
('spatial_error_map','Spatial Error Map','节点/道路传感器的误差空间分布',draw66),('node_improvement','Node-Level Improvement Plot','不同节点的修正收益',draw67),('temporal_error_profile','Temporal Error Profile','不同预测 horizon 的误差曲线',draw68),('spatiotemporal_surface','Spatio-Temporal Surface','时间、节点与误差的联合变化',draw69),('graph_overlay','Graph Overlay Plot','在空间网络上展示关键决策',draw70),('event_trajectory','Event-Aligned Trajectory','异常事件前后新旧预测比较',draw71),('revision_timeline','Revision Timeline','逐时刻记录节点修正动作',draw72),('gain_distribution','Gain Distribution Plot','有益与有害修正的分布',draw73),('budget_performance','Budget–Performance Curve','有限修正预算下的误差变化',draw74),('state_transition','State Transition Diagram','策略中的状态和行动路径',draw75)])]

if __name__=='__main__':
    import sys
    for cat,items in CATEGORIES:
        print('Generating:',cat,len(items),flush=True)
        for slug,title,caption,func in items:
            try:export(slug,title,func,cat,caption)
            except Exception as exc:
                print('FAILED',slug,type(exc).__name__,str(exc),flush=True)
                raise
    print('Saved',len(meta),'PNG and SVG pairs',flush=True)
