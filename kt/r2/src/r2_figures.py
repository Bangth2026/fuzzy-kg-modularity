"""Figures of R2. Fig. 2: gain over the empty-graph control against training interactions per KC.
Fig. 3: architecture ablations and the second channel."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
K='/home/claude/kt'; df=pd.read_csv(f'{K}/paper/r2_runs.csv')
D10=['assist09','assist12','assist15','assist17','junyi','algebra05','eedi','ednet','bridge06','statics11']
NAMES=dict(zip(D10,['AS09','AS12','AS15','AS17','Junyi','Alg05','Eedi','EdNet','Bridge06','Statics11']))
plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
def gains(a,b,met):
    x=df[df.cfg==a].set_index(['data','frac','sub','seed'])[met]; y=df[df.cfg==b].set_index(['data','frac','sub','seed'])[met]
    j=x.index.intersection(y.index); d=(x[j]-y[j]).reset_index(); d.columns=['data','frac','sub','seed','d']
    ipk=df.drop_duplicates(['data','frac','sub']).set_index(['data','frac','sub']).ipk
    d['ipk']=[ipk[(r.data,r.frac,r['sub'])] for _,r in d.iterrows()]
    return d[d.data.isin(D10)]
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7),sharey=False)
COL={'P/support':'#1f5fa8','P/cag':'#c4581b','P/gate2':'#3a8a3a','P/support-pooled':'#7a4fa0'}; LAB={'P/support':'support graph (own sample)','P/cag':'CAG (monotone gate)','P/gate2':'free-sign gate','P/support-pooled':'support graph (all learners)'}
for k,met in enumerate(('auc','new')):
    a_=ax[k]; a_.axhline(0,color='0.5',lw=0.6)
    for cfg in ('P/support','P/support-pooled','P/cag','P/gate2'):
        d=gains(cfg,'P/empty',met)
        if len(d)==0: continue
        if cfg=='P/support':
            for c,g in d.groupby('data'):
                m=g.groupby('frac')[['ipk','d']].mean().sort_index(); a_.plot(m.ipk,m.d,color=COL[cfg],lw=0.5,alpha=0.25)
        m=d.groupby(['frac','data'])[['ipk','d']].mean().groupby('frac').agg(['mean','sem'])
        a_.errorbar(np.exp(np.log(d.groupby(['frac','data']).ipk.mean()).groupby('frac').mean()),m[('d','mean')],yerr=m[('d','sem')],color=COL[cfg],marker='o',ms=3,lw=1.2,capsize=2,label=LAB[cfg])
    a_.set_xscale('log'); a_.set_xlabel('training interactions per KC (log scale)')
    a_.set_ylabel('AUC gain over empty-graph control' if k==0 else 'first-attempt AUC gain')
    a_.set_title('(a) all predictions' if k==0 else '(b) first attempt on a KC',loc='left',fontsize=8)
ax[0].legend(frameon=False,fontsize=7)
fig.tight_layout(); fig.savefig(f'{K}/paper/fig2_learning_curve.png',dpi=300)
# Fig 3: ablations and channel
o=json.load(open(f'{K}/paper/r2_results.json'))
items=[('remove GRU',o['ablation'].get('gru')),('remove attention',o['ablation'].get('att')),('remove same-KC bias',o['ablation'].get('same')),('remove h⊙z',o['ablation'].get('hz')),
       ('add support graph',o['graph_vs_empty'].get('0.1|auc|P/support|P/empty')),('add support graph + CAG',o['graph_vs_empty'].get('0.1|auc|P/cag|P/empty')),('add PC loss, graded',o['pc_loss'].get('graded|auc')),('add PC loss, permuted',o['pc_loss'].get('permuted|auc'))]
items=[(n,v) for n,v in items if v]
fig,a_=plt.subplots(figsize=(4.6,2.6))
for i,(n,v) in enumerate(items):
    pc=np.array(list(v['per_corpus'].values())); a_.scatter(pc,np.full(len(pc),i),s=7,color='0.55',zorder=2)
    a_.plot([v['mean']],[i],'D',color='#1f5fa8' if v['mean']>=0 else '#a8321f',ms=4,zorder=3)
a_.axvline(0,color='0.5',lw=0.6); a_.set_yticks(range(len(items))); a_.set_yticklabels([n for n,_ in items]); a_.invert_yaxis()
a_.set_xlabel('AUC change vs. empty-graph control (10% of learners)')
fig.tight_layout(); fig.savefig(f'{K}/paper/fig3_ablation.png',dpi=300)
print('ok')
