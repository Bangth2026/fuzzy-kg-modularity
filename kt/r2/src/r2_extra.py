"""Numbers for the two additions of the second round: the pyKT-protocol comparison and the DKT-influence graph at
matched density. Writes paper/r2_extra.json; r2_stats.py reads it."""
import json, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
K='/home/claude/kt'
out={}
# ---- pyKT protocol
R=[json.load(open(f)) for f in glob.glob('/home/claude/pyktp/runs/*.json')]
pk=pd.DataFrame(R)
PUB={  # pyKT (NeurIPS 2022) Table 10 and simpleKT (ICLR 2023) Table 2: test AUC, mean ± sd over 5 folds, tuned by Bayesian search
 ('statics2011','dkt'):(0.8222,0.0013),('statics2011','sakt'):(0.7965,0.0014),('statics2011','akt'):(0.8309,0.0009),('statics2011','simplekt'):(0.8199,0.0011),
 ('assist2015','dkt'):(0.7271,0.0005),('assist2015','sakt'):(0.7114,0.0003),('assist2015','akt'):(0.7281,0.0004),('assist2015','simplekt'):(0.7248,0.0005)}
if len(pk):
    t=pk.groupby(['dataset','model']).agg(m=('test_auc','mean'),s=('test_auc','std'),n=('test_auc','size'),ep=('epochs','mean'),secs=('secs','mean'))
    out['pykt']={f'{d}|{m}':{'mean':float(r.m),'sd':None if r.n<2 else float(r.s),'n':int(r.n),'epochs':float(r.ep),'secs':float(r.secs),
                            'published':PUB.get((d,m),(None,None))[0],'published_sd':PUB.get((d,m),(None,None))[1]} for (d,m),r in t.iterrows()}
    # paired by fold: support graph minus empty graph
    for d in pk.dataset.unique():
        x=pk[(pk.dataset==d)&(pk.model=='fkg_support')].set_index('fold').test_auc; y=pk[(pk.dataset==d)&(pk.model=='fkg_empty')].set_index('fold').test_auc
        j=x.index.intersection(y.index)
        if len(j): out['pykt'][f'{d}|support-empty']={'mean':float((x[j]-y[j]).mean()),'pos':int(((x[j]-y[j])>0).sum()),'n':int(len(j))}
        e=pk[(pk.dataset==d)&(pk.model=='fkg_support')].get('edges')
        if e is not None and len(e.dropna()): out['pykt'][f'{d}|edges']=float(e.dropna().mean())
# ---- DKT influence at matched density (10%, sample 0, seeds 0-2, fresh runs)
rows=[]
for f in glob.glob(f'{K}/runs/*_f10__fkgkt3__*__s[012].json'):
    j=json.load(open(f)); t=j['tag']
    if not (t.startswith(tuple(['assist09_f10__','assist12_f10__','assist15_f10__','assist17_f10__','junyi_f10__','algebra05_f10__','eedi_f10__','ednet_f10__']))): continue
    p=t.split('__');
    if p[4]!='c0': continue
    rows.append(dict(data=p[0].split('_')[0],cons=p[2],graph=p[3],seed=j['seed'],auc=j['test_auc'],new=j['test_auc_newkc']))
d=pd.DataFrame(rows); D8=['assist09','assist12','assist15','assist17','junyi','algebra05','eedi','ednet']
d['cfg']=d.cons+':'+d.graph; m=d.groupby(['cfg','data'])[['auc','new']].mean()
res={}; fam=[]
for c in ('transition:support','dkteq:support','dkteq:rank','dkt10:support','dkt10:rank'):
    for met in ('auc','new'):
        ds=[x for x in D8 if (c,x) in m.index and ('na:empty',x) in m.index]
        if len(ds)<8: continue
        diff=np.array([m.loc[(c,x),met]-m.loc[('na:empty',x),met] for x in ds])
        res[f'{c}|{met}']={'mean':float(diff.mean()),'pos':int((diff>0).sum()),'n':len(ds),'p':float(wilcoxon(diff).pvalue),'per':{x:round(float(v),5) for x,v in zip(ds,diff)}}; fam.append(f'{c}|{met}')
for met in ('auc','new'):
    ds=[x for x in D8 if ('transition:support',x) in m.index and ('dkteq:support',x) in m.index]
    if len(ds)==8:
        diff=np.array([m.loc[('transition:support',x),met]-m.loc[('dkteq:support',x),met] for x in ds])
        res[f'op-minus-dkteq|{met}']={'mean':float(diff.mean()),'pos':int((diff>0).sum()),'n':8,'p':float(wilcoxon(diff).pvalue)}; fam.append(f'op-minus-dkteq|{met}')
# Holm
p=np.array([res[k]['p'] for k in fam]); o=np.argsort(p); adj=np.empty(len(p)); run=0
for i,j in enumerate(o): run=max(run,(len(p)-i)*p[j]); adj[j]=min(1,run)
for k,a in zip(fam,adj): res[k]['p_holm']=float(a)
res['n_seeds']=d.groupby(['cfg','data']).size().to_dict().__len__()
eq={}
for x in D8:
    a=json.load(open(f'{K}/graphs/{x}_f10_dkteq_audit.json')); eq[x]={'op':a['edges_target'],'dkteq':a['dkteq']['edges'],'dkt10':a['dkt10']['edges']}
res['edges']=eq
out['dkteq']=res
out['pykt_pub']={f'{d}|{m}':v for (d,m),v in PUB.items()}
json.dump(out,open(f'{K}/paper/r2_extra.json','w'),indent=1)
if __name__=='__main__':
    for k,v in out.get('pykt',{}).items(): print(k,v)
    for k,v in res.items():
        if isinstance(v,dict) and 'mean' in v: print(k,round(v['mean'],4),v['pos'],v.get('p_holm'))
