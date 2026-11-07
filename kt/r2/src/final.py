import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
sys.path.insert(0,'/home/claude/kt/src'); import analyse as A
df=A.df; D=['assist09','assist12','assist15','assist17','junyi','algebra05','eedi','ednet']
R={}
P='FKG-KT/transition:product3'
main=['DKT','SAKT','AKT','GKT','FKG-KT/none',P,'FKG-KT/transition:rank','FKG-KT/dktinf:rank','FKG-KT/copractice:rank','FKG-KT/expert']
for fr in (0.1,1.0):
    for metric in ('test_auc','test_auc_newkc'):
        t=df[df.frac==fr].groupby(['cfg','data'])[metric].agg(['mean','std','count'])
        R[f'{metric}_{fr}']={f'{c}|{d}':[float(t.loc[(c,d),'mean']),float(t.loc[(c,d),'std']) if t.loc[(c,d),'count']>1 else None,int(t.loc[(c,d),'count'])] for c,d in t.index if c in main}
# paired bootstraps
R['boot']={}
for fr in (0.1,1.0):
    for d in D:
        for a,b in ((P,'FKG-KT/none'),(P,'DKT'),('FKG-KT/expert','FKG-KT/none'),('FKG-KT/none','DKT'),('FKG-KT/transition:support','FKG-KT/none')):
            r=A.paired_boot(d,fr,a,b,B=500)
            if r: R['boot'][f'{fr}|{d}|{a}|{b}']=r
# Wilcoxon across corpora on seed-mean diffs
R['wilcoxon']={}
for fr in (0.1,1.0):
    for a,b in ((P,'FKG-KT/none'),(P,'DKT'),(P,'SAKT'),(P,'AKT'),(P,'GKT'),('FKG-KT/none','AKT'),('FKG-KT/none','DKT'),('FKG-KT/transition:support','FKG-KT/none'),('FKG-KT/transition:rank','FKG-KT/transition:permuted'),('FKG-KT/transition:rank','FKG-KT/transition:support')):
        for metric in ('test_auc','test_auc_newkc'):
            m=df[df.frac==fr].groupby(['cfg','data'])[metric].mean()
            ds=[d for d in D if (a,d) in m.index and (b,d) in m.index]
            if len(ds)<5: continue
            diff=np.array([m[(a,d)]-m[(b,d)] for d in ds])
            R['wilcoxon'][f'{fr}|{metric}|{a}|{b}']={'n':len(ds),'mean':float(diff.mean()),'min':float(diff.min()),'max':float(diff.max()),'pos':int((diff>0).sum()),'p':float(wilcoxon(diff).pvalue)}
# audit validity
v=A.audit_validity(0.1); v.to_csv('/home/claude/kt/paper/audit_validity.csv',index=False)
R['audit']={'n':len(v)}
for k,s in (('rank_drop',1),('qf_drop',1),('retained_mass',-1),('mean_outdeg',-1)):
    R['audit'][k]=float(spearmanr(s*v[k],v.loss).statistic)
    cuts=v[v.graph.str.startswith('cut')]; R['audit'][k+'_cuts']=float(spearmanr(s*cuts[k],cuts.loss).statistic)
R['audit']['loss_by_graph']=v.groupby('graph').loss.agg(['mean','std','min','max']).round(5).to_dict()
# seed noise reference: sd across seeds of rank variant
R['seed_sd_sparse']=float(df[(df.frac==0.1)&(df.cfg=='FKG-KT/transition:rank')].groupby('data').test_auc.std().mean())
# variants summary sparse: mean over corpora of diff vs none
m=df[df.frac==0.1].groupby(['cfg','data']).test_auc.mean(); mn=df[df.frac==0.1].groupby(['cfg','data']).test_auc_newkc.mean()
R['variants_sparse']={c:{'auc_vs_none':float(np.mean([m[(c,d)]-m[('FKG-KT/none',d)] for d in D if (c,d) in m.index])),
                          'newkc_vs_none':float(np.mean([mn[(c,d)]-mn[('FKG-KT/none',d)] for d in D if (c,d) in mn.index])),
                          'pos':int(sum(m[(c,d)]>m[('FKG-KT/none',d)] for d in D if (c,d) in m.index)),'n':int(sum((c,d) in m.index for d in D))}
                       for c in sorted(set(df[(df.frac==0.1)&df.cfg.str.startswith('FKG-KT/')].cfg)) if c!='FKG-KT/none'}
# moderator: gain vs interactions per KC
st=pd.read_csv('/home/claude/kt/paper/table1.csv').set_index('data')
xs=[];ys=[];yn=[]
for fr in (0.1,1.0):
    mm=df[df.frac==fr].groupby(['cfg','data'])[['test_auc','test_auc_newkc']].mean()
    for d in D:
        if (P,d) in mm.index and ('FKG-KT/none',d) in mm.index:
            xs.append(st.loc[d,'train_int_per_kc']*(0.1 if fr<1 else 1)); ys.append(mm.loc[(P,d),'test_auc']-mm.loc[('FKG-KT/none',d),'test_auc']); yn.append(mm.loc[(P,d),'test_auc_newkc']-mm.loc[('FKG-KT/none',d),'test_auc_newkc'])
R['moderator']={'rho_auc':float(spearmanr(xs,ys).statistic),'p_auc':float(spearmanr(xs,ys).pvalue),'rho_new':float(spearmanr(xs,yn).statistic),'p_new':float(spearmanr(xs,yn).pvalue),'n':len(xs)}
json.dump(R,open('/home/claude/kt/paper/results.json','w'),indent=1)
print(json.dumps({k:R[k] for k in ('wilcoxon','audit','moderator','seed_sd_sparse')},indent=1)[:6000])
