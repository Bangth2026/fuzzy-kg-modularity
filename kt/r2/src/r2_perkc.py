"""Gain of a graph input over the empty-graph control by KC data tercile (training interactions on the target KC).
Test predictions are matched to their target KC by rebuilding the test sequences in the order evaluate() uses."""
import sys, json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/home/claude/kt/src'); from graphs import load, gsuffix
K_='/home/claude/kt'
def targets(data,frac,sub):
    df=load(data,frac,sub); K=int(df.kc.max())+1; d=df[df.split=='test']; T=[]
    for u,g in d.groupby('user',sort=False):
        q=g.kc.values[:200]
        if len(q)<2: continue
        T.append(q[1:])
    return np.concatenate(T),K
def run(corpora,frac=0.1,subs=range(5),cfgs=(('support','transition__support__c0'),('cag','transition__support__c0_g')),ref='na__empty__c0'):
    rows=[]
    for c in corpora:
        for s in subs:
            sfx=gsuffix(frac,s); f0=f'{K_}/runs/{c}{sfx}__fkgkt3__{ref}__s{s}_pred.npz'
            if not os.path.exists(f0): continue
            kc,K=targets(c,frac,s); n=np.load(f'{K_}/graphs/{c}{sfx}_counts.npy')
            e=np.load(f0); assert len(e['p'])==len(kc), (c,s,len(e['p']),len(kc))
            nk=n[kc]; edges=np.quantile(n[n>0],[1/3,2/3]); bin_=np.digitize(nk,edges)
            for name,tag in cfgs:
                f=f'{K_}/runs/{c}{sfx}__fkgkt3__{tag}__s{s}_pred.npz'
                if not os.path.exists(f): continue
                p=np.load(f)['p']
                for b in range(3):
                    m=bin_==b
                    if m.sum()<200 or len(set(e['y'][m]))<2: continue
                    rows.append(dict(data=c,sub=s,cfg=name,bin=b,n=int(m.sum()),d=roc_auc_score(e['y'][m],p[m])-roc_auc_score(e['y'][m],e['p'][m])))
    return pd.DataFrame(rows)
if __name__=='__main__':
    C=['assist09','assist12','assist15','assist17','junyi','algebra05','eedi','ednet','bridge06','statics11']
    out={}
    for fr in (0.05,0.1,0.25):
        r=run(C,fr)
        if len(r)==0: continue
        r.to_csv(f'{K_}/paper/r2_perkc_{fr}.csv',index=False)
        m=r.groupby(['cfg','bin','data']).d.mean().groupby(['cfg','bin']).agg(['mean','count'])
        out[str(fr)]={f'{c}|{b}':[float(v['mean']),int(v['count'])] for (c,b),v in m.iterrows()}
        print(fr); print(m.round(4))
    json.dump(out,open(f'{K_}/paper/r2_perkc.json','w'),indent=1)
