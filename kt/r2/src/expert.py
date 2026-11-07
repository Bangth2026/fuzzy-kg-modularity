"""Expert knowledge graphs: Junyi knowledge map (prerequisite links); grouping graphs for Eedi
(subject tree), Algebra 2005 (curriculum unit), EdNet (TOEIC part). Saved as dense KxK arrays."""
import json, numpy as np, pandas as pd, sys
sys.path.insert(0,'/home/claude/kt/src'); from graphs import load, G
def kcnames(name):
    df=pd.read_parquet(f'/home/claude/kt/data/{name}.parquet',columns=['kc','kc_name']).drop_duplicates('kc')
    names=['__none__']*(int(df.kc.max())+1)
    for k,n in zip(df.kc,df.kc_name): names[int(k)]=n
    return names
def junyi():
    names=kcnames('junyi'); idx={n:i for i,n in enumerate(names)}; K=len(names)
    ex=pd.read_csv('/home/claude/raw/junyi/junyi_Exercise_table.csv'); ex.columns=[c.strip('﻿') for c in ex.columns]
    A=np.zeros((K,K),dtype=np.float32); n=0
    for _,r in ex.iterrows():
        if r['name'] not in idx or pd.isna(r['prerequisites']): continue
        for p in str(r['prerequisites']).split(','):
            p=p.strip()
            if p in idx and p!=r['name']: A[idx[p],idx[r['name']]]=1; n+=1
    return A,{'expert_edges':n,'type':'prerequisite map'}
def grouping(name):
    names=kcnames(name); g=json.load(open(f'/home/claude/kt/data/{name}_groups.json')); K=len(names)
    lab=np.array([str(g.get(n)) for n in names]); A=(lab[:,None]==lab[None,:]).astype(np.float32); np.fill_diagonal(A,0)
    return A,{'groups':int(len(set(lab))),'type':'grouping'}
if __name__=='__main__':
    info={}
    for name in sys.argv[1:]:
        A,i=junyi() if name=='junyi' else grouping(name); info[name]=i
        np.save(f'{G}/{name}_expert.npy',A); print(name,i,int(A.sum()))
