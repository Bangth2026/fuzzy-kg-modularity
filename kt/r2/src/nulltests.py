"""Algorithm 1 (reduced draws) on the KG of each corpus and construction: degree-preserving null,
global and row weights-only nulls; expert-grouping agreement where a grouping exists."""
import sys, json, numpy as np
sys.path.insert(0,'/home/claude/code/fuzzy-kg-modularity-1.4.3'); sys.path.insert(0,'/home/claude/kt/src')
from fuzzykg.graph import FuzzyGraph
from fuzzykg.nulltest import best_qf, degree_test, weights_only_test
from sklearn.metrics import adjusted_rand_score as ari
G='/home/claude/kt/graphs'
def fg(name,c):
    A=np.load(f'{G}/{name}_{c}.npz')['graded']; s,d=np.nonzero(A); return FuzzyGraph(A.shape[0],s,d,A[s,d].astype(float))
def run(name,c,draws=50):
    Gf=fg(name,c); b=best_qf(Gf,1.0,(0,1,2))
    keep=np.zeros(Gf.n,bool); keep[Gf.src]=True; keep[Gf.dst]=True
    r={'n_active':int(keep.sum()),'m':Gf.m,'Q_F':b.Q_F,'k_active':int(len(set(b.membership[keep])))}
    r['degree']=degree_test(Gf,draws,1.0,(0,1,2),obs=b.Q_F)
    r['global']=weights_only_test(Gf,'global',draws,1.0,(0,1,2),obs=b.Q_F)
    r['row']=weights_only_test(Gf,'row',draws,1.0,(0,1,2),obs=b.Q_F)
    try:
        grp=json.load(open(f'/home/claude/kt/data/{name}_groups.json'))
        import pandas as pd
        names=pd.read_parquet(f'/home/claude/kt/data/{name}.parquet',columns=['kc','kc_name']).drop_duplicates('kc').sort_values('kc').kc_name.tolist()
        lab=np.array([str(grp.get(n)) for n in names])[:Gf.n]
        r['grouping_ari']=float(ari(lab[keep],b.membership[keep]))
        rng=np.random.default_rng(0); perm=[]
        for _ in range(20):
            P=best_qf(FuzzyGraph(Gf.n,Gf.src,Gf.dst,rng.permutation(Gf.mu)),1.0,(0,1,2)); perm.append(ari(lab[keep],P.membership[keep]))
        r['grouping_ari_permuted']=[float(np.mean(perm)),float(np.std(perm))]
    except FileNotFoundError: pass
    return r
if __name__=='__main__':
    out={}
    for name in sys.argv[1:]:
        for c in ('transition','copractice','dktinf'):
            try: out[f'{name}/{c}']=run(name,c); x=out[f'{name}/{c}']; print(name,c,round(x['Q_F'],3),'zdeg',round(x['degree']['z'],1),'zglob',round(x['global']['z'],1),'zrow',round(x['row']['z'],1),x.get('grouping_ari'),flush=True)
            except FileNotFoundError: print('missing',name,c)
        json.dump(out,open(f'{G}/nulltests_{"_".join(sys.argv[1:])}.json','w'),indent=1)
