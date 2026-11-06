"""Covariance term of Lemma 2: bound B(Pi)=1/4 sum_k R(x_k)R(y_k) and decision flips."""
import sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/code/fuzzy-kg-modularity-1.4.3')
from fuzzykg.graph import FuzzyGraph
from fuzzykg.modularity import fuzzy_modularity
from fuzzykg.leiden import fuzzy_leiden
from fuzzykg.multislice import level_grid, qbar
def shares(G,memb,alphas):
    memb=np.asarray(memb); K=memb.max()+1; m=np.array([(G.mu>=a).sum() for a in alphas],float)
    X=np.zeros((K,len(alphas)));Y=np.zeros((K,len(alphas)))
    for l,a in enumerate(alphas):
        e=G.mu>=a; X[:,l]=np.bincount(memb[G.src[e]],minlength=K)/m[l]; Y[:,l]=np.bincount(memb[G.dst[e]],minlength=K)/m[l]
    return X,Y,m/m.sum()
def cov_and_bound(G,memb,alphas):
    X,Y,p=shares(G,memb,alphas)
    cov=((p*X*Y).sum(1)-(p*X).sum(1)*(p*Y).sum(1)).sum()
    sd=np.sqrt(np.maximum((p*X*X).sum(1)-(p*X).sum(1)**2,0)*np.maximum((p*Y*Y).sum(1)-(p*Y).sum(1)**2,0)).sum()
    B=0.25*((X.max(1)-X.min(1))*(Y.max(1)-Y.min(1))).sum()
    return float(cov),float(sd),float(B)
def rankQ(G,memb,alphas):
    N=(G.mu[:,None]>=alphas[None,:]).sum(1)/len(alphas); return fuzzy_modularity(FuzzyGraph(G.n,G.src,G.dst,N),memb)
def best(G,seeds=range(3)): return max((fuzzy_leiden(G,1.0,seed=s,n_iterations=2) for s in seeds),key=lambda r:r.Q_F).membership
def audit_graph(G):
    al=level_grid(G); rows=[]
    parts={'fuzzy':best(G)}
    for q in (0.25,0.5,0.75):
        a=np.quantile(G.mu,q); e=G.mu>=a; parts[f'cut{int(q*100)}']=best(FuzzyGraph(G.n,G.src[e],G.dst[e],np.ones(e.sum())))
    parts['support']=best(FuzzyGraph(G.n,G.src,G.dst,np.ones(G.m)))
    for k,P in parts.items():
        c,sd,B=cov_and_bound(G,P,al); rows.append({'partition':k,'qbar':qbar(G,P,al),'rankQ':rankQ(G,P,al),'cov':c,'cs_bound':sd,'range_bound':B})
    return rows
def flips(rows):
    n=0;tot=0;margins=[]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            a,b=rows[i],rows[j]; tot+=1
            d1=a['qbar']-b['qbar']; d2=a['rankQ']-b['rankQ']
            if np.sign(d1)!=np.sign(d2) and abs(d1)>1e-12: n+=1
            margins.append(abs(d2)-(a['range_bound']+b['range_bound']))
    return n,tot,margins
if __name__=='__main__':
    out={}
    files=sorted(glob.glob('/home/claude/code/fuzzy-kg-modularity-1.4.3/data/graphs/*_edges.csv'))
    for f in files:
        name=f.split('/')[-1][:-10]; e=pd.read_csv(f); n=int(max(e.src.max(),e.dst.max()))+1
        G=FuzzyGraph(n,e.src.values,e.dst.values,e.mu.values.astype(float)); rows=audit_graph(G); nf,tot,mg=flips(rows)
        out[name]={'rows':rows,'flips':nf,'pairs':tot,'max_abs_cov':max(abs(r['cov']) for r in rows),'certified_pairs':int(sum(m>0 for m in mg))}
        print(name,'max|cov|',round(out[name]['max_abs_cov'],4),'flips',nf,'/',tot,'certified',out[name]['certified_pairs'])
    json.dump(out,open('/home/claude/kt/theory/covbound_real.json','w'),indent=1)
