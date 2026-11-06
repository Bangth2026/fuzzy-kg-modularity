"""Directed weighted LFR benchmark (Lancichinetti & Fortunato, PRE 80, 016118, 2009), re-implemented,
and the validity of the cut audits against the recovery shortfall on it.

Generator: power-law in-degrees (exponent tau1, mean k, max kmax) and out-degrees with the same law and total;
power-law community sizes (exponent tau2, in [cmin, cmax]); a fraction 1-mu_t of every node's in- and
out-stubs is wired inside its community (stub matching, self-loops and multi-edges rejected); weights follow
the LFR strength rule s_i = k_i^beta with a fraction 1-mu_w of each node's strength on internal edges,
w_ij proportional to (s_i^out s_j^in)^(1/2) within the internal and the external edge set separately; weights are
scaled to (0,1] to serve as memberships."""
import sys, json, numpy as np
sys.path.insert(0,'/home/claude/code/fuzzy-kg-modularity-1.4.3')
from fuzzykg.graph import FuzzyGraph
from fuzzykg.modularity import fuzzy_modularity
from fuzzykg.leiden import fuzzy_leiden
from sklearn.metrics import adjusted_rand_score as ari
from scipy.stats import spearmanr

def powerlaw_int(rng,n,expo,xmin,xmax):
    u=rng.random(n); a=1-expo
    if abs(a)<1e-9: return np.floor(xmin*(xmax/xmin)**u).astype(int)
    return np.floor(((xmax**a-xmin**a)*u+xmin**a)**(1/a)).astype(int)
def degrees(rng,N,k,kmax,tau1):
    for xmin in np.arange(1,k,0.25):
        d=powerlaw_int(rng,200000,tau1,xmin,kmax)
        if d.mean()>=k: break
    return powerlaw_int(rng,N,tau1,xmin,kmax)
def lfr_dw(N=1000,k=10,kmax=50,tau1=2.0,tau2=1.0,mu_t=0.3,mu_w=0.3,beta=1.5,cmin=20,cmax=100,seed=0):
    rng=np.random.default_rng(seed)
    kin=degrees(rng,N,k,kmax,tau1); kout=rng.permutation(kin)
    # community sizes
    sizes=[]
    while sum(sizes)<N: sizes.append(int(powerlaw_int(rng,1,tau2,cmin,cmax)[0]))
    sizes[-1]-=sum(sizes)-N
    if sizes[-1]<cmin: sizes[-2]+=sizes[-1]; sizes=sizes[:-1]
    # assign nodes to communities with room for their internal degree
    order=np.argsort(-np.maximum(kin,kout)); comm=-np.ones(N,int); free=np.array(sizes)
    for v in order:
        need=(1-mu_t)*max(kin[v],kout[v])
        ok=np.flatnonzero((free>0)&(np.array(sizes)-1>=need))
        c=rng.choice(ok if len(ok) else np.flatnonzero(free>0)); comm[v]=c; free[c]-=1
    # stub matching inside and between communities
    E=set()
    def match(outs,ins,internal):
        for _ in range(20):
            rng.shuffle(outs); rng.shuffle(ins); lo=[];li=[]
            for u,v in zip(outs,ins):
                bad=(u==v) or ((u,v) in E) or (internal and comm[u]!=comm[v]) or ((not internal) and comm[u]==comm[v])
                if bad: lo.append(u); li.append(v)
                else: E.add((u,v))
            if not lo: break
            outs=np.array(lo); ins=np.array(li)
    kin_int=np.round((1-mu_t)*kin).astype(int); kout_int=np.round((1-mu_t)*kout).astype(int)
    for c in range(len(sizes)):
        mem=np.flatnonzero(comm==c)
        outs=np.repeat(mem,kout_int[mem]); ins=np.repeat(mem,kin_int[mem]); m=min(len(outs),len(ins))
        match(rng.permutation(outs)[:m],rng.permutation(ins)[:m],True)
    outs=np.repeat(np.arange(N),kout-kout_int); ins=np.repeat(np.arange(N),kin-kin_int); m=min(len(outs),len(ins))
    match(rng.permutation(outs)[:m],rng.permutation(ins)[:m],False)
    e=np.array(sorted(E)); src,dst=e[:,0],e[:,1]
    dout=np.bincount(src,minlength=N); din=np.bincount(dst,minlength=N)
    s_out=np.maximum(dout,1)**beta; s_in=np.maximum(din,1)**beta
    raw=np.sqrt(s_out[src]*s_in[dst]); internal=comm[src]==comm[dst]
    w=np.empty(len(e))
    tot=(s_out.sum()+s_in.sum())/2
    w[internal]=raw[internal]/raw[internal].sum()*(1-mu_w)*tot
    if (~internal).any(): w[~internal]=raw[~internal]/raw[~internal].sum()*mu_w*tot
    w=w/w.max()
    return FuzzyGraph(N,src,dst,np.clip(w,1e-6,1.0)),comm,float(internal.mean())

def best(G,seeds=(0,1)): return max((fuzzy_leiden(G,1.0,seed=s,n_iterations=2) for s in seeds),key=lambda r:r.Q_F).membership
def rank_code(mu,L=8):
    qs=np.unique(np.quantile(mu,np.arange(L)/L)); return (mu[:,None]>=qs[None,:]).sum(1)/len(qs)
def run_graph(G,truth):
    rk=rank_code(G.mu); Gr=FuzzyGraph(G.n,G.src,G.dst,rk)
    PF=best(G); PR=best(Gr); QF=fuzzy_modularity(G,PF); QR=fuzzy_modularity(Gr,PR); aF=ari(truth,PF)
    rows=[]
    for q in (0.25,0.5,0.75):
        a=np.quantile(G.mu,q); keep=G.mu>=a
        Pc=best(FuzzyGraph(G.n,G.src[keep],G.dst[keep],np.ones(keep.sum())))
        rows.append({'cut':q,'shortfall':1-ari(truth,Pc),'loss_vs_F':aF-ari(truth,Pc),'qf_drop':QF-fuzzy_modularity(G,Pc),
                     'rank_drop':QR-fuzzy_modularity(Gr,Pc),'retained_mass':float(G.mu[keep].sum()/G.mu.sum()),'mean_outdeg':float(keep.sum()/G.n)})
    return rows,aF
if __name__=='__main__':
    out=[]
    for N in (1000,2000):
        for mu_t in (0.1,0.2,0.3,0.4,0.5):
            for mu_w in (0.1,0.3,0.5):
                for seed in range(3):
                    G,truth,intf=lfr_dw(N=N,mu_t=mu_t,mu_w=mu_w,seed=seed)
                    rows,aF=run_graph(G,truth)
                    for r in rows: out.append({'N':N,'mu_t':mu_t,'mu_w':mu_w,'seed':seed,'internal_frac':intf,'ari_F':aF,**r})
                    print(N,mu_t,mu_w,seed,round(intf,2),round(aF,3),[round(r['shortfall'],3) for r in rows],flush=True)
    json.dump(out,open('/home/claude/kt/theory/lfr_dw.json','w'),indent=1)
    import pandas as pd
    d=pd.DataFrame(out); res={}
    for k,s in (('rank_drop',1),('qf_drop',1),('retained_mass',-1),('mean_outdeg',-1)):
        res[k]={'rho_shortfall':float(spearmanr(s*d[k],d.shortfall).statistic),'rho_loss':float(spearmanr(s*d[k],d.loss_vs_F).statistic)}
        conc=[];
        for _,g in d.groupby(['N','mu_t','mu_w','seed']):
            v=g[k].values*s; y=g.shortfall.values
            for i in range(3):
                for j in range(i+1,3):
                    if abs(y[i]-y[j])>=0.01: conc.append(np.sign(v[i]-v[j])==np.sign(y[i]-y[j]))
        res[k]['concordance']=float(np.mean(conc)); res[k]['pairs']=len(conc)
    json.dump(res,open('/home/claude/kt/theory/lfr_dw_summary.json','w'),indent=1); print(json.dumps(res,indent=1))
