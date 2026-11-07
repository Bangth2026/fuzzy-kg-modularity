"""Fuzzy prerequisite KGs built from TRAINING learners only, their variants, and the audits."""
import os, sys, json, hashlib, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0,'/home/claude/code/fuzzy-kg-modularity-1.4.3')
from fuzzykg.graph import FuzzyGraph
from fuzzykg.construction import build_transition_graph, TransitionConfig, build_fuzzy_graph, ConstructionConfig
from fuzzykg.modularity import fuzzy_modularity
from fuzzykg.leiden import fuzzy_leiden
from fuzzykg.paths_fast import strengths
D='/home/claude/kt/data'; G='/home/claude/kt/graphs'
MAXLEN=200
def split_of(u):
    h=int(hashlib.md5(str(u).encode()).hexdigest(),16)%100
    return 'train' if h<70 else ('valid' if h<80 else 'test')
def gsuffix(frac=1.0,sub=0):
    """Graph-file suffix of a training subsample: '' (all), '_f10' (sub 0, as in R1), '_f10s3' (sub 3)."""
    if frac>=1.0: return ''
    return f'_f{int(round(frac*100))}'+(f's{sub}' if sub else '')
def load(name,frac=1.0,sub=0):
    df=pd.read_parquet(f'{D}/{name}.parquet')
    df=df.groupby('user',sort=False).head(MAXLEN).reset_index(drop=True)   # first MAXLEN interactions per learner
    df['split']=df.user.map(split_of)
    if frac<1.0:
        # sub=0 reproduces the single 10% sample of R1; sub>0 draws independent subsamples of training learners
        tr=np.array(sorted(df.loc[df.split=='train','user'].unique())); keep=set(np.random.default_rng(12345+1000*sub).choice(tr,max(2,int(round(frac*len(tr)))),replace=False))
        df=df[(df.split!='train')|df.user.isin(keep)].reset_index(drop=True)
    return df
def to_dense(Gf,K,mu=None):
    A=np.zeros((K,K),dtype=np.float32)
    A[Gf.src,Gf.dst]=Gf.mu if mu is None else mu
    return A
def best_part(Gf,seeds=range(5)):
    runs=[fuzzy_leiden(Gf,1.0,seed=s,n_iterations=2) for s in seeds]
    return max(runs,key=lambda r:r.Q_F).membership
def fg_from(Gf,mu): return FuzzyGraph.from_edges(Gf.n,Gf.src,Gf.dst,mu) if hasattr(FuzzyGraph,'from_edges') else None
def mk(K,s,d,w): return FuzzyGraph(n=K,src=np.asarray(s,dtype=np.int64),dst=np.asarray(d,dtype=np.int64),mu=np.asarray(w,dtype=float))
def rank_code(mu,L=8):
    qs=np.unique(np.quantile(mu,np.arange(L)/L))
    return (mu[:,None]>=qs[None,:]).sum(1)/len(qs)
def build(name, constructions=('transition','copractice'),frac=1.0,sub=0):
    df=load(name,frac,sub); sfx=gsuffix(frac,sub); K=int(df.kc.max())+1
    tr=df[df.split=='train'].rename(columns={'user':'user_id'})[['user_id','kc','correct','order']]
    out={'name':name,'K':K,'n_train_learners':int(tr.user_id.nunique()),'n_train_interactions':len(tr)}
    for c in constructions:
        if c=='transition':
            Gf,tab=build_transition_graph(tr,TransitionConfig(min_support=10,tau_E=0.0))
        else:
            Gf,tab=build_fuzzy_graph(tr,ConstructionConfig(min_support=20,tau_E=0.2))
        save_variants(name+sfx,c,Gf,K,out)
    json.dump(out,open(f'{G}/{name}{sfx}_audit.json','w'),indent=1)
    return out
def save_variants(name,c,Gf,K,out,seed=0):
    rng=np.random.default_rng(seed)
    # node ids: FuzzyGraph labels are the kc ints as strings or ints
    lab=np.array([int(x) for x in Gf.labels])
    src,dst,mu=lab[Gf.src],lab[Gf.dst],Gf.mu.astype(float)
    def dense(w,s=src,d=dst):
        A=np.zeros((K,K),dtype=np.float32); A[s,d]=w; return A
    full=mk(K,src,dst,mu)
    rk=rank_code(mu)
    Grank=mk(K,src,dst,rk)
    PiF=best_part(full); PiR=best_part(Grank)
    QF=fuzzy_modularity(full,PiF); QR=fuzzy_modularity(Grank,PiR)
    V={'graded':dense(mu),'rank':dense(rk),'support':dense(np.ones_like(mu)),'permuted':dense(rank_code(rng.permutation(mu)))}
    rec={'nodes_with_edges':int(len(set(src)|set(dst))),'edges':int(len(mu)),'mu_mean':float(mu.mean()),'mu_sd':float(mu.std()),
         'reciprocity':float(full.reciprocity()) if hasattr(full,'reciprocity') else None,'Q_F':float(QF),'k':int(len(set(PiF))),'Q_F_rank':float(QR),'cuts':{}}
    np.save(f'{G}/{name}_{c}_partition.npy',np.asarray(PiF))
    for q in (0.25,0.5,0.75):
        a=np.quantile(mu,q); keep=mu>=a
        Gc=mk(K,src[keep],dst[keep],np.ones(keep.sum()))
        Pic=best_part(Gc)
        drop=QF-fuzzy_modularity(full,Pic); rdrop=QR-fuzzy_modularity(Grank,Pic)
        V[f'cut{int(q*100)}']=dense(np.ones(keep.sum()),src[keep],dst[keep])
        rec['cuts'][f'cut{int(q*100)}']={'alpha':float(a),'retained_edges':int(keep.sum()),'retained_mass':float(mu[keep].sum()/mu.sum()),
            'mean_outdeg':float(keep.sum()/K),'qf_drop':float(drop),'rank_drop':float(rdrop)}
    # support (no-cut) baseline audit
    Pis=best_part(mk(K,src,dst,np.ones_like(mu)))
    rec['cuts']['support']={'qf_drop':float(QF-fuzzy_modularity(full,Pis)),'rank_drop':float(QR-fuzzy_modularity(Grank,Pis)),'retained_mass':1.0}
    # permuted-memberships audit
    Gp=mk(K,src,dst,rng.permutation(mu)); Pip=best_part(Gp)
    rec['cuts']['permuted']={'qf_drop':float(QF-fuzzy_modularity(full,Pip)),'rank_drop':float(QR-fuzzy_modularity(Grank,Pip)),'retained_mass':1.0}
    # product closure (<=3 hops) on the graded graph, rank-coded
    S,H=strengths(full,'product',max_hops=3); S=np.where(S>0,S,0.0); np.fill_diagonal(S,0)
    nz=S>0; Sr=np.zeros_like(S,dtype=np.float32); Sr[nz]=rank_code(S[nz]); V['product3']=Sr
    rec['product3_edges']=int(nz.sum())
    np.savez_compressed(f'{G}/{name}_{c}.npz',**V)
    out[c]=rec
def kc_counts(name,frac=1.0,sub=0):
    """Training interactions per KC in the (sub)sample: the input of the count-aware gate."""
    df=load(name,frac,sub); K=int(df.kc.max())+1; tr=df[df.split=='train']
    n=np.bincount(tr.kc.values,minlength=K).astype(np.float32); np.save(f'{G}/{name}{gsuffix(frac,sub)}_counts.npy',n); return n
def phi_rel(name,frac=1.0,sub=0,win=10):
    """RKT-style exercise relation: phi coefficient between correctness on c_i and on a later c_j (within win steps),
    training learners only; negative values set to 0 (Pandey and Srivastava 2020, no text available)."""
    df=load(name,frac,sub); K=int(df.kc.max())+1; tr=df[df.split=='train']
    N=np.zeros((2,2,K,K))
    for _,g in tr.groupby('user',sort=False):
        q=g.kc.values; r=g.correct.values
        for d in range(1,win+1):
            if len(q)<=d: break
            np.add.at(N,(r[:-d],r[d:],q[:-d],q[d:]),1)
    n11,n10,n01,n00=N[1,1],N[1,0],N[0,1],N[0,0]
    num=n11*n00-n10*n01; den=np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    P=np.where(den>0,num/np.maximum(den,1e-12),0.0); P=np.clip(P,0,None); np.fill_diagonal(P,0)
    np.save(f'{G}/{name}{gsuffix(frac,sub)}_phi.npy',P.astype(np.float32)); return P
def gkt_adj(name,frac=1.0,sub=0):
    """GKT 'transition graph': A_ij = n(i->j)/sum_k n(i->k) over consecutive answers of training learners."""
    df=load(name,frac,sub); K=int(df.kc.max())+1; tr=df[df.split=='train']; name=name+gsuffix(frac,sub)
    q=tr.kc.values; u=tr.user.values; same=u[1:]==u[:-1]; a=q[:-1][same]; b=q[1:][same]; keep=a!=b
    C=np.zeros((K,K)); np.add.at(C,(a[keep],b[keep]),1)
    A=C/np.maximum(C.sum(1,keepdims=True),1); np.save(f'{G}/{name}_gktA.npy',A.astype(np.float32))

if __name__=='__main__':
    fr=float(os.environ.get('FRAC','1')); sub=int(os.environ.get('SUB','0')); only=os.environ.get('CONS','transition,copractice').split(',')
    for n in sys.argv[1:]:
        gkt_adj(n,fr,sub); kc_counts(n,fr,sub)
        if os.environ.get('PHI','0')=='1': phi_rel(n,fr,sub)
        import time; t=time.time(); r=build(n,constructions=tuple(only),frac=fr,sub=sub); print(n,round(time.time()-t,1),'s',{c:(r[c]['edges'],round(r[c]['Q_F'],3),r[c]['k']) for c in ('transition','copractice') if c in r})

