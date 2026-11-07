import json, glob, numpy as np, pandas as pd, itertools
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
R='/home/claude/kt/runs'; G='/home/claude/kt/graphs'
rows=[json.load(open(f)) for f in glob.glob(f'{R}/*.json')]
df=pd.DataFrame(rows)
df['frac']=df.frac.astype(float)
def cfg(r):
    if r['model']!='fkgkt2': return r['model'].upper()
    if r['graph']=='none': return 'FKG-KT/none'
    if r['graph']=='expert': return 'FKG-KT/expert'
    return f"FKG-KT/{r['construction']}:{r['graph']}"
df['cfg']=df.apply(cfg,axis=1)
def table(frac):
    t=df[df.frac==frac].groupby(['data','cfg']).test_auc.agg(['mean','std','count']).reset_index()
    return t
def paired_boot(d,frac,a,b,B=1000,seed=0):
    """Seed-averaged predictions, paired bootstrap over test learners of AUC(a)-AUC(b)."""
    def avg(c):
        fs=sorted(df[(df.data==d)&(df.frac==frac)&(df.cfg==c)].tag)
        if not fs: return None
        Z=[np.load(f'{R}/{t}_pred.npz') for t in fs]; return np.mean([z['p'] for z in Z],0),Z[0]['y'],Z[0]['u'],Z[0]['new']
    A=avg(a);Bb=avg(b)
    if A is None or Bb is None: return None
    pa,y,u,new=A; pb=Bb[0]; rng=np.random.default_rng(seed); us=np.unique(u); idx={x:np.flatnonzero(u==x) for x in us}
    obs=roc_auc_score(y,pa)-roc_auc_score(y,pb); obs_new=roc_auc_score(y[new],pa[new])-roc_auc_score(y[new],pb[new])
    bs=[];bn=[]
    for _ in range(B):
        s=np.concatenate([idx[x] for x in rng.choice(us,len(us))]); yy=y[s]
        bs.append(roc_auc_score(yy,pa[s])-roc_auc_score(yy,pb[s])); nn=new[s]
        bn.append(roc_auc_score(yy[nn],pa[s][nn])-roc_auc_score(yy[nn],pb[s][nn]))
    return {'diff':obs,'ci':list(np.percentile(bs,[2.5,97.5])),'diff_new':obs_new,'ci_new':list(np.percentile(bn,[2.5,97.5]))}
if __name__=='__main__':
    for fr in (0.1,1.0):
        t=table(fr); print('frac',fr); print(t.pivot(index='cfg',columns='data',values='mean').round(4).to_string())

def audit_validity(frac=0.1):
    out=[]
    sfx='_f10' if frac<1 else ''
    for d in sorted(df.data.unique()):
        A=json.load(open(f'{G}/{d}{sfx}_audit.json'))
        for c in ('transition','copractice'):
            base=df[(df.data==d)&(df.frac==frac)&(df.cfg==f'FKG-KT/{c}:rank')].test_auc.mean()
            for g in ('support','permuted','cut25','cut50','cut75'):
                a=df[(df.data==d)&(df.frac==frac)&(df.cfg==f'FKG-KT/{c}:{g}')].test_auc
                if len(a)==0 or np.isnan(base): continue
                au=A[c]['cuts'][g]
                out.append({'data':d,'construction':c,'graph':g,'loss':base-a.mean(),'n':len(a),'rank_drop':au['rank_drop'],'qf_drop':au['qf_drop'],
                            'retained_mass':au.get('retained_mass',1.0),'mean_outdeg':au.get('mean_outdeg',A[c]['edges']/A['K'])})
    return pd.DataFrame(out)
