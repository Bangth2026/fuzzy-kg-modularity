"""R2 analysis: every number reported in the manuscript is computed here from run JSON files.
Inputs: runs/ (R2 runs) and the R1 runs of release 2.0.0. Output: paper/r2_results.json, paper/r2_*.csv."""
import json, glob, re, os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
warnings.filterwarnings('ignore')
K='/home/claude/kt'; R1='/home/claude/rel/fuzzy-kg-modularity-2.0.0/kt/results/runs'
D8=['assist09','assist12','assist15','assist17','junyi','algebra05','eedi','ednet']
DNEW=['bridge06','statics11']; D10=D8+DNEW
TAGRE=re.compile(r'^([a-z0-9]+?)(?:_f(\d+)(?:s(\d+))?)?$')

def label(model,cons,graph,cx):
    extra=cx[2:] if len(cx)>2 else ''
    if model=='fkgkt3':
        pooled='_gfull' in extra; extra=extra.replace('_gfull','')
        mg=re.match(r'_g(\d?)(?![a-z])',extra); gv=(mg.group(1) or '1') if mg else ''
        g=graph if not gv else ({'1':'cag','2':'gate2'}[gv] if graph=='support' else graph+'+gate'+gv)
        m=re.search(r'_a-([a-z-]+)',extra); p=re.search(r'_pc-([a-z]+)-([0-9.]+)',extra)
        lab='P/'+g+('-pooled' if pooled else '')
        if m: lab+='/-'+m.group(1)
        if p: lab+=f'/pc-{p.group(1)}-{p.group(2)}'
        return lab
    if model=='fkgkt2':
        return 'R1/none' if graph=='none' else f'R1/{cons}:{graph}'
    return {'dkt':'DKT','sakt':'SAKT','akt':'AKT','gkt':'GKT','simplekt':'simpleKT','rkt':'RKT'}.get(model,model)

def load_runs():
    rows={}
    for src in (R1,f'{K}/runs'):
        for f in glob.glob(f'{src}/*.json'):
            r=json.load(open(f)); parts=r['tag'].split('__')
            m=TAGRE.match(parts[0]); data=m.group(1); frac=int(m.group(2))/100 if m.group(2) else 1.0; sub=int(m.group(3) or 0)
            rows[r['tag']]=dict(tag=r['tag'],data=data,frac=frac,sub=sub,seed=int(parts[5][1:]),cfg=label(parts[1],parts[2],parts[3],parts[4]),
                                auc=r['test_auc'],new=r.get('test_auc_newkc'),valid=r['valid_auc'],secs=r['secs'],ep=r['epochs_run'],src='R1' if src==R1 else 'R2')
    return pd.DataFrame(rows.values())

def holm(pvals):
    p=np.asarray(pvals,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,j in enumerate(o):
        run=max(run,(m-i)*p[j]); adj[j]=min(1.0,run)
    return adj

def ipk(data,frac,sub):
    sfx='' if frac>=1 else f'_f{int(round(frac*100))}'+(f's{sub}' if sub else '')
    f=f'{K}/graphs/{data}{sfx}_counts.npy'
    if not os.path.exists(f): return np.nan
    n=np.load(f); return float(n.sum()/len(n))

def paired(df,a,b,metric='auc',fr=None,keys=('data','frac','sub','seed')):
    """Differences a-b matched on corpus, fraction, subsample and seed."""
    x=df[df.cfg==a].set_index(list(keys))[metric]; y=df[df.cfg==b].set_index(list(keys))[metric]
    j=x.index.intersection(y.index); d=(x[j]-y[j]).reset_index(); d.columns=list(keys)+['d']
    if fr is not None: d=d[d.frac==fr]
    return d

def across(d,corpora):
    """Corpus-level mean differences and a Wilcoxon signed-rank test over corpora."""
    m=d.groupby('data').d.mean().reindex(corpora).dropna()
    if len(m)<5: return None
    p=float(wilcoxon(m.values).pvalue) if np.any(m.values!=0) else 1.0
    return {'n':int(len(m)),'mean':float(m.mean()),'median':float(m.median()),'min':float(m.min()),'max':float(m.max()),'pos':int((m>0).sum()),'p':p,'per_corpus':{k:round(float(v),5) for k,v in m.items()}}

def main():
    df=load_runs(); out={'n_runs':{'R1':int((df.src=='R1').sum()),'R2':int((df.src=='R2').sum())}}
    df['ipk']=[ipk(a,b,c) for a,b,c in zip(df.data,df.frac,df['sub'])]
    df.to_csv(f'{K}/paper/r2_runs.csv',index=False)
    # ---- 1. graph over the empty control, by fraction (seed == sub for the learning curve; sub0 seed0 at 100%)
    fam=[]; res={}
    for fr in (0.05,0.1,0.25,0.5,1.0):
        for a,b in (('P/support','P/empty'),('P/cag','P/empty'),('P/cag','P/support'),('P/gate2','P/empty'),('P/gate2','P/support'),('P/support-pooled','P/empty'),('P/support-pooled','P/support')):
            for met in ('auc','new'):
                d=paired(df,a,b,met,fr)
                r=across(d,D10)
                if r: res[f'{fr}|{met}|{a}|{b}']=r; fam.append(f'{fr}|{met}|{a}|{b}')
    # Holm families: (F1) support vs empty, 4 fractions x 2 metrics = the primary question;
    # (F2) the two gates against the empty control and against the support graph
    for famk in ([k for k in fam if k.endswith('|P/support|P/empty')],[k for k in fam if not k.endswith('|P/support|P/empty')]):
        adj=holm([res[k]['p'] for k in famk])
        for k,a in zip(famk,adj): res[k]['p_holm']=float(a); res[k]['holm_family_size']=len(famk)
    out['graph_vs_empty']=res
    # ---- 2. dose-response: cell-level gain vs log10 training interactions per KC, corpus random intercept
    import statsmodels.formula.api as smf
    dr={}
    for a,b in (('P/support','P/empty'),('P/cag','P/empty'),('P/cag','P/support'),('P/gate2','P/empty')):
        for met in ('auc','new'):
            d=paired(df,a,b,met); d['x']=np.log10([ipk(a_,b_,c_) for a_,b_,c_ in zip(d.data,d.frac,d['sub'])]); d=d.dropna()
            d=d[d.data.isin(D10)]
            if len(d)<20: continue
            fit=smf.mixedlm('d ~ x',d,groups=d['data']).fit(reml=True)
            fq=smf.mixedlm('d ~ x + I(x**2)',d,groups=d['data']).fit(reml=False); fl=smf.mixedlm('d ~ x',d,groups=d['data']).fit(reml=False)
            from scipy.stats import chi2
            quad={'b1':float(fq.params['x']),'b2':float(fq.params['I(x ** 2)']),'p_b2':float(fq.pvalues['I(x ** 2)']),
                  'lr_p':float(chi2.sf(2*(fq.llf-fl.llf),1)),'peak_ipk':float(10**(-fq.params['x']/(2*fq.params['I(x ** 2)']))) if fq.params['I(x ** 2)']<0 else None}
            within={}
            for fr in sorted(d.frac.unique()):
                m=d[d.frac==fr].groupby('data')[['d','x']].mean()
                if len(m)>=5: s=spearmanr(m.x,m.d); within[str(fr)]={'rho':float(s.statistic),'p':float(s.pvalue),'n':len(m)}
            dr[f'{met}|{a}|{b}']={'slope':float(fit.params['x']),'se':float(fit.bse['x']),'p':float(fit.pvalues['x']),'intercept':float(fit.params['Intercept']),
                                  'n_cells':int(len(d)),'within_fraction_spearman':within,'quadratic':quad,
                                  'gain_by_fraction':{str(fr):float(d[d.frac==fr].groupby('data').d.mean().mean()) for fr in sorted(d.frac.unique())}}
    out['dose_response']=dr
    # ---- 3. architecture: ablations against the empty control (10%, subs 0-2, seed=sub)
    ab={}; fam=[]
    for x in ('gru','att','same','hz'):
        d=paired(df,f'P/empty/-{x}','P/empty','auc',0.1); r=across(d,D10)
        if r: ab[x]=r; fam.append(x)
    d=paired(df,'R1/none','P/empty','auc',0.1); d=d[d['sub']==0]; r=across(d,D8)
    if r: ab['r1_none_vs_empty']=r; fam.append('r1_none_vs_empty')
    adj=holm([ab[k]['p'] for k in fam])
    for k,a in zip(fam,adj): ab[k]['p_holm']=float(a)
    out['ablation']=ab
    # ---- 4. second channel: prerequisite-consistency loss
    pc={}; fam=[]
    for w in ('graded','permuted'):
        for met in ('auc','new'):
            d=paired(df,f'P/empty/pc-{w}-0.1','P/empty',met,0.1); r=across(d,D10)
            if r: pc[f'{w}|{met}']=r; fam.append(f'{w}|{met}')
    for met in ('auc','new'):
        d=paired(df,'P/empty/pc-graded-0.1','P/empty/pc-permuted-0.1',met,0.1); r=across(d,D10)
        if r: pc[f'graded-permuted|{met}']=r; fam.append(f'graded-permuted|{met}')
    adj=holm([pc[k]['p'] for k in fam])
    for k,a in zip(fam,adj): pc[k]['p_holm']=float(a)
    out['pc_loss']=pc
    # ---- 5. R1 graph inputs against the empty control (10%, sub 0, seeds matched)
    v={}; fam=[]
    for c in sorted(set(df[(df.frac==0.1)&df.cfg.str.startswith('R1/')&(df.cfg!='R1/none')].cfg)):
        for met in ('auc','new'):
            d=paired(df,c,'P/empty',met,0.1); d=d[d['sub']==0]; r=across(d,D8)
            if r: v[f'{c}|{met}']=r; fam.append(f'{c}|{met}')
    adj=holm([v[k]['p'] for k in fam])
    for k,a in zip(fam,adj): v[k]['p_holm']=float(a)
    out['r1_inputs_vs_empty']=v
    # ---- 6. main table: seed means at 10% sub 0 and at 100%
    main=['DKT','SAKT','AKT','GKT','simpleKT','RKT','R1/none','P/empty','P/support','P/cag','R1/transition:product3']
    tab={}
    for fr in (0.1,1.0):
        t=df[(df.frac==fr)&(df['sub']==0)&df.cfg.isin(main)].groupby(['cfg','data']).agg(auc=('auc','mean'),sd=('auc','std'),n=('auc','size'),new=('new','mean'))
        tab[str(fr)]={f'{c}|{d}':[round(float(r.auc),4),None if np.isnan(r.sd) else round(float(r.sd),4),int(r.n),None if pd.isna(r.new) else round(float(r.new),4)] for (c,d),r in t.iterrows()}
    out['main_table']=tab
    # model comparisons across corpora (seed-mean AUC, 10% sub 0 and 100%)
    cmp={}; fam=[]
    for fr in (0.1,1.0):
        m=df[(df.frac==fr)&(df['sub']==0)].groupby(['cfg','data']).auc.mean()
        for a in ('P/support','P/empty'):
            for b in ('DKT','SAKT','AKT','GKT','simpleKT','RKT'):
                ds=[d for d in D10 if (a,d) in m.index and (b,d) in m.index]
                if len(ds)<5: continue
                diff=np.array([m[(a,d)]-m[(b,d)] for d in ds])
                cmp[f'{fr}|{a}|{b}']={'n':len(ds),'mean':float(diff.mean()),'min':float(diff.min()),'max':float(diff.max()),'pos':int((diff>0).sum()),'p':float(wilcoxon(diff).pvalue)}
                fam.append(f'{fr}|{a}|{b}')
    adj=holm([cmp[k]['p'] for k in fam])
    for k,a in zip(fam,adj): cmp[k]['p_holm']=float(a)
    out['model_comparisons']=cmp
    # ---- 7. gate parameters are not stored; report per-KC analysis instead: gain by KC data quartile (10%, subs)
    out['seed_sd']={str(fr):float(df[(df.frac==fr)&(df.cfg=='P/empty')].groupby(['data','sub']).auc.std().mean()) for fr in (0.1,)}
    out['sub_sd']={str(fr):float(df[(df.frac==fr)&(df.cfg=='P/empty')].groupby('data').auc.std().mean()) for fr in (0.05,0.1,0.25)}
    # graph quality: share of the edges of the all-learner graph recovered by each subsample graph, and edges per KC
    gq={}
    for c in D10:
        full=np.load(f'{K}/graphs/{c}_transition.npz')['support']>0
        for fr,ns in ((0.05,5),(0.1,5),(0.25,3)):
            rec=[];ne=[]
            for sb in range(ns):
                sfx=f'_f{int(round(fr*100))}'+(f's{sb}' if sb else '')
                g=np.load(f'{K}/graphs/{c}{sfx}_transition.npz')['support']>0
                n_=min(g.shape[0],full.shape[0]); rec.append((g[:n_,:n_]&full[:n_,:n_]).sum()/max(full.sum(),1)); ne.append(g.sum())
            gq[f'{c}|{fr}']={'recall_of_full_edges':float(np.mean(rec)),'edges':float(np.mean(ne)),'full_edges':int(full.sum())}
    out['graph_quality']=gq
    out['secs']=df.groupby(['cfg','frac']).secs.mean().round(1).reset_index().query("cfg in ['DKT','AKT','SAKT','GKT','simpleKT','RKT','P/empty','P/support','P/cag']").to_dict('records')
    json.dump(out,open(f'{K}/paper/r2_results.json','w'),indent=1)
    return out

if __name__=='__main__':
    o=main()
    for k in ('graph_vs_empty','ablation','pc_loss','dose_response'):
        print('==',k)
        for kk,v in o[k].items():
            print(kk,{x:(round(y,4) if isinstance(y,float) else y) for x,y in v.items() if x not in ('per_corpus','within_fraction_spearman','gain_by_fraction')})
