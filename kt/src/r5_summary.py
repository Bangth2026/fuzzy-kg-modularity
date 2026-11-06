import json, glob, collections, numpy as np, sys
F=sys.argv[1] if len(sys.argv)>1 else '0.1'
R=[json.load(open(f)) for f in glob.glob('/home/claude/kt/runs_r5/*.json')]
R=[r for r in R if abs(r['frac']-float(F))<1e-9 and r['d']==64 and r['wd']==0 and r['seed']==r['sub'] and r.get('patience',3)==3 and r.get('max_epochs',30)==30]
T=collections.defaultdict(dict)
for r in R: T[(r['data'],r['sub'])][r['cond']]=r
corp=sorted({k[0] for k in T})
for c in corp:
    subs=sorted(s for (cc,s) in T if cc==c)
    row=[]
    for cond in ['support','recency','suprec','rewired']:
        d=[T[(c,s)][cond]['test_auc']-T[(c,s)]['empty']['test_auc'] for s in subs if cond in T[(c,s)] and 'empty' in T[(c,s)]]
        f=[T[(c,s)][cond]['test_auc_first']-T[(c,s)]['empty']['test_auc_first'] for s in subs if cond in T[(c,s)] and 'empty' in T[(c,s)]]
        row.append(f'{cond}:{np.mean(d):+.4f}/{np.mean(f):+.4f}(n{len(d)})')
    e=[T[(c,s)]['empty'] for s in subs if 'empty' in T[(c,s)]]
    sp=[T[(c,s)]['support'] for s in subs if 'support' in T[(c,s)]]
    print(c,' '.join(row),'| first%',round(np.mean([x['n_first']/x['n_pred'] for x in e])*100,1),'| prevprereq',round(np.mean([x['first_prev_is_prereq'] for x in sp]),2),'| bpre',round(np.mean([x['beta_pre'] for x in sp]),3),'| hitmax',sum(x['hit_max'] for x in e+sp),'/',len(e+sp))
