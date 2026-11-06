import sys,json,numpy as np
sys.path.insert(0,'/home/claude/kt/theory'); from covbound import *
out={}
for d in ['assist09','assist12','assist15','assist17','junyi','algebra05','eedi','ednet']:
    for c in ('transition','copractice','dktinf'):
        A=np.load(f'/home/claude/kt/graphs/{d}_{c}.npz')['graded']; s,t=np.nonzero(A)
        G=FuzzyGraph(A.shape[0],s,t,A[s,t].astype(float)); rows=audit_graph(G); nf,tot,mg=flips(rows)
        out[f'{d}/{c}']={'rows':rows,'flips':nf,'pairs':tot,'max_abs_cov':max(abs(r['cov']) for r in rows),'certified_pairs':int(sum(m>0 for m in mg))}
        print(d,c,round(out[f'{d}/{c}']['max_abs_cov'],4),nf,tot,out[f'{d}/{c}']['certified_pairs'],flush=True)
json.dump(out,open('/home/claude/kt/theory/covbound_new.json','w'),indent=1)
