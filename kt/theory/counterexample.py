import sys, json, numpy as np
sys.path.insert(0,'/home/claude/kt/theory'); from covbound import *
def make(nA,nB,pA,pX,pB,muA,muX,muB,seed):
    rng=np.random.default_rng(seed); n=nA+nB; S=[];D=[];M=[]
    for u in range(n):
        for v in range(n):
            if u==v: continue
            a,b=u<nA,v<nA
            p,mu=(pA,muA) if a and b else ((pB,muB) if (not a and not b) else (pX,muX))
            if rng.random()<p: S.append(u);D.append(v);M.append(mu*(1+0.02*rng.standard_normal()))
    return FuzzyGraph(n,np.array(S),np.array(D),np.clip(np.array(M),0.01,1.0))
res=[]
for seed in range(200):
    rng=np.random.default_rng(1000+seed)
    G=make(10,10,rng.uniform(.5,1),rng.uniform(.1,.6),rng.uniform(.05,.5),rng.uniform(.7,1),rng.uniform(.1,.4),rng.uniform(.1,.5),seed)
    al=level_grid(G); rows=audit_graph(G)
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            a,b=rows[i],rows[j]; d1=a['qbar']-b['qbar']; d2=a['rankQ']-b['rankQ']
            if np.sign(d1)!=np.sign(d2) and min(abs(d1),abs(d2))>0.005:
                res.append({'seed':seed,'pair':(a['partition'],b['partition']),'dqbar':d1,'drankQ':d2,'cov':(a['cov'],b['cov']),'bounds':(a['range_bound'],b['range_bound']),'m':G.m})
    if len(res)>=5: break
print(len(res)); [print(r) for r in res[:5]]
json.dump(res,open('/home/claude/kt/theory/counterexample.json','w'),indent=1,default=float)
