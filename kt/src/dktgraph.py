"""Neural construction: DKT conditional-influence graph (Piech et al. 2015), J_ij = y(j|i)/sum_k y(j|k),
top-10 successors per source, memberships J scaled to (0,1]. Built from DKT trained on training learners."""
import sys, json, numpy as np, torch
sys.path.insert(0,'/home/claude/kt/src'); from models import DKT; from graphs import load, save_variants, G, mk
def build(name,topk=10,frac=1.0):
    df=load(name,frac); K=int(df.kc.max())+1; name=name+('' if frac>=1 else f'_f{int(frac*100)}')
    m=DKT(K); m.load_state_dict(torch.load(f'/home/claude/kt/runs/{name}_dkt_s0.pt')); m.eval()
    with torch.no_grad():
        q=torch.arange(K).unsqueeze(1); r=torch.ones_like(q)
        Y=m.all_probs(q,r)[:,0,:].numpy()          # Y[i,j] = P(correct on j | one correct answer on i)
    J=Y/Y.sum(0,keepdims=True); np.fill_diagonal(J,0)
    src=[];dst=[];w=[]
    for i in range(K):
        top=np.argsort(-J[i])[:topk]; src+= [i]*topk; dst+=top.tolist(); w+=J[i,top].tolist()
    w=np.array(w); w=w/w.max()
    Gf=mk(K,src,dst,w); Gf.labels=list(range(K))
    out={'name':name}; save_variants(name,'dktinf',Gf,K,out)
    json.dump(out,open(f'{G}/{name}_dktinf_audit.json','w'),indent=1); print(name,out['dktinf']['edges'],round(out['dktinf']['Q_F'],3),out['dktinf']['k'])
if __name__=='__main__':
    import os
    for n in sys.argv[1:]: build(n,frac=float(os.environ.get('FRAC','1')))
