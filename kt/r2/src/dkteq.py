"""Neural construction at matched density: DKT conditional-influence graph keeping the |E| strongest
off-diagonal entries of J, |E| = edges of the order-preserving graph of the same sample (Minor 5)."""
import sys, os, json, numpy as np, torch
sys.path.insert(0,'/home/claude/kt/src'); from models import DKT; from graphs import load, save_variants, G, mk, gsuffix
def build(name,frac=0.1,sub=0):
    df=load(name,frac,sub); K=int(df.kc.max())+1; gname=name+gsuffix(frac,sub)
    m=DKT(K); m.load_state_dict(torch.load(f'/home/claude/kt/runs_dkt/{gname}_dkt_s0.pt')); m.eval()
    with torch.no_grad():
        q=torch.arange(K).unsqueeze(1); r=torch.ones_like(q); Y=m.all_probs(q,r)[:,0,:].numpy()
    J=Y/Y.sum(0,keepdims=True); np.fill_diagonal(J,0)
    E=int((np.load(f'{G}/{gname}_transition.npz')['support']>0).sum())
    flat=np.argsort(-J,axis=None)[:E]; src,dst=np.unravel_index(flat,J.shape); w=J[src,dst]; w=w/w.max()
    Gf=mk(K,src,dst,w); Gf.labels=list(range(K)); out={'name':gname,'edges_target':E}
    save_variants(gname,'dkteq',Gf,K,out)
    # the construction of the preliminary analysis from the same DKT: ten strongest targets per source
    s2=[];d2=[];w2=[]
    for i in range(K):
        top=np.argsort(-J[i])[:10]; s2+=[i]*10; d2+=top.tolist(); w2+=J[i,top].tolist()
    w2=np.array(w2); Gf2=mk(K,s2,d2,w2/w2.max()); Gf2.labels=list(range(K)); save_variants(gname,'dkt10',Gf2,K,out)
    json.dump(out,open(f'{G}/{gname}_dkteq_audit.json','w'),indent=1)
    print(gname,'edges: order-pres',E,'dkteq',out['dkteq']['edges'],'dkt10',out['dkt10']['edges'])
if __name__=='__main__':
    for n in sys.argv[1:]: build(n)
