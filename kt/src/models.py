import torch, torch.nn as nn, torch.nn.functional as F
class DKT(nn.Module):
    def __init__(s,K,d=64,drop=0.2):
        super().__init__(); s.K=K; s.emb=nn.Embedding(2*K+1,d,padding_idx=2*K); s.rnn=nn.GRU(d,d,batch_first=True); s.out=nn.Linear(d,K); s.dp=nn.Dropout(drop)
    def forward(s,q,r,qn):
        x=s.emb(torch.where(q<s.K,q+s.K*r,torch.full_like(q,2*s.K)))
        h,_=s.rnn(x); y=s.out(s.dp(h))
        return y.gather(2,qn.clamp(max=s.K-1).unsqueeze(2)).squeeze(2)
    def all_probs(s,q,r):
        x=s.emb(q+s.K*r); h,_=s.rnn(x); return torch.sigmoid(s.out(h))
class SAKT(nn.Module):
    def __init__(s,K,d=64,heads=4,drop=0.2,L=200):
        super().__init__(); s.K=K; s.inter=nn.Embedding(2*K+1,d,padding_idx=2*K); s.ex=nn.Embedding(K+1,d,padding_idx=K); s.pos=nn.Embedding(L,d)
        s.att=nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True); s.ln1=nn.LayerNorm(d); s.ffn=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Dropout(drop),nn.Linear(d,d)); s.ln2=nn.LayerNorm(d); s.out=nn.Linear(d,1)
    def forward(s,q,r,qn):
        T=q.size(1); p=s.pos(torch.arange(T,device=q.device))
        kv=s.inter(torch.where(q<s.K,q+s.K*r,torch.full_like(q,2*s.K)))+p; qq=s.ex(qn.clamp(max=s.K))
        mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=q.device),1)
        a,_=s.att(qq,kv,kv,attn_mask=mask); a=s.ln1(a+qq); a=s.ln2(a+s.ffn(a)); return s.out(a).squeeze(2)
class GKT(nn.Module):
    """Graph-based KT (Nakagawa et al. 2019) with a fixed statistics-based adjacency, re-implemented."""
    def __init__(s,K,A,d=32,drop=0.2):
        super().__init__(); s.K=K; s.register_buffer('A',torch.tensor(A,dtype=torch.float32))
        s.emb_x=nn.Embedding(2*K+1,d,padding_idx=2*K); s.emb_c=nn.Embedding(K,d)
        s.f_self=nn.Sequential(nn.Linear(2*d,d),nn.ReLU()); s.f_out=nn.Sequential(nn.Linear(3*d,d),nn.ReLU()); s.f_in=nn.Sequential(nn.Linear(3*d,d),nn.ReLU())
        s.cell=nn.GRUCell(d,d); s.out=nn.Linear(d,1); s.d=d; s.dp=nn.Dropout(drop)
    def forward(s,q,r,qn):
        B,T=q.shape; K,d=s.K,s.d; h=s.emb_c.weight.unsqueeze(0).expand(B,K,d).contiguous()*0
        outs=[]
        for t in range(T):
            qt=q[:,t]; valid=(qt<K)
            qc=qt.clamp(max=K-1); x=s.emb_x(torch.where(valid,qt+K*r[:,t],torch.full_like(qt,2*K)))
            hi=h[torch.arange(B,device=q.device),qc]; selfin=torch.cat([hi,x],1)
            m_self=s.f_self(selfin)
            ctx=selfin.unsqueeze(1).expand(B,K,2*d)
            pair=torch.cat([ctx,h+s.emb_c.weight.unsqueeze(0)],2)
            w_out=s.A[qc].unsqueeze(2); w_in=s.A[:,qc].T.unsqueeze(2)
            m=w_out*s.f_out(pair)+w_in*s.f_in(pair)
            m[torch.arange(B,device=q.device),qc]=m_self
            upd=(w_out+w_in).squeeze(2)>0; upd[torch.arange(B,device=q.device),qc]=True; upd=upd&valid.unsqueeze(1)
            hn=s.cell(m.reshape(B*K,d),h.reshape(B*K,d)).reshape(B,K,d)
            h=torch.where(upd.unsqueeze(2),hn,h)
            qnt=qn[:,t].clamp(max=K-1)
            outs.append(s.out(s.dp(h[torch.arange(B,device=q.device),qnt]+s.emb_c(qnt))).squeeze(1))
        return torch.stack(outs,1)
def rownorm(A):
    s=A.sum(1,keepdim=True); return torch.where(s>0,A/s.clamp(min=1e-12),torch.zeros_like(A))
class FKGKT(nn.Module):
    """Proposed: KC embeddings refined over a fuzzy prerequisite KG (prerequisite and successor
    contexts, row-normalised graded adjacency) plus an optional community embedding; GRU over
    graph-aware interaction embeddings; target-conditioned readout."""
    def __init__(s,K,A=None,comm=None,d=64,drop=0.2,layers=1):
        super().__init__(); s.K=K; s.E=nn.Embedding(K+1,d,padding_idx=K); s.R=nn.Embedding(2,d)
        s.use_g=A is not None
        if s.use_g:
            A=torch.tensor(A,dtype=torch.float32); A.fill_diagonal_(0)
            s.register_buffer('Ain',rownorm(A.T)); s.register_buffer('Aout',rownorm(A))
            s.Win=nn.ModuleList([nn.Linear(d,d) for _ in range(layers)]); s.Wout=nn.ModuleList([nn.Linear(d,d) for _ in range(layers)]); s.ln=nn.LayerNorm(d)
        s.use_c=comm is not None
        if s.use_c:
            s.register_buffer('comm',torch.tensor(comm,dtype=torch.long)); s.C=nn.Embedding(int(max(comm))+1,d)
        s.rnn=nn.GRU(d,d,batch_first=True); s.dp=nn.Dropout(drop)
        s.out=nn.Sequential(nn.Linear(3*d,d),nn.ReLU(),nn.Dropout(drop),nn.Linear(d,1))
    def kc_repr(s):
        Z=s.E.weight[:s.K]
        if s.use_g:
            for Wi,Wo in zip(s.Win,s.Wout): Z=Z+torch.tanh(Wi(s.Ain@Z))+torch.tanh(Wo(s.Aout@Z))
            Z=s.ln(Z)
        if s.use_c: Z=Z+s.C(s.comm)
        return torch.cat([Z,torch.zeros(1,Z.size(1),device=Z.device)],0)
    def forward(s,q,r,qn):
        Z=s.kc_repr(); zq=Z[q.clamp(max=s.K)]; x=s.dp(zq+s.R(r))
        h,_=s.rnn(x); zn=Z[qn.clamp(max=s.K)]
        return s.out(torch.cat([h,zn,h*zn],2)).squeeze(2)
class FKGKT2(nn.Module):
    """FKG-KT: GRU knowledge state + graph-biased attention over past interactions.
    The attention logit between a past interaction on c' and the target c adds
    beta_pre * S[c', c] + beta_suc * S[c, c'], where S is the (composed, rank-coded)
    fuzzy prerequisite strength; KC embeddings are refined over the same graph."""
    def __init__(s,K,A=None,comm=None,d=64,drop=0.2):
        super().__init__(); s.K=K; s.base=FKGKT(K,A,comm,d,drop); s.use_g=A is not None
        if s.use_g:
            S=torch.tensor(A,dtype=torch.float32); S.fill_diagonal_(0)
            S=torch.cat([torch.cat([S,torch.zeros(K,1)],1),torch.zeros(1,K+1)],0); s.register_buffer('S',S)
            s.bpre=nn.Parameter(torch.tensor(1.0)); s.bsuc=nn.Parameter(torch.tensor(1.0)); s.bsame=nn.Parameter(torch.tensor(1.0))
        s.Wq=nn.Linear(d,d); s.Wk=nn.Linear(d,d); s.Wv=nn.Linear(d,d); s.d=d
        s.out=nn.Sequential(nn.Linear(4*d,d),nn.ReLU(),nn.Dropout(drop),nn.Linear(d,1)); s.dp=nn.Dropout(drop)
    def forward(s,q,r,qn):
        b=s.base; Z=b.kc_repr(); qc=q.clamp(max=s.K); qnc=qn.clamp(max=s.K)
        x=b.dp(Z[qc]+b.R(r)); h,_=b.rnn(x); zn=Z[qnc]
        T=q.size(1); logit=(s.Wq(zn)@s.Wk(x).transpose(1,2))/s.d**0.5
        same=(qnc.unsqueeze(2)==qc.unsqueeze(1)).float()
        if s.use_g:
            pre=s.S[qc.unsqueeze(1).expand(-1,T,-1),qnc.unsqueeze(2).expand(-1,-1,T)]
            suc=s.S[qnc.unsqueeze(2).expand(-1,-1,T),qc.unsqueeze(1).expand(-1,T,-1)]
            logit=logit+s.bpre*pre+s.bsuc*suc+s.bsame*same
        mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=q.device),1)|(q>=s.K).unsqueeze(1)
        logit=logit.masked_fill(mask,-1e9); a=torch.softmax(logit,-1)
        c=s.dp(a@s.Wv(x))
        return s.out(torch.cat([h,c,zn,h*zn],2)).squeeze(2)

class _AKTLayer(nn.Module):
    """Transformer layer with AKT's monotonic (distance-decayed) attention."""
    def __init__(s,d,h,drop):
        super().__init__(); s.h=h; s.dk=d//h
        s.k=nn.Linear(d,d); s.v=nn.Linear(d,d); s.o=nn.Linear(d,d)
        s.gammas=nn.Parameter(torch.zeros(h,1,1)); nn.init.xavier_uniform_(s.gammas)
        s.ln1=nn.LayerNorm(d); s.ln2=nn.LayerNorm(d); s.drop=nn.Dropout(drop)
        s.ff=nn.Sequential(nn.Linear(d,4*d),nn.ReLU(),nn.Dropout(drop),nn.Linear(4*d,d))
    def forward(s,query,key,values,mask_k,zero_pad,apply_pos=True):
        B,T,d=query.shape; H,dk=s.h,s.dk
        q=s.k(query).view(B,T,H,dk).transpose(1,2); k=s.k(key).view(B,T,H,dk).transpose(1,2); v=s.v(values).view(B,T,H,dk).transpose(1,2)
        nopeek=torch.triu(torch.ones(T,T,device=query.device),diagonal=mask_k)==0   # allowed positions
        scores=q@k.transpose(-2,-1)/dk**0.5
        x1=torch.arange(T,device=query.device).expand(T,-1); x2=x1.transpose(0,1).contiguous()
        with torch.no_grad():
            sc=torch.softmax(scores.masked_fill(~nopeek,-1e32),-1)*nopeek.float()
            distcum=torch.cumsum(sc,-1); disttot=sc.sum(-1,keepdim=True)
            pos=torch.abs(x1-x2)[None,None].float()
            dist=torch.clamp((disttot-distcum)*pos,min=0.).sqrt().detach()
        gamma=-1.*F.softplus(s.gammas).unsqueeze(0)
        eff=torch.clamp((dist*gamma).exp(),min=1e-5,max=1e5)
        scores=(scores*eff).masked_fill(~nopeek,-1e32); a=torch.softmax(scores,-1)
        if zero_pad: a=torch.cat([torch.zeros_like(a[:,:,:1,:]),a[:,:,1:,:]],2)
        out=(s.drop(a)@v).transpose(1,2).reshape(B,T,d)
        x=s.ln1(query+s.drop(s.o(out)))
        if apply_pos: x=s.ln2(x+s.drop(s.ff(x)))
        return x
class AKT(nn.Module):
    """AKT (Ghosh, Heffernan, Lan 2020) at the KC level (no Rasch variation: items are KCs).
    One question-encoder layer, one knowledge-encoder layer, two knowledge-retriever layers with
    monotonic attention; prediction from [retrieved state; target KC embedding]."""
    def __init__(s,K,d=64,heads=4,drop=0.2):
        super().__init__(); s.K=K
        s.q=nn.Embedding(K+1,d,padding_idx=K); s.qa=nn.Embedding(2*K+1,d,padding_idx=2*K)
        s.qenc=_AKTLayer(d,heads,drop); s.kenc=_AKTLayer(d,heads,drop); s.ret=nn.ModuleList([_AKTLayer(d,heads,drop) for _ in range(2)])
        s.out=nn.Sequential(nn.Linear(2*d,d),nn.ReLU(),nn.Dropout(drop),nn.Linear(d,d//2),nn.ReLU(),nn.Dropout(drop),nn.Linear(d//2,1))
    def forward(s,q,r,qn):
        K=s.K
        Qf=torch.cat([q[:,:1],qn],1)                                   # q_1..q_{T+1}
        Rf=torch.cat([r,torch.zeros_like(r[:,:1])],1)
        qe=s.q(Qf.clamp(max=K)); qa=s.qa(torch.where(Qf<K,Qf+K*Rf,torch.full_like(Qf,2*K)))
        x=s.qenc(qe,qe,qe,mask_k=1,zero_pad=False)                     # sees current question
        y=s.kenc(qa,qa,qa,mask_k=1,zero_pad=False)                     # sees current interaction
        h=x
        for i,L in enumerate(s.ret): h=L(h,h,y,mask_k=0,zero_pad=True) # strictly past interactions
        o=s.out(torch.cat([h,qe],-1)).squeeze(-1)
        return o[:,1:]
