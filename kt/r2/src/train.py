import sys, json, time, argparse, os, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
sys.path.insert(0,'/home/claude/kt/src'); from models import *; from graphs import load, gsuffix
torch.set_num_threads(int(os.environ.get('NT','1')))
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R=os.environ.get('RUNS','/home/claude/kt/runs'); G='/home/claude/kt/graphs'; os.makedirs(R,exist_ok=True)
def seqs(df,K,split,L=200):
    d=df[df.split==split]; Q=[];Rr=[];U=[]
    for u,g in d.groupby('user',sort=False):
        q=g.kc.values[:L]; r=g.correct.values[:L]
        if len(q)<2: continue
        Q.append(q);Rr.append(r);U.append(u)
    n=len(Q); T=max(len(x) for x in Q)
    q=np.full((n,T),K);r=np.zeros((n,T),dtype=int)
    for i,(a,b) in enumerate(zip(Q,Rr)): q[i,:len(a)]=a;r[i,:len(b)]=b
    return torch.tensor(q),torch.tensor(r),U
def batches(q,r,bs,shuffle,rng):
    idx=rng.permutation(len(q)) if shuffle else np.arange(len(q))
    for i in range(0,len(q),bs):
        j=idx[i:i+bs]; qq=q[j]; rr=r[j]; yield j,qq[:,:-1],rr[:,:-1],qq[:,1:],rr[:,1:]
def evaluate(m,q,r,K,bs=128,users=None):
    m.eval(); P=[];Y=[];Uidx=[];NEW=[]
    with torch.no_grad():
        for j,a,b,c,y in batches(q,r,bs,False,None):
            p=torch.sigmoid(m(a.to(DEV),b.to(DEV),c.to(DEV))).cpu(); mask=c<K
            P.append(p[mask].numpy());Y.append(y[mask].numpy()); Uidx.append(np.repeat(j,mask.sum(1).numpy()))
            qq=torch.cat([a[:,:1],c],1).numpy(); seen=np.zeros(c.shape,dtype=bool)
            for i in range(qq.shape[0]):
                _,fi=np.unique(qq[i],return_index=True); first=dict(zip(_,fi))
                seen[i]=np.array([first[v]<t+1 for t,v in enumerate(qq[i,1:])])
            NEW.append((~seen)[mask.numpy()])
    P=np.concatenate(P);Y=np.concatenate(Y)
    NEW=np.concatenate(NEW); evaluate.new_auc=roc_auc_score(Y[NEW],P[NEW]) if NEW.any() and len(set(Y[NEW]))>1 else None; evaluate.new=NEW
    return roc_auc_score(Y,P),accuracy_score(Y,P>0.5),P,Y,np.concatenate(Uidx)
def build_model(a,K,name):
    if a.model=='dkt': return DKT(K)
    if a.model=='sakt': return SAKT(K)
    if a.model=='akt': return AKT(K)
    if a.model=='gkt':
        A=np.load(f'{G}/{name}_gktA.npy'); return GKT(K,A)
    if a.model in ('fkgkt','fkgkt2'):
        A=None; comm=None
        if a.graph!='none':
            if a.graph=='expert':
                E=np.load(f'{G}/{a.data}_expert.npy'); A=np.zeros((K,K),np.float32); n=min(K,E.shape[0]); A[:n,:n]=E[:n,:n]
            elif a.graph.startswith('dkt'): A=np.load(f'{G}/{name}_dktinf.npz')[a.graph.split(':')[1]]
            else: A=np.load(f'{G}/{name}_{a.construction}.npz')[a.graph]
        if a.comm: comm=np.load(f'{G}/{name}_{a.construction}_partition.npy')
        return (FKGKT if a.model=='fkgkt' else FKGKT2)(K,A,comm)
    if a.model=='fkgkt3':
        if a.graph in ('empty','none'): A=np.zeros((K,K),np.float32)
        elif a.graph=='expert':
            E=np.load(f'{G}/{a.data}_expert.npy'); A=np.zeros((K,K),np.float32); n=min(K,E.shape[0]); A[:n,:n]=E[:n,:n]
        elif a.graph.startswith('dkt'): A=np.load(f'{G}/{name}_dktinf.npz')[a.graph.split(':')[1]]
        else: A=np.load(f'{G}/{(a.data if a.gsrc=="full" else name)}_{a.construction}.npz')[a.graph]   # gsrc=full: graph from all training learners (larger cohort)
        if A.shape[0]!=K: B=np.zeros((K,K),np.float32); n=min(K,A.shape[0]); B[:n,:n]=A[:n,:n]; A=B   # KC index range can differ between samples
        cnt=np.load(f'{G}/{name}_counts.npy') if a.gate else None
        return FKGKT3(K,A,cnt,abl=tuple(x for x in a.abl.split(',') if x),gate=int(a.gate))
    if a.model=='simplekt': return SimpleKT(K)
    if a.model=='rkt': return RKT(K,np.load(f'{G}/{name}_phi.npy'))
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--data'); p.add_argument('--model'); p.add_argument('--graph',default='none'); p.add_argument('--construction',default='transition')
    p.add_argument('--comm',type=int,default=0); p.add_argument('--seed',type=int,default=0); p.add_argument('--epochs',type=int,default=30); p.add_argument('--bs',type=int,default=128); p.add_argument('--lr',type=float,default=2e-3); p.add_argument('--patience',type=int,default=3); p.add_argument('--frac',type=float,default=1.0)
    p.add_argument('--sub',type=int,default=0); p.add_argument('--abl',default=''); p.add_argument('--gate',type=int,default=0); p.add_argument('--gsrc',default='')
    p.add_argument('--pc',default=''); p.add_argument('--pclam',type=float,default=0.0)
    a=p.parse_args(argv)
    if a.model in ('gkt','akt'): a.bs=min(a.bs,32)
    extra=('_g' if a.gate==1 else (f'_g{a.gate}' if a.gate else ''))+(('_a-'+a.abl.replace(',','-')) if a.abl else '')+(f'_pc-{a.pc}-{a.pclam:g}' if a.pclam>0 else '')+('_gfull' if a.gsrc=='full' else '')
    uses_c=a.model.startswith('fkgkt') and a.graph not in ('none','empty','expert') and not a.graph.startswith('dkt')
    tag=f"{a.data}{gsuffix(a.frac,a.sub)}__{a.model}__{a.construction if (uses_c or (a.model in ('fkgkt','fkgkt2') and a.graph!='none')) else 'na'}__{a.graph}__c{a.comm}{extra}__s{a.seed}"
    if os.path.exists(f'{R}/{tag}.json'): print('skip',tag); return
    torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed)
    df=load(a.data,a.frac,a.sub); K=int(df.kc.max())+1; gname=a.data+gsuffix(a.frac,a.sub)
    tq,tr,_=seqs(df,K,'train'); vq,vr,_=seqs(df,K,'valid'); sq,sr,su=seqs(df,K,'test')
    if a.model=='gkt' and K>150: a.bs=16 if K<=400 else 8
    m=build_model(a,K,gname).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=a.lr)
    best=-1;bad=0;t0=time.time();hist=[]
    if a.pclam>0:   # prerequisite-consistency loss: second channel through which graded edges can enter the tracer
        Wg=np.load(f'{G}/{gname}_transition.npz')['rank']; es,ed=np.nonzero(Wg); ew=Wg[es,ed].astype(np.float32)
        if a.pc=='support': ew=np.ones_like(ew)
        elif a.pc=='permuted': ew=np.random.default_rng(777).permutation(ew)
        es_t,ed_t,ew_t=torch.tensor(es),torch.tensor(ed),torch.tensor(ew); pcrng=torch.Generator().manual_seed(a.seed)
    for ep in range(a.epochs):
        m.train()
        for j,qa,ra,qn,y in batches(tq,tr,a.bs,True,rng):
            qa,ra,qn,y=qa.to(DEV),ra.to(DEV),qn.to(DEV),y.to(DEV)
            lo=m(qa,ra,qn); mask=qn<K
            loss=F.binary_cross_entropy_with_logits(lo[mask],y[mask].float())
            if a.pclam>0 and len(ew)>0:
                idx=torch.randint(len(ew),qn.shape,generator=pcrng)
                pa=torch.sigmoid(m(qa,ra,es_t[idx].to(DEV))); pb=torch.sigmoid(m(qa,ra,ed_t[idx].to(DEV)))
                loss=loss+a.pclam*(ew_t[idx].to(DEV)*F.relu(pb-pa))[mask].mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
        va=evaluate(m,vq,vr,K)[0]; hist.append(va)
        if va>best+1e-5: best=va;bad=0; state={k:v.detach().clone() for k,v in m.state_dict().items()}
        else:
            bad+=1
            if bad>=a.patience: break
    m.load_state_dict(state)
    auc,acc,P,Y,Ui=evaluate(m,sq,sr,K)
    res={'tag':tag,**vars(a),'test_auc_newkc':evaluate.new_auc,'n_newkc':int(evaluate.new.sum()),'K':K,'valid_auc':best,'test_auc':auc,'test_acc':acc,'epochs_run':len(hist),'valid_hist':hist,'secs':time.time()-t0,
         'n_params':sum(x.numel() for x in m.parameters())}
    if getattr(m,'gate',0):
        with torch.no_grad(): gk=m.g()[:K].cpu().numpy()
        res['gate']={'w0':float(m.w0),'w1':float(m.w1),'g_mean':float(gk.mean()),'g_min':float(gk.min()),'g_max':float(gk.max())}
    np.savez_compressed(f'{R}/{tag}_pred.npz',new=evaluate.new,p=P.astype(np.float32),y=Y.astype(np.int8),u=Ui.astype(np.int32))
    if a.model=='dkt' and a.seed==0: torch.save(m.state_dict(),f'{R}/{gname}_dkt_s0.pt')
    json.dump(res,open(f'{R}/{tag}.json','w'))
    print(tag,'valid',round(best,4),'test',round(auc,4),'ep',len(hist),'s',round(time.time()-t0))
if __name__=='__main__': main()
