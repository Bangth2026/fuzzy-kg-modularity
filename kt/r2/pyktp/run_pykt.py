"""pyKT protocol run (Liu et al., NeurIPS 2022): pyKT preprocessing, 20% test learners, 5-fold CV on the rest,
sequences cut into windows of 200, min length 3, Adam, early stopping on validation AUC with patience 10, at most
200 epochs, test AUC over all masked next-step predictions (pyKT `evaluate`, non-window test file).
Models: pyKT's own DKT, SAKT, AKT (installed pykt-toolkit 0.0.38) and simpleKT (pykt main branch), with the
default hyperparameters of the pyKT example scripts; FKG-KT (empty graph / support graph) and our DKT with the
settings of the main study. Usage: run_pykt.py dataset model fold"""
import sys, os, json, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.nn.functional import one_hot, binary_cross_entropy
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/home/claude/pyktp'); sys.path.insert(0,'/home/claude/kt/src')
from pykt.datasets.data_loader import KTDataset
torch.set_num_threads(int(os.environ.get('NT','1')))
OUT='/home/claude/pyktp/runs'; os.makedirs(OUT,exist_ok=True)
ds,mname,fold=sys.argv[1],sys.argv[2],int(sys.argv[3]); seed=int(os.environ.get('SEED','42'))
MAXEP=int(os.environ.get('MAXEP','200'))
tag=f'{ds}__{mname}__f{fold}'
if os.path.exists(f'{OUT}/{tag}.json'): print('skip',tag); sys.exit()
cfg=json.load(open('/home/claude/pyktp/configs/data_config.json'))[ds]; dp=f'/home/claude/pyktp/data/{ds}'
K=cfg['num_c']; torch.manual_seed(seed); np.random.seed(seed)
tv=f'{dp}/train_valid_sequences.csv'; te=f'{dp}/test_sequences.csv'
folds=set(cfg['folds'])
tr_ds=KTDataset(tv,cfg['input_type'],folds-{fold}); va_ds=KTDataset(tv,cfg['input_type'],{fold}); te_ds=KTDataset(te,cfg['input_type'],{-1})
bs={'dkt':256,'sakt':64,'akt':int(os.environ.get('AKTBS','16')),'simplekt':64}.get(mname,256)
tl=DataLoader(tr_ds,bs,shuffle=True); vl=DataLoader(va_ds,bs); sl=DataLoader(te_ds,bs)
def graph_support():
    """order-preserving fuzzy graph (s=10) from the TRAINING folds of this split, support only"""
    from fuzzykg.construction import build_transition_graph, TransitionConfig
    d=pd.read_csv(tv); d=d[d.fold!=fold]; rows=[]
    for uid,g in d.groupby('uid',sort=False):
        o=0
        for _,r in g.iterrows():
            c=[int(x) for x in r.concepts.split(',')]; y=[int(x) for x in r.responses.split(',')]; sm=[int(x) for x in r.selectmasks.split(',')]
            for ci,yi,mi in zip(c,y,sm):
                if ci<0: continue
                rows.append((uid,ci,yi,o)); o+=1
    df=pd.DataFrame(rows,columns=['user_id','kc','correct','order']).drop_duplicates(['user_id','order'])
    Gf,_=build_transition_graph(df,TransitionConfig(min_support=10,tau_E=0.0))
    lab=np.array([int(x) for x in Gf.labels]); A=np.zeros((K,K),np.float32); A[lab[Gf.src],lab[Gf.dst]]=1.0
    return A,int(A.sum())
info={}
if mname=='dkt':
    from pykt.models.dkt import DKT; m=DKT(K,emb_size=200,dropout=0.2); lr=1e-3
elif mname=='sakt':
    from pykt.models.sakt import SAKT; m=SAKT(K,seq_len=200,emb_size=256,num_attn_heads=8,dropout=0.2,num_en=1); lr=1e-3
elif mname=='akt':
    from pykt.models.akt import AKT; m=AKT(K,0,d_model=256,n_blocks=4,dropout=0.2,d_ff=512,num_attn_heads=8); lr=1e-4
elif mname=='simplekt':
    from pkx.simplekt import simpleKT; m=simpleKT(K,0,d_model=256,n_blocks=2,dropout=0.1,d_ff=256,final_fc_dim=256,final_fc_dim2=256,num_attn_heads=4,seq_len=200); lr=1e-4
elif mname in ('fkg_empty','fkg_support'):
    from models import FKGKT3
    if mname=='fkg_support': A,ne=graph_support(); info['edges']=ne
    else: A=np.zeros((K,K),np.float32)
    m=FKGKT3(K,A); m.model_name='fkg'; lr=2e-3
elif mname=='ourdkt':
    from models import DKT as ODKT; m=ODKT(K); m.model_name='odkt'; lr=2e-3
opt=torch.optim.Adam(m.parameters(),lr=lr)
def fwd(d,train):
    c,r,cs,rs=d['cseqs'].long(),d['rseqs'].long(),d['shft_cseqs'].long(),d['shft_rseqs'].long(); q,qs=d['qseqs'].long(),d['shft_qseqs'].long()
    mm,sm=d['masks'].bool(),d['smasks'].bool(); n=m.model_name; reg=0.0
    if n=='dkt': y=(m(c,r)*one_hot(cs,K)).sum(-1)
    elif n=='sakt': y=m(c,r,cs)
    elif n=='akt':
        cc=torch.cat((c[:,0:1],cs),1); cr=torch.cat((r[:,0:1],rs),1); cq=torch.cat((q[:,0:1],qs),1)
        y,reg=m(cc,cr,cq); y=y[:,1:]
    elif n=='simplekt':
        y=m(d,train=True)[0][:,1:] if train else m(d)[:,1:]
    else:
        cin=torch.where(mm,c,torch.full_like(c,K)); cn=torch.where(mm,cs,torch.full_like(cs,K)); rin=torch.where(mm,r,torch.zeros_like(r))
        y=torch.sigmoid(m(cin,rin,cn))
    return y,reg,sm,rs
def evaluate(loader):
    m.eval(); P=[];Y=[]
    with torch.no_grad():
        for d in loader:
            y,_,sm,rs=fwd(d,False); P.append(torch.masked_select(y,sm).numpy()); Y.append(torch.masked_select(rs,sm).numpy())
    P=np.concatenate(P);Y=np.concatenate(Y); return roc_auc_score(Y,P),float(((P>=0.5)==Y).mean())
best=-1;bad=0;t0=time.time();hist=[]
for ep in range(MAXEP):
    m.train()
    for d in tl:
        y,reg,sm,rs=fwd(d,True)
        loss=binary_cross_entropy(torch.masked_select(y,sm).double(),torch.masked_select(rs,sm).double())+reg
        opt.zero_grad(); loss.backward()
        if m.model_name in ('fkg','odkt'): torch.nn.utils.clip_grad_norm_(m.parameters(),5.0)
        opt.step()
    va,_=evaluate(vl); hist.append(va)
    if va>best+1e-5: best=va;bad=0;state={k:v.detach().clone() for k,v in m.state_dict().items()}
    else:
        bad+=1
        if bad>=10: break
m.load_state_dict(state); ta,tacc=evaluate(sl)
res={'tag':tag,'dataset':ds,'model':mname,'fold':fold,'seed':seed,'valid_auc':best,'test_auc':ta,'test_acc':tacc,'epochs':len(hist),'secs':time.time()-t0,'maxep':MAXEP,**info}
json.dump(res,open(f'{OUT}/{tag}.json','w')); print(tag,'valid',round(best,4),'test',round(ta,4),'ep',len(hist),'s',round(time.time()-t0))
