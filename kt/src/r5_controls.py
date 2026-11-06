"""Release 3.1.0, revision R5: controls that separate the graph from recency.

Conditions (all FKG-KT, same code path as release 2.0.0 models.FKGKT2, A always passed so the
layer normalisation and the same-KC bias are kept):
  empty    A = 0                                   (empty-graph control)
  support  A = support of the order-preserving graph of the subsample
  rewired  degree-preserving random rewiring of that support (in- and out-degrees kept)
  recency  A = 0, plus beta_rec * 1[j in the run that ends at t and c_j != c_{t+1}]
  suprec   support graph plus the recency term
For every run the script records test AUC on all predictions and on first attempts, ACC, RMSE,
ECE (15 bins), the number of first-attempt predictions, epochs run, and the learned scalars
beta_pre, beta_suc, beta_same, beta_rec and the mean norm of the graph message m_k.
"""
import sys, os, json, time, argparse, hashlib, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import FKGKT2
from train import seqs, batches
torch.set_num_threads(int(os.environ.get('NT', '1')))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
D = f'{ROOT}/data'; OUT = f'{ROOT}/runs_r5'; GR = f'{ROOT}/graphs_r5'
os.makedirs(OUT, exist_ok=True); os.makedirs(GR, exist_ok=True)
MAXLEN = 200

def split_of(u):
    h = int(hashlib.md5(str(u).encode()).hexdigest(), 16) % 100
    return 'train' if h < 70 else ('valid' if h < 80 else 'test')

def load(name, frac, sub):
    """Subsample `sub` of the training learners. sub = 0 reproduces the single 10% sample of
    release 2.0.0 (seed 12345); sub >= 1 uses seed 12345 + sub. Each subsample is drawn without
    replacement from the training learners and independently of the others."""
    df = pd.read_parquet(f'{D}/{name}.parquet')
    df = df.groupby('user', sort=False).head(MAXLEN).reset_index(drop=True)
    df['split'] = df.user.map(split_of)
    if frac < 1.0:
        tr = np.array(sorted(df.loc[df.split == 'train', 'user'].unique()))
        keep = set(np.random.default_rng(12345 + sub).choice(tr, max(2, int(round(frac * len(tr)))), replace=False))
        df = df[(df.split != 'train') | df.user.isin(keep)].reset_index(drop=True)
    return df

def transition_support(tr, K, s=10):
    """Order-preserving graph of Eq. (1) (Algorithm 1), support only."""
    T = np.zeros((K, K))
    for _, g in tr.groupby('user', sort=False):
        q = g.kc.values
        runs = q[np.r_[True, q[1:] != q[:-1]]]
        if len(runs) > 1: np.add.at(T, (runs[:-1], runs[1:]), 1)
    tot = T + T.T
    asym = np.where(tot >= s, np.maximum(0, 2 * T / np.maximum(tot, 1e-12) - 1), 0)
    mx = T.max(1, keepdims=True)
    pref = np.where(mx > 0, T / np.maximum(mx, 1e-12), 0)
    mu = np.minimum(asym, pref); np.fill_diagonal(mu, 0)
    return mu

def rewire(A, seed, n_swaps_per_edge=10):
    """Degree-preserving rewiring of a directed 0/1 graph by double edge swaps
    (a->b, c->d) -> (a->d, c->b); no self-loops, no duplicate or reciprocal edges."""
    rng = np.random.default_rng(seed)
    src, dst = np.nonzero(A); E = len(src)
    if E < 2: return A.copy(), 0
    S = set(zip(src.tolist(), dst.tolist())); src = src.copy(); dst = dst.copy(); done = 0
    for _ in range(n_swaps_per_edge * E * 3):
        if done >= n_swaps_per_edge * E: break
        i, j = rng.integers(E, size=2)
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        if i == j or a == d or c == b: continue
        if (a, d) in S or (c, b) in S or (d, a) in S or (b, c) in S: continue
        S.discard((a, b)); S.discard((c, d)); S.add((a, d)); S.add((c, b))
        dst[i], dst[j] = d, b; done += 1
    R = np.zeros_like(A); R[src, dst] = 1
    return R, done

class FKGKT3(FKGKT2):
    """FKG-KT with an optional graph-free recency term in the attention logit."""
    def __init__(s, K, A, recency=False, d=64, drop=0.2):
        super().__init__(K, A, None, d, drop); s.recency = recency
        if recency: s.brec = nn.Parameter(torch.tensor(1.0))
    def forward(s, q, r, qn):
        b = s.base; Z = b.kc_repr(); qc = q.clamp(max=s.K); qnc = qn.clamp(max=s.K)
        x = b.dp(Z[qc] + b.R(r)); h, _ = b.rnn(x); zn = Z[qnc]
        T = q.size(1); logit = (s.Wq(zn) @ s.Wk(x).transpose(1, 2)) / s.d ** 0.5
        same = (qnc.unsqueeze(2) == qc.unsqueeze(1)).float()
        pre = s.S[qc.unsqueeze(1).expand(-1, T, -1), qnc.unsqueeze(2).expand(-1, -1, T)]
        suc = s.S[qnc.unsqueeze(2).expand(-1, -1, T), qc.unsqueeze(1).expand(-1, T, -1)]
        logit = logit + s.bpre * pre + s.bsuc * suc + s.bsame * same
        if s.recency:
            rid = torch.cumsum(torch.cat([torch.zeros_like(qc[:, :1]), (qc[:, 1:] != qc[:, :-1]).long()], 1), 1)
            inrun = (rid.unsqueeze(1) == rid.unsqueeze(2))            # [b, t, j]: j in the run that contains t
            rec = (inrun & (qnc.unsqueeze(2) != qc.unsqueeze(1))).float()
            logit = logit + s.brec * rec
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1) | (q >= s.K).unsqueeze(1)
        logit = logit.masked_fill(mask, -1e9); a = torch.softmax(logit, -1)
        c = s.dp(a @ s.Wv(x))
        return s.out(torch.cat([h, c, zn, h * zn], 2)).squeeze(2)
    def msg_norm(s):
        b = s.base; Z0 = b.E.weight[:s.K]
        m = torch.tanh(b.Win[0](b.Ain @ Z0)) + torch.tanh(b.Wout[0](b.Aout @ Z0))
        return float(m.norm(dim=1).mean())

def evaluate(m, q, r, K, bs=128):
    m.eval(); P = []; Y = []; NEW = []; PREV = []
    with torch.no_grad():
        for j, a, b, c, y in batches(q, r, bs, False, None):
            p = torch.sigmoid(m(a, b, c)); mask = c < K
            P.append(p[mask].numpy()); Y.append(y[mask].numpy())
            qq = torch.cat([a[:, :1], c], 1).numpy(); seen = np.zeros(c.shape, dtype=bool)
            for i in range(qq.shape[0]):
                u, fi = np.unique(qq[i], return_index=True); first = dict(zip(u, fi))
                seen[i] = np.array([first[v] < t + 1 for t, v in enumerate(qq[i, 1:])])
            NEW.append((~seen)[mask.numpy()]); PREV.append(np.stack([a.numpy(), c.numpy()], -1)[mask.numpy()])
    P = np.concatenate(P); Y = np.concatenate(Y); NEW = np.concatenate(NEW); PREV = np.concatenate(PREV)
    return P, Y, NEW, PREV

def ece(p, y, n=15):
    bins = np.minimum((p * n).astype(int), n - 1); e = 0.0
    for k in range(n):
        s = bins == k
        if s.any(): e += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(e)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--frac', type=float); ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--cond'); ap.add_argument('--seed', type=int, default=None); ap.add_argument('--d', type=int, default=64); ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--lr_scalar', type=float, default=None); ap.add_argument('--bs', type=int, default=128)
    a = ap.parse_args(argv)
    if a.seed is None: a.seed = a.sub
    extra = ('' if (a.d == 64 and a.wd == 0) else f'__d{a.d}_wd{a.wd:g}') + ('' if a.seed == a.sub else f'__seed{a.seed}') + ('' if a.lr_scalar is None else f'__lrs{a.lr_scalar:g}')
    tag = f"{a.data}__f{int(round(a.frac*100))}__s{a.sub}__{a.cond}{extra}"
    if os.path.exists(f'{OUT}/{tag}.json'): print('skip', tag); return
    df = load(a.data, a.frac, a.sub); K = int(df.kc.max()) + 1
    tr = df[df.split == 'train']
    gpath = f'{GR}/{a.data}__f{int(round(a.frac*100))}__s{a.sub}.npz'
    if os.path.exists(gpath):
        g = np.load(gpath); sup = g['support']; rw = g['rewired']
    else:
        mu = transition_support(tr, K); sup = (mu > 0).astype(np.float32)
        rw, _ = rewire(sup, seed=1000 + a.sub); rw = rw.astype(np.float32)
        np.savez_compressed(gpath, support=sup, rewired=rw)
    A = {'empty': np.zeros((K, K), np.float32), 'support': sup, 'rewired': rw,
         'recency': np.zeros((K, K), np.float32), 'suprec': sup}[a.cond]
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tq, trr, _ = seqs(df, K, 'train'); vq, vr, _ = seqs(df, K, 'valid'); sq, sr, _ = seqs(df, K, 'test')
    m = FKGKT3(K, A, recency=a.cond in ('recency', 'suprec'), d=a.d)
    sc = [p for n, p in m.named_parameters() if n in ('bpre', 'bsuc', 'bsame', 'brec')]
    other = [p for n, p in m.named_parameters() if n not in ('bpre', 'bsuc', 'bsame', 'brec')]
    opt = torch.optim.Adam([{'params': other}, {'params': sc, 'lr': a.lr_scalar or a.lr, 'weight_decay': 0.0}], lr=a.lr, weight_decay=a.wd)
    best = -1; bad = 0; t0 = time.time(); hist = []; state = None
    for ep in range(a.epochs):
        m.train()
        for j, qa, ra, qn, y in batches(tq, trr, a.bs, True, rng):
            lo = m(qa, ra, qn); mk = qn < K
            loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        P, Y, _, _ = evaluate(m, vq, vr, K); va = roc_auc_score(Y, P); hist.append(va)
        if va > best + 1e-5: best = va; bad = 0; state = {k: v.detach().clone() for k, v in m.state_dict().items()}; best_ep = ep + 1
        else:
            bad += 1
            if bad >= a.patience: break
    m.load_state_dict(state)
    P, Y, NEW, PREV = evaluate(m, sq, sr, K)
    prevkc, tgt = PREV[:, 0], PREV[:, 1]
    nb = sup[np.minimum(prevkc, K - 1), np.minimum(tgt, K - 1)] > 0   # c_t -> c_{t+1} is an edge of the support
    res = dict(tag=tag, data=a.data, frac=a.frac, sub=a.sub, seed=a.seed, cond=a.cond, d=a.d, wd=a.wd, K=K,
               n_train_learners=int(tr.user.nunique()), n_train_inter=int(len(tr)),
               edges=int(sup.sum()), edges_rewired=int(rw.sum()),
               test_auc=float(roc_auc_score(Y, P)),
               test_auc_first=float(roc_auc_score(Y[NEW], P[NEW])) if len(set(Y[NEW])) > 1 else None,
               test_acc=float(((P > 0.5) == Y).mean()), test_rmse=float(np.sqrt(((P - Y) ** 2).mean())), test_ece=ece(P, Y),
               n_pred=int(len(Y)), n_first=int(NEW.sum()),
               first_prev_is_prereq=float(nb[NEW].mean()) if NEW.any() else None,
               valid_auc=float(best), epochs_run=len(hist), best_epoch=best_ep, hit_max=bool(len(hist) == a.epochs), valid_hist=hist,
               beta_pre=float(m.bpre.detach()), beta_suc=float(m.bsuc.detach()), beta_same=float(m.bsame.detach()),
               beta_rec=float(m.brec.detach()) if m.recency else None, msg_norm=m.msg_norm(), lr=a.lr, lr_scalar=a.lr_scalar, patience=a.patience, max_epochs=a.epochs, secs=time.time() - t0)
    np.savez_compressed(f'{OUT}/{tag}_pred.npz', p=P.astype(np.float32), y=Y.astype(np.int8), new=NEW)
    json.dump(res, open(f'{OUT}/{tag}.json', 'w'))
    print(tag, 'auc', round(res['test_auc'], 4), 'first', round(res['test_auc_first'] or 0, 4), 'ep', len(hist), 'b', round(res['beta_pre'], 3), round(res['beta_same'], 3), 's', round(res['secs']))

if __name__ == '__main__': main()
