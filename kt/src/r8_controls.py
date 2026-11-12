"""Release 3.5.0: the null-graph ladder beyond the probe (Section 5.4 of the Neural Networks manuscript).

Same data loading, subsamples, graph construction (support of the order-preserving graph of each
subsample), Erdos-Renyi null graph, update budget and evaluation as r6_controls.py / r7_controls.py.
Runs on a GPU when one is available.

Conditions (--cond):
  Probe (FKG-KT, models.FKGKT2 via r5_controls.FKGKT3), graph read in the embedding refinement only:
    empty          A = 0 (empty-graph control)
    uniform_emb    complete graph without self-loops in the embedding refinement (as r7_controls.py)
    er_emb         Erdos-Renyi graph with as many edges as the support graph (as r6_controls.py)
    support_emb    support graph (as r6_controls.py)
    hash           B1, parameter-sharing control: A = 0, and the KC embedding table is replaced by
                   E_k = mean of k rows of a shared table with K rows, the rows of KC k drawn at
                   random (seed 3000 + sub) without replacement; k = mean number of neighbours per
                   KC in the support graph (in plus out degree), rounded and clipped to [2, 8].
  A2, second backbone (SAKT [Pandey and Karypis 2019] with the embedding refinement of Eq. (2)):
    sakte_empty, sakte_uniform, sakte_er, sakte_support
                   SAKT-E: interaction embedding Z[c_t] + R[r_t] + position, query Z[c_{t+1}], where
                   Z is the KC table refined over the graph as in Eq. (2) (tanh projections of the
                   in- and out-neighbour means, layer normalisation); the attention block, the
                   feed-forward block and the output layer are those of models.SAKT.
  A1, a published graph-based tracer (GKT [Nakagawa et al. 2019], models.GKT, d = 32):
    gkt_empty, gkt_er, gkt_support, gkt_rewired
                   GKT with the empty graph, the Erdos-Renyi graph, the support graph and the
                   degree-preserving rewiring of the support graph. With A = 0 a KC state is updated
                   only when that KC is answered. Batch size 32 (16 above 150 KCs, 8 above 400), as
                   in train.py.
Budget: at most --max_steps updates (2,000), validation AUC every 25 updates, patience 8 evaluations,
best validation checkpoint evaluated on the test learners (as Table 4 of the manuscript).
Output: runs_r8/<tag>.json and <tag>_pred.npz (p, y, new = first attempt, kc = target KC).
"""
import sys, os, json, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_controls import FKGKT3, load, transition_support, rewire, ece
from r6_controls import erdos_renyi
from models import GKT, rownorm
from train import seqs, batches
torch.set_num_threads(int(os.environ.get('NT', '1')))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
GR = f'{ROOT}/graphs_r5'; os.makedirs(GR, exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FKGKT3Hash(FKGKT3):
    """Empty-graph probe whose KC table shares parameters between KCs by hashing (B1)."""
    def __init__(s, K, k, seed, d=64, drop=0.2):
        super().__init__(K, np.zeros((K, K), np.float32), recency=False, d=d, drop=drop)
        rng = np.random.default_rng(seed)
        H = np.stack([rng.choice(K, size=k, replace=False) for _ in range(K)])
        s.register_buffer('H', torch.tensor(H, dtype=torch.long))
        s.T = nn.Embedding(K, d); s.k = k
        base = s.base
        def kc_repr():
            Z = s.T.weight[s.H].mean(1)
            for Wi, Wo in zip(base.Win, base.Wout): Z = Z + torch.tanh(Wi(base.Ain @ Z)) + torch.tanh(Wo(base.Aout @ Z))
            Z = base.ln(Z)
            return torch.cat([Z, torch.zeros(1, Z.size(1), device=Z.device)], 0)
        base.kc_repr = kc_repr


class SAKTE(nn.Module):
    """SAKT with a KC embedding table refined over a graph (entry point E of Section 4.2)."""
    def __init__(s, K, A, d=64, heads=4, drop=0.2, L=200):
        super().__init__(); s.K = K
        A = torch.tensor(A, dtype=torch.float32); A.fill_diagonal_(0)
        s.register_buffer('Ain', rownorm(A.T)); s.register_buffer('Aout', rownorm(A))
        s.E = nn.Embedding(K + 1, d, padding_idx=K); s.R = nn.Embedding(2, d); s.pos = nn.Embedding(L, d)
        s.Win = nn.Linear(d, d); s.Wout = nn.Linear(d, d); s.lnE = nn.LayerNorm(d)
        s.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True); s.ln1 = nn.LayerNorm(d)
        s.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(drop), nn.Linear(d, d)); s.ln2 = nn.LayerNorm(d)
        s.out = nn.Linear(d, 1); s.dp = nn.Dropout(drop)
    def kc_repr(s):
        Z = s.E.weight[:s.K]
        Z = s.lnE(Z + torch.tanh(s.Win(s.Ain @ Z)) + torch.tanh(s.Wout(s.Aout @ Z)))
        return torch.cat([Z, torch.zeros(1, Z.size(1), device=Z.device)], 0)
    def forward(s, q, r, qn):
        Z = s.kc_repr(); T = q.size(1); p = s.pos(torch.arange(T, device=q.device))
        kv = s.dp(Z[q.clamp(max=s.K)] + s.R(r)) + p; qq = Z[qn.clamp(max=s.K)]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
        a, _ = s.att(qq, kv, kv, attn_mask=mask); a = s.ln1(a + qq); a = s.ln2(a + s.ffn(a))
        return s.out(a).squeeze(2)


def evaluate(m, q, r, K, bs):
    m.eval(); P = []; Y = []; NEW = []; KC = []
    with torch.no_grad():
        for j, a, b, c, y in batches(q, r, bs, False, None):
            p = torch.sigmoid(m(a.to(DEV), b.to(DEV), c.to(DEV))).cpu(); mask = c < K
            P.append(p[mask].numpy()); Y.append(y[mask].numpy()); KC.append(c[mask].numpy())
            qq = torch.cat([a[:, :1], c], 1).numpy(); seen = np.zeros(c.shape, dtype=bool)
            for i in range(qq.shape[0]):
                u, fi = np.unique(qq[i], return_index=True); first = dict(zip(u, fi))
                seen[i] = np.array([first[v] < t + 1 for t, v in enumerate(qq[i, 1:])])
            NEW.append((~seen)[mask.numpy()])
    return np.concatenate(P), np.concatenate(Y), np.concatenate(NEW), np.concatenate(KC)


def hash_k(sup):
    K = sup.shape[0]
    return int(np.clip(round(2 * sup.sum() / max(K, 1)), 2, 8))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--frac', type=float, default=0.1); ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--cond'); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--max_steps', type=int, default=2000); ap.add_argument('--eval_every', type=int, default=25); ap.add_argument('--patience_evals', type=int, default=8)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--out', default=f'{ROOT}/runs_r8')
    a = ap.parse_args(argv); seed = a.sub; os.makedirs(a.out, exist_ok=True)
    tag = f"{a.data}__f{int(round(a.frac*100))}__s{a.sub}__{a.cond}__steps"
    if os.path.exists(f'{a.out}/{tag}.json'): print('skip', tag); return
    df = load(a.data, a.frac, a.sub); K = int(df.kc.max()) + 1; tr = df[df.split == 'train']
    gpath = f'{GR}/{a.data}__f{int(round(a.frac*100))}__s{a.sub}.npz'
    if os.path.exists(gpath):
        g = np.load(gpath); sup = g['support']; rw = g['rewired']
    else:
        mu = transition_support(tr, K); sup = (mu > 0).astype(np.float32)
        rw, _ = rewire(sup, seed=1000 + a.sub); rw = rw.astype(np.float32)
        np.savez_compressed(gpath, support=sup, rewired=rw)
    er = erdos_renyi(K, int(sup.sum()), seed=2000 + a.sub)
    Z0 = np.zeros((K, K), np.float32); U = np.ones((K, K), np.float32); np.fill_diagonal(U, 0)
    graph = {'empty': Z0, 'uniform': U, 'er': er, 'support': sup, 'rewired': rw}
    fam, _, gname = a.cond.partition('_')
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    bs = a.bs
    if a.cond in ('empty', 'uniform_emb', 'er_emb', 'support_emb'):
        A = graph[a.cond.replace('_emb', '')]
        m = FKGKT3(K, A, recency=False)
        with torch.no_grad():
            if a.cond != 'empty': m.S.zero_()
        family = 'probe'
    elif a.cond == 'hash':
        A = Z0; m = FKGKT3Hash(K, hash_k(sup), seed=3000 + a.sub); family = 'probe'
    elif fam == 'sakte':
        A = graph[gname]; m = SAKTE(K, A); family = 'sakte'
    elif fam == 'gkt':
        A = graph[gname]; m = GKT(K, A); family = 'gkt'
        bs = 32 if K <= 150 else (16 if K <= 400 else 8)
    else:
        raise ValueError(a.cond)
    m = m.to(DEV)
    tq, trr, _ = seqs(df, K, 'train'); vq, vr, _ = seqs(df, K, 'valid'); sq, sr, _ = seqs(df, K, 'test')
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    best = -1; bad = 0; t0 = time.time(); hist = []; state = None; best_at = 0; steps = 0; stop = False
    ebs = max(bs, 32)
    def step(qa, ra, qn, y):
        m.train(); qa, ra, qn, y = qa.to(DEV), ra.to(DEV), qn.to(DEV), y.to(DEV)
        lo = m(qa, ra, qn); mk = qn < K
        loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    def check():
        nonlocal best, bad, state, best_at
        P, Y, _, _ = evaluate(m, vq, vr, K, ebs); va = roc_auc_score(Y, P); hist.append(va)
        if va > best + 1e-5: best = va; bad = 0; state = {k: v.detach().clone() for k, v in m.state_dict().items()}; best_at = steps
        else: bad += 1
        return bad >= a.patience_evals
    while not stop:
        for j, qa, ra, qn, y in batches(tq, trr, bs, True, rng):
            step(qa, ra, qn, y); steps += 1
            if steps % a.eval_every == 0 and check(): stop = True; break
            if steps >= a.max_steps: stop = True; break
    if state is None: check()
    hit = steps >= a.max_steps and bad < a.patience_evals
    m.load_state_dict(state)
    P, Y, NEW, KC = evaluate(m, sq, sr, K, ebs)
    res = dict(tag=tag, data=a.data, frac=a.frac, sub=a.sub, seed=seed, cond=a.cond, family=family, budget='steps', K=K,
               device=str(DEV), batch_size=bs, lr=a.lr,
               n_train_learners=int(tr.user.nunique()), n_train_inter=int(len(tr)), edges=int(sup.sum()), edges_cond=int(np.asarray(A).sum()),
               hash_k=(m.k if a.cond == 'hash' else None),
               test_auc=float(roc_auc_score(Y, P)), test_auc_first=float(roc_auc_score(Y[NEW], P[NEW])) if len(set(Y[NEW])) > 1 else None,
               test_acc=float(((P > 0.5) == Y).mean()), test_rmse=float(np.sqrt(((P - Y) ** 2).mean())), test_ece=ece(P, Y),
               n_pred=int(len(Y)), n_first=int(NEW.sum()), valid_auc=float(best), steps=steps, best_step=best_at, hit_max=bool(hit),
               n_evals=len(hist), valid_hist=hist, max_steps=a.max_steps, eval_every=a.eval_every, patience_evals=a.patience_evals,
               secs=time.time() - t0)
    np.savez_compressed(f'{a.out}/{tag}_pred.npz', p=P.astype(np.float32), y=Y.astype(np.int8), new=NEW, kc=KC.astype(np.int32))
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    print(tag, 'auc', round(res['test_auc'], 4), 'first', round(res['test_auc_first'] or 0, 4), 'steps', steps, 'best', best_at, 's', round(res['secs']), flush=True)


if __name__ == '__main__': main()
