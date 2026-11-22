"""Release 3.13.0: runs added in the fourth version of the IEEE TLT manuscript.

Same data pipeline, subsamples, update budget (at most 2,000 updates, validation every 25, patience 8),
training loop and evaluation as r14_controls.py (CPU, one thread, NT=1). Every condition of
r14_controls.py is passed through to it unchanged (and reproduces its archived run); this script adds two
graphs built from the training learners of the subsample in other ways than Eq. (2), each with as many
directed edges as the support graph G of the subsample and, like G, no reciprocal pair and no self-loop:

  dkt   conditional-influence graph of a DKT tracer (Piech et al., 2015). A DKT model (models.DKT, d = 64)
        is trained on the training learners of the subsample under the same update budget and seed as the
        probe; J[i, j] = y(j | i) / sum_k y(j | k), with y(j | i) the predicted probability of a correct
        answer on j after one correct answer on i. For every unordered pair the direction with the larger
        J is kept, and the |E(G)| pairs with the largest J become edges.
  corr  co-performance graph: for every pair of KCs practised by at least 10 common training learners, the
        Pearson correlation across those learners of their proportion correct on the two KCs; the |E(G)|
        pairs with the largest positive correlation become edges, each pointing from the KC the training
        learners meet earlier (mean rank of the first encounter, as UR in r14_controls.py) to the later.
  dpdkt degree-preserving rewiring of dkt (double edge swaps, ten per edge, seed 1000 + s, as DP(G)).

Conditions added: dkt_emb, dpdkt_emb, corr_emb (graph in the embedding refinement only, as support_emb) and
st_dkt, st_dpdkt, st_corr (graph in the state-routing channel, as st_support). The DKT model behind dkt is
trained once per subsample and its graph cached in graphs_r16/<data>__f10__s<s>__dkt.npz together with the
DKT test AUC; the run file records the DKT validation AUC and the overlap of each new graph with G.
Output: runs_r16/<tag>.json, <tag>_pred.npz, as r14_controls.py.
"""
import sys, os, json, time, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r14_controls as R14
from r5_controls import FKGKT3, load, transition_support, rewire, ece
from r6_controls import erdos_renyi
from r9_controls import neighbour_sets
from models import DKT
import r13_controls as R13
from train import seqs, batches
torch.set_num_threads(int(os.environ.get('NT', '1')))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
GR5 = f'{ROOT}/graphs_r5'; GR = f'{ROOT}/graphs_r16'; os.makedirs(GR, exist_ok=True); os.makedirs(GR5, exist_ok=True)
NEW = ('dkt_emb', 'dpdkt_emb', 'corr_emb', 'st_dkt', 'st_dpdkt', 'st_corr')


def support_of(df, a, K):
    tr = df[df.split == 'train']
    gpath = f'{GR5}/{a.data}__f{int(round(a.frac*100))}__s{a.sub}.npz'
    if os.path.exists(gpath):
        g = np.load(gpath); return g['support'], g['rewired']
    mu = transition_support(tr, K); sup = (mu > 0).astype(np.float32)
    rw, _ = rewire(sup, seed=1000 + a.sub); rw = rw.astype(np.float32)
    np.savez_compressed(gpath, support=sup, rewired=rw); return sup, rw


def top_pairs(W, E):
    """|E| directed edges from a score matrix W: per unordered pair the larger direction, largest first."""
    K = W.shape[0]; iu, ju = np.triu_indices(K, 1)
    fw, bw = W[iu, ju], W[ju, iu]; score = np.maximum(fw, bw)
    ok = np.isfinite(score) & (score > 0)
    idx = np.flatnonzero(ok)[np.argsort(-score[ok], kind='stable')][:E]
    A = np.zeros((K, K), np.float32)
    for e in idx:
        a, b = (iu[e], ju[e]) if fw[e] >= bw[e] else (ju[e], iu[e]); A[a, b] = 1
    return A


def train_dkt(df, K, seed, max_steps=2000, eval_every=25, patience=8, bs=128):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = DKT(K)
    tq, trr, _ = seqs(df, K, 'train'); vq, vr, _ = seqs(df, K, 'valid'); sq, sr, _ = seqs(df, K, 'test')
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    def ev(q, r):
        m.eval(); P = []; Y = []
        with torch.no_grad():
            for j, qa, ra, qn, y in batches(q, r, 256, False, rng):
                lo = m(qa, ra, qn); mk = qn < K; P.append(torch.sigmoid(lo[mk]).numpy()); Y.append(y[mk].numpy())
        return roc_auc_score(np.concatenate(Y), np.concatenate(P))
    best = -1; bad = 0; steps = 0; state = None; stop = False
    while not stop:
        for j, qa, ra, qn, y in batches(tq, trr, bs, True, rng):
            m.train(); lo = m(qa, ra, qn); mk = qn < K
            loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); steps += 1
            if steps % eval_every == 0:
                va = ev(vq, vr)
                if va > best + 1e-5: best = va; bad = 0; state = {k: v.clone() for k, v in m.state_dict().items()}
                else: bad += 1
                if bad >= patience: stop = True; break
            if steps >= max_steps: stop = True; break
    if state is not None: m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        q = torch.arange(K).unsqueeze(1); r = torch.ones_like(q); Y = m.all_probs(q, r)[:, 0, :].numpy()
    J = Y / Y.sum(0, keepdims=True); np.fill_diagonal(J, 0)
    return J, float(best), float(ev(sq, sr)), steps


def dkt_graph(df, a, K, sup):
    p = f'{GR}/{a.data}__f{int(round(a.frac*100))}__s{a.sub}__dkt.npz'
    if os.path.exists(p):
        g = np.load(p); return g['A'], g['dp'], dict(dkt_valid_auc=float(g['va']), dkt_test_auc=float(g['ta']), dkt_steps=int(g['steps']))
    J, va, ta, steps = train_dkt(df, K, seed=a.sub)
    A = top_pairs(J, int(sup.sum())); dp, _ = rewire(A, seed=1000 + a.sub); dp = dp.astype(np.float32)
    np.savez_compressed(p, A=A, dp=dp, J=J.astype(np.float32), va=va, ta=ta, steps=steps)
    return A, dp, dict(dkt_valid_auc=va, dkt_test_auc=ta, dkt_steps=steps)


def corr_graph(tr, K, E, min_common=10):
    pc = tr.groupby(['user', 'kc']).correct.mean().unstack()          # learners x KCs, NaN if not practised
    X = np.full((pc.shape[0], K), np.nan); X[:, pc.columns.values.astype(int)] = pc.values
    M = ~np.isnan(X); Xz = np.where(M, X, 0.0); Mf = M.astype(float)
    n = Mf.T @ Mf
    s1 = Xz.T @ Mf; s2 = (Xz ** 2).T @ Mf; sxy = Xz.T @ Xz          # s1[i,j] = sum over common learners of x_i
    mi = s1 / np.maximum(n, 1); mj = s1.T / np.maximum(n, 1)
    cov = sxy / np.maximum(n, 1) - mi * mj
    vi = s2 / np.maximum(n, 1) - mi ** 2; vj = s2.T / np.maximum(n, 1) - mj ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        C = cov / np.sqrt(vi * vj)
    C[(n < min_common) | ~np.isfinite(C)] = 0; np.fill_diagonal(C, 0)
    U = top_pairs(np.triu(C, 1), E)    # undirected scores in the upper triangle
    rank = R14.first_encounter_rank(tr, K); A = np.zeros_like(U)
    for x, y in zip(*np.nonzero(U)):
        if rank[y] < rank[x]: x, y = y, x
        A[x, y] = 1
    return A


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--frac', type=float, default=0.1); ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--cond'); ap.add_argument('--rep', type=int, default=0); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--out', default=f'{ROOT}/runs_r16')
    a, rest = ap.parse_known_args(argv)
    if a.cond not in NEW:
        return R14.main(argv if '--out' in argv else argv + ['--out', a.out])
    seed = a.sub + 100 * a.rep; os.makedirs(a.out, exist_ok=True)
    tag = f"{a.data}__f{int(round(a.frac*100))}__s{a.sub}__{a.cond}" + (f"__rep{a.rep}" if a.rep else '') + "__steps"
    if os.path.exists(f'{a.out}/{tag}.json'): print('skip', tag); return
    df = load(a.data, a.frac, a.sub); K = int(df.kc.max()) + 1; tr = df[df.split == 'train']
    sup, rw = support_of(df, a, K); er = erdos_renyi(K, int(sup.sum()), seed=2000 + a.sub)
    extra = {}
    gname = a.cond[3:] if a.cond.startswith('st_') else a.cond[:-4]
    if gname in ('dkt', 'dpdkt'):
        A_dkt, A_dp, extra = dkt_graph(df, a, K, sup); A = A_dkt if gname == 'dkt' else A_dp
    else:
        A = corr_graph(tr, K, int(sup.sum()))
    S = (sup + sup.T) > 0; B = (A + A.T) > 0
    extra.update(overlap_with_G_directed=float((A * sup).sum() / max(1, A.sum())),
                 overlap_with_G_undirected=float((B & S).sum() / max(1, B.sum())))
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if a.cond.startswith('st_'):
        m = R13.FKGKT3State(K, A); family = 'probe_state'
    else:
        m = FKGKT3(K, A, recency=False); family = 'probe'
        with torch.no_grad(): m.S.zero_()
    tq, trr, _ = seqs(df, K, 'train'); vq, vr, _ = seqs(df, K, 'valid'); sq, sr, _ = seqs(df, K, 'test')
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)
    best = -1; bad = 0; t0 = time.time(); hist = []; state = None; best_at = 0; steps = 0; stop = False
    bs = 128; ebs = 128; max_steps, eval_every, patience = 2000, 25, 8
    while not stop:
        for j, qa, ra, qn, y in batches(tq, trr, bs, True, rng):
            m.train(); lo = m(qa, ra, qn); mk = qn < K
            loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); steps += 1
            if steps % eval_every == 0:
                o = R13.evaluate(m, vq, vr, K, ebs, {}); va = roc_auc_score(o['y'], o['p']); hist.append(va)
                if va > best + 1e-5: best = va; bad = 0; state = {k: v.detach().clone() for k, v in m.state_dict().items()}; best_at = steps
                else: bad += 1
                if bad >= patience: stop = True; break
            if steps >= max_steps: stop = True; break
    m.load_state_dict(state)
    o = R13.evaluate(m, sq, sr, K, ebs, {'sup': neighbour_sets(sup), 'er': neighbour_sets(er)})
    P, Y, NW = o['p'], o['y'], o['new']
    res = dict(tag=tag, data=a.data, frac=a.frac, sub=a.sub, rep=a.rep, seed=seed, cond=a.cond, family=family, budget='steps',
               K=K, device='cpu', torch=torch.__version__, batch_size=bs, lr=a.lr, d=64, maxlen=200,
               n_params=int(sum(p.numel() for p in m.parameters())), n_train_learners=int(tr.user.nunique()),
               n_train_inter=int(len(tr)), edges=int(sup.sum()), edges_cond=int(A.sum()),
               test_auc=float(roc_auc_score(Y, P)), test_auc_first=float(roc_auc_score(Y[NW], P[NW])),
               test_acc=float(((P > 0.5) == Y).mean()), test_rmse=float(np.sqrt(((P - Y) ** 2).mean())), test_ece=ece(P, Y),
               n_pred=int(len(Y)), n_first=int(NW.sum()), valid_auc=float(best), steps=steps, best_step=best_at,
               hit_max=bool(steps >= max_steps and bad < patience), n_evals=len(hist), valid_hist=hist, **extra)
    np.savez_compressed(f'{a.out}/{tag}_pred.npz', p=P.astype(np.float32), y=Y.astype(np.int8), new=NW,
                        kc=o['kc'].astype(np.int32), pos=o['pos'].astype(np.int16), uid=o['uid'].astype(np.int32))
    res['secs_total'] = time.time() - t0
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    print(tag, 'auc', round(res['test_auc'], 4), 'first', round(res['test_auc_first'], 4), 'steps', steps, flush=True)


if __name__ == '__main__': main()
