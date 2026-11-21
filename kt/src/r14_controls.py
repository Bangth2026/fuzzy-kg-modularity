"""Release 3.11.0: runs added in the second revision for IEEE Transactions on Learning Technologies.

Same data pipeline, subsamples, graphs, update budget (at most 2,000 updates, validation every 25,
patience 8), training loop and evaluation as r13_controls.py (CPU, one thread, NT=1). Every condition
of r13_controls.py is available and reproduces its archived run; this script adds graphs that need the
curriculum units of a corpus or an authored map, two corpora prepared by prep_r14.py, and a longer cap
on the interactions kept per learner.

Graphs added (each built on the KCs of the training subsample, read at the entry points named below):
  unit    unit-preserving random graph UR(G): as many edges as the support graph G, of which as many lie
          inside one curriculum unit as in G; each edge is drawn uniformly over the pairs of its class
          (same unit / different units) that are not yet linked, and points from the KC that the
          training learners of the subsample meet earlier (mean index of the first encounter) to the
          one they meet later. It keeps the size of G, its share of within-unit edges and the curriculum
          order of every edge, but not its degrees or the identity of its edges (seed 7000 + s).
  clique  the authored grouping of the release (Junyi topic, Algebra 2005 and Bridge to Algebra unit of
          the problem hierarchy, Statics 2011 module, Eedi level-2 subject, EdNet TOEIC part): every pair
          of KCs in the same group is linked in both directions.
  map     Junyi only: the published prerequisite map (prerequisite -> exercise), as in expert.py.
Conditions added:
  <g>_emb   for g in unit, clique, map: the graph in the embedding refinement only (as support_emb)
  st_<g>    for g in unit, clique, map: the graph in the state-routing channel (as st_support)
  gkt_<g>   GKT with that graph (models.GKT, d = 32, batch size as in r9_controls.py)
Environment KT_MAXLEN=n (default 200) keeps the first n interactions per learner instead of 200; the tag
then ends in __len<n>.
Output: runs_r14/<tag>.json, <tag>_pred.npz (and <tag>_cand.npz with --dump), as r13_controls.py.
"""
import sys, os, json, time, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r5_controls
MAXLEN = int(os.environ.get('KT_MAXLEN', '200')); r5_controls.MAXLEN = MAXLEN
from r5_controls import FKGKT3, load, transition_support, rewire, ece
from r6_controls import erdos_renyi
from r9_controls import neighbour_sets
from models import GKT
import r13_controls as R13
from train import seqs, batches
torch.set_num_threads(int(os.environ.get('NT', '1')))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt'); RAW = os.environ.get('KT_RAW', '/home/claude/raw')
GR = f'{ROOT}/graphs_r5'; os.makedirs(GR, exist_ok=True)


def kc_names(df, K):
    names = ['__none__'] * K
    for k, n in df[['kc', 'kc_name']].drop_duplicates('kc').itertuples(index=False): names[int(k)] = str(n)
    return names


def groups_of(name, names):
    p = f'{ROOT}/data/{name}_groups.json'
    if not os.path.exists(p): return None
    g = json.load(open(p)); return np.array([str(g.get(n)) for n in names])


def first_encounter_rank(tr, K):
    """Mean over training learners of the index (0, 1, ...) at which the learner first meets each KC,
    among the KCs that learner meets; KCs never met get +inf."""
    s = np.zeros(K); n = np.zeros(K)
    for _, g in tr.groupby('user', sort=False):
        q = pd.unique(g.kc.values); s[q] += np.arange(len(q)) / max(1, len(q) - 1); n[q] += 1
    return np.where(n > 0, s / np.maximum(n, 1), np.inf)


def unit_random(sup, grp, rank, seed):
    """UR(G): as many within-unit and between-unit edges as sup, drawn uniformly, directed by rank."""
    rng = np.random.default_rng(seed); K = len(grp)
    same = grp[:, None] == grp[None, :]; np.fill_diagonal(same, False)
    n_in = int((sup * same).sum()); n_out = int(sup.sum()) - n_in
    A = np.zeros((K, K), np.float32)
    iu, ju = np.triu_indices(K, 1)
    for want, cls in ((n_in, same[iu, ju]), (n_out, ~same[iu, ju])):
        pool = np.flatnonzero(cls); want = min(want, len(pool))
        for e in rng.choice(pool, want, replace=False):
            a, b = iu[e], ju[e]
            if rank[b] < rank[a] or (rank[a] == rank[b] and rng.random() < 0.5): a, b = b, a
            A[a, b] = 1
    return A


def clique(grp):
    A = (grp[:, None] == grp[None, :]).astype(np.float32); np.fill_diagonal(A, 0); return A


def junyi_map(names):
    idx = {n: i for i, n in enumerate(names)}; K = len(names)
    ex = pd.read_csv(f'{RAW}/junyi/junyi_Exercise_table.csv'); ex.columns = [c.strip('﻿') for c in ex.columns]
    A = np.zeros((K, K), np.float32)
    for nm, pre in zip(ex['name'], ex['prerequisites']):
        if nm not in idx or pd.isna(pre): continue
        for p in str(pre).split(','):
            p = p.strip()
            if p in idx and p != nm: A[idx[p], idx[nm]] = 1
    return A


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--frac', type=float, default=0.1); ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--cond'); ap.add_argument('--rep', type=int, default=0); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--max_steps', type=int, default=2000); ap.add_argument('--eval_every', type=int, default=25); ap.add_argument('--patience_evals', type=int, default=8)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--out', default=f'{ROOT}/runs_r14'); ap.add_argument('--dump', action='store_true')
    a = ap.parse_args(argv); seed = a.sub + 100 * a.rep; os.makedirs(a.out, exist_ok=True)
    tag = (f"{a.data}__f{int(round(a.frac*100))}__s{a.sub}__{a.cond}" + (f"__rep{a.rep}" if a.rep else '')
           + ('' if abs(a.lr - 2e-3) < 1e-12 else f"__lr{a.lr:g}") + "__steps" + (f"__len{MAXLEN}" if MAXLEN != 200 else ''))
    if os.path.exists(f'{a.out}/{tag}.json'): print('skip', tag); return
    df = load(a.data, a.frac, a.sub); K = int(df.kc.max()) + 1; tr = df[df.split == 'train']
    gpath = f'{GR}/{a.data}__f{int(round(a.frac*100))}__s{a.sub}' + (f'__len{MAXLEN}' if MAXLEN != 200 else '') + '.npz'
    if os.path.exists(gpath):
        g = np.load(gpath); sup = g['support']; rw = g['rewired']
    else:
        mu = transition_support(tr, K); sup = (mu > 0).astype(np.float32)
        rw, _ = rewire(sup, seed=1000 + a.sub); rw = rw.astype(np.float32)
        np.savez_compressed(gpath, support=sup, rewired=rw)
    er = erdos_renyi(K, int(sup.sum()), seed=2000 + a.sub)
    Z0 = np.zeros((K, K), np.float32); U = np.ones((K, K), np.float32); np.fill_diagonal(U, 0)
    graph = {'empty': Z0, 'uniform': U, 'er': er, 'support': sup, 'rewired': rw, 'dp': rw}
    gname = a.cond.split('_', 1)[1] if a.cond.startswith(('st_', 'gkt_')) else a.cond.replace('_emb', '')
    if gname in ('unit', 'clique', 'map'):
        names = kc_names(df, K)
        if gname == 'map':
            graph['map'] = junyi_map(names)
        else:
            grp = groups_of(a.data, names)
            if grp is None: raise SystemExit(f'no curriculum units for {a.data}')
            graph['clique'] = clique(grp)
            if gname == 'unit': graph['unit'] = unit_random(sup, grp, first_encounter_rank(tr, K), seed=7000 + a.sub)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    bs = a.bs
    if a.cond in ('unit_emb', 'clique_emb', 'map_emb'):
        A = graph[gname]; m = FKGKT3(K, A, recency=False)
        with torch.no_grad(): m.S.zero_()
        family = 'probe'
    elif a.cond.startswith('gkt_'):
        A = graph[gname]; m = GKT(K, A); family = 'gkt'
        bs = 32 if K <= 150 else (16 if K <= 400 else 8)
    else:
        m, A, family = R13.build(a, K, graph, sup, er)
    tq, trr, _ = seqs(df, K, 'train', L=MAXLEN); vq, vr, _ = seqs(df, K, 'valid', L=MAXLEN); sq, sr, _ = seqs(df, K, 'test', L=MAXLEN)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)
    best = -1; bad = 0; t0 = time.time(); hist = []; state = None; best_at = 0; steps = 0; stop = False
    ebs = max(bs, 32)
    def step(qa, ra, qn, y):
        m.train(); lo = m(qa, ra, qn); mk = qn < K
        loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    def check():
        nonlocal best, bad, state, best_at
        o = R13.evaluate(m, vq, vr, K, ebs, {}); va = roc_auc_score(o['y'], o['p']); hist.append(va)
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
    o = R13.evaluate(m, sq, sr, K, ebs, {'sup': neighbour_sets(sup), 'er': neighbour_sets(er)})
    P, Y, NEW = o['p'], o['y'], o['new']
    def auc(mk): return float(roc_auc_score(Y[mk], P[mk])) if len(set(Y[mk])) > 1 else None
    res = dict(tag=tag, data=a.data, frac=a.frac, sub=a.sub, rep=a.rep, seed=seed, cond=a.cond, family=family, budget='steps',
               K=K, device='cpu', torch=torch.__version__, batch_size=bs, lr=a.lr, d=int(getattr(m, 'd', 64)), maxlen=MAXLEN,
               n_params=int(sum(p.numel() for p in m.parameters())),
               n_trainable=int(sum(p.numel() for p in m.parameters() if p.requires_grad)),
               n_train_learners=int(tr.user.nunique()), n_train_inter=int(len(tr)),
               edges=int(sup.sum()), edges_cond=int(np.asarray(A).sum()),
               test_auc=float(roc_auc_score(Y, P)), test_auc_first=auc(NEW),
               test_acc=float(((P > 0.5) == Y).mean()), test_rmse=float(np.sqrt(((P - Y) ** 2).mean())), test_ece=ece(P, Y),
               n_pred=int(len(Y)), n_first=int(NEW.sum()), valid_auc=float(best), steps=steps, best_step=best_at,
               hit_max=bool(hit), n_evals=len(hist), valid_hist=hist, max_steps=a.max_steps, eval_every=a.eval_every,
               patience_evals=a.patience_evals, secs=time.time() - t0)
    np.savez_compressed(f'{a.out}/{tag}_pred.npz', p=P.astype(np.float32), y=Y.astype(np.int8), new=NEW,
                        kc=o['kc'].astype(np.int32), nb_sup=o['nb_sup'], nb_er=o['nb_er'], pos=o['pos'].astype(np.int16),
                        uid=o['uid'].astype(np.int32))
    if a.dump and isinstance(m, FKGKT3) and not isinstance(m, R13.FKGKT3State):
        Sc, SEEN = R13.candidate_scores(m, sq, sr, K)
        np.savez_compressed(f'{a.out}/{tag}_cand.npz', score=Sc, seen=SEEN, t=np.array(R13.DUMP_T))
        res['dump'] = True
    res['secs_total'] = time.time() - t0
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    print(tag, 'auc', round(res['test_auc'], 4), 'first', round(res['test_auc_first'] or 0, 4), 'steps', steps, 'best', best_at,
          's', round(res['secs']), flush=True)


if __name__ == '__main__': main()
