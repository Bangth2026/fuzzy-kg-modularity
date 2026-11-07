"""Release 3.1.0: training budget by update steps, and two further graph controls.

Two additions to r5_controls.py (same model FKGKT3 = models.FKGKT2 plus optional recency term,
same data loading, graph construction, subsamples and evaluation):

1. Budget by steps (--budget steps). Release 2.1.0 trains for at most 30 epochs with batches of 128
   learner sequences, so the number of Adam updates is 30 * ceil(n_train / 128): 30 on a corpus with
   fewer than 128 training learners. With --budget steps the tracer is trained for at most
   --max_steps updates (default 2,000); validation AUC is computed every --eval_every updates
   (default 25) and training stops after --patience_evals evaluations without improvement
   (default 8, i.e. 200 updates). Batches are drawn by reshuffling the training learners each time
   they are exhausted, as in the epoch schedule. The checkpoint with the best validation AUC is
   evaluated.

2. Two further conditions (--cond):
   er       Erdos-Renyi graph with as many directed edges as the support graph of the subsample,
            drawn uniformly over ordered pairs without self-loops and without reciprocal pairs;
            it keeps the density and loses the degree sequence.
   degfeat  empty graph (A = 0) plus the log in- and out-degree of every KC in the support graph,
            standardised over the KCs of the corpus, mapped by a learned linear layer into the KC
            embedding before the layer normalisation of Eq. (2); the tracer is told how many
            neighbours each KC has and nothing about which ones.
The conditions of release 2.1.0 (empty, support, rewired) are unchanged.
"""
import sys, os, json, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r5_controls as R5
from r5_controls import FKGKT3, load, transition_support, rewire, evaluate, ece
from train import seqs, batches
torch.set_num_threads(int(os.environ.get('NT', '1')))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
GR = f'{ROOT}/graphs_r5'; os.makedirs(GR, exist_ok=True)


def erdos_renyi(K, E, seed):
    """Directed graph with E edges uniform over ordered pairs a != b, no reciprocal pair."""
    rng = np.random.default_rng(seed); A = np.zeros((K, K), np.float32); n = 0
    E = min(E, K * (K - 1) // 2)
    while n < E:
        a, b = rng.integers(K, size=2)
        if a == b or A[a, b] or A[b, a]: continue
        A[a, b] = 1; n += 1
    return A


class FKGKT3Deg(FKGKT3):
    """Empty-graph control plus a degree feature in the KC embedding."""
    def __init__(s, K, A, deg, d=64, drop=0.2):
        super().__init__(K, A, recency=False, d=d)
        s.register_buffer('phi', torch.tensor(deg, dtype=torch.float32)); s.Wdeg = nn.Linear(deg.shape[1], d)
        base = s.base
        def kc_repr():
            Z = base.E.weight[:base.K] + s.Wdeg(s.phi)
            for Wi, Wo in zip(base.Win, base.Wout): Z = Z + torch.tanh(Wi(base.Ain @ Z)) + torch.tanh(Wo(base.Aout @ Z))
            Z = base.ln(Z)
            return torch.cat([Z, torch.zeros(1, Z.size(1), device=Z.device)], 0)
        base.kc_repr = kc_repr


def degree_features(sup):
    f = np.stack([np.log1p(sup.sum(0)), np.log1p(sup.sum(1))], 1)   # in-degree, out-degree
    return (f - f.mean(0)) / np.maximum(f.std(0), 1e-6)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--frac', type=float); ap.add_argument('--sub', type=int, default=0); ap.add_argument('--cond')
    ap.add_argument('--budget', default='epochs', choices=['epochs', 'steps'])
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--max_steps', type=int, default=2000); ap.add_argument('--eval_every', type=int, default=25); ap.add_argument('--patience_evals', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--out', default=f'{ROOT}/runs_r6')
    a = ap.parse_args(argv); seed = a.sub; os.makedirs(a.out, exist_ok=True)
    tag = f"{a.data}__f{int(round(a.frac*100))}__s{a.sub}__{a.cond}__{a.budget}"
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
    Z0 = np.zeros((K, K), np.float32)
    A = {'empty': Z0, 'support': sup, 'rewired': rw, 'er': er, 'degfeat': Z0}[a.cond]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tq, trr, _ = seqs(df, K, 'train'); vq, vr, _ = seqs(df, K, 'valid'); sq, sr, _ = seqs(df, K, 'test')
    m = FKGKT3Deg(K, A, degree_features(sup)) if a.cond == 'degfeat' else FKGKT3(K, A, recency=False)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    best = -1; bad = 0; t0 = time.time(); hist = []; state = None; best_at = 0; steps = 0; stop = False
    def step(qa, ra, qn, y):
        m.train(); lo = m(qa, ra, qn); mk = qn < K
        loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    def check():
        nonlocal best, bad, state, best_at
        P, Y, _, _ = evaluate(m, vq, vr, K); va = roc_auc_score(Y, P); hist.append(va)
        if va > best + 1e-5: best = va; bad = 0; state = {k: v.detach().clone() for k, v in m.state_dict().items()}; best_at = steps
        else: bad += 1
        return bad >= (a.patience if a.budget == 'epochs' else a.patience_evals)
    if a.budget == 'epochs':
        for ep in range(a.epochs):
            for j, qa, ra, qn, y in batches(tq, trr, a.bs, True, rng): step(qa, ra, qn, y); steps += 1
            if check(): break
        hit = len(hist) == a.epochs
    else:
        while not stop:
            for j, qa, ra, qn, y in batches(tq, trr, a.bs, True, rng):
                step(qa, ra, qn, y); steps += 1
                if steps % a.eval_every == 0 and check(): stop = True; break
                if steps >= a.max_steps: stop = True; break
        hit = steps >= a.max_steps and bad < a.patience_evals
    m.load_state_dict(state)
    P, Y, NEW, PREV = evaluate(m, sq, sr, K)
    res = dict(tag=tag, data=a.data, frac=a.frac, sub=a.sub, seed=seed, cond=a.cond, budget=a.budget, K=K,
               n_train_learners=int(tr.user.nunique()), n_train_inter=int(len(tr)), edges=int(sup.sum()), edges_cond=int(A.sum()),
               test_auc=float(roc_auc_score(Y, P)), test_auc_first=float(roc_auc_score(Y[NEW], P[NEW])) if len(set(Y[NEW])) > 1 else None,
               test_acc=float(((P > 0.5) == Y).mean()), test_rmse=float(np.sqrt(((P - Y) ** 2).mean())), test_ece=ece(P, Y),
               n_pred=int(len(Y)), n_first=int(NEW.sum()), valid_auc=float(best), steps=steps, best_step=best_at, hit_max=bool(hit),
               n_evals=len(hist), valid_hist=hist, steps_per_epoch=int(np.ceil(len(tq) / a.bs)),
               beta_pre=float(m.bpre.detach()), beta_suc=float(m.bsuc.detach()), beta_same=float(m.bsame.detach()),
               lr=a.lr, max_steps=a.max_steps, eval_every=a.eval_every, patience_evals=a.patience_evals, max_epochs=a.epochs, secs=time.time() - t0)
    np.savez_compressed(f'{a.out}/{tag}_pred.npz', p=P.astype(np.float32), y=Y.astype(np.int8), new=NEW)
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    print(tag, 'auc', round(res['test_auc'], 4), 'first', round(res['test_auc_first'] or 0, 4), 'steps', steps, 'best', best_at, 'b', round(res['beta_pre'], 3), 's', round(res['secs']))

if __name__ == '__main__': main()
