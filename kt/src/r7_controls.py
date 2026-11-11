"""Release 3.4.0: controls that separate neighbour mixing from shrinkage, extra capacity and
optimisation settings. Update budget (as Table 6), 10% of the training learners.

Same model, data, subsamples, graph construction, training loop and evaluation as r6_controls.py.
Conditions (--cond):
  empty, support     as in r6_controls.py (rerun here so that per-prediction files with the target
                     KC are kept for the analyses by KC)
  uniform_emb        the embedding refinement of Eq. (2) reads the complete graph, A = 11^T - I, so
                     after row normalisation every KC is mixed with the mean embedding of all other
                     KCs; the attention bias of Eq. (3) gets a zero matrix (as in support_emb, er_emb)
  empty_l2, support_l2
                     as empty / support, with the penalty lambda * mean_k ||E_k - mean_j E_j||^2 on
                     the KC embedding table added to the training loss (--l2, shrinkage of every KC
                     embedding toward the mean embedding)
  --lr sets the shared learning rate (default 0.002) for the learning-rate sweep.
Per-prediction files (_pred.npz) hold p, y, new (first attempt) and kc (target KC).
"""
import sys, os, json, time, argparse, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_controls import FKGKT3, load, transition_support, rewire, evaluate, ece
from train import seqs, batches
torch.set_num_threads(int(os.environ.get('NT', '1')))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
GR = f'{ROOT}/graphs_r5'; os.makedirs(GR, exist_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--frac', type=float, default=0.1); ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--cond'); ap.add_argument('--l2', type=float, default=0.0); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--max_steps', type=int, default=2000); ap.add_argument('--eval_every', type=int, default=25); ap.add_argument('--patience_evals', type=int, default=8)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--out', default=f'{ROOT}/runs_r7')
    a = ap.parse_args(argv); seed = a.sub; os.makedirs(a.out, exist_ok=True)
    extra = (f'__l2{a.l2:g}' if a.cond.endswith('_l2') else '') + ('' if a.lr == 2e-3 else f'__lr{a.lr:g}')
    tag = f"{a.data}__f{int(round(a.frac*100))}__s{a.sub}__{a.cond}{extra}__steps"
    if os.path.exists(f'{a.out}/{tag}.json'): print('skip', tag); return
    df = load(a.data, a.frac, a.sub); K = int(df.kc.max()) + 1; tr = df[df.split == 'train']
    gpath = f'{GR}/{a.data}__f{int(round(a.frac*100))}__s{a.sub}.npz'
    if os.path.exists(gpath):
        g = np.load(gpath); sup = g['support']
    else:
        mu = transition_support(tr, K); sup = (mu > 0).astype(np.float32)
        rw, _ = rewire(sup, seed=1000 + a.sub); np.savez_compressed(gpath, support=sup, rewired=rw.astype(np.float32))
    Z0 = np.zeros((K, K), np.float32); U = np.ones((K, K), np.float32); np.fill_diagonal(U, 0)
    base_cond = a.cond.replace('_l2', '')
    A = {'empty': Z0, 'support': sup, 'uniform_emb': U}[base_cond]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tq, trr, _ = seqs(df, K, 'train'); vq, vr, _ = seqs(df, K, 'valid'); sq, sr, _ = seqs(df, K, 'test')
    m = FKGKT3(K, A, recency=False)
    with torch.no_grad():
        if a.cond == 'uniform_emb': m.S.zero_()
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    best = -1; bad = 0; t0 = time.time(); hist = []; state = None; best_at = 0; steps = 0; stop = False
    def step(qa, ra, qn, y):
        m.train(); lo = m(qa, ra, qn); mk = qn < K
        loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float())
        if a.l2 > 0 and a.cond.endswith('_l2'):
            E = m.base.E.weight[:K]; loss = loss + a.l2 * ((E - E.mean(0, keepdim=True)) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    def check():
        nonlocal best, bad, state, best_at
        P, Y, _, _ = evaluate(m, vq, vr, K); va = roc_auc_score(Y, P); hist.append(va)
        if va > best + 1e-5: best = va; bad = 0; state = {k: v.detach().clone() for k, v in m.state_dict().items()}; best_at = steps
        else: bad += 1
        return bad >= a.patience_evals
    while not stop:
        for j, qa, ra, qn, y in batches(tq, trr, a.bs, True, rng):
            step(qa, ra, qn, y); steps += 1
            if steps % a.eval_every == 0 and check(): stop = True; break
            if steps >= a.max_steps: stop = True; break
    hit = steps >= a.max_steps and bad < a.patience_evals
    m.load_state_dict(state)
    P, Y, NEW, PREV = evaluate(m, sq, sr, K)
    res = dict(tag=tag, data=a.data, frac=a.frac, sub=a.sub, seed=seed, cond=a.cond, l2=a.l2, lr=a.lr, budget='steps', K=K,
               n_train_learners=int(tr.user.nunique()), n_train_inter=int(len(tr)), edges=int(sup.sum()), edges_cond=int(A.sum()),
               test_auc=float(roc_auc_score(Y, P)), test_auc_first=float(roc_auc_score(Y[NEW], P[NEW])) if len(set(Y[NEW])) > 1 else None,
               test_acc=float(((P > 0.5) == Y).mean()), test_rmse=float(np.sqrt(((P - Y) ** 2).mean())), test_ece=ece(P, Y),
               n_pred=int(len(Y)), n_first=int(NEW.sum()), valid_auc=float(best), steps=steps, best_step=best_at, hit_max=bool(hit),
               n_evals=len(hist), valid_hist=hist, beta_pre=float(m.bpre.detach()), beta_suc=float(m.bsuc.detach()),
               beta_same=float(m.bsame.detach()), max_steps=a.max_steps, eval_every=a.eval_every, patience_evals=a.patience_evals,
               secs=time.time() - t0)
    np.savez_compressed(f'{a.out}/{tag}_pred.npz', p=P.astype(np.float32), y=Y.astype(np.int8), new=NEW, kc=PREV[:, 1].astype(np.int32))
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    print(tag, 'auc', round(res['test_auc'], 4), 'first', round(res['test_auc_first'] or 0, 4), 'steps', steps, 'best', best_at, 's', round(res['secs']))

if __name__ == '__main__': main()
