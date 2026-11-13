"""Release 3.6.0: a complete null-graph ladder, a frozen-random-table control, per-prediction
neighbour flags and seed replicates (Sections 5.4 and 6.5 of the Neural Networks manuscript).

Same data, subsamples, graphs, update budget (at most 2,000 updates, validation every 25, patience 8),
training loop and evaluation as r6_controls.py, r7_controls.py and r8_controls.py. Run on CPU with one
thread, so that the runs pair with the archived CPU runs of runs_r6 and runs_r7 (a rerun of an archived
condition reproduces its test AUC to the printed digit).

Conditions (--cond):
  Probe (FKG-KT, graph read in the embedding refinement of Eq. (2) only; attention bias zero):
    empty, uniform_emb, er_emb, support_emb    as in r6_controls.py / r7_controls.py
    dp_emb      degree-preserving rewiring of the support graph (seed 1000+s), embedding refinement only;
                the rung that completes the ladder empty -> U -> ER -> DP -> G in one setting
    erR_emb     frozen-random-table control: the KC embedding is
                    LN(E_k + tanh(W_in (A_in R)_k) + tanh(W_out (A_out R)_k)),
                with A the Erdos-Renyi graph (seed 2000+s) and R a K x d table drawn once from N(0, 1)
                (the initialisation of E; seed 4000+s) and never trained. E, W_in, W_out are trained as in
                er_emb. If erR_emb recovers the gain of er_emb, the gain needs no learned neighbour embedding.
  Probe with both entry points: support, empty (the comparison of Tables 3 and 4).
  GKT (models.GKT, d = 32, graph routes the state): gkt_empty, gkt_uniform, gkt_er, gkt_rewired,
    gkt_support; gkt_uniform is the complete graph without self-loops.
--rep j (default 0) changes the training seed only (parameter initialisation, batch order, dropout):
seed = sub + 100 j. The subsample and its graphs are those of seed 0.

Output: runs_r9/<tag>.json and <tag>_pred.npz with p, y, new (first attempt at the target KC), kc (target
KC), nb_sup and nb_er (an in- or out-neighbour of the target KC in the support graph, respectively the
Erdos-Renyi graph, occurs earlier in the learner's sequence), and pos (position in the sequence).
"""
import sys, os, json, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_controls import FKGKT3, load, transition_support, rewire, ece
from r6_controls import erdos_renyi
from models import GKT
from train import seqs, batches
torch.set_num_threads(int(os.environ.get('NT', '1')))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
GR = f'{ROOT}/graphs_r5'; os.makedirs(GR, exist_ok=True)
DEV = torch.device('cpu')


class FKGKT3FrozenR(FKGKT3):
    """Probe whose embedding refinement mixes a frozen random table R instead of E."""
    def __init__(s, K, A, seed, d=64, drop=0.2):
        super().__init__(K, A, recency=False, d=d, drop=drop)
        g = torch.Generator().manual_seed(seed)
        s.register_buffer('Rfix', torch.randn(K, d, generator=g))
        base = s.base
        def kc_repr():
            Z = base.E.weight[:base.K]
            for Wi, Wo in zip(base.Win, base.Wout):
                Z = Z + torch.tanh(Wi(base.Ain @ s.Rfix)) + torch.tanh(Wo(base.Aout @ s.Rfix))
            Z = base.ln(Z)
            return torch.cat([Z, torch.zeros(1, Z.size(1), device=Z.device)], 0)
        base.kc_repr = kc_repr


def neighbour_sets(A):
    B = (np.asarray(A) + np.asarray(A).T) > 0
    return [set(np.flatnonzero(B[k])) for k in range(B.shape[0])]


def evaluate(m, q, r, K, bs, nbs):
    m.eval(); P = []; Y = []; NEW = []; KC = []; POS = []; NB = {k: [] for k in nbs}
    with torch.no_grad():
        for j, a, b, c, y in batches(q, r, bs, False, None):
            p = torch.sigmoid(m(a, b, c)); mask = (c < K).numpy()
            P.append(p.numpy()[mask]); Y.append(y.numpy()[mask]); KC.append(c.numpy()[mask])
            qq = torch.cat([a[:, :1], c], 1).numpy(); T = c.shape[1]
            seen = np.zeros(c.shape, bool); nb = {k: np.zeros(c.shape, bool) for k in nbs}
            pos = np.broadcast_to(np.arange(1, T + 1), c.shape)
            for i in range(qq.shape[0]):
                prev = set()
                for t in range(T):
                    prev.add(int(qq[i, t])); v = int(qq[i, t + 1])
                    if v >= K: break
                    seen[i, t] = v in prev
                    for k, N in nbs.items(): nb[k][i, t] = not N[v].isdisjoint(prev)
            NEW.append((~seen)[mask]); POS.append(pos[mask])
            for k in nbs: NB[k].append(nb[k][mask])
    out = dict(p=np.concatenate(P), y=np.concatenate(Y), new=np.concatenate(NEW), kc=np.concatenate(KC),
               pos=np.concatenate(POS))
    for k in nbs: out['nb_' + k] = np.concatenate(NB[k])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--frac', type=float, default=0.1); ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--cond'); ap.add_argument('--rep', type=int, default=0); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--max_steps', type=int, default=2000); ap.add_argument('--eval_every', type=int, default=25); ap.add_argument('--patience_evals', type=int, default=8)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--out', default=f'{ROOT}/runs_r9')
    a = ap.parse_args(argv); seed = a.sub + 100 * a.rep; os.makedirs(a.out, exist_ok=True)
    tag = f"{a.data}__f{int(round(a.frac*100))}__s{a.sub}__{a.cond}" + (f"__rep{a.rep}" if a.rep else '') + "__steps"
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
    graph = {'empty': Z0, 'uniform': U, 'er': er, 'support': sup, 'rewired': rw, 'dp': rw}
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    bs = a.bs
    if a.cond in ('empty', 'uniform_emb', 'er_emb', 'support_emb', 'dp_emb'):
        A = graph[a.cond.replace('_emb', '')]; m = FKGKT3(K, A, recency=False)
        with torch.no_grad():
            if a.cond != 'empty': m.S.zero_()
        family = 'probe'
    elif a.cond == 'support':
        A = sup; m = FKGKT3(K, A, recency=False); family = 'probe'
    elif a.cond == 'erR_emb':
        A = er; m = FKGKT3FrozenR(K, A, seed=4000 + a.sub)
        with torch.no_grad(): m.S.zero_()
        family = 'probe'
    elif a.cond.startswith('gkt_'):
        A = graph[a.cond[4:]]; m = GKT(K, A); family = 'gkt'
        bs = 32 if K <= 150 else (16 if K <= 400 else 8)
    else:
        raise ValueError(a.cond)
    tq, trr, _ = seqs(df, K, 'train'); vq, vr, _ = seqs(df, K, 'valid'); sq, sr, _ = seqs(df, K, 'test')
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)
    best = -1; bad = 0; t0 = time.time(); hist = []; state = None; best_at = 0; steps = 0; stop = False
    ebs = max(bs, 32)
    def step(qa, ra, qn, y):
        m.train(); lo = m(qa, ra, qn); mk = qn < K
        loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    def check():
        nonlocal best, bad, state, best_at
        o = evaluate(m, vq, vr, K, ebs, {}); va = roc_auc_score(o['y'], o['p']); hist.append(va)
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
    o = evaluate(m, sq, sr, K, ebs, {'sup': neighbour_sets(sup), 'er': neighbour_sets(er)})
    P, Y, NEW = o['p'], o['y'], o['new']
    def auc(mk): return float(roc_auc_score(Y[mk], P[mk])) if len(set(Y[mk])) > 1 else None
    res = dict(tag=tag, data=a.data, frac=a.frac, sub=a.sub, rep=a.rep, seed=seed, cond=a.cond, family=family, budget='steps',
               K=K, device='cpu', batch_size=bs, lr=a.lr, n_train_learners=int(tr.user.nunique()), n_train_inter=int(len(tr)),
               edges=int(sup.sum()), edges_cond=int(np.asarray(A).sum()),
               test_auc=float(roc_auc_score(Y, P)), test_auc_first=auc(NEW),
               test_auc_first_nbsup=auc(NEW & o['nb_sup']), test_auc_first_nonbsup=auc(NEW & ~o['nb_sup']),
               test_acc=float(((P > 0.5) == Y).mean()), test_rmse=float(np.sqrt(((P - Y) ** 2).mean())), test_ece=ece(P, Y),
               n_pred=int(len(Y)), n_first=int(NEW.sum()), n_first_nbsup=int((NEW & o['nb_sup']).sum()),
               n_first_nber=int((NEW & o['nb_er']).sum()), valid_auc=float(best), steps=steps, best_step=best_at,
               hit_max=bool(hit), n_evals=len(hist), valid_hist=hist, max_steps=a.max_steps, eval_every=a.eval_every,
               patience_evals=a.patience_evals, secs=time.time() - t0)
    np.savez_compressed(f'{a.out}/{tag}_pred.npz', p=P.astype(np.float32), y=Y.astype(np.int8), new=NEW,
                        kc=o['kc'].astype(np.int32), nb_sup=o['nb_sup'], nb_er=o['nb_er'], pos=o['pos'].astype(np.int16))
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    print(tag, 'auc', round(res['test_auc'], 4), 'first', round(res['test_auc_first'] or 0, 4), 'steps', steps, 'best', best_at,
          's', round(res['secs']), flush=True)


if __name__ == '__main__': main()
