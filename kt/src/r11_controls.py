"""Release 3.8.0: where does the gain of the random-vector control come from? The maps W_in, W_out frozen.

Conditions added to r10_controls.py (same data, subsamples, initialisation, update budget, training loop, evaluation and
per-prediction files, CPU with one thread, training seed 0):

  rvecF_emb   rvec_emb with W_in, W_out and their biases frozen at their initial values. The model is built with the same
              generator calls as rvec_emb, so the component tanh(W_in q_in,k + b_in) + tanh(W_out q_out,k + b_out) equals the
              one rvec_emb starts from, but it stays fixed: each KC receives a fixed random offset to E_k and nothing else.
              Separates a shared, trained map of fixed codes (rvec_emb) from a change of the initial geometry only.
  rvec1F_emb  the same for rvec1_emb (unit variance).

Earlier documentation of r10_controls.py follows.

Release 3.7.0: is a graph needed for the gain of sparse mixing? Controls without a graph.

Same data, subsamples, graphs, update budget (at most 2,000 updates, validation every 25, patience 8), training loop,
evaluation and per-prediction files as r9_controls.py, on a CPU with one thread, so that the runs pair with runs_r9
(empty, er_emb, erR_emb) and runs_r7 (uniform_emb). The graph is read in the embedding refinement only (attention bias
zero), and the KC embedding is LN(E_k + tanh(W_in q_in,k) + tanh(W_out q_out,k)) with

  rvec_emb    q_in, q_out fixed random K x d matrices drawn once (seed 5000 + subsample) and never trained, NO graph:
              row k is N(0, s_k^2 I) with s_k^2 the sum of squared weights of row k of the row-normalised Erdos-Renyi
              matrix of erR_emb (1 / in-degree, respectively 1 / out-degree; 0 for a KC without neighbours). Each row
              has the distribution of the corresponding row of A_in R in erR_emb, but rows are drawn independently,
              so no structure shared through neighbours remains.
  rvec1_emb   as rvec_emb with unit variance for every KC (a fixed N(0, 1) vector per KC and channel; seed 6000 + s).
  erE2_emb    as erR_emb (Erdos-Renyi mixing, seed 2000 + s, of a second K x d table initialised as R, seed 4000 + s),
              but the second table is trained: separates a fixed component from a second set of free parameters.
  empty       the empty-graph control, rerun to check that a run of runs_r9 is reproduced exactly.

Output: runs_r10/<tag>.json and <tag>_pred.npz, in the format of r9_controls.py.
"""
import sys, os, json, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_controls import FKGKT3, load, transition_support, rewire, ece
from r6_controls import erdos_renyi
from r9_controls import evaluate, neighbour_sets
from models import rownorm
from train import seqs, batches
torch.set_num_threads(int(os.environ.get('NT', '1')))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
GR = f'{ROOT}/graphs_r5'; os.makedirs(GR, exist_ok=True)


class FKGKT3Fixed(FKGKT3):
    """Probe whose embedding refinement adds tanh(W_in q_in) + tanh(W_out q_out) for fixed or trainable q."""
    def __init__(s, K, A, kind, seed, d=64, drop=0.2):
        super().__init__(K, A, recency=False, d=d, drop=drop)
        base = s.base; g = torch.Generator().manual_seed(seed)
        if kind in ('rvec', 'rvec1'):
            Qin = torch.randn(K, d, generator=g); Qout = torch.randn(K, d, generator=g)
            if kind == 'rvec':
                sin = (base.Ain ** 2).sum(1, keepdim=True).sqrt(); sout = (base.Aout ** 2).sum(1, keepdim=True).sqrt()
                Qin, Qout = Qin * sin, Qout * sout
            s.register_buffer('Qin', Qin); s.register_buffer('Qout', Qout)
            def kc_repr():
                Z = base.E.weight[:base.K]
                for Wi, Wo in zip(base.Win, base.Wout):
                    Z = Z + torch.tanh(Wi(s.Qin)) + torch.tanh(Wo(s.Qout))
                Z = base.ln(Z)
                return torch.cat([Z, torch.zeros(1, Z.size(1), device=Z.device)], 0)
        elif kind == 'erE2':
            s.E2 = nn.Parameter(torch.randn(K, d, generator=g))
            def kc_repr():
                Z = base.E.weight[:base.K]
                for Wi, Wo in zip(base.Win, base.Wout):
                    Z = Z + torch.tanh(Wi(base.Ain @ s.E2)) + torch.tanh(Wo(base.Aout @ s.E2))
                Z = base.ln(Z)
                return torch.cat([Z, torch.zeros(1, Z.size(1), device=Z.device)], 0)
        else:
            raise ValueError(kind)
        base.kc_repr = kc_repr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--frac', type=float, default=0.1); ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--cond'); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--max_steps', type=int, default=2000); ap.add_argument('--eval_every', type=int, default=25); ap.add_argument('--patience_evals', type=int, default=8)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--out', default=f'{ROOT}/runs_r11')
    a = ap.parse_args(argv); seed = a.sub; os.makedirs(a.out, exist_ok=True)
    tag = f"{a.data}__f{int(round(a.frac*100))}__s{a.sub}__{a.cond}__steps"
    if os.path.exists(f'{a.out}/{tag}.json'): print('skip', tag); return
    df = load(a.data, a.frac, a.sub); K = int(df.kc.max()) + 1; tr = df[df.split == 'train']
    gpath = f'{GR}/{a.data}__f{int(round(a.frac*100))}__s{a.sub}.npz'
    if os.path.exists(gpath):
        g = np.load(gpath); sup = g['support']
    else:
        mu = transition_support(tr, K); sup = (mu > 0).astype(np.float32)
        rw, _ = rewire(sup, seed=1000 + a.sub); np.savez_compressed(gpath, support=sup, rewired=rw.astype(np.float32))
    er = erdos_renyi(K, int(sup.sum()), seed=2000 + a.sub)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if a.cond == 'empty':
        A = np.zeros((K, K), np.float32); m = FKGKT3(K, A, recency=False)
    elif a.cond in ('rvec_emb', 'rvec1_emb', 'erE2_emb', 'rvecF_emb', 'rvec1F_emb'):
        frozen = a.cond.endswith('F_emb'); kind = a.cond[:-4].rstrip('F'); A = er
        m = FKGKT3Fixed(K, A, kind, seed={'rvec': 5000, 'rvec1': 6000, 'erE2': 4000}[kind] + a.sub)
        with torch.no_grad(): m.S.zero_()
        if frozen:
            for lin in list(m.base.Win) + list(m.base.Wout):
                for p in lin.parameters(): p.requires_grad_(False)
    else:
        raise ValueError(a.cond)
    graph_used = a.cond == 'erE2_emb'
    tq, trr, _ = seqs(df, K, 'train'); vq, vr, _ = seqs(df, K, 'valid'); sq, sr, _ = seqs(df, K, 'test')
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)
    best = -1; bad = 0; t0 = time.time(); hist = []; state = None; best_at = 0; steps = 0; stop = False
    bs = a.bs; ebs = max(bs, 32)
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
    res = dict(tag=tag, data=a.data, frac=a.frac, sub=a.sub, rep=0, seed=seed, cond=a.cond, family='probe', budget='steps',
               K=K, device='cpu', batch_size=bs, lr=a.lr, n_train_learners=int(tr.user.nunique()), n_train_inter=int(len(tr)),
               edges=int(sup.sum()), graph_used=graph_used, torch=torch.__version__,
               test_auc=float(roc_auc_score(Y, P)), test_auc_first=auc(NEW),
               test_acc=float(((P > 0.5) == Y).mean()), test_rmse=float(np.sqrt(((P - Y) ** 2).mean())), test_ece=ece(P, Y),
               n_pred=int(len(Y)), n_first=int(NEW.sum()), valid_auc=float(best), steps=steps, best_step=best_at,
               hit_max=bool(hit), n_evals=len(hist), valid_hist=hist, max_steps=a.max_steps, eval_every=a.eval_every,
               patience_evals=a.patience_evals, secs=time.time() - t0)
    np.savez_compressed(f'{a.out}/{tag}_pred.npz', p=P.astype(np.float32), y=Y.astype(np.int8), new=NEW,
                        kc=o['kc'].astype(np.int32), nb_sup=o['nb_sup'], nb_er=o['nb_er'], pos=o['pos'].astype(np.int16))
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    print(tag, 'auc', round(res['test_auc'], 4), 'first', round(res['test_auc_first'] or 0, 4), 'steps', steps, 'best', best_at,
          's', round(res['secs']), flush=True)


if __name__ == '__main__': main()
