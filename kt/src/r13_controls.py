"""Release 3.10.0: runs added in the revision for IEEE Transactions on Learning Technologies.

Same data, subsamples, graphs, update budget (at most 2,000 updates, validation every 25, patience 8),
training loop and evaluation as r9_controls.py, r10_controls.py and r11_controls.py; CPU with one thread
(NT=1), torch 2.15.0+cpu. With an archived condition the script reproduces the archived run.

Conditions (--cond):
  Probe (FKG-KT), as in r9/r10/r11:
    empty, support (both entry points), support_emb, er_emb, uniform_emb, dp_emb, rvec_emb, rvecF_emb
  Capacity and parameterisation controls without a graph:
    empty_d128  the empty-graph probe with every width doubled (d = 128): more capacity everywhere
    tvec_emb    rvec_emb with the per-KC codes q_in, q_out trained (a second trainable KC table, 2 x d
                per KC, read through the trained maps of Eq. (3)); initial values those of rvec_emb
  State-routing channel inside the probe (entry point S):
    st_empty, st_uniform, st_er, st_dp, st_support
                the probe with no graph at (E) or (B) and one more input to the output MLP,
                    v_t = sum_{j<=t} B(c_j, c_{t+1}) W_n x_j / max(1, sum_{j<=t} B(c_j, c_{t+1})),
                with B the symmetric 0/1 version of the graph (A + A^T > 0, zero diagonal): the learner's
                past interactions on graph neighbours of the target, routed to the prediction for the
                target. With the empty graph v_t = 0, so st_empty is the control of this channel.
  SAKT-E (SAKT with the refined KC table of Eq. (3), as in r8_controls.py), on a CPU:
    sakte_empty, sakte_uniform, sakte_er, sakte_support, sakte_rvec, sakte_rvecF
                sakte_rvec replaces A_in E and A_out E by fixed random codes drawn without a graph, with
                the row variance of Erdos-Renyi mixing (seed 5000 + s), as rvec_emb in the probe;
                sakte_rvecF freezes W_in, W_out and their biases at their initial values.
  GKT runs use r9_controls.py with --out runs_r13/gkt.

--rep j changes the training seed only (seed = sub + 100 j). --lr x is added to the tag when x != 0.002.
--dump writes, for at most 400 test learners and positions t in {10, 25, 50, 100, 150}, the predicted
probability of a correct first attempt at every KC the learner has not yet attempted (decision analysis,
Section VI-G of the TLT manuscript), to <tag>_cand.npz (probe conditions only).

Output: runs_r13/<tag>.json and <tag>_pred.npz (p, y, new, kc, pos, nb_sup, nb_er, uid = index of the test
learner).
"""
import sys, os, json, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_controls import FKGKT3, load, transition_support, rewire, ece
from r6_controls import erdos_renyi
from r8_controls import SAKTE
from r9_controls import neighbour_sets
from r10_controls import FKGKT3Fixed
from train import seqs, batches
torch.set_num_threads(int(os.environ.get('NT', '1')))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
GR = f'{ROOT}/graphs_r5'; os.makedirs(GR, exist_ok=True)
DUMP_T = (10, 25, 50, 100, 150); DUMP_N = 400


class FKGKT3Trainable(FKGKT3Fixed):
    """rvec_emb with the codes q_in, q_out turned into trainable parameters (same initial values)."""
    def __init__(s, K, A, seed, d=64, drop=0.2):
        super().__init__(K, A, 'rvec', seed, d=d, drop=drop)
        Qin, Qout = s.Qin.clone(), s.Qout.clone(); del s.Qin; del s.Qout
        s.Qin = nn.Parameter(Qin); s.Qout = nn.Parameter(Qout)


class FKGKT3State(FKGKT3):
    """Probe with a state-routing channel: past interactions on graph neighbours of the target."""
    def __init__(s, K, A, d=64, drop=0.2):
        super().__init__(K, np.zeros((K, K), np.float32), recency=False, d=d, drop=drop)
        B = (np.asarray(A) + np.asarray(A).T) > 0; np.fill_diagonal(B, False)
        Bt = torch.zeros(K + 1, K + 1); Bt[:K, :K] = torch.tensor(B, dtype=torch.float32)
        s.register_buffer('Bn', Bt)
        s.Wn = nn.Linear(d, d)
        s.out = nn.Sequential(nn.Linear(5 * d, d), nn.ReLU(), nn.Dropout(drop), nn.Linear(d, 1))
    def forward(s, q, r, qn):
        b = s.base; Z = b.kc_repr(); qc = q.clamp(max=s.K); qnc = qn.clamp(max=s.K)
        x = b.dp(Z[qc] + b.R(r)); h, _ = b.rnn(x); zn = Z[qnc]
        T = q.size(1); logit = (s.Wq(zn) @ s.Wk(x).transpose(1, 2)) / s.d ** 0.5
        same = (qnc.unsqueeze(2) == qc.unsqueeze(1)).float()
        logit = logit + s.bsame * same
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1) | (q >= s.K).unsqueeze(1)
        logit = logit.masked_fill(mask, -1e9); a = torch.softmax(logit, -1)
        c = s.dp(a @ s.Wv(x))
        M = s.Bn[qc.unsqueeze(1).expand(-1, T, -1), qnc.unsqueeze(2).expand(-1, -1, T)].masked_fill(mask, 0.0)
        v = (M @ s.Wn(x)) / M.sum(-1, keepdim=True).clamp(min=1.0)
        return s.out(torch.cat([h, c, zn, h * zn, v], 2)).squeeze(2)


class SAKTEFixed(SAKTE):
    """SAKT-E whose refinement reads fixed random codes drawn without a graph (rvec_emb for SAKT-E)."""
    def __init__(s, K, A, seed, frozen=False, d=64):
        super().__init__(K, A, d=d)
        g = torch.Generator().manual_seed(seed)
        Qin = torch.randn(K, d, generator=g); Qout = torch.randn(K, d, generator=g)
        sin = (s.Ain ** 2).sum(1, keepdim=True).sqrt(); sout = (s.Aout ** 2).sum(1, keepdim=True).sqrt()
        s.register_buffer('Qin', Qin * sin); s.register_buffer('Qout', Qout * sout)
        if frozen:
            for p in list(s.Win.parameters()) + list(s.Wout.parameters()): p.requires_grad_(False)
    def kc_repr(s):
        Z = s.E.weight[:s.K]
        Z = s.lnE(Z + torch.tanh(s.Win(s.Qin)) + torch.tanh(s.Wout(s.Qout)))
        return torch.cat([Z, torch.zeros(1, Z.size(1), device=Z.device)], 0)


def evaluate(m, q, r, K, bs, nbs):
    """As r9_controls.evaluate, plus the index of the learner of every prediction."""
    m.eval(); P = []; Y = []; NEW = []; KC = []; POS = []; UID = []; NB = {k: [] for k in nbs}
    with torch.no_grad():
        for j, a, b, c, y in batches(q, r, bs, False, None):
            p = torch.sigmoid(m(a, b, c)); mask = (c < K).numpy()
            P.append(p.numpy()[mask]); Y.append(y.numpy()[mask]); KC.append(c.numpy()[mask])
            UID.append(np.repeat(j, mask.sum(1)))
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
               pos=np.concatenate(POS), uid=np.concatenate(UID))
    for k in nbs: out['nb_' + k] = np.concatenate(NB[k])
    return out


def candidate_scores(m, q, r, K):
    """P(correct) at every KC for the first DUMP_N test learners at positions DUMP_T (FKGKT3 models).

    Position t means: after the interactions 0..t of the learner (t + 1 of them), predicting interaction t + 1,
    exactly as the forward pass of the model for step t. Returns scores [n, len(DUMP_T), K] (float16; NaN where
    the learner's sequence is shorter) and a mask of the KCs already attempted by then."""
    m.eval(); n = min(DUMP_N, len(q)); L = len(DUMP_T)
    S = np.full((n, L, K), np.nan, np.float32); SEEN = np.zeros((n, L, K), bool)
    b = m.base
    with torch.no_grad():
        Z = b.kc_repr(); zc = Z[:K]; wq = m.Wq(zc)                   # [K, d]
        for i0 in range(0, n, 16):
            qa = q[i0:min(n, i0 + 16)]; ra = r[i0:min(n, i0 + 16)]
            qc = qa.clamp(max=K); x = Z[qc] + b.R(ra); h, _ = b.rnn(x)
            wk = m.Wk(x); wv = m.Wv(x)                                   # [B, T, d]
            for li, t in enumerate(DUMP_T):
                valid = (qa[:, t + 1] < K) if t + 1 < qa.shape[1] else torch.zeros(len(qa), dtype=torch.bool)
                if not valid.any(): continue
                lg = torch.einsum('kd,btd->bkt', wq, wk[:, :t + 1]) / m.d ** 0.5     # [B, K, t+1]
                past = qc[:, :t + 1]
                same = (past.unsqueeze(1) == torch.arange(K).view(1, K, 1)).float()
                pre = m.S[past.unsqueeze(1).expand(-1, K, -1), torch.arange(K).view(1, K, 1).expand(len(qa), -1, t + 1)]
                suc = m.S[torch.arange(K).view(1, K, 1).expand(len(qa), -1, t + 1), past.unsqueeze(1).expand(-1, K, -1)]
                lg = lg + m.bpre * pre + m.bsuc * suc + m.bsame * same
                lg = lg.masked_fill((past >= K).unsqueeze(1), -1e9)
                u = torch.softmax(lg, -1) @ wv[:, :t + 1]                            # [B, K, d]
                ht = h[:, t].unsqueeze(1).expand(-1, K, -1); zk = zc.unsqueeze(0).expand(len(qa), -1, -1)
                p = torch.sigmoid(m.out(torch.cat([ht, u, zk, ht * zk], 2)).squeeze(2)).numpy()
                for bi in range(len(qa)):
                    if not bool(valid[bi]): continue
                    S[i0 + bi, li] = p[bi]
                    seen = np.unique(qa[bi, :t + 1].numpy()); seen = seen[seen < K]; SEEN[i0 + bi, li, seen] = True
    return S.astype(np.float16), SEEN


def build(a, K, graph, sup, er):
    c = a.cond
    if c in ('empty', 'uniform_emb', 'er_emb', 'support_emb', 'dp_emb'):
        A = graph[c.replace('_emb', '')]; m = FKGKT3(K, A, recency=False)
        with torch.no_grad():
            if c != 'empty': m.S.zero_()
        return m, A, 'probe'
    if c == 'support':
        return FKGKT3(K, sup, recency=False), sup, 'probe'
    if c == 'empty_d128':
        return FKGKT3(K, graph['empty'], recency=False, d=128), graph['empty'], 'probe'
    if c in ('rvec_emb', 'rvecF_emb'):
        m = FKGKT3Fixed(K, er, 'rvec', seed=5000 + a.sub)
        with torch.no_grad(): m.S.zero_()
        if c == 'rvecF_emb':
            for lin in list(m.base.Win) + list(m.base.Wout):
                for p in lin.parameters(): p.requires_grad_(False)
        return m, er, 'probe'
    if c == 'tvec_emb':
        m = FKGKT3Trainable(K, er, seed=5000 + a.sub)
        with torch.no_grad(): m.S.zero_()
        return m, er, 'probe'
    if c.startswith('st_'):
        A = graph[c[3:]]; return FKGKT3State(K, A), A, 'probe_state'
    if c in ('sakte_rvec', 'sakte_rvecF'):
        return SAKTEFixed(K, er, seed=5000 + a.sub, frozen=c.endswith('F')), er, 'sakte'
    if c.startswith('sakte_'):
        A = graph[c[6:]]; return SAKTE(K, A), A, 'sakte'
    raise ValueError(c)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--frac', type=float, default=0.1); ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--cond'); ap.add_argument('--rep', type=int, default=0); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--max_steps', type=int, default=2000); ap.add_argument('--eval_every', type=int, default=25); ap.add_argument('--patience_evals', type=int, default=8)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--out', default=f'{ROOT}/runs_r13'); ap.add_argument('--dump', action='store_true')
    a = ap.parse_args(argv); seed = a.sub + 100 * a.rep; os.makedirs(a.out, exist_ok=True)
    tag = (f"{a.data}__f{int(round(a.frac*100))}__s{a.sub}__{a.cond}" + (f"__rep{a.rep}" if a.rep else '')
           + ('' if abs(a.lr - 2e-3) < 1e-12 else f"__lr{a.lr:g}") + "__steps")
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
    m, A, family = build(a, K, graph, sup, er)
    bs = a.bs
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
               K=K, device='cpu', torch=torch.__version__, batch_size=bs, lr=a.lr, d=int(getattr(m, 'd', 64)),
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
    if a.dump and isinstance(m, FKGKT3) and not isinstance(m, FKGKT3State):
        Sc, SEEN = candidate_scores(m, sq, sr, K)
        np.savez_compressed(f'{a.out}/{tag}_cand.npz', score=Sc, seen=SEEN, t=np.array(DUMP_T))
        res['dump'] = True
    res['secs_total'] = time.time() - t0
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    print(tag, 'auc', round(res['test_auc'], 4), 'first', round(res['test_auc_first'] or 0, 4), 'steps', steps, 'best', best_at,
          's', round(res['secs']), flush=True)


if __name__ == '__main__': main()
