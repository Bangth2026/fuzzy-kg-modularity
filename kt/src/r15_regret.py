"""Release 3.12.0: regret of next-KC choices under simulated learners of two families (no new training runs).

Decision models (25% of the training learners, subsample 0, runs_r13): the support graph (seed 0) and the empty-graph
control (seeds 0 and 100). Simulated learners: (a) the empty-graph probe, (b) the support-graph probe and (c) the
random-code probe trained on all training learners (seed 0, runs_r13), and (d) a Rasch model, P(correct) =
sigmoid(theta_learner - b_KC), fitted by penalised maximum likelihood to every observed answer of the test learners
(the prediction files of the 100% empty-graph run hold them); (d) shares no architecture or training data with
either decision model. Rules: the unattempted KC whose predicted P is closest to 0.65, and the easiest one.
Regret: |P_sim(chosen) - 0.65| - min over unattempted KCs |P_sim - 0.65| (target rule); max P_sim - P_sim(chosen)
(easiest rule). Writes results_r15/r15_regret.json.
"""
import os, json, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); R = f'{REPO}/runs_r13'
OUT = f'{REPO}/results_r15'; os.makedirs(OUT, exist_ok=True)
CORPORA = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
TARGET = 0.65


def cand(c, frac, cond, rep=0):
    return np.load(f'{R}/{c}__f{frac}__s0__{cond}{"__rep%d" % rep if rep else ""}__steps_cand.npz')


def rasch(c, n, K):
    P = np.load(f'{R}/{c}__f100__s0__empty__steps_pred.npz')
    u = P['uid'].astype(int); k = P['kc'].astype(int); y = P['y'].astype(float); N = int(u.max()) + 1
    lam = 1.0
    def f(w):
        th, b = w[:N], w[N:]; z = th[u] - b[k]; p = 1 / (1 + np.exp(-z))
        ll = np.sum(y * z - np.logaddexp(0, z)); g = y - p
        gth = np.bincount(u, g, N) - lam * th; gb = -np.bincount(k, g, K) - lam * b
        return -(ll - 0.5 * lam * (th @ th + b @ b)), -np.concatenate([gth, gb])
    w = minimize(f, np.zeros(N + K), jac=True, method='L-BFGS-B').x
    th, b = w[:N], w[N:]
    S = 1 / (1 + np.exp(-(th[:n, None] - b[None, :])))
    return np.repeat(S[:, None, :], 5, axis=1)


def regrets(model, sim, seen):
    ok = ~np.isnan(model[..., 0].astype(np.float32)); tr = []; er = []
    for i, t in zip(*np.nonzero(ok)):
        cnd = ~seen[i, t]
        if cnd.sum() < 2: continue
        a = model[i, t, cnd].astype(np.float32); s = sim[i, t, cnd].astype(np.float64)
        dist = np.abs(s - TARGET); tr.append(dist[np.argmin(np.abs(a - TARGET))] - dist.min())
        er.append(s.max() - s[np.argmax(a)])
    return float(np.mean(tr)), float(np.mean(er))


rows = []
for c in CORPORA:
    M = {'G': cand(c, 25, 'support'), 'C0': cand(c, 25, 'empty'), 'C1': cand(c, 25, 'empty', 1)}
    seen = M['G']['seen']
    for v in M.values(): assert (v['seen'] == seen).all()
    n, _, K = seen.shape
    sims = {}
    for lab, cond in (('probe_empty_100', 'empty'), ('probe_support_100', 'support'), ('probe_rvec_100', 'rvec_emb')):
        S = cand(c, 100, cond); Ss = S['score'][..., :K]
        # Junyi: the KC table of the 25% models has one KC fewer (the last index, never seen in their training learners);
        # the shared indices refer to the same KCs, and the extra KC is left out of the candidate set.
        assert (S['seen'][..., :K] == seen).all(); sims[lab] = Ss
    sims['rasch'] = rasch(c, n, max(K, int(np.load(f'{R}/{c}__f100__s0__empty__steps_pred.npz')['kc'].max()) + 1))[..., :K]
    rng = np.random.default_rng(0); rnd = rng.random(seen.shape)
    for sl, S in sims.items():
        for m, A in list(M.items()) + [('random', None)]:
            sc = rnd if A is None else A['score']
            if A is None: sc = np.where(np.isnan(M['G']['score'][..., :1].astype(np.float32)), np.nan, sc)
            t_, e_ = regrets(sc, S, seen)
            rows.append(dict(data=c, sim=sl, model=m, target=t_, easiest=e_))
df = pd.DataFrame(rows)


def mixed_graph(d):
    """regret ~ graph + (1 | corpus) on the three decision models per corpus; the seed variance of the control enters
    the residual, so the interval of the graph term is set against retraining noise."""
    import statsmodels.formula.api as smf
    d = d[d.model.isin(['G', 'C0', 'C1'])].copy(); d['graph'] = (d.model == 'G').astype(float)
    r = smf.mixedlm('y ~ graph', d, groups=d['data']).fit(reml=True)
    lo, hi = r.conf_int().loc['graph']
    return dict(est=float(r.params['graph']), lo=float(lo), hi=float(hi), p=float(r.pvalues['graph']),
                resid_sd=float(np.sqrt(r.scale)))


out = {}
for sl in df.sim.unique():
    o = {}
    for rule in ('target', 'easiest'):
        w = df[df.sim == sl].pivot(index='data', columns='model', values=rule)
        crossed = w.G - w.C1; shared = w.G - w.C0; seed = w.C1 - w.C0; vs_mean = w.G - (w.C0 + w.C1) / 2
        d = df[df.sim == sl].rename(columns={rule: 'y'})
        o[rule] = dict(
            mean_regret={m: float(w[m].mean()) for m in w.columns},
            graph_minus_control_mean_of_two_seeds=dict(mean=float(vs_mean.mean()), lower=int((vs_mean < 0).sum()),
                                                       p_wilcoxon=float(wilcoxon(vs_mean).pvalue)),
            graph_minus_control_crossed_seed=dict(mean=float(crossed.mean()), lower=int((crossed < 0).sum())),
            graph_minus_control_shared_init=dict(mean=float(shared.mean()), lower=int((shared < 0).sum())),
            control_seed_difference_signed=dict(mean=float(seed.mean()), sd=float(seed.std(ddof=1)),
                                                q=[float(seed.quantile(.1)), float(seed.quantile(.9))],
                                                mean_abs=float(seed.abs().mean())),
            mixed_seed_as_residual=mixed_graph(d),
            per_corpus=w.round(5).to_dict(orient='index'))
    out[sl] = o
json.dump(out, open(f'{OUT}/r15_regret.json', 'w'), indent=1)
for sl, o in out.items():
    for rule, v in o.items():
        mr = v['mean_regret']; g = v['graph_minus_control_mean_of_two_seeds']; s = v['control_seed_difference_signed']; mx = v['mixed_seed_as_residual']
        print(f"{sl:18s} {rule:8s} G={mr['G']:.4f} C0={mr['C0']:.4f} C1={mr['C1']:.4f} rnd={mr['random']:.4f} | G-meanC={g['mean']:+.4f} lower {g['lower']}/10 p={g['p_wilcoxon']:.3f} "
              f"| crossed {v['graph_minus_control_crossed_seed']['mean']:+.4f} shared {v['graph_minus_control_shared_init']['mean']:+.4f} "
              f"| seed signed {s['mean']:+.4f} sd {s['sd']:.4f} |abs| {s['mean_abs']:.4f} | mixed {mx['est']:+.4f} [{mx['lo']:+.4f},{mx['hi']:+.4f}] p={mx['p']:.2f}")
