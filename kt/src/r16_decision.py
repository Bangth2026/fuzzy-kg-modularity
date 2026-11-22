"""Release 3.13.0: a decision-based reference point for the smallest effect of interest (no new training runs).

How large must a difference in first-attempt AUC be before it changes the quality of next-KC choices by more
than retraining with another seed does? Decision models: every probe trained on 25% of the training learners
under the update budget whose candidate file exists (runs_r13, runs_r14: subsamples 0 to 2; empty-graph control,
support graph at both entry points, support graph in the embedding refinement, random code; on subsample 0 also
seeds 100 and 200 of the control and of the support graph), 16 models per corpus. Every model scores the same
test learners at the same positions, so the candidate sets are shared. Simulated learner: the Rasch model of
r15_regret.py (fitted to every observed answer of the test learners; it shares no architecture or weights with
the decision models). For each model: first-attempt test AUC (its run file) and the mean regret of the target
rule (closest to 0.65) and of the easiest-KC rule.

Fit: regret ~ AUC_first + (1 | corpus), so the slope is estimated from differences between models of one corpus.
Reference point: delta_dec = s_seed / |slope|, with s_seed the standard deviation of the signed regret difference
between two seeds of the control (r15_regret.py: 0.0100 for the target rule, 0.0429 for the easiest rule): the
AUC difference whose expected effect on regret equals the retraining noise of one decision model. Also reported:
the expected change of regret for 0.001, 0.002 and 0.003 AUC, as a share of s_seed.
Writes results_r16/r16_decision.json.
"""
import os, glob, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
import importlib.util
OUT = f'{REPO}/results_r16'; os.makedirs(OUT, exist_ok=True)
CORPORA = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
TARGET = 0.65
R13 = f'{REPO}/runs_r13'; R14 = f'{REPO}/runs_r14'


def rasch(c, n, K):
    from scipy.optimize import minimize
    P = np.load(f'{R13}/{c}__f100__s0__empty__steps_pred.npz')
    u = P['uid'].astype(int); k = P['kc'].astype(int); y = P['y'].astype(float); N = int(u.max()) + 1; lam = 1.0
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
    files = sorted(set(glob.glob(f'{R13}/{c}__f25__*_cand.npz')) | set(glob.glob(f'{R14}/{c}__f25__*_cand.npz')))
    seen = None; mods = []
    for f in files:
        tag = os.path.basename(f)[:-len('_cand.npz')]
        js = f[:-len('_cand.npz')] + '.json'
        if not os.path.exists(js): continue
        j = json.load(open(js)); z = np.load(f)
        if seen is None: seen = z['seen']
        assert (z['seen'] == seen).all()
        mods.append((tag, j, z['score']))
    n, _, K = seen.shape
    Kr = max(K, int(np.load(f'{R13}/{c}__f100__s0__empty__steps_pred.npz')['kc'].max()) + 1)
    S = rasch(c, n, Kr)[..., :K]
    for tag, j, sc in mods:
        t_, e_ = regrets(sc[..., :K], S, seen)
        rows.append(dict(data=c, tag=tag, cond=j['cond'], sub=int(j['sub']), rep=int(j.get('rep', 0) or 0),
                         auc_first=float(j['test_auc_first']), auc=float(j['test_auc']), target=t_, easiest=e_))
    print(c, len(mods), flush=True)
df = pd.DataFrame(rows)

import statsmodels.formula.api as smf
S_SEED = {'target': 0.0100, 'easiest': 0.0429}   # r15_regret.json, Rasch, control seed pairs (Supplementary Table S10.4)
out = dict(models_per_corpus={c: int((df.data == c).sum()) for c in CORPORA}, s_seed=S_SEED, fits={})
for rule in ('target', 'easiest'):
    for x in ('auc_first', 'auc'):
        d = df.rename(columns={rule: 'y', x: 'x'}).copy()
        d['xc'] = d.x - d.groupby('data').x.transform('mean'); d['yc'] = d.y - d.groupby('data').y.transform('mean')
        r = smf.mixedlm('y ~ xc', d, groups=d['data']).fit(reml=True)
        b = float(r.params['xc']); lo, hi = [float(v) for v in r.conf_int().loc['xc']]
        rho = float(np.corrcoef(d.xc, d.yc)[0, 1])
        o = dict(slope=b, ci95=[lo, hi], p=float(r.pvalues['xc']), within_corpus_r=rho,
                 sd_auc_within=float(d.xc.std(ddof=1)), sd_regret_within=float(d.yc.std(ddof=1)))
        o['delta_dec'] = float(S_SEED[rule] / abs(b)) if b != 0 else None
        o['delta_dec_ci'] = ([float(S_SEED[rule] / abs(v)) for v in (hi, lo)] if lo < 0 and hi < 0 else
                             ([float(S_SEED[rule] / abs(lo)), None] if lo < 0 else None))
        o['regret_change_per_auc'] = {str(dl): dict(change=float(b * dl), share_of_seed_sd=float(abs(b) * dl / S_SEED[rule]))
                                      for dl in (0.001, 0.002, 0.003, 0.005)}
        out['fits'][f'{rule}~{x}'] = o
out['per_model'] = df.round(5).to_dict(orient='records')
json.dump(out, open(f'{OUT}/r16_decision.json', 'w'), indent=1)
for k, o in out['fits'].items():
    print(f"{k:20s} slope {o['slope']:+.3f} [{o['ci95'][0]:+.3f},{o['ci95'][1]:+.3f}] p={o['p']:.3g} r={o['within_corpus_r']:+.2f} "
          f"sdAUC={o['sd_auc_within']:.4f} delta_dec={o['delta_dec']} ci={o['delta_dec_ci']} "
          f"per0.002={o['regret_change_per_auc']['0.002']['change']:+.5f} ({o['regret_change_per_auc']['0.002']['share_of_seed_sd']:.2f} sd)")
