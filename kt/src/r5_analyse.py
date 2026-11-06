"""Tables and statistics of revision R5 from runs_r5/*.json -> results_r5/r5_results.json."""
import json, glob, collections, os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt'); OUT = f'{ROOT}/results_r5'; os.makedirs(OUT, exist_ok=True)
CORP = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
R = [json.load(open(f)) for f in glob.glob(f'{ROOT}/runs_r5/*.json')]
main = [r for r in R if r['d'] == 64 and r['wd'] == 0 and r['seed'] == r['sub'] and r.get('lr_scalar') is None]
LRS = [r for r in R if r.get('lr_scalar') is not None]
T = collections.defaultdict(dict)
for r in main: T[(r['data'], round(r['frac'], 2), r['sub'])][r['cond']] = r

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj

def wil(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.all(x == 0): return 1.0
    return float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)

def gains(frac, cond, ref='empty', key='test_auc'):
    """corpus -> mean over subsamples of AUC(cond) - AUC(ref), on subsamples where both exist."""
    out = {}
    for c in CORP:
        d = [T[(c, frac, s)][cond][key] - T[(c, frac, s)][ref][key] for s in range(5)
             if (c, frac, s) in T and cond in T[(c, frac, s)] and ref in T[(c, frac, s)] and T[(c, frac, s)][cond][key] is not None and T[(c, frac, s)][ref][key] is not None]
        if d: out[c] = (float(np.mean(d)), len(d))
    return out

res = {'n_runs': len(R), 'n_main': len(main)}
# ---- Table 5 (controls): family = all rows of the table at both fractions, both metrics
rows = []
for frac in (0.1, 0.25):
    for cond, ref in [('support', 'empty'), ('recency', 'empty'), ('support', 'recency'), ('rewired', 'empty'), ('support', 'rewired')]:
        for key in ('test_auc', 'test_auc_first'):
            g = gains(frac, cond, ref, key)
            if len(g) < 2: continue
            v = np.array([g[c][0] for c in g])
            rows.append(dict(frac=frac, cond=cond, ref=ref, key=key, mean=float(v.mean()), pos=int((v > 0).sum()), n=len(v),
                             nsub=int(np.median([g[c][1] for c in g])), p=wil(v), per_corpus={c: g[c][0] for c in g}))
adj = holm([r['p'] for r in rows])
for r, a in zip(rows, adj): r['holm'] = float(a)
res['table5'] = rows; res['table5_family_size'] = len(rows)
# descriptive (outside the family): graph added on top of the recency term, 10% only
extra = []
for key in ('test_auc', 'test_auc_first'):
    for cond, ref in [('suprec', 'recency'), ('suprec', 'empty')]:
        g = gains(0.1, cond, ref, key)
        if len(g) >= 2:
            v = np.array([g[c][0] for c in g]); extra.append(dict(cond=cond, ref=ref, key=key, mean=float(v.mean()), pos=int((v > 0).sum()), n=len(v), p=wil(v)))
res['table5_extra'] = extra

# ---- learning curve: support - empty at every fraction (replication of Table 3)
lc = []
for frac in (0.05, 0.1, 0.25, 1.0):
    for key in ('test_auc', 'test_auc_first'):
        g = gains(frac, 'support', 'empty', key)
        if len(g) < 2: continue
        v = np.array([g[c][0] for c in g])
        lc.append(dict(frac=frac, key=key, mean=float(v.mean()), pos=int((v > 0).sum()), n=len(v), p=wil(v), per_corpus={c: g[c][0] for c in g}))
for r, a in zip(lc, holm([r['p'] for r in lc])): r['holm'] = float(a)
res['curve'] = lc

# ---- mixed model on the subsample-level gain, LOCO, and categorical-fraction model
try:
    import statsmodels.formula.api as smf
    from scipy.stats import chi2
    recs = []
    for (c, frac, s), v in T.items():
        if 'support' in v and 'empty' in v:
            ipk = v['empty']['n_train_inter'] / v['empty']['K']
            for key in ('test_auc', 'test_auc_first'):
                if v['support'][key] is None or v['empty'][key] is None: continue
                recs.append(dict(corpus=c, frac=frac, key=key, x=np.log10(ipk), g=v['support'][key] - v['empty'][key]))
    df = pd.DataFrame(recs); mm = {}
    def lr(d):
        d = d.copy(); d['x2'] = d.x ** 2
        m1 = smf.mixedlm('g ~ x + x2', d, groups=d['corpus']).fit(reml=False)
        m0 = smf.mixedlm('g ~ x', d, groups=d['corpus']).fit(reml=False)
        b1, b2 = m1.params['x'], m1.params['x2']
        return dict(b2=float(b2), p=float(chi2.sf(2 * (m1.llf - m0.llf), 1)), peak=float(10 ** (-b1 / (2 * b2))) if b2 < 0 else None, n=len(d))
    for key in ('test_auc', 'test_auc_first'):
        d = df[df.key == key]
        if d.frac.nunique() < 3: continue
        full = lr(d); loco = {c: lr(d[d.corpus != c]) for c in d.corpus.unique()}
        cat = smf.mixedlm('g ~ C(frac)', d, groups=d['corpus']).fit(reml=False)
        catc = {str(k): float(v) for k, v in cat.params.items() if k.startswith('C(') or k == 'Intercept'}
        mm[key] = dict(full=full, loco=loco, loco_b2_range=[min(v['b2'] for v in loco.values()), max(v['b2'] for v in loco.values())],
                       loco_p_max=max(v['p'] for v in loco.values()), loco_neg=sum(v['b2'] < 0 for v in loco.values()),
                       categorical=catc, cat_means={str(f): float(d[d.frac == f].g.mean()) for f in sorted(d.frac.unique())})
    res['mixed'] = mm
except Exception as e:
    res['mixed_error'] = repr(e)

# ---- diagnostics: share of first attempts, prev KC is a prerequisite neighbour, betas, epochs, metrics
diag = {}
for frac in (0.05, 0.1, 0.25, 1.0):
    rr = [r for r in main if round(r['frac'], 2) == frac]
    if not rr: continue
    e = [r for r in rr if r['cond'] == 'empty']; s = [r for r in rr if r['cond'] == 'support']
    fs = {c: float(np.mean([r['n_first'] / r['n_pred'] for r in e if r['data'] == c])) for c in CORP if any(r['data'] == c for r in e)}
    diag[str(frac)] = dict(
        first_share=fs,
        prev_is_prereq={c: float(np.mean([r['first_prev_is_prereq'] for r in s if r['data'] == c])) for c in CORP if any(r['data'] == c for r in s)},
        beta={cond: {b: [float(np.min([r[b] for r in rr if r['cond'] == cond and r[b] is not None])), float(np.median([r[b] for r in rr if r['cond'] == cond and r[b] is not None])), float(np.max([r[b] for r in rr if r['cond'] == cond and r[b] is not None]))]
                     for b in ('beta_pre', 'beta_suc', 'beta_same', 'beta_rec') if any(r['cond'] == cond and r[b] is not None for r in rr)} for cond in ('support', 'recency', 'suprec', 'rewired', 'empty')},
        epochs={'median': float(np.median([r['epochs_run'] for r in rr])), 'hit_max_share': float(np.mean([r['hit_max'] for r in rr])),
                'hit_max_by_corpus': {c: float(np.mean([r['hit_max'] for r in rr if r['data'] == c])) for c in CORP if any(r['data'] == c for r in rr)},
                'early_le10_share': float(np.mean([r['epochs_run'] <= 10 for r in rr]))},
        msg_norm={c: float(np.mean([r['msg_norm'] for r in s if r['data'] == c])) for c in CORP if any(r['data'] == c for r in s)},
        metrics={k: gains(frac, 'support', 'empty', k) for k in ('test_acc', 'test_rmse', 'test_ece')},
        edges={c: float(np.mean([r['edges'] for r in s if r['data'] == c])) for c in CORP if any(r['data'] == c for r in s)},
        n_runs=len(rr))
res['diag'] = diag

# ---- one cell with a larger learning rate for the graph scalars (10%, subsamples 0-2)
if LRS:
    L = collections.defaultdict(dict)
    for r in LRS: L[(r['data'], r['sub'])][r['cond']] = r
    ga, gf, ra, rf, bp = [], [], [], [], []
    for c in CORP:
        da, df_, xa, xf = [], [], [], []
        for s in range(3):
            if (c, s) in L and 'support' in L[(c, s)] and 'empty' in L[(c, s)] and (c, 0.1, s) in T and 'support' in T[(c, 0.1, s)] and 'empty' in T[(c, 0.1, s)]:
                hs, he = L[(c, s)]['support'], L[(c, s)]['empty']; ns, ne = T[(c, 0.1, s)]['support'], T[(c, 0.1, s)]['empty']
                da.append(hs['test_auc'] - he['test_auc']); df_.append(hs['test_auc_first'] - he['test_auc_first'])
                xa.append(ns['test_auc'] - ne['test_auc']); xf.append(ns['test_auc_first'] - ne['test_auc_first']); bp.append(hs['beta_pre'])
        if da: ga.append(np.mean(da)); gf.append(np.mean(df_)); ra.append(np.mean(xa)); rf.append(np.mean(xf))
    if ga:
        res['lrs'] = dict(n_corpora=len(ga), gain_all=float(np.mean(ga)), gain_first=float(np.mean(gf)), ref_all=float(np.mean(ra)), ref_first=float(np.mean(rf)),
                          pos_first=int(sum(np.array(gf) > 0)), bpre=[float(min(bp)), float(max(bp))], p_first_vs_ref=wil(np.array(gf) - np.array(rf)))
# ---- seeds of the empty-graph control on sample 0 at 10% (for Table 4)
ctl = collections.defaultdict(dict)
for r in R:
    if r['cond'] == 'empty' and round(r['frac'], 2) == 0.1 and r['sub'] == 0 and r['d'] == 64: ctl[r['data']][r['seed']] = (r['test_auc'], r['test_auc_first'], r['epochs_run'])
res['control_s0'] = {c: v for c, v in ctl.items()}
json.dump(res, open(f'{OUT}/r5_results.json', 'w'), indent=1)
print('runs', len(R), 'table5 family', len(rows))
for r in rows: print(f"{r['frac']} {r['cond']:8s}-{r['ref']:8s} {r['key']:15s} {r['mean']:+.4f} {r['pos']}/{r['n']} p={r['p']:.3f} holm={r['holm']:.3f}")
for r in lc: print('curve', r['frac'], r['key'], f"{r['mean']:+.4f}", f"{r['pos']}/{r['n']}", round(r['holm'], 3))
print(json.dumps(res.get('mixed', res.get('mixed_error')), indent=0)[:1500])
