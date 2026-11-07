"""Release 3.1.0 analysis -> results_r6/r6_results.json.

1. Update steps actually taken by the learning-curve runs (kt/r2, 957 runs) and the runs of
   release 2.1.0 (runs_r5): steps = epochs run * ceil(training learners / batch size).
2. Pooled support-versus-control comparison over the two sets of runs. Both use the same model
   (r2/src/models.FKGKT3 and src/models.FKGKT2 compute the same function; they create their
   parameters in a different order, so the same seed gives a different initialisation) and the
   same order-preserving graph (edge counts agree; one edge differs on Statics 2011 at 10%).
   Subsample 0 is the same sample in both; subsamples s >= 1 differ (seed 12345 + 1000 s in kt/r2,
   12345 + s in runs_r5). Pooled: for every corpus, the mean over distinct subsamples, the two runs
   on subsample 0 averaged.
3. Prespecified comparisons of the controls table with 95% intervals: corpus-then-subsample
   bootstrap of the mean of corpus means, and a mixed model (random intercept per corpus) on
   subsample-level differences.
4. Runs of release 3.1.0 (runs_r6): training budget by steps; Erdos-Renyi and degree-feature
   controls.
"""
import json, glob, os, re, collections, numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt'); OUT = f'{ROOT}/results_r6'; os.makedirs(OUT, exist_ok=True)
CORP = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
BS = 128
rng0 = np.random.default_rng(20261107)

def wil(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.all(x == 0): return 1.0
    return float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj

def boot_ci(per_corpus_lists, B=10000, rng=rng0):
    """per_corpus_lists: list over corpora of arrays of subsample-level differences.
    Resample corpora with replacement, then subsamples within each drawn corpus."""
    L = [np.asarray(v, float) for v in per_corpus_lists if len(v)]
    n = len(L); est = float(np.mean([v.mean() for v in L])); bs = np.empty(B)
    for b in range(B):
        idx = rng.integers(n, size=n)
        bs[b] = np.mean([L[i][rng.integers(len(L[i]), size=len(L[i]))].mean() for i in idx])
    return est, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

def mixed_ci(rows):
    """rows: list of (corpus, diff). Intercept-only mixed model, random intercept per corpus."""
    import statsmodels.formula.api as smf
    d = pd.DataFrame(rows, columns=['corpus', 'g'])
    try:
        f = smf.mixedlm('g ~ 1', d, groups=d['corpus']).fit(reml=True)
        b = float(f.params['Intercept']); se = float(f.bse['Intercept'])
        return dict(est=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=float(f.pvalues['Intercept']), n=len(d))
    except Exception as e:
        return dict(error=repr(e))

res = {}
# ------------------------------------------------------------------ run sets
R5 = [json.load(open(f)) for f in glob.glob(f'{ROOT}/runs_r5/*.json')]
R5m = [r for r in R5 if r['d'] == 64 and r['wd'] == 0 and r['seed'] == r['sub'] and r.get('lr_scalar') is None]
ntr = {(r['data'], round(r['frac'], 2)): r['n_train_learners'] for r in R5}
r2 = pd.read_csv(f'{ROOT}/r2/results/r2_runs.csv'); r2 = r2[r2.src == 'R2']
# training learners at every size from the 10% count (subsamples are round(frac * n_train)); 100% and 25% recorded in runs_r5
full_tr = {c: ntr[(c, 1.0)] for c in CORP if (c, 1.0) in ntr}
def n_train(c, fr): return ntr.get((c, round(fr, 2)), max(2, int(round(fr * full_tr[c]))))

# 1. update steps -------------------------------------------------------------------------------
steps = {}
for name, rows in (('learning_curve', [(x.data, x.frac, x.ep) for x in r2[r2.cfg.isin(['P/empty', 'P/support'])].itertuples()]),
                   ('controls', [(r['data'], round(r['frac'], 2), r['epochs_run']) for r in R5m])):
    T = collections.defaultdict(list)
    for c, fr, ep in rows: T[(c, round(fr, 2))].append(ep * int(np.ceil(n_train(c, fr) / BS)))
    steps[name] = {f'{c}|{fr}': dict(per_epoch=int(np.ceil(n_train(c, fr) / BS)), n_train=n_train(c, fr), median=float(np.median(v)), min=int(min(v)), max=int(max(v)), n=len(v))
                   for (c, fr), v in sorted(T.items())}
    for fr in (0.05, 0.1, 0.25, 1.0):
        v = [s for (c, f), L in T.items() if f == fr for s in L]
        if v: steps[name][f'all|{fr}'] = dict(median=float(np.median(v)), min=int(min(v)), max=int(max(v)), share_le_60=float(np.mean(np.array(v) <= 60)), n=len(v))
res['steps'] = steps

# 2. pooled support - empty ---------------------------------------------------------------------
A = collections.defaultdict(dict)   # (corpus, frac, sample_id) -> list of (key -> diff)
def add(c, fr, sid, da, df_):
    A[(c, fr)].setdefault(sid, []).append((da, df_))
x = r2[r2.cfg.isin(['P/empty', 'P/support'])]
piv = x.pivot_table(index=['data', 'frac', 'sub', 'seed'], columns='cfg', values=['auc', 'new'])
for (c, fr, s, seed), v in piv.iterrows():
    if seed != s or np.isnan(v[('auc', 'P/support')]) or np.isnan(v[('auc', 'P/empty')]): continue
    add(c, round(fr, 2), ('r2', s) if s else ('shared', 0), v[('auc', 'P/support')] - v[('auc', 'P/empty')], v[('new', 'P/support')] - v[('new', 'P/empty')])
T5 = collections.defaultdict(dict)
for r in R5m: T5[(r['data'], round(r['frac'], 2), r['sub'])][r['cond']] = r
for (c, fr, s), v in T5.items():
    if 'support' in v and 'empty' in v:
        add(c, fr, ('r5', s) if s else ('shared', 0), v['support']['test_auc'] - v['empty']['test_auc'], v['support']['test_auc_first'] - v['empty']['test_auc_first'])
pooled = []
for fr in (0.05, 0.1, 0.25, 1.0):
    for k, key in enumerate(('all', 'first')):
        per = {}
        for c in CORP:
            S = A.get((c, fr), {})
            if not S: continue
            per[c] = [float(np.mean([t[k] for t in L])) for sid, L in S.items()]
        v = np.array([np.mean(per[c]) for c in per])
        est, lo, hi = boot_ci(list(per.values()))
        mm = mixed_ci([(c, g) for c in per for g in per[c]])
        pooled.append(dict(frac=fr, key=key, mean=float(v.mean()), pos=int((v > 0).sum()), n=len(v), p=wil(v), boot=[lo, hi], mixed=mm,
                           n_samples=int(np.median([len(per[c]) for c in per])), per_corpus={c: float(np.mean(per[c])) for c in per}))
for r, a in zip(pooled, holm([r['p'] for r in pooled])): r['holm'] = float(a)
res['pooled'] = pooled
# pooled mixed model of the learning curve (quadratic in log10 interactions per KC)
try:
    import statsmodels.formula.api as smf
    from scipy.stats import chi2
    ipk = {}
    for r in R5m: ipk[(r['data'], round(r['frac'], 2))] = r['n_train_inter'] / r['K']
    recs = []
    for (c, fr), S in A.items():
        if (c, fr) not in ipk: continue
        for sid, L in S.items():
            for k, key in enumerate(('all', 'first')):
                recs.append(dict(corpus=c, frac=fr, key=key, x=float(np.log10(ipk[(c, fr)])), g=float(np.mean([t[k] for t in L]))))
    D = pd.DataFrame(recs); curve = {}
    for key in ('all', 'first'):
        d = D[D.key == key].copy(); d['x2'] = d.x ** 2
        def fit(f, e):
            for meth in (['bfgs'], ['lbfgs'], ['powell'], ['nm']):
                try: return smf.mixedlm(f, e, groups=e['corpus']).fit(reml=False, method=meth)
                except Exception: continue
            raise RuntimeError('mixedlm failed')
        m1 = fit('g ~ x + x2', d); m0 = fit('g ~ x', d)
        b1, b2 = m1.params['x'], m1.params['x2']
        loco = {}
        for c in d.corpus.unique():
            e = d[d.corpus != c]
            try: n1 = fit('g ~ x + x2', e); n0 = fit('g ~ x', e)
            except Exception: continue
            loco[c] = dict(b2=float(n1.params['x2']), p=float(chi2.sf(2 * (n1.llf - n0.llf), 1)))
        curve[key] = dict(b2=float(b2), p=float(chi2.sf(2 * (m1.llf - m0.llf), 1)), peak=float(10 ** (-b1 / (2 * b2))) if b2 < 0 else None, n=len(d),
                          loco_neg=sum(v['b2'] < 0 for v in loco.values()), loco_p=[min(v['p'] for v in loco.values()), max(v['p'] for v in loco.values())])
    res['pooled_curve'] = curve
except Exception as e:
    res['pooled_curve_error'] = repr(e)

# 3. prespecified comparisons of the controls table --------------------------------------------
pre = []
for fr in (0.1, 0.25):
    for cond, ref in (('support', 'recency'), ('support', 'rewired'), ('support', 'empty'), ('rewired', 'empty')):
        per = collections.defaultdict(list)
        for (c, f, s), v in T5.items():
            if f == fr and cond in v and ref in v: per[c].append(v[cond]['test_auc_first'] - v[ref]['test_auc_first'])
        est, lo, hi = boot_ci([per[c] for c in CORP if c in per])
        pre.append(dict(frac=fr, cond=cond, ref=ref, key='first', mean=est, boot=[lo, hi], mixed=mixed_ci([(c, g) for c in per for g in per[c]])))
res['controls_ci'] = pre

# 4. second channel: draws per epoch --------------------------------------------------------------
pc = {}
for c in ('statics11', 'algebra05'):
    r = [r for r in R5m if r['data'] == c and round(r['frac'], 2) == 0.1 and r['cond'] == 'support']
    if r:
        n_pred = float(np.mean([x['n_train_inter'] - x['n_train_learners'] for x in r])); E = float(np.mean([x['edges'] for x in r]))
        pc[c] = dict(train_predictions=n_pred, edges=E, draws_per_edge_per_epoch=n_pred / E)
res['pc_draws'] = pc

# 5. runs of release 3.1.0 -----------------------------------------------------------------------
R6 = [json.load(open(f)) for f in glob.glob(f'{ROOT}/runs_r6/*.json')]
T6 = collections.defaultdict(dict)
for r in R6: T6[(r['data'], round(r['frac'], 2), r['sub'], r['budget'])][r['cond']] = r
def tab(budget, fr, cond, ref, key, src=None):
    per = collections.defaultdict(list)
    for (c, f, s, b), v in T6.items():
        if b != budget or f != fr: continue
        vv = dict(v)
        if src is not None and (c, f, s) in T5:   # reference from runs_r5 (epoch budget, same code, same subsample)
            for k2 in ('empty', 'support', 'rewired'):
                if k2 in T5[(c, f, s)]: vv.setdefault(k2, T5[(c, f, s)][k2])
        if cond in vv and ref in vv and vv[cond][key] is not None and vv[ref][key] is not None: per[c].append(vv[cond][key] - vv[ref][key])
    if len(per) < 2: return None
    v = np.array([np.mean(per[c]) for c in per]); est, lo, hi = boot_ci(list(per.values()))
    return dict(budget=budget, frac=fr, cond=cond, ref=ref, key=key, mean=float(v.mean()), pos=int((v > 0).sum()), n=len(v), p=wil(v), boot=[lo, hi],
                nsub=int(np.median([len(x) for x in per.values()])), per_corpus={c: float(np.mean(per[c])) for c in per})
steps_tab = []
for fr in (0.05, 0.1):
    for cond, ref in (('support', 'empty'), ('rewired', 'empty'), ('er', 'empty'), ('support', 'rewired'), ('support', 'er')):
        for key in ('test_auc', 'test_auc_first'):
            t = tab('steps', fr, cond, ref, key)
            if t: steps_tab.append(t)
for r, a in zip(steps_tab, holm([r['p'] for r in steps_tab])): r['holm'] = float(a)
res['steps_table'] = steps_tab
# step budget against the epoch budget, same subsample, same condition
gain_budget = collections.defaultdict(list); steps_taken = collections.defaultdict(list)
for (c, f, s, b), v in T6.items():
    if b != 'steps': continue
    for cond in ('empty', 'support'):
        if cond in v and (c, f, s) in T5 and cond in T5[(c, f, s)]:
            for key in ('test_auc', 'test_auc_first'):
                gain_budget[(f, cond, key, c)].append(v[cond][key] - T5[(c, f, s)][cond][key])
    for cond, r in v.items(): steps_taken[(c, f)].append((r['best_step'], r['steps'], r['hit_max']))
res['budget_effect'] = {}
for f in (0.05, 0.1):
    for cond in ('empty', 'support'):
        for key in ('test_auc', 'test_auc_first'):
            pc_ = {c: float(np.mean(v)) for (f2, c2, k2, c), v in gain_budget.items() if (f2, c2, k2) == (f, cond, key)}
            if pc_:
                res['budget_effect'][f'{f}|{cond}' + ('' if key == 'test_auc' else '|first')] = dict(mean=float(np.mean(list(pc_.values()))), per_corpus=pc_, pos=int(sum(x > 0 for x in pc_.values())), n=len(pc_), p=wil(list(pc_.values())))
res['steps_taken'] = {f'{c}|{f}': dict(best_step_median=float(np.median([x[0] for x in v])), best_step_max=int(max(x[0] for x in v)), steps_median=float(np.median([x[1] for x in v])),
                                       hit_max=float(np.mean([x[2] for x in v])), n=len(v)) for (c, f), v in sorted(steps_taken.items())}
e3 = []
for fr in (0.1, 0.25):
    for cond, ref in (('support', 'empty'), ('rewired', 'empty'), ('er', 'empty'), ('degfeat', 'empty'), ('support', 'er'), ('rewired', 'er'), ('support', 'degfeat')):
        for key in ('test_auc', 'test_auc_first'):
            t = tab('epochs', fr, cond, ref, key, src='r5')
            if t: e3.append(t)
for r, a in zip(e3, holm([r['p'] for r in e3])): r['holm'] = float(a)
res['e3_table'] = e3
res['steps_subs'] = {f'{c}|{f}': len({k[2] for k in T6 if k[0] == c and k[1] == f and k[3] == 'steps' and 'support' in T6[k] and 'empty' in T6[k]}) for c in CORP for f in (0.05, 0.1)}
res['n_runs_r6'] = len(R6); res['n_runs_r6_by'] = dict(collections.Counter(f"{r['budget']}|{r['frac']}|{r['cond']}" for r in R6))
# run counts by set (for the Supplementary)
cnt = collections.Counter()
for x in r2.itertuples(): cnt[('kt/r2 (release 3.0.0)', x.frac, x.cfg)] += 1
for r in R5: cnt[('runs_r5 (release 2.1.0)', round(r['frac'], 2), r['cond'] + ('' if r['seed'] == r['sub'] and r.get('lr_scalar') is None else ' (extra seed or scalar lr)'))] += 1
for r in R6: cnt[('runs_r6 (release 3.1.0)', round(r['frac'], 2), r['cond'] + '/' + r['budget'])] += 1
res['run_counts'] = [dict(set=a, frac=b, cond=c, n=n) for (a, b, c), n in sorted(cnt.items(), key=lambda t: (t[0][0], t[0][1], t[0][2]))]
json.dump(res, open(f'{OUT}/r6_results.json', 'w'), indent=1)
print('pooled'); [print(f"  {r['frac']} {r['key']:5s} {r['mean']:+.4f} {r['pos']}/{r['n']} p={r['p']:.3f} holm={r['holm']:.3f} boot=[{r['boot'][0]:+.4f},{r['boot'][1]:+.4f}] mixed={r['mixed'].get('est',0):+.4f} [{r['mixed'].get('lo',0):+.4f},{r['mixed'].get('hi',0):+.4f}] nsub={r['n_samples']}") for r in pooled]
print('curve', json.dumps(res.get('pooled_curve', res.get('pooled_curve_error'))))
print('controls CI'); [print(f"  {r['frac']} {r['cond']}-{r['ref']} {r['mean']:+.4f} boot=[{r['boot'][0]:+.4f},{r['boot'][1]:+.4f}] mixed=[{r['mixed'].get('lo',0):+.4f},{r['mixed'].get('hi',0):+.4f}]") for r in pre]
print('steps (all)', {k: v for k, v in steps['learning_curve'].items() if k.startswith('all')})
print('pc', pc); print('r6 runs', len(R6))
for t in steps_tab + e3: print(f"  {t['budget']} {t['frac']} {t['cond']}-{t['ref']} {t['key']} {t['mean']:+.4f} {t['pos']}/{t['n']} p={t['p']:.3f} holm={t['holm']:.3f} [{t['boot'][0]:+.4f},{t['boot'][1]:+.4f}]")
print('budget', json.dumps(res['budget_effect']))
