"""Release 3.4.0 analysis -> results_r6/r34_results.json.

Reads only archived run files (kt/r2, runs_r5, runs_r6); no new training runs.

1. Robustness of the main comparisons: for every comparison, the mean of corpus means, a 95%
   corpus-then-subsample bootstrap interval, a mixed-model interval (random intercept per corpus),
   and leave-one-corpus-out (LOCO) summaries: range of the mean, smallest number of corpora with a
   positive difference, largest Wilcoxon p over the nine remaining corpora.
2. Decision-level and calibration metrics of the support graph against the empty-graph control:
   accuracy at the 0.5 threshold, RMSE and ECE (15 bins) on all test predictions, by training
   budget and size; number of test predictions and first attempts per corpus.
3. Overlap between subsamples: two subsamples of m learners drawn without replacement and
   independently from N training learners share m^2 / N learners in expectation, a share m / N of
   each subsample.
"""
import json, glob, os, collections, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import wilcoxon
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt'); OUT = f'{ROOT}/results_r6'
CORP = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
rng0 = np.random.default_rng(20261111)


def wil(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.all(x == 0): return 1.0
    return float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)


def boot_ci(L, B=10000, rng=rng0):
    L = [np.asarray(v, float) for v in L if len(v)]
    n = len(L); bs = np.empty(B)
    for b in range(B):
        idx = rng.integers(n, size=n)
        bs[b] = np.mean([L[i][rng.integers(len(L[i]), size=len(L[i]))].mean() for i in idx])
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def mixed_ci(rows):
    import statsmodels.formula.api as smf
    d = pd.DataFrame(rows, columns=['corpus', 'g'])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = smf.mixedlm('g ~ 1', d, groups=d['corpus']).fit(reml=True)
        b = float(f.params['Intercept']); se = float(f.bse['Intercept'])
        return dict(est=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=float(f.pvalues['Intercept']), n=len(d))
    except Exception as e:
        return dict(error=repr(e))


def summary(per):
    cs = [c for c in CORP if c in per]
    m = np.array([np.mean(per[c]) for c in cs])
    lo, hi = boot_ci([per[c] for c in cs])
    loco = []
    for drop in cs:
        mm = np.array([np.mean(per[c]) for c in cs if c != drop])
        loco.append(dict(drop=drop, mean=float(mm.mean()), pos=int((mm > 0).sum()), p=wil(mm)))
    return dict(mean=float(m.mean()), pos=int((m > 0).sum()), n=len(cs), p=wil(m), boot=[lo, hi],
                mixed=mixed_ci([(c, g) for c in cs for g in per[c]]),
                loco_mean=[min(x['mean'] for x in loco), max(x['mean'] for x in loco)],
                loco_pos_min=min(x['pos'] for x in loco), loco_p_max=max(x['p'] for x in loco), loco=loco,
                per_corpus={c: float(np.mean(per[c])) for c in cs})


# ---------------------------------------------------------------- run sets (as in r6/r32_analyse.py)
R5 = [json.load(open(f)) for f in glob.glob(f'{ROOT}/runs_r5/*.json')]
R5m = [r for r in R5 if r['d'] == 64 and r['wd'] == 0 and r['seed'] == r['sub'] and r.get('lr_scalar') is None]
T5 = collections.defaultdict(dict)
for r in R5m: T5[(r['data'], round(r['frac'], 2), r['sub'])][r['cond']] = r
R6 = [json.load(open(f)) for f in glob.glob(f'{ROOT}/runs_r6/*.json')]
R6 = [r for r in R6 if r['cond'] in ('empty', 'support', 'rewired', 'er', 'degfeat', 'degfeat0')]
T = collections.defaultdict(dict)
for r in R6: T[(r['data'], round(r['frac'], 2), r['sub'], r['budget'])][r['cond']] = r
for (c, f, s), v in T5.items():
    for k, r in v.items():
        if k in ('empty', 'support', 'rewired', 'recency'): T[(c, f, s, 'epochs')].setdefault(k, r)
KEYS = {'all': 'test_auc', 'first': 'test_auc_first'}


def units(fr, cond, ref, metric, budgets, subs=None):
    per = collections.defaultdict(list)
    for (c, f, s, b), v in T.items():
        if f != fr or b not in budgets or (subs is not None and s not in subs): continue
        if cond in v and ref in v and v[cond].get(metric) is not None and v[ref].get(metric) is not None:
            per[c].append(v[cond][metric] - v[ref][metric])
    return per


# pooled support - empty over the learning-curve runs (kt/r2) and the runs of release 2.1.0
r2 = pd.read_csv(f'{ROOT}/r2/results/r2_runs.csv'); r2 = r2[r2.src == 'R2']
A = collections.defaultdict(dict)
x = r2[r2.cfg.isin(['P/empty', 'P/support'])]
piv = x.pivot_table(index=['data', 'frac', 'sub', 'seed'], columns='cfg', values=['auc', 'new'])
for (c, fr, s, seed), v in piv.iterrows():
    if seed != s or np.isnan(v[('auc', 'P/support')]) or np.isnan(v[('auc', 'P/empty')]): continue
    A[(c, round(fr, 2))].setdefault(('r2', s) if s else ('shared', 0), []).append(
        (v[('auc', 'P/support')] - v[('auc', 'P/empty')], v[('new', 'P/support')] - v[('new', 'P/empty')]))
for (c, fr, s), v in T5.items():
    if 'support' in v and 'empty' in v:
        A[(c, fr)].setdefault(('r5', s) if s else ('shared', 0), []).append(
            (v['support']['test_auc'] - v['empty']['test_auc'], v['support']['test_auc_first'] - v['empty']['test_auc_first']))


def pooled(fr, k):
    per = {}
    for c in CORP:
        S = A.get((c, fr), {})
        if S: per[c] = [float(np.mean([t[k] for t in L])) for L in S.values()]
    return per


res = {}
rows = []
for fr in (0.1, 0.25):
    for k, key in enumerate(('all', 'first')):
        rows.append(dict(label='support-empty, pooled', budget='epochs', frac=fr, key=key, **summary(pooled(fr, k))))
for key in ('all', 'first'):
    rows.append(dict(label='support-empty', budget='steps', frac=0.1, key=key, **summary(units(0.1, 'support', 'empty', KEYS[key], ('steps',)))))
for cond, ref in (('recency', 'empty'), ('rewired', 'empty'), ('er', 'empty'), ('support', 'rewired'), ('support', 'er')):
    rows.append(dict(label=f'{cond}-{ref}', budget='epochs', frac=0.1, key='first',
                     **summary(units(0.1, cond, ref, 'test_auc_first', ('epochs',), (0, 1, 2, 3, 4)))))
res['robustness'] = rows

# 2. decision-level and calibration metrics ----------------------------------------------------
dec = []
for budget, fracs in (('epochs', (0.05, 0.1, 0.25)), ('steps', (0.05, 0.1))):
    for fr in fracs:
        for metric in ('test_acc', 'test_rmse', 'test_ece'):
            per = units(fr, 'support', 'empty', metric, (budget,))
            if len(per) < 2: continue
            s = summary(per); s.pop('loco')
            dec.append(dict(budget=budget, frac=fr, metric=metric, **s))
res['decision_metrics'] = dec
cnt = {}
for c in CORP:
    v = [r for (cc, f, s), d in T5.items() if cc == c and f == 0.1 for r in d.values()]
    if v: cnt[c] = dict(n_pred=float(np.median([r['n_pred'] for r in v])), n_first=float(np.median([r['n_first'] for r in v])))
res['test_counts_10'] = cnt

# 3. subsample overlap -------------------------------------------------------------------------
ntr = {(r['data'], round(r['frac'], 2)): r['n_train_learners'] for r in R5}
ov = {}
for c in CORP:
    N = ntr.get((c, 1.0))
    if not N: continue
    ov[c] = {}
    for fr in (0.05, 0.1, 0.25):
        m = ntr.get((c, fr), max(2, int(round(fr * N))))
        ov[c][str(fr)] = dict(N=N, m=m, expected_shared=m * m / N, share=m / N)
res['subsample_overlap'] = ov

json.dump(res, open(f'{OUT}/r34_results.json', 'w'), indent=1)
for r in rows:
    print(f"{r['label']:24s} {r['budget']:6s} {r['frac']:.2f} {r['key']:5s} {r['mean']:+.4f} {r['pos']}/{r['n']} "
          f"boot=[{r['boot'][0]:+.4f},{r['boot'][1]:+.4f}] mixed=[{r['mixed'].get('lo',0):+.4f},{r['mixed'].get('hi',0):+.4f}] "
          f"LOCO mean [{r['loco_mean'][0]:+.4f},{r['loco_mean'][1]:+.4f}] pos>={r['loco_pos_min']}/9 pmax={r['loco_p_max']:.3f}")
for d in dec:
    print(f"{d['budget']:6s} {d['frac']:.2f} {d['metric']:9s} {d['mean']:+.5f} {d['pos']}/{d['n']} p={d['p']:.3f} boot=[{d['boot'][0]:+.5f},{d['boot'][1]:+.5f}]")
print(json.dumps(cnt))
print({c: {f: round(v['share'], 3) for f, v in o.items()} for c, o in ov.items()})
