"""Release 3.2.0 analysis -> results_r6/r32_results.json.

Adds to r6_analyse.py (whose output, results_r6/r6_results.json, is unchanged):

1. Degree feature with a zero-initialised map (condition degfeat0): against the empty-graph control
   and the support graph, under the update budget (subsamples as in Table 6) and under the epoch
   budget (subsamples 0-2, paired with the runs of release 2.1.0).
2. Erdos-Renyi and degree-preserving graphs on every subsample run: epoch budget, subsamples 0-4 at
   10% (ER of subsamples 3-4 added in this release); update budget, five subsamples on the four small
   corpora and two on the six larger ones (ER on the larger corpora and rewired on their subsample 1
   added in this release).
3. Pooled comparisons of the graph controls at 10%: for every corpus, the subsample-level
   differences of both budgets; bootstrap over corpora, then units within corpus, and a mixed model
   with a random intercept per corpus. Also per budget.
4. Update-budget table with the rows that have fewer than six corpora reported as descriptive
   (outside the Holm family: with n = 4 the smallest two-sided exact Wilcoxon p is 0.125).
5. Learned scalars under the update budget; share of first attempts among test predictions by
   corpus; run counts by set including this release.
"""
import json, glob, os, collections, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import wilcoxon
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt'); OUT = f'{ROOT}/results_r6'
CORP = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
SMALL = ['statics11', 'algebra05', 'bridge06', 'assist17']
rng0 = np.random.default_rng(20261107)


def wil(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.all(x == 0): return 1.0
    return float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj


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
        f = smf.mixedlm('g ~ 1', d, groups=d['corpus']).fit(reml=True)
        b = float(f.params['Intercept']); se = float(f.bse['Intercept'])
        return dict(est=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=float(f.pvalues['Intercept']), n=len(d))
    except Exception as e:
        return dict(error=repr(e))


R5 = [json.load(open(f)) for f in glob.glob(f'{ROOT}/runs_r5/*.json')]
R5m = [r for r in R5 if r['d'] == 64 and r['wd'] == 0 and r['seed'] == r['sub'] and r.get('lr_scalar') is None]
T5 = collections.defaultdict(dict)
for r in R5m: T5[(r['data'], round(r['frac'], 2), r['sub'])][r['cond']] = r
R6 = [json.load(open(f)) for f in glob.glob(f'{ROOT}/runs_r6/*.json')]
T = collections.defaultdict(dict)        # (corpus, frac, sub, budget) -> cond -> run
for r in R6: T[(r['data'], round(r['frac'], 2), r['sub'], r['budget'])][r['cond']] = r
for (c, f, s), v in T5.items():          # epoch-budget references from release 2.1.0 (same code, same subsample)
    for k, r in v.items():
        if k in ('empty', 'support', 'rewired', 'recency'): T[(c, f, s, 'epochs')].setdefault(k, r)

KEYS = {'all': 'test_auc', 'first': 'test_auc_first'}


def units(fr, cond, ref, key, budgets, subs=None):
    per = collections.defaultdict(list)
    for (c, f, s, b), v in T.items():
        if f != fr or b not in budgets or (subs is not None and s not in subs): continue
        if cond in v and ref in v and v[cond][KEYS[key]] is not None and v[ref][KEYS[key]] is not None:
            per[c].append(v[cond][KEYS[key]] - v[ref][KEYS[key]])
    return per


def summary(per, with_ci=True):
    cs = [c for c in CORP if c in per]
    m = np.array([np.mean(per[c]) for c in cs])
    out = dict(mean=float(m.mean()), pos=int((m > 0).sum()), n=len(cs), p=wil(m), units=int(sum(len(per[c]) for c in cs)),
               nsub=float(np.median([len(per[c]) for c in cs])), per_corpus={c: float(np.mean(per[c])) for c in cs})
    if with_ci:
        out['boot'] = list(boot_ci([per[c] for c in cs])); out['mixed'] = mixed_ci([(c, g) for c in cs for g in per[c]])
    return out


res = {}
# 1-2. tables -------------------------------------------------------------------------------------
PAIRS_STEPS = [('support', 'empty'), ('rewired', 'empty'), ('er', 'empty'), ('degfeat0', 'empty'),
               ('support', 'rewired'), ('support', 'er'), ('rewired', 'er'), ('support', 'degfeat0')]
steps_tab = []
for fr in (0.05, 0.1):
    for cond, ref in PAIRS_STEPS:
        for key in ('all', 'first'):
            per = units(fr, cond, ref, key, ('steps',))
            if len(per) < 2: continue
            row = dict(budget='steps', frac=fr, cond=cond, ref=ref, key=key, **summary(per))
            row['in_family'] = row['n'] >= 6
            steps_tab.append(row)
fam = [r for r in steps_tab if r['in_family']]
for r, a in zip(fam, holm([r['p'] for r in fam])): r['holm'] = float(a)
for r in steps_tab: r.setdefault('holm', None)
res['steps_table'] = steps_tab; res['steps_family_size'] = len(fam)

PAIRS_EP = [('support', 'empty'), ('rewired', 'empty'), ('er', 'empty'), ('degfeat', 'empty'), ('degfeat0', 'empty'),
            ('support', 'er'), ('rewired', 'er'), ('support', 'rewired'), ('support', 'degfeat'), ('support', 'degfeat0')]
ep_tab = []
for cond, ref in PAIRS_EP:
    subs = (0, 1, 2) if 'degfeat' in cond + ref else (0, 1, 2, 3, 4)
    for key in ('all', 'first'):
        per = units(0.1, cond, ref, key, ('epochs',), subs)
        if len(per) < 2: continue
        ep_tab.append(dict(budget='epochs', frac=0.1, subs=list(subs), cond=cond, ref=ref, key=key, **summary(per)))
for r, a in zip(ep_tab, holm([r['p'] for r in ep_tab])): r['holm'] = float(a)
res['epochs_controls_table'] = ep_tab; res['epochs_family_size'] = len(ep_tab)
# lower block of the controls table (epoch budget, 10%): the comparisons that are not in its upper block
low = []
for cond, ref, subs in (('er', 'empty', (0, 1, 2, 3, 4)), ('support', 'er', (0, 1, 2, 3, 4)), ('rewired', 'er', (0, 1, 2, 3, 4)),
                        ('degfeat0', 'empty', (0, 1, 2)), ('support', 'degfeat0', (0, 1, 2))):
    for key in ('all', 'first'):
        per = units(0.1, cond, ref, key, ('epochs',), subs)
        low.append(dict(budget='epochs', frac=0.1, subs=list(subs), cond=cond, ref=ref, key=key, **summary(per)))
for r, a in zip(low, holm([r['p'] for r in low])): r['holm'] = float(a)
res['table5_lower'] = low
# recency against the empty-graph control, first attempts, with intervals (release 2.1.0 runs)
rec = []
for fr in (0.1, 0.25):
    for key in ('all', 'first'):
        per = units(fr, 'recency', 'empty', key, ('epochs',))
        rec.append(dict(frac=fr, key=key, **summary(per)))
res['recency_ci'] = rec

# 3. pooled over both budgets at 10% ------------------------------------------------------------
pooled = []
for cond, ref in (('support', 'empty'), ('rewired', 'empty'), ('er', 'empty'), ('support', 'rewired'), ('rewired', 'er'), ('support', 'er'),
                  ('degfeat0', 'empty'), ('support', 'degfeat0')):
    for key in ('all', 'first'):
        for budgets in (('epochs', 'steps'), ('epochs',), ('steps',)):
            subs = (0, 1, 2) if 'degfeat0' in (cond, ref) and budgets == ('epochs',) else None
            per = units(0.1, cond, ref, key, budgets, subs)
            if len(per) < 2: continue
            pooled.append(dict(frac=0.1, cond=cond, ref=ref, key=key, budgets='+'.join(budgets), **summary(per)))
res['pooled_controls'] = pooled
# ratio of rewired and ER to support on first attempts, by budget and subsample set
ratio = []
for budgets in (('epochs',), ('steps',), ('epochs', 'steps')):
    for subs in ((0, 1, 2), (3, 4), None):
        if subs is not None and budgets != ('epochs',): continue
        row = {}
        for cond in ('support', 'rewired', 'er'):
            per = units(0.1, cond, 'empty', 'first', budgets, subs)
            if len(per) >= 2: row[cond] = float(np.mean([np.mean(per[c]) for c in per])); row[cond + '_n'] = len(per)
        ratio.append(dict(budgets='+'.join(budgets), subs=None if subs is None else list(subs), **row))
res['first_attempt_ratios'] = ratio

# 5. diagnostics --------------------------------------------------------------------------------
beta = collections.defaultdict(list); best = collections.defaultdict(list)
for r in R6:
    if r['budget'] != 'steps': continue
    if r['cond'] != 'empty': beta[r['cond']].append([r['beta_pre'], r['beta_suc'], r['beta_same']])
    else: beta['empty_same'].append([r['beta_same']])
    best[r['data'] in SMALL].append(r['best_step'])
res['beta_steps'] = {k: dict(min=float(np.min(v)), max=float(np.max(v)), mean_abs_dev=float(np.mean(np.abs(np.array(v) - 1))), n=len(v)) for k, v in beta.items()}
allb = np.array([x for k, v in beta.items() if k != 'empty_same' for x in v])
res['beta_steps_all'] = dict(min=float(allb.min()), max=float(allb.max()), n=len(allb))
res['best_step'] = {('small' if k else 'large'): dict(median=float(np.median(v)), max=int(max(v)), n=len(v)) for k, v in best.items()}
dm = [r['deg_map_norm'] for r in R6 if r['cond'] == 'degfeat0' and r.get('deg_map_norm') is not None]
if dm: res['degfeat0_map_norm'] = dict(median=float(np.median(dm)), max=float(max(dm)), n=len(dm))
share = collections.defaultdict(list)
for r in R5m:
    if round(r['frac'], 2) == 0.1 and r['cond'] == 'support': share[r['data']].append(r['n_first'] / r['n_pred'])
res['first_share_10'] = {c: float(np.mean(share[c])) for c in CORP if c in share}
prev = json.load(open(f'{OUT}/r6_results.json'))['run_counts']
cnt = collections.Counter()
for x in prev:
    if not x['set'].startswith('runs_r6'): cnt[(x['set'], x['frac'], x['cond'])] += x['n']
for r in R6:
    rel = 'runs_r6 (release 3.2.0)' if r['cond'] == 'degfeat0' or (r['cond'] == 'er' and (r['budget'] == 'epochs' and r['sub'] >= 3 or r['budget'] == 'steps' and r['data'] not in SMALL)) \
        or (r['cond'] == 'rewired' and r['budget'] == 'steps' and r['data'] not in SMALL and r['sub'] == 1) else 'runs_r6 (release 3.1.0)'
    cnt[(rel, round(r['frac'], 2), r['cond'] + '/' + r['budget'])] += 1
res['run_counts'] = [dict(set=a, frac=b, cond=c, n=n) for (a, b, c), n in sorted(cnt.items())]
res['run_totals'] = {s: sum(n for (a, b, c), n in cnt.items() if a == s) for s in sorted({k[0] for k in cnt})}
json.dump(res, open(f'{OUT}/r32_results.json', 'w'), indent=1)
pd.DataFrame(res['run_counts']).to_csv(f'{OUT}/run_counts.csv', index=False)


def fmt(r):
    h = '' if r.get('holm') is None else f" holm={r['holm']:.3f}"
    ci = f" boot=[{r['boot'][0]:+.4f},{r['boot'][1]:+.4f}] mixed=[{r['mixed'].get('lo', 0):+.4f},{r['mixed'].get('hi', 0):+.4f}]" if 'boot' in r else ''
    return f"{r['cond']}-{r['ref']} {r['key']:5s} {r['mean']:+.4f} {r['pos']}/{r['n']} units={r['units']} p={r['p']:.3f}{h}{ci}"


print('== steps table (family', len(fam), ')'); [print(' ', r['frac'], fmt(r), '' if r['in_family'] else '(descriptive)') for r in steps_tab]
print('== epochs controls (family', len(ep_tab), ')'); [print(' ', fmt(r)) for r in ep_tab]
print('== table 5 lower'); [print(' ', r['subs'], fmt(r)) for r in low]
print('== recency'); [print(' ', r['frac'], 'recency-empty', r['key'], round(r['mean'],4), r['boot'], r['mixed']) for r in rec]
print('== pooled 10%'); [print(' ', r['budgets'], fmt(r)) for r in pooled]
print('== ratios', json.dumps(ratio, indent=0))
print('== beta', json.dumps(res['beta_steps']), res['beta_steps_all'], res['best_step'], res.get('degfeat0_map_norm'))
print('== share', {k: round(v * 100, 1) for k, v in res['first_share_10'].items()})
print('== runs', res['run_totals'])
