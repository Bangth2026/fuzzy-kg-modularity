"""Release 3.3.0 analysis -> results_r6/r33_results.json (run files of releases 2.1.0, 3.1.0, 3.2.0 and 3.3.0).

1. Rise of the graph gain from 5% to 10% under the two budgets on the same corpus x subsample units
   (epoch budget: runs of release 2.1.0, subsamples with seed 12345 + s; update budget: runs_r6), with
   a bootstrap interval (corpora, then units within corpus) for the difference of the two rises.
2. Updates to the selected checkpoint under the two budgets, by corpus (epoch budget: best epoch x
   updates per epoch), and the per-corpus values of the upper rows of Table 6.
3. Graph controls pooled over both budgets with a fixed effect for the budget (effect-coded, so the
   intercept is the mean of the two budgets) and a random intercept per corpus; the budget
   coefficient is the graph x budget interaction. Intervals are unadjusted.
4. Channel-separating controls (release 3.3.0, update budget, 10%): the graph only in the embedding
   refinement of Eq. (2) (*_emb) or only in the attention bias of Eq. (3) (*_att), for the support and
   the Erdos-Renyi graphs, against the empty-graph control and against the full graph.
"""
import json, glob, os, collections, warnings, math, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.stats import wilcoxon
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt'); OUT = f'{ROOT}/results_r6'
CORP = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
SMALL = ['statics11', 'algebra05', 'bridge06', 'assist17']
rng0 = np.random.default_rng(20261108)


def wil(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.all(x == 0): return 1.0
    return float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj


def boot(L, B=10000, rng=rng0):
    L = [np.asarray(v, float) for v in L if len(v)]; n = len(L); bs = np.empty(B)
    for b in range(B):
        idx = rng.integers(n, size=n)
        bs[b] = np.mean([L[i][rng.integers(len(L[i]), size=len(L[i]))].mean() for i in idx])
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def mixed(df, formula='g ~ 1'):
    import statsmodels.formula.api as smf
    f = smf.mixedlm(formula, df, groups=df['corpus']).fit(reml=True)
    out = {}
    for k in f.params.index:
        if k == 'Group Var': continue
        b, se = float(f.params[k]), float(f.bse[k])
        out[k] = dict(est=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=float(f.pvalues[k]))
    return out


R5 = [json.load(open(f)) for f in glob.glob(f'{ROOT}/runs_r5/*.json')]
R5 = [r for r in R5 if r['d'] == 64 and r['wd'] == 0 and r['seed'] == r['sub'] and r.get('lr_scalar') is None]
R6 = [json.load(open(f)) for f in glob.glob(f'{ROOT}/runs_r6/*.json')]
T = collections.defaultdict(dict)
for r in R6: T[(r['data'], round(r['frac'], 2), r['sub'], r['budget'])][r['cond']] = r
for r in R5: T[(r['data'], round(r['frac'], 2), r['sub'], 'epochs')].setdefault(r['cond'], r)
KEY = {'all': 'test_auc', 'first': 'test_auc_first'}
res = {}

# 1. rise from 5% to 10% on the same units -----------------------------------------------------
rise = {}
for key in ('all', 'first'):
    per = collections.defaultdict(list); rows = []
    for c in CORP:
        for s in range(5):
            try:
                g = {(b, f): T[(c, f, s, b)]['support'][KEY[key]] - T[(c, f, s, b)]['empty'][KEY[key]]
                     for b in ('epochs', 'steps') for f in (0.05, 0.1)}
            except KeyError:
                continue
            re, rs = g[('epochs', 0.1)] - g[('epochs', 0.05)], g[('steps', 0.1)] - g[('steps', 0.05)]
            per[c].append(re - rs); rows.append(dict(corpus=c, sub=s, **{f'{b}_{int(f*100)}': v for (b, f), v in g.items()}))
    d = pd.DataFrame(rows); cm = d.groupby('corpus').mean(numeric_only=True)
    out = {k: float(cm[k].mean()) for k in ('epochs_5', 'epochs_10', 'steps_5', 'steps_10')}
    out['rise_epochs'] = out['epochs_10'] - out['epochs_5']; out['rise_steps'] = out['steps_10'] - out['steps_5']
    out['share_explained'] = 1 - out['rise_steps'] / out['rise_epochs'] if out['rise_epochs'] else None
    diff = np.array([np.mean(per[c]) for c in CORP if c in per])
    out.update(diff=float(diff.mean()), diff_pos=int((diff > 0).sum()), n=len(diff), units=len(d), p=wil(diff), boot=boot([per[c] for c in CORP if c in per]))
    rise[key] = out
res['rise_same_units'] = rise

# 2. updates to the selected checkpoint and per-corpus upper rows of Table 6 ---------------------------
ck = []
for c in CORP:
    for f in (0.05, 0.1):
        e, u = [], []
        for s in range(5):
            v = T.get((c, f, s, 'steps'), {}); w = T.get((c, f, s, 'epochs'), {})
            for cond in ('empty', 'support'):
                if cond in v and cond in w:
                    spe = v[cond]['steps_per_epoch']
                    e.append(w[cond]['best_epoch'] * spe); u.append(v[cond]['best_step'])
        if e: ck.append(dict(corpus=c, frac=f, updates_epoch_budget=float(np.median(e)), updates_step_budget=float(np.median(u)), n=len(e),
                             steps_per_epoch=int(T[(c, f, 0, 'steps')]['empty']['steps_per_epoch'])))
res['checkpoint_updates'] = ck
up = []
for c in CORP:
    for f in (0.05, 0.1):
        row = dict(corpus=c, frac=f)
        for cond in ('empty', 'support'):
            for key in ('all', 'first'):
                vals = [T[(c, f, s, 'steps')][cond][KEY[key]] - T[(c, f, s, 'epochs')][cond][KEY[key]]
                        for s in range(5) if cond in T.get((c, f, s, 'steps'), {}) and cond in T.get((c, f, s, 'epochs'), {})]
                if vals: row[f'{cond}_{key}'] = float(np.mean(vals)); row['n_sub'] = len(vals)
        up.append(row)
res['table6_upper_by_corpus'] = up

# 3. pooled with a budget fixed effect -------------------------------------------------------------
pooled = []
for cond, ref in (('support', 'empty'), ('rewired', 'empty'), ('er', 'empty'), ('support', 'rewired'), ('rewired', 'er'),
                  ('support', 'er'), ('degfeat0', 'empty'), ('support', 'degfeat0')):
    for key in ('all', 'first'):
        rows = []
        for (c, f, s, b), v in T.items():
            if f != 0.1 or cond not in v or ref not in v: continue
            if b == 'epochs' and 'degfeat0' in (cond, ref) and s > 2: continue
            rows.append(dict(corpus=c, budget=1.0 if b == 'steps' else -1.0, g=v[cond][KEY[key]] - v[ref][KEY[key]]))
        d = pd.DataFrame(rows); m = mixed(d, 'g ~ budget')
        pooled.append(dict(cond=cond, ref=ref, key=key, units=len(d), mean=m['Intercept'], budget_half_diff=m['budget']))
res['pooled_budget_fe'] = pooled

# 4. channel-separating controls ----------------------------------------------------------------
PAIRS = [('support_emb', 'empty'), ('support_att', 'empty'), ('er_emb', 'empty'), ('er_att', 'empty'),
         ('support', 'support_emb'), ('support', 'support_att'), ('er', 'er_emb'), ('er', 'er_att'),
         ('support_att', 'er_att'), ('support_emb', 'er_emb')]
ch = []
for cond, ref in PAIRS:
    for key in ('all', 'first'):
        per = collections.defaultdict(list)
        for (c, f, s, b), v in T.items():
            if f == 0.1 and b == 'steps' and cond in v and ref in v: per[c].append(v[cond][KEY[key]] - v[ref][KEY[key]])
        if len(per) < 2: continue
        cs = [c for c in CORP if c in per]; mm = np.array([np.mean(per[c]) for c in cs])
        d = pd.DataFrame([dict(corpus=c, g=g) for c in cs for g in per[c]])
        ch.append(dict(cond=cond, ref=ref, key=key, mean=float(mm.mean()), pos=int((mm > 0).sum()), n=len(cs), units=len(d), p=wil(mm),
                       boot=boot([per[c] for c in cs]), mixed=mixed(d)['Intercept']))
for r, a in zip(ch, holm([r['p'] for r in ch])): r['holm'] = float(a)
res['channels'] = ch
res['channels_family_size'] = len(ch)
nch = sum(1 for r in R6 if r['cond'].endswith(('_emb', '_att')))
res['n_channel_runs'] = nch
json.dump(res, open(f'{OUT}/r33_results.json', 'w'), indent=1)
pd.DataFrame(up).to_csv(f'{OUT}/table6_upper_by_corpus.csv', index=False)
pd.DataFrame(ck).to_csv(f'{OUT}/checkpoint_updates.csv', index=False)

f4 = lambda x: f'{x:+.4f}'
print('== rise'); [print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}) for k, v in rise.items()]
print('== checkpoints'); [print(r) for r in ck]
print('== upper by corpus'); [print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}) for r in up]
print('== pooled with budget FE')
for r in pooled: print(r['cond'], r['ref'], r['key'], f4(r['mean']['est']), f"[{f4(r['mean']['lo'])},{f4(r['mean']['hi'])}]", 'budget', f4(r['budget_half_diff']['est']), f"p={r['budget_half_diff']['p']:.3f}")
print('== channels (family', len(ch), ', runs', nch, ')')
for r in ch: print(r['cond'], '-', r['ref'], r['key'], f4(r['mean']), f"{r['pos']}/{r['n']}", f"holm={r['holm']:.3f}", 'mixed', f"[{f4(r['mixed']['lo'])},{f4(r['mixed']['hi'])}]")
