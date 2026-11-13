"""Analysis of the release-3.6.0 runs (runs_r9) together with the archived CPU runs (runs_r6, runs_r7)
and the release-3.5.0 runs (runs_r8). Writes results_r9/r9_results.json; reads run files only.

1. Reproduction: every rerun of an archived CPU condition (empty, er_emb, support_emb, support) against
   runs_r6 / runs_r7 (largest absolute difference in test AUC).
2. Complete null-graph ladder with the graph read in the embedding refinement only (update budget, 10%):
   empty -> uniform_emb -> er_emb -> dp_emb -> support_emb on every unit where all five rungs exist.
   Each term (adjacent difference) and the total: mean over corpora of the subsample means, corpora with a
   positive difference, Wilcoxon p over corpora (unadjusted), 95% mixed-model interval (random intercept
   per corpus) and a corpus-then-subsample bootstrap interval. Also the reverse path in which DP precedes
   ER (empty -> uniform -> dp -> er -> support is not a ladder; the alternative order reported is
   empty -> dp -> support with the ER rung dropped, and the term degree-feature-free).
3. Frozen-random-table control: erR_emb against empty, uniform_emb and er_emb.
4. First attempts split by whether a graph neighbour of the target KC occurs earlier in the learner's
   sequence (support graph, Erdos-Renyi graph), for support_emb - empty and er_emb - empty.
5. First attempts by data per KC (support - empty, both entry points; support_emb - empty): terciles within
   corpus with the number of first-attempt predictions per tercile; quintiles of log n_k pooled across
   corpora; a mixed model of the bin-level gain on log10 n_k (random intercept per corpus).
6. Seed replicates (rep 0-2 on subsamples 0-1 of the four small corpora): seed and subsample components
   of the variance of the paired difference, for the probe (support - empty, support_emb - empty,
   er_emb - empty) and for GKT (support - rewired).
7. GKT on CPU: ladder empty -> uniform -> er -> rewired -> support on the units run.
8. GKT runs of release 3.5.0 (runs_r8): absolute test AUCs per corpus and condition, batch size, updates,
   and per-corpus values of every comparison of Table 8.
"""
import os, sys, json, glob, collections, numpy as np, pandas as pd, warnings
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
REPO = os.path.dirname(HERE)
R9 = os.environ.get('R9_RUNS', f'{ROOT}/runs_r9'); OUT = os.environ.get('R9_OUT', f'{REPO}/results_r9'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def runs(path, steps_only=True):
    rows = []
    for f in glob.glob(f'{path}/*.json'):
        r = json.load(open(f))
        if steps_only and r.get('budget') != 'steps': continue
        if abs(r.get('frac', 0.1) - 0.1) > 1e-9: continue
        if r.get('lr', 0.002) != 0.002 or r.get('l2', 0) not in (0, 0.0, None): continue
        rows.append(r)
    return pd.DataFrame(rows)


R6 = runs(f'{REPO}/runs_r6'); R7 = runs(f'{REPO}/runs_r7'); R8 = runs(f'{REPO}/runs_r8'); R9r = runs(R9)
if 'rep' not in R9r: R9r['rep'] = 0
R9r['rep'] = R9r['rep'].fillna(0).astype(int)
res = {'n_runs_r9': int(len(R9r)), 'runs_by_cond_r9': {f'{c}|rep{r}': int(n) for (c, r), n in R9r.groupby(['cond', 'rep']).size().items()}}


def mixed(df):
    try:
        import statsmodels.formula.api as smf
        if df.data.nunique() < 3: return None
        r = smf.mixedlm('d ~ 1', df, groups=df['data']).fit(reml=True)
        lo, hi = r.conf_int().loc['Intercept']
        if not (np.isfinite(lo) and np.isfinite(hi)):
            # corpus variance estimated at zero: refit with a derivative-free optimiser
            # corpus variance estimated at zero: the model reduces to a common mean over units
            r = smf.ols('d ~ 1', df).fit()
            lo, hi = r.conf_int().loc['Intercept']
            return dict(est=float(r.params['Intercept']), lo=float(lo), hi=float(hi), singular=True)
        return dict(est=float(r.params['Intercept']), lo=float(lo), hi=float(hi))
    except Exception as e:
        return dict(error=str(e)[:100])


def boot(df, B=4000):
    g = {c: v.d.values for c, v in df.groupby('data')}; cs = list(g)
    out = []
    for _ in range(B):
        pick = rng.choice(len(cs), len(cs))
        out.append(np.mean([rng.choice(g[cs[i]], len(g[cs[i]])).mean() for i in pick]))
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def summ(df):
    """df: columns data, sub, d (subsample-level differences)."""
    df = df.dropna()
    if df.empty: return None
    per = df.groupby('data').d.mean()
    o = dict(units=int(len(df)), n=int(len(per)), mean=float(per.mean()), pos=int((per > 0).sum()),
             per_corpus={k: float(v) for k, v in per.items()})
    if len(per) >= 4: o['p_wilcoxon'] = float(wilcoxon(per.values).pvalue)
    o['mixed'] = mixed(df); o['boot'] = boot(df) if len(per) >= 3 else None
    return o


def table(D, key):
    col = 'test_auc' if key == 'all' else ('test_auc_first' if key == 'first' else key)
    return D.pivot_table(index=['data', 'sub'], columns='cond', values=col, aggfunc='first')


def diff(T, a, b):
    if a not in T or b not in T: return None
    j = (T[a] - T[b]).dropna().rename('d').reset_index()
    return summ(j)


# 1. reproduction ------------------------------------------------------------------------------
rep0 = R9r[R9r.rep == 0]
arch = pd.concat([R6, R7]).drop_duplicates(['data', 'sub', 'cond'])
m = rep0.merge(arch[['data', 'sub', 'cond', 'test_auc', 'test_auc_first']], on=['data', 'sub', 'cond'], suffixes=('', '_arch'))
res['reproduction'] = dict(n=int(len(m)), max_abs_diff_all=float((m.test_auc - m.test_auc_arch).abs().max()) if len(m) else None,
                           max_abs_diff_first=float((m.test_auc_first - m.test_auc_first_arch).abs().max()) if len(m) else None)

# 2. complete ladder ---------------------------------------------------------------------------
# rung values: r9 reruns where present (identical to the archived runs), archived runs otherwise
lad = pd.concat([rep0[rep0.family == 'probe'], arch]).drop_duplicates(['data', 'sub', 'cond'])
LAD = {}
for key in ('first', 'all'):
    T = table(lad, key)
    need = ['empty', 'uniform_emb', 'er_emb', 'dp_emb', 'support_emb']
    T = T[[c for c in T.columns if c in need + ['erR_emb']]].dropna(subset=[c for c in need if c in T.columns])
    terms = [('uniform_emb', 'empty', 'U - empty (mixing)'), ('er_emb', 'uniform_emb', 'ER - U (sparse mixing)'),
             ('dp_emb', 'er_emb', 'DP - ER (degrees)'), ('support_emb', 'dp_emb', 'G - DP (edges)'),
             ('support_emb', 'empty', 'G - empty (total)'),
             ('er_emb', 'empty', 'ER - empty'), ('dp_emb', 'empty', 'DP - empty'), ('support_emb', 'er_emb', 'G - ER'),
             ('dp_emb', 'uniform_emb', 'DP - U')]
    LAD[key] = {lab: diff(T, a, b) for a, b, lab in terms}
    LAD[key]['_units'] = int(len(T)); LAD[key]['_corpora'] = sorted(T.reset_index().data.unique().tolist())
res['ladder_emb'] = LAD

# 3. frozen random table -----------------------------------------------------------------------
FR = {}
for key in ('first', 'all'):
    T = table(lad, key)
    FR[key] = {lab: diff(T, a, b) for a, b, lab in [('erR_emb', 'empty', 'ER(frozen R) - empty'),
                                                     ('erR_emb', 'uniform_emb', 'ER(frozen R) - U'),
                                                     ('er_emb', 'erR_emb', 'ER(learned E) - ER(frozen R)'),
                                                     ('support_emb', 'erR_emb', 'G - ER(frozen R)')]}
res['frozen_random_table'] = FR


# 4. neighbour-seen split ----------------------------------------------------------------------
def pred(tag): return np.load(f'{R9}/{tag}_pred.npz')
NB = collections.defaultdict(list); share = collections.defaultdict(list)
for (c, s), g in rep0.groupby(['data', 'sub']):
    tags = dict(zip(g.cond, g.tag))
    if not {'empty', 'support_emb', 'er_emb'} <= set(tags): continue
    P = {k: pred(tags[k]) for k in ('empty', 'support_emb', 'er_emb')}
    new = P['empty']['new'].astype(bool); y = P['empty']['y'].astype(int)
    for k in ('support_emb', 'er_emb'): assert (P[k]['kc'] == P['empty']['kc']).all()
    for gname in ('sup', 'er'):
        nb = P['empty'][f'nb_{gname}'].astype(bool)
        share[gname].append(dict(data=c, sub=s, d=float(nb[new].mean())))
        for cond in ('support_emb', 'er_emb'):
            for lab, sel in (('seen', new & nb), ('unseen', new & ~nb)):
                if sel.sum() > 20 and 0 < y[sel].mean() < 1:
                    dd = roc_auc_score(y[sel], P[cond]['p'][sel]) - roc_auc_score(y[sel], P['empty']['p'][sel])
                    NB[(gname, cond, lab)].append(dict(data=c, sub=s, d=dd, n=int(sel.sum())))
            # difference seen - unseen within unit
    for gname in ('sup', 'er'):
        for cond in ('support_emb', 'er_emb'):
            a = [r for r in NB[(gname, cond, 'seen')] if r['data'] == c and r['sub'] == s]
            b = [r for r in NB[(gname, cond, 'unseen')] if r['data'] == c and r['sub'] == s]
            if a and b: NB[(gname, cond, 'seen-unseen')].append(dict(data=c, sub=s, d=a[0]['d'] - b[0]['d']))
res['neighbour_split'] = {f'{g}|{c}|{l}': dict(summ(pd.DataFrame(v)) or {}, n_first_median=float(np.median([r.get('n', np.nan) for r in v])))
                          for (g, c, l), v in NB.items()}
res['share_first_with_seen_neighbour'] = {g: summ(pd.DataFrame(v)) for g, v in share.items()}

# 5. first attempts by data per KC ----------------------------------------------------------------
from r5_controls import load
TER = collections.defaultdict(list); CNT = collections.defaultdict(list); POOL = []
for (c, s), g in rep0.groupby(['data', 'sub']):
    tags = dict(zip(g.cond, g.tag))
    if 'empty' not in tags: continue
    df = load(c, 0.1, s); tr = df[df.split == 'train']; K = int(df.kc.max()) + 1
    nk = tr.groupby('kc').size().reindex(range(K)).fillna(0)
    cuts = np.quantile(nk.values, [1 / 3, 2 / 3])
    E = pred(tags['empty']); new = E['new'].astype(bool); y = E['y'].astype(int)[new]; kc = E['kc'][new]
    n_of = nk.values[kc]; band = np.digitize(n_of, cuts, right=True)
    for cond in ('support', 'support_emb'):
        if cond not in tags: continue
        P = pred(tags[cond])['p'][new]; pe = E['p'][new]
        for b in range(3):
            sel = band == b
            CNT[(cond, b)].append(dict(data=c, sub=s, n=int(sel.sum()), npos=int(y[sel].sum()), kcs=int(len(np.unique(kc[sel])))))
            if sel.sum() > 20 and 0 < y[sel].mean() < 1:
                TER[(cond, b)].append(dict(data=c, sub=s, d=roc_auc_score(y[sel], P[sel]) - roc_auc_score(y[sel], pe[sel])))
        POOL.append(dict(data=c, sub=s, cond=cond, ln=np.log10(n_of + 1), y=y, p=P, pe=pe))
res['nk_terciles'] = {f'{c}|T{b + 1}': summ(pd.DataFrame(v)) for (c, b), v in TER.items()}
res['nk_tercile_counts'] = {f'{c}|T{b + 1}': dict(first_median=float(np.median([r['n'] for r in v])), first_min=int(min(r['n'] for r in v)),
                                                  first_total=int(sum(r['n'] for r in v)), units=len(v),
                                                  units_le20=int(sum(r['n'] <= 20 for r in v)))
                            for (c, b), v in CNT.items()}
# pooled quintiles of log n_k across corpora, per condition
PQ = {}
for cond in ('support', 'support_emb'):
    rows = [r for r in POOL if r['cond'] == cond]
    if not rows: continue
    allln = np.concatenate([r['ln'] for r in rows]); qs = np.quantile(allln, [.2, .4, .6, .8])
    bins = collections.defaultdict(list); pts = []
    for r in rows:
        b = np.digitize(r['ln'], qs, right=True)
        for k in range(5):
            sel = b == k
            if sel.sum() > 20 and 0 < r['y'][sel].mean() < 1:
                dd = roc_auc_score(r['y'][sel], r['p'][sel]) - roc_auc_score(r['y'][sel], r['pe'][sel])
                bins[k].append(dict(data=r['data'], sub=r['sub'], d=dd))
                pts.append(dict(data=r['data'], x=float(np.mean(r['ln'][sel])), d=dd, w=int(sel.sum())))
    out = {f'Q{k + 1}': summ(pd.DataFrame(v)) for k, v in bins.items()}
    out['cuts_log10_nk_plus1'] = [float(x) for x in qs]
    try:
        import statsmodels.formula.api as smf
        D = pd.DataFrame(pts); f = smf.mixedlm('d ~ x', D, groups=D['data']).fit(reml=True)
        lo, hi = f.conf_int().loc['x']
        out['slope_per_log10_nk'] = dict(est=float(f.params['x']), lo=float(lo), hi=float(hi), n_points=int(len(D)))
    except Exception as e:
        out['slope_per_log10_nk'] = dict(error=str(e)[:100])
    PQ[cond] = out
res['nk_pooled_quintiles'] = PQ

# 6. seed replicates ---------------------------------------------------------------------------
SEED = {}
for fam, pairs in (('probe', [('support', 'empty'), ('support_emb', 'empty'), ('er_emb', 'empty'), ('support_emb', 'er_emb')]),
                   ('gkt', [('gkt_support', 'gkt_rewired'), ('gkt_support', 'gkt_empty')])):
    for a, b in pairs:
        for key, col in (('first', 'test_auc_first'), ('all', 'test_auc')):
            X = R9r[R9r.cond.isin([a, b])].pivot_table(index=['data', 'sub', 'rep'], columns='cond', values=col, aggfunc='first')
            if a not in X or b not in X: continue
            d = (X[a] - X[b]).dropna().rename('d').reset_index()
            full = d.groupby(['data', 'sub']).filter(lambda g: g.rep.nunique() >= 3)
            if full.empty: continue
            seed_sd = full.groupby(['data', 'sub']).d.std(ddof=1)
            unit_mean = full.groupby(['data', 'sub']).d.mean()
            # variance components: within-unit (seed) and between units within corpus
            w = float((seed_sd ** 2).mean())
            b_ = full.groupby(['data', 'sub']).d.mean().groupby('data').var(ddof=1).mean()
            SEED[f'{a}-{b}|{key}'] = dict(units=int(full.groupby(['data', 'sub']).ngroups), seeds=3,
                                          seed_var=w, seed_sd=float(np.sqrt(w)),
                                          between_subsample_var_of_unit_means=float(b_) if b_ == b_ else None,
                                          mean_seed0=float(full[full.rep == 0].groupby('data').d.mean().mean()),
                                          mean_all_seeds=float(full.groupby('data').d.mean().mean()),
                                          pos_units_all_seeds=int((unit_mean > 0).sum()),
                                          per_unit={f'{k[0]}|{k[1]}': [float(x) for x in g.sort_values('rep').d] for k, g in full.groupby(['data', 'sub'])})
res['seed_replicates'] = SEED

# 7. GKT ladder on CPU -------------------------------------------------------------------------
G9 = R9r[(R9r.family == 'gkt') & (R9r.rep == 0)]
GK = {}
for key in ('first', 'all'):
    T = table(G9, key)
    GK[key] = {lab: diff(T, a, b) for a, b, lab in [('gkt_uniform', 'gkt_empty', 'U - empty'), ('gkt_er', 'gkt_uniform', 'ER - U'),
                                                    ('gkt_rewired', 'gkt_er', 'DP - ER'), ('gkt_support', 'gkt_rewired', 'G - DP'),
                                                    ('gkt_support', 'gkt_empty', 'G - empty'), ('gkt_er', 'gkt_empty', 'ER - empty')]}
    GK[key]['_abs'] = {f'{k[0]}|{k[1]}': {c: float(v) for c, v in row.items() if v == v} for k, row in T.iterrows()}
res['gkt_cpu_ladder'] = GK

# 8. GKT runs of release 3.5.0 -----------------------------------------------------------------
g8 = R8[R8.family == 'gkt']
res['gkt_r8_absolute'] = {c: {cond: dict(first=float(v.test_auc_first.mean()), all=float(v.test_auc.mean()),
                                          first_min=float(v.test_auc_first.min()), all_min=float(v.test_auc.min()),
                                          updates=float(v.steps.mean()), best_update=float(v.best_step.mean()),
                                          hit_max=int(v.hit_max.sum()), batch=int(v.batch_size.iloc[0]), n=int(len(v)),
                                          device=sorted(v.device.unique().tolist()))
                              for cond, v in gc.groupby('cond')} for c, gc in g8.groupby('data')}
P8 = {}
for fam, pairs in (('probe', [('uniform_emb', 'empty'), ('er_emb', 'uniform_emb'), ('support_emb', 'er_emb'), ('support_emb', 'empty'),
                              ('hash', 'empty'), ('er_emb', 'hash'), ('support_emb', 'hash')]),
                   ('sakte', [('sakte_uniform', 'sakte_empty'), ('sakte_er', 'sakte_uniform'), ('sakte_support', 'sakte_er'), ('sakte_support', 'sakte_empty')]),
                   ('gkt', [('gkt_er', 'gkt_empty'), ('gkt_rewired', 'gkt_er'), ('gkt_support', 'gkt_rewired'), ('gkt_support', 'gkt_empty')])):
    for key in ('first', 'all'):
        T = table(R8, key)
        for a, b in pairs:
            s_ = diff(T, a, b)
            if s_: P8[f'{a}-{b}|{key}'] = dict(mean=s_['mean'], pos=s_['pos'], per_corpus=s_['per_corpus'])
res['table8_per_corpus'] = P8

json.dump(res, open(f'{OUT}/r9_results.json', 'w'), indent=1, default=float)
f4 = lambda x: 'NA' if x is None else f'{x:+.4f}'
def line(lab, s):
    if not s: return
    mx = s.get('mixed') or {}
    print(f"  {lab:34s} {f4(s['mean'])} ({s['pos']}/{s['n']}) mixed {f4(mx.get('lo'))}..{f4(mx.get('hi'))} p={s.get('p_wilcoxon', float('nan')):.3f}")
print('runs r9', len(R9r), 'reproduction', res['reproduction'])
for key in ('first', 'all'):
    print('LADDER', key, LAD[key]['_units'], 'units', LAD[key]['_corpora'])
    for k, v in LAD[key].items():
        if not k.startswith('_'): line(k, v)
    print('FROZEN R', key)
    for k, v in FR[key].items(): line(k, v)
print('NEIGHBOUR SPLIT')
for k, v in res['neighbour_split'].items(): line(k, v)
print('SHARE seen neighbour', {g: round(v['mean'], 3) for g, v in res['share_first_with_seen_neighbour'].items() if v})
print('NK terciles'); [line(k, v) for k, v in res['nk_terciles'].items()]
print('NK counts', res['nk_tercile_counts'])
print('NK pooled', {c: v.get('slope_per_log10_nk') for c, v in PQ.items()})
for c, v in PQ.items():
    for k, s in v.items():
        if k.startswith('Q'): line(c + ' ' + k, s)
print('SEEDS'); [print(' ', k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'per_unit'}) for k, v in SEED.items()]
for key in ('first', 'all'):
    print('GKT CPU', key); [line(k, v) for k, v in GK[key].items() if not k.startswith('_')]
