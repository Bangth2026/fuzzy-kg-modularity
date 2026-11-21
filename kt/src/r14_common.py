"""Shared loaders and summaries for r14_analyse.py (release 3.11.0).

load_runs, pairs and the reference set of archived runs at 10% follow r13_analyse.py exactly, so a comparison made
here on archived runs reproduces the value reported in the earlier releases. Added: 90% mixed-model intervals,
two one-sided tests against the smallest effect of interest DELTA, the minimal detectable effect at 80% power, and a
corpus-then-subsample bootstrap for every comparison.
"""
import os, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, norm
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
SMALL = ['statics11', 'algebra05', 'bridge06', 'assist17']
CORPORA = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
U10 = [(c, s) for c in SMALL for s in range(5)] + [(c, s) for c in CORPORA if c not in SMALL for s in range(2)]
U25 = [(c, s) for c in CORPORA for s in range(3)]; U100 = [(c, 0) for c in CORPORA]
DELTA = 0.002   # smallest effect of interest, AUC (fixed before any equivalence test was computed; Section V-E)


def load_runs(paths):
    R = {}
    for path in paths:
        for f in glob.glob(f'{path}/*.json'):
            r = json.load(open(f))
            if r.get('budget') != 'steps' or r.get('l2', 0) not in (0, 0.0, None): continue
            if int(r.get('maxlen', 200) or 200) != 200: continue
            key = (r['data'], round(float(r.get('frac', 0.1)), 2), int(r['sub']), r['cond'], int(r.get('rep', 0) or 0),
                   round(float(r.get('lr', 0.002)), 4))
            R[key] = (r, f[:-5])
    return R


def load_len(paths, n):
    R = {}
    for path in paths:
        for f in glob.glob(f'{path}/*.json'):
            r = json.load(open(f))
            if int(r.get('maxlen', 200) or 200) != n: continue
            R[(r['data'], round(float(r['frac']), 2), int(r['sub']), r['cond'], int(r.get('rep', 0) or 0), 0.002)] = (r, f[:-5])
    return R


def reference_runs():
    """The archived runs at 10% used by Tables III and IV (as r13_analyse.py), plus runs_r13 at 10%."""
    ARCH = {}
    for path, conds in ((f'{REPO}/runs_r9', {'empty', 'support_emb', 'er_emb', 'erR_emb', 'dp_emb'}), (f'{REPO}/runs_r7', {'uniform_emb', 'empty', 'support'}),
                        (f'{REPO}/runs_r10', {'rvec_emb', 'rvec1_emb', 'erE2_emb'}), (f'{REPO}/runs_r11', {'rvecF_emb', 'rvec1F_emb'})):
        for k, v in load_runs([path]).items():
            if k[3] in conds and k[0] != 'ednet' and k[1] == 0.1: ARCH.setdefault(k, v)
    for path in (f'{REPO}/runs_r10/repro', f'{REPO}/runs_r10/base', f'{REPO}/runs_r11/base', f'{REPO}/runs_r11', f'{REPO}/runs_r10'):
        for k, v in load_runs([path]).items():
            if k[0] == 'ednet' and k[1] == 0.1: ARCH.setdefault(k, v)
    N13 = load_runs([f'{REPO}/runs_r13'])
    REF = dict(ARCH)
    REF.update({k: v for k, v in N13.items() if k[0] == 'ednet' and k[1] == 0.1})
    REF.update({k: v for k, v in N13.items() if k[1] == 0.1})
    return ARCH, N13, REF


def ladder_runs():
    """The runs of the embedding-only ladder of Table III (upper and frozen-table blocks): runs_r9 on all ten corpora
    (EdNet on the data of release 3.6.0, as in r9_analyse.py) and uniform mixing from runs_r7."""
    L = {}
    for k, v in load_runs([f'{REPO}/runs_r7']).items():
        if k[3] in ('uniform_emb',) and k[1] == 0.1: L[k] = v
    for k, v in load_runs([f'{REPO}/runs_r9']).items():
        if k[1] == 0.1: L[k] = v
    return L


def pairs(R, units, a, b, frac=0.1, col='test_auc_first', rep_a=0, rep_b=0, lr=0.002):
    rows = []
    for c, s in units:
        ka = (c, frac, s, a, rep_a, lr); kb = (c, frac, s, b, rep_b, lr)
        if ka in R and kb in R and R[ka][0][col] is not None and R[kb][0][col] is not None:
            rows.append(dict(data=c, sub=s, d=R[ka][0][col] - R[kb][0][col]))
    return rows


def mixed(df):
    """Random intercept per corpus; returns estimate, SE and 95%/90% intervals (OLS on corpus means if singular)."""
    import statsmodels.formula.api as smf
    if df.data.nunique() < 2: return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _mixed(df, smf)


def _mixed(df, smf):
    try:
        if df.groupby('data').size().max() == 1: raise ValueError('one unit per corpus')
        r = smf.mixedlm('d ~ 1', df, groups=df['data']).fit(reml=True)
        est, se = float(r.params['Intercept']), float(r.bse['Intercept'])
        if not (np.isfinite(se) and se > 0): raise ValueError('singular')
        kind = 'mixed'
    except Exception:
        r = smf.ols('d ~ 1', df).fit(); est, se = float(r.params['Intercept']), float(r.bse['Intercept']); kind = 'ols'
    z95, z90 = norm.ppf(.975), norm.ppf(.95)
    p_lo = 1 - norm.cdf((est + DELTA) / se); p_hi = norm.cdf((est - DELTA) / se)
    return dict(est=est, se=se, lo=est - z95 * se, hi=est + z95 * se, lo90=est - z90 * se, hi90=est + z90 * se,
                p_tost=float(max(p_lo, p_hi)), equivalent=bool(max(p_lo, p_hi) < 0.05),
                mde80=float((z95 + norm.ppf(.8)) * se), kind=kind)


def boot(df, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    g = {c: v.d.values for c, v in df.groupby('data')}; cs = list(g); out = []
    for _ in range(B):
        pick = rng.choice(len(cs), len(cs))
        out.append(np.mean([rng.choice(g[cs[i]], len(g[cs[i]])).mean() for i in pick]))
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def summ(rows, do_boot=True):
    if not rows: return None
    df = pd.DataFrame(rows).dropna(subset=['d'])
    if df.empty: return None
    per = df.groupby('data').d.mean()
    o = dict(units=int(len(df)), n=int(len(per)), mean=float(per.mean()), pos=int((per > 0).sum()),
             per_corpus={k: float(v) for k, v in per.items()})
    if len(per) >= 4: o['p_wilcoxon'] = float(wilcoxon(per.values).pvalue)
    o['mixed'] = mixed(df)
    if do_boot and len(per) >= 3: o['boot'] = boot(df)
    return o


def both(R, units, a, b, **kw):
    return {'first': summ(pairs(R, units, a, b, col='test_auc_first', **kw)), 'all': summ(pairs(R, units, a, b, col='test_auc', **kw))}


def holm(ps):
    k = [i for i, p in enumerate(ps) if p is not None]; order = sorted(k, key=lambda i: ps[i]); out = [None] * len(ps); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(k) - r) * ps[i])); out[i] = run
    return out
