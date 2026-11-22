"""Release 3.12.0: analyses for the third revision of the TLT manuscript (no new training runs).

1. Entry point x edge term: (G - DP) in the state channel minus (G - DP) in the embedding refinement, paired on the
   32 units at 10% under the update budget that carry both ladders (runs_r9 for the embedding ladder, runs_r13 for the
   state channel); also the same contrast for the total gain (G - empty) and the KC-input term.
2. Every term with its 90% mixed-model interval, TOST p and a minimal-effect test against delta, for
   delta in {0.001, 0.002, 0.003}, on every ladder that can be recomputed from archived runs.
Writes results_r15/r15_results.json.
"""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import norm, wilcoxon
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
OUT = f'{REPO}/results_r15'; os.makedirs(OUT, exist_ok=True)
SMALL = ['statics11', 'algebra05', 'bridge06', 'assist17']
CORPORA = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
U10 = [(c, s) for c in SMALL for s in range(5)] + [(c, s) for c in CORPORA if c not in SMALL for s in range(2)]
DELTAS = (0.001, 0.002, 0.003)
rng = np.random.default_rng(2026)


def load(paths):
    R = {}
    for path in paths:
        for f in glob.glob(f'{path}/*.json'):
            r = json.load(open(f))
            if r.get('budget') != 'steps' or r.get('l2', 0) not in (0, 0.0, None): continue
            if abs(float(r.get('lr', 0.002)) - 0.002) > 1e-9 or int(r.get('rep', 0) or 0) != 0: continue
            if round(float(r.get('frac', 0.1)), 2) != 0.1: continue
            R.setdefault((r['data'], int(r['sub']), r['cond']), r)
    return R


# archived embedding ladder (as in r13_analyse.py: EdNet from the rebuilt data of release 3.7.0 where present)
REF = {}
for path, conds in ((f'{REPO}/runs_r9', {'empty', 'support_emb', 'er_emb', 'erR_emb', 'dp_emb'}),
                    (f'{REPO}/runs_r7', {'uniform_emb', 'empty', 'support'})):
    for k, v in load([path]).items():
        if k[2] in conds and k[0] != 'ednet': REF.setdefault(k, v)
for path in (f'{REPO}/runs_r10/repro', f'{REPO}/runs_r10/base', f'{REPO}/runs_r11/base', f'{REPO}/runs_r11', f'{REPO}/runs_r10',
             f'{REPO}/runs_r9', f'{REPO}/runs_r7'):
    for k, v in load([path]).items():
        if k[0] == 'ednet': REF.setdefault(k, v)
N13 = load([f'{REPO}/runs_r13'])
for k, v in N13.items():
    if k[0] == 'ednet' and k[2] in ('empty',): REF[k] = v
ALL = {**REF, **{k: v for k, v in N13.items() if k[2].startswith('st_')}}
R6 = load([f'{REPO}/runs_r6'])
for k, v in R6.items():
    if k[2] in ('support', 'empty'): ALL[(k[0], k[1], 'r6_' + k[2])] = v


def diff(a, b, col):
    rows = []
    for c, s in U10:
        if (c, s, a) in ALL and (c, s, b) in ALL:
            rows.append(dict(data=c, sub=s, d=ALL[(c, s, a)][col] - ALL[(c, s, b)][col]))
    return pd.DataFrame(rows)


def contrast(terms, col):
    """terms: list of (sign, a, b); returns per-unit sum of sign * (a - b) on units carrying every term."""
    dfs = [diff(a, b, col).set_index(['data', 'sub']).d * sg for sg, a, b in terms]
    d = pd.concat(dfs, axis=1).dropna().sum(axis=1)
    return d.reset_index().rename(columns={0: 'd'})


def mixed(df):
    import statsmodels.formula.api as smf
    r = smf.mixedlm('d ~ 1', df, groups=df['data']).fit(reml=True)
    est = float(r.params['Intercept']); se = float(r.bse['Intercept'])
    if not np.isfinite(se) or se <= 0:
        r = smf.ols('d ~ 1', df).fit(); est = float(r.params['Intercept']); se = float(r.bse['Intercept'])
    return est, se


def boot(df, B=10000):
    g = {c: v.d.values for c, v in df.groupby('data')}; cs = list(g); out = np.empty(B)
    for i in range(B):
        pick = rng.choice(len(cs), len(cs))
        out[i] = np.mean([rng.choice(g[cs[j]], len(g[cs[j]])).mean() for j in pick])
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def describe(df):
    per = df.groupby('data').d.mean(); est, se = mixed(df)
    o = dict(units=int(len(df)), corpora=int(len(per)), mean=float(per.mean()), pos=int((per > 0).sum()),
             mixed=est, se=se, ci95=[est - 1.96 * se, est + 1.96 * se], ci90=[est - 1.645 * se, est + 1.645 * se],
             p_two_sided=float(2 * norm.sf(abs(est) / se)), mde80=2.80 * se, boot95=boot(df),
             p_wilcoxon=float(wilcoxon(per.values).pvalue), per_corpus={k: float(v) for k, v in per.items()})
    o['by_delta'] = {str(dl): dict(tost_p=float(max(norm.sf((est + dl) / se), norm.cdf((est - dl) / se))),
                                   equivalent=bool(est - 1.645 * se > -dl and est + 1.645 * se < dl),
                                   exceeds=bool(est - 1.645 * se > dl),
                                   p_minimal_effect=float(norm.sf((abs(est) - dl) / se))) for dl in DELTAS}
    return o


res = {'units': len(U10)}
INTER = {
    'edges: (G - DP)_state - (G - DP)_emb': [(1, 'st_support', 'st_dp'), (-1, 'support_emb', 'dp_emb')],
    'degrees: (DP - ER)_state - (DP - ER)_emb': [(1, 'st_dp', 'st_er'), (-1, 'dp_emb', 'er_emb')],
    'KC input: (ER - U)_state - (ER - U)_emb': [(1, 'st_er', 'st_uniform'), (-1, 'er_emb', 'uniform_emb')],
    'total: (G - empty)_state - (G - empty)_emb': [(1, 'st_support', 'st_empty'), (-1, 'support_emb', 'empty')],
}
TERMS = {
    'both entry points G - empty (runs_r6)': [(1, 'r6_support', 'r6_empty')],
    'emb U - empty': [(1, 'uniform_emb', 'empty')], 'emb ER - U': [(1, 'er_emb', 'uniform_emb')],
    'emb DP - ER': [(1, 'dp_emb', 'er_emb')], 'emb G - DP': [(1, 'support_emb', 'dp_emb')],
    'emb G - empty': [(1, 'support_emb', 'empty')], 'emb ER - ER(R)': [(1, 'er_emb', 'erR_emb')],
    'state U - empty': [(1, 'st_uniform', 'st_empty')], 'state ER - U': [(1, 'st_er', 'st_uniform')],
    'state DP - ER': [(1, 'st_dp', 'st_er')], 'state G - DP': [(1, 'st_support', 'st_dp')],
    'state G - empty': [(1, 'st_support', 'st_empty')], 'state ER - empty': [(1, 'st_er', 'st_empty')],
}
for name, block in (('interaction', INTER), ('terms', TERMS)):
    res[name] = {lab: {k: describe(contrast(t, col)) for k, col in (('first', 'test_auc_first'), ('all', 'test_auc'))}
                 for lab, t in block.items()}
json.dump(res, open(f'{OUT}/r15_results.json', 'w'), indent=1)
for name in ('interaction', 'terms'):
    for lab, v in res[name].items():
        for k in ('first', 'all'):
            o = v[k]
            print(f"{name[:5]} {lab[:44]:44s} {k:5s} n={o['units']} mean={o['mean']:+.4f} ({o['pos']}/{o['corpora']}) "
                  f"mixed={o['mixed']:+.4f} 95%[{o['ci95'][0]:+.4f},{o['ci95'][1]:+.4f}] 90%[{o['ci90'][0]:+.4f},{o['ci90'][1]:+.4f}] "
                  f"p={o['p_two_sided']:.3f} boot[{o['boot95'][0]:+.4f},{o['boot95'][1]:+.4f}] "
                  + ' '.join(f"d{dl}:{'EQ' if b['equivalent'] else ('>d' if b['exceeds'] else '--')}" for dl, b in o['by_delta'].items()))
