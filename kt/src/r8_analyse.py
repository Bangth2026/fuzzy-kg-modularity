"""Analysis of the r8 runs: ladder terms per family, with the statistics of the manuscript.

For every comparison a - b and both prediction sets (all, first attempts): the subsample-level
differences are averaged within corpus; the summary is the mean over corpora, the number of corpora
with a positive difference, a Wilcoxon signed-rank test over corpora, and a 95% interval from a
mixed model with a random intercept per corpus on the subsample-level differences.
Also: reproduction check of the probe runs against the archived release-3.4.0 runs (runs_r6, runs_r7).
Writes results_r8/r8_results.json and prints a summary table.
"""
import os, sys, json, glob, numpy as np, pandas as pd, warnings
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
RUNS = os.environ.get('R8_RUNS', f'{ROOT}/runs_r8'); OUT = f'{ROOT}/results_r8'; os.makedirs(OUT, exist_ok=True)

COMPARISONS = {
    'probe': [('uniform_emb', 'empty'), ('er_emb', 'uniform_emb'), ('support_emb', 'er_emb'), ('support_emb', 'empty'),
              ('er_emb', 'empty'), ('hash', 'empty'), ('er_emb', 'hash'), ('support_emb', 'hash')],
    'sakte': [('sakte_uniform', 'sakte_empty'), ('sakte_er', 'sakte_uniform'), ('sakte_support', 'sakte_er'),
              ('sakte_support', 'sakte_empty'), ('sakte_er', 'sakte_empty')],
    'gkt': [('gkt_er', 'gkt_empty'), ('gkt_rewired', 'gkt_er'), ('gkt_support', 'gkt_rewired'), ('gkt_support', 'gkt_er'),
            ('gkt_support', 'gkt_empty'), ('gkt_rewired', 'gkt_empty')],
}


def load_runs(path):
    rows = [json.load(open(f)) for f in glob.glob(f'{path}/*.json')]
    return pd.DataFrame(rows)


def mixed_ci(df):
    try:
        import statsmodels.formula.api as smf
        if df.data.nunique() < 3: return None
        r = smf.mixedlm('d ~ 1', df, groups=df['data']).fit(reml=True)
        lo, hi = r.conf_int().loc['Intercept']
        return dict(est=float(r.params['Intercept']), lo=float(lo), hi=float(hi), p=float(r.pvalues['Intercept']))
    except Exception as e:
        return dict(error=str(e)[:120])


def compare(R, a, b, key):
    col = 'test_auc' if key == 'all' else 'test_auc_first'
    A = R[R.cond == a].set_index(['data', 'sub'])[col]; B = R[R.cond == b].set_index(['data', 'sub'])[col]
    j = pd.concat([A.rename('a'), B.rename('b')], axis=1).dropna()
    if j.empty: return None
    j['d'] = j.a - j.b; j = j.reset_index()
    per = j.groupby('data').d.mean()
    out = dict(a=a, b=b, key=key, units=int(len(j)), n=int(len(per)), mean=float(per.mean()), pos=int((per > 0).sum()),
               per_corpus={k: float(v) for k, v in per.items()})
    if len(per) >= 4 and (per != 0).any():
        out['p'] = float(wilcoxon(per.values).pvalue)
    out['mixed'] = mixed_ci(j)
    return out


def main():
    R = load_runs(RUNS)
    if R.empty: print('no runs in', RUNS); return
    res = {'n_runs': int(len(R)), 'runs_by_cond': R.groupby('cond').size().to_dict(), 'comparisons': []}
    for fam, comps in COMPARISONS.items():
        for a, b in comps:
            for key in ('first', 'all'):
                c = compare(R, a, b, key)
                if c: c['family'] = fam; res['comparisons'].append(c)
    # reproduction check against the archived probe runs of release 3.4.0
    rep = []
    for cond, arch in [('empty', 'runs_r7'), ('uniform_emb', 'runs_r7'), ('support_emb', 'runs_r6'), ('er_emb', 'runs_r6')]:
        for _, r in R[R.cond == cond].iterrows():
            f = f"{ROOT}/{arch}/{r['data']}__f10__s{r['sub']}__{cond}__steps.json"
            if os.path.exists(f):
                o = json.load(open(f)); rep.append(dict(cond=cond, data=r['data'], sub=int(r['sub']),
                                                        d_all=r['test_auc'] - o['test_auc'],
                                                        d_first=(r['test_auc_first'] or 0) - (o['test_auc_first'] or 0)))
    if rep:
        rp = pd.DataFrame(rep)
        res['reproduction'] = dict(n=int(len(rp)), max_abs_all=float(rp.d_all.abs().max()), max_abs_first=float(rp.d_first.abs().max()),
                                   mean_abs_first=float(rp.d_first.abs().mean()))
    json.dump(res, open(f'{OUT}/r8_results.json', 'w'), indent=1)
    print(f"{'family':6} {'comparison':32} {'pred':5} {'mean':>8} {'pos':>6} {'p':>6}  interval")
    for c in res['comparisons']:
        mx = c.get('mixed') or {}
        iv = f"{mx['lo']:+.4f} to {mx['hi']:+.4f}" if 'lo' in mx else '-'
        print(f"{c['family']:6} {c['a']+' - '+c['b']:32} {c['key']:5} {c['mean']:+.4f} {c['pos']:>3}/{c['n']:<2} {c.get('p', float('nan')):6.3f}  {iv}")
    if 'reproduction' in res: print('reproduction vs release 3.4.0:', res['reproduction'])


if __name__ == '__main__': main()
