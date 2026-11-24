"""Release 3.14.0: a descriptive look across the ten corpora (no new runs).

Per corpus: the first-attempt gain of the support graph at both entry points over the empty-graph control at 10% under
the update budget (mean over the subsamples of runs_r6 / runs_r16, as in r16_analyse.py), on the 32 units of Table II
and on the 50 units with five subsamples per corpus; mean and median over corpora. Corpus features from the run files
of the empty-graph control (subsample 0): number of KCs K, training interactions per KC at 10%, share of first attempts
among test predictions, edges of the support graph per KC; and, for the six corpora with curriculum units, the share of
support edges inside one unit (Section VI-H of the article). Spearman rank correlations of the gain with each feature
over the ten corpora (six for the unit share), with an exact permutation p-value; ten points, so descriptive only.
Writes results_r17/r17_between_corpora.json.
"""
import os, json, itertools, math
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
OUT = f'{REPO}/results_r17'; os.makedirs(OUT, exist_ok=True)
R = json.load(open(f'{REPO}/results_r16/r16_results.json'))['balanced_both_entry_points']
g32 = R['32 units (as Table II)']['first']['per_corpus']; g50 = R['50 units (five subsamples per corpus)']['first']['per_corpus']
a32 = R['32 units (as Table II)']['all']['per_corpus']; a50 = R['50 units (five subsamples per corpus)']['all']['per_corpus']
UNIT_SHARE = {'junyi': 0.69, 'bridge06': 0.89, 'algebra05': 0.36, 'statics11': 0.94, 'ednet': 0.82, 'eedi': 0.08}
rows = []
for c in g50:
    j = json.load(open(f'{REPO}/runs_r6/{c}__f10__s0__empty__steps.json'))
    s = json.load(open(f'{REPO}/runs_r6/{c}__f10__s0__support__steps.json'))
    rows.append(dict(data=c, gain_first_32=g32[c], gain_first_50=g50[c], gain_all_32=a32[c], gain_all_50=a50[c], K=j['K'],
                     inter_per_kc=j['n_train_inter'] / j['K'], first_share=j['n_first'] / j['n_pred'],
                     edges_per_kc=s['edges'] / j['K'], unit_share=UNIT_SHARE.get(c, np.nan)))
df = pd.DataFrame(rows).sort_values('gain_first_50', ascending=False)


def perm_p(x, y):
    x, y = np.asarray(x), np.asarray(y); r0 = abs(spearmanr(x, y).correlation); n = len(x)
    if n <= 8:
        perms = itertools.permutations(range(n)); tot = math.factorial(n)
    else:
        rng = np.random.default_rng(2026); perms = (rng.permutation(n) for _ in range(20000)); tot = 20000
    hit = sum(abs(spearmanr(x, y[list(p)]).correlation) >= r0 - 1e-12 for p in perms)
    return hit / tot


out = dict(per_corpus=df.round(5).to_dict(orient='records'), summary={}, spearman={})
for k in ('gain_first_32', 'gain_first_50', 'gain_all_32', 'gain_all_50'):
    v = df[k].values
    out['summary'][k] = dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                             positive=int((v > 0).sum()), above_mean=int((v > v.mean()).sum()))
for feat in ('K', 'inter_per_kc', 'first_share', 'edges_per_kc', 'unit_share'):
    d = df.dropna(subset=[feat])
    for k in ('gain_first_32', 'gain_first_50'):
        out['spearman'][f'{k}~{feat}'] = dict(n=int(len(d)), rho=float(spearmanr(d[feat], d[k]).correlation), p_perm=float(perm_p(d[feat].values, d[k].values)))
json.dump(out, open(f'{OUT}/r17_between_corpora.json', 'w'), indent=1)
print(df.round(4).to_string(index=False))
print(json.dumps(out['summary'], indent=0))
for k, v in out['spearman'].items(): print(f"{k:32s} n={v['n']} rho={v['rho']:+.2f} p={v['p_perm']:.3f}")
