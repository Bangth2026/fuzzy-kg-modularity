"""Release 3.12.0: the update-budget gain at 10% (Table II, lower block) with subsamples 0 and 1 only on every corpus."""
import os, json, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
R = {}
for f in glob.glob(f'{REPO}/runs_r6/*__steps.json'):
    r = json.load(open(f))
    if r.get('budget') == 'steps' and round(float(r['frac']), 2) == 0.1 and r['cond'] in ('support', 'empty') and int(r.get('rep', 0) or 0) == 0:
        R[(r['data'], int(r['sub']), r['cond'])] = r
out = {}
for col in ('test_auc_first', 'test_auc'):
    rows = [dict(data=c, sub=s, d=R[(c, s, 'support')][col] - R[(c, s, 'empty')][col]) for (c, s, k) in R if k == 'support' and s < 2 and (c, s, 'empty') in R]
    df = pd.DataFrame(rows); per = df.groupby('data').d.mean()
    m = smf.mixedlm('d ~ 1', df, groups=df['data']).fit(reml=True); lo, hi = m.conf_int().loc['Intercept']
    out[col] = dict(units=len(df), mean=float(per.mean()), pos=int((per > 0).sum()), n=int(len(per)), p_wilcoxon=float(wilcoxon(per).pvalue), ci95=[float(lo), float(hi)])
    print(col, out[col])
json.dump(out, open(f'{REPO}/results_r15/r15_subsample2.json', 'w'), indent=1)
