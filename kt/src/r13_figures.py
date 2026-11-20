"""Release 3.10.0: Fig. 2 of the TLT article, the learning curve of r9_figures.py (Fig. 3 there) with the update-budget
runs at 25% and 100% of runs_r13 added to those of runs_r6 at 5% and 10%. Writes kt/results_r13/fig2_learning_curve.png."""
import os, json, glob, numpy as np, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); KT = os.path.dirname(HERE); OUT = f'{KT}/results_r13'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
D10 = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']

# ---------------- Fig. 3
df = pd.read_csv(f'{KT}/r2/results/r2_runs.csv')
def gains(a, b, met):
    x = df[df.cfg == a].set_index(['data', 'frac', 'sub', 'seed'])[met]; y = df[df.cfg == b].set_index(['data', 'frac', 'sub', 'seed'])[met]
    j = x.index.intersection(y.index); d = (x[j] - y[j]).reset_index(); d.columns = ['data', 'frac', 'sub', 'seed', 'd']
    ipk = df.drop_duplicates(['data', 'frac', 'sub']).set_index(['data', 'frac', 'sub']).ipk
    d['ipk'] = [ipk[(r.data, r.frac, r['sub'])] for _, r in d.iterrows()]
    return d[d.data.isin(D10)]
R6 = pd.DataFrame([json.load(open(f)) for f in glob.glob(f'{KT}/runs_r6/*.json')])
R6 = R6[(R6.budget == 'steps') & R6.cond.isin(['support', 'empty'])]
R13 = pd.DataFrame([json.load(open(f)) for f in glob.glob(f'{KT}/runs_r13/*.json')])
R13 = R13[R13.cond.isin(['support', 'empty']) & (R13.frac > 0.2) & (R13.rep == 0) & ((R13.lr - 0.002).abs() < 1e-9)]
R6 = pd.concat([R6, R13[[c for c in R6.columns if c in R13.columns]]], ignore_index=True)
R6['ipk'] = R6.n_train_inter / R6.K
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for k, (met, col6) in enumerate((('auc', 'test_auc'), ('new', 'test_auc_first'))):
    a_ = ax[k]; a_.axhline(0, color='0.5', lw=0.6)
    d = gains('P/support', 'P/empty', met)
    for c, g in d.groupby('data'):
        m = g.groupby('frac')[['ipk', 'd']].mean().sort_index(); a_.plot(m.ipk, m.d, color='#1f5fa8', lw=0.5, alpha=0.25)
    m = d.groupby(['frac', 'data'])[['ipk', 'd']].mean().groupby('frac').agg(['mean', 'sem'])
    x = np.exp(np.log(d.groupby(['frac', 'data']).ipk.mean()).groupby('frac').mean())
    a_.errorbar(x, m[('d', 'mean')], yerr=m[('d', 'sem')], color='#1f5fa8', marker='o', ms=3, lw=1.2, capsize=2, label='epoch budget (5%, 10%, 25%, 100%)')
    P = R6.pivot_table(index=['data', 'frac', 'sub'], columns='cond', values=col6).dropna()
    P['d'] = P['support'] - P['empty']
    ip = R6.drop_duplicates(['data', 'frac', 'sub']).set_index(['data', 'frac', 'sub']).ipk
    P['ipk'] = ip.reindex(P.index).values; P = P.reset_index()
    pc = P.groupby(['frac', 'data'])[['ipk', 'd']].mean()
    mm = pc.groupby('frac').d.agg(['mean', 'sem']); xx = np.exp(np.log(pc.ipk).groupby('frac').mean())
    a_.errorbar(xx * 1.04, mm['mean'], yerr=mm['sem'], color='#c4581b', marker='s', ms=3.5, lw=0, elinewidth=1.2, capsize=2,
                label='update budget (5%, 10%, 25%, 100%)')
    a_.set_xscale('log'); a_.set_xlabel('training interactions per KC (log scale)')
    a_.set_ylabel('AUC gain over empty-graph control' if k == 0 else 'first-attempt AUC gain')
    a_.set_title('(a) all predictions' if k == 0 else '(b) first attempt on a KC', loc='left', fontsize=8)
ax[0].legend(frameon=False, fontsize=6.5, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_learning_curve.png', dpi=300); plt.close(fig)

