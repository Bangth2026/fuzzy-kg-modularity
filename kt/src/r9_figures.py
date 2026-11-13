"""Figures of release 3.6.0.
Fig. 3: gain of the support graph over the empty-graph control against training interactions per KC,
        epoch budget (learning-curve runs, kt/r2/results/r2_runs.csv) and update budget (runs_r6, 5% and 10%).
Fig. 4: terms of the null-graph ladder with 95% mixed-model intervals for the probe (complete ladder,
        results_r9), SAKT-E and GKT (results_r8), first attempts and all predictions.
Writes kt/results_r9/fig3_learning_curve.png and fig4_ladder.png.
"""
import os, json, glob, numpy as np, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); KT = os.path.dirname(HERE); OUT = f'{KT}/results_r9'; os.makedirs(OUT, exist_ok=True)
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
                label='update budget (5%, 10%)')
    a_.set_xscale('log'); a_.set_xlabel('training interactions per KC (log scale)')
    a_.set_ylabel('AUC gain over empty-graph control' if k == 0 else 'first-attempt AUC gain')
    a_.set_title('(a) all predictions' if k == 0 else '(b) first attempt on a KC', loc='left', fontsize=8)
ax[0].legend(frameon=False, fontsize=6.5, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_learning_curve.png', dpi=300); plt.close(fig)

# ---------------- Fig. 4
r9p = f'{OUT}/r9_results.json'; r8p = f'{KT}/results_r8/r8_results.json'
if os.path.exists(r9p) and os.path.exists(r8p):
    r9 = json.load(open(r9p)); r8 = json.load(open(r8p))
    def r8get(a, b, key):
        for c in r8['comparisons']:
            if c['a'] == a and c['b'] == b and c['key'] == key: return c
    panels = [('Probe (FKG-KT)', [('U − ∅', 'U - empty (mixing)'), ('ER − U', 'ER - U (sparse mixing)'), ('DP − ER', 'DP - ER (degrees)'),
                                  ('G − DP', 'G - DP (edges)'), ('G − ∅', 'G - empty (total)')], 'r9'),
              ('SAKT-E', [('U − ∅', ('sakte_uniform', 'sakte_empty')), ('ER − U', ('sakte_er', 'sakte_uniform')),
                          ('G − ER', ('sakte_support', 'sakte_er')), ('G − ∅', ('sakte_support', 'sakte_empty'))], 'r8'),
              ('GKT', [('ER − ∅', ('gkt_er', 'gkt_empty')), ('DP − ER', ('gkt_rewired', 'gkt_er')), ('G − DP', ('gkt_support', 'gkt_rewired')),
                       ('G − ∅', ('gkt_support', 'gkt_empty'))], 'r8')]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=False)
    for ax_, (title, rows, src) in zip(axs, panels):
        for j, (key, colr, off) in enumerate((('first', '#1f5fa8', -0.15), ('all', '#9a9a9a', 0.15))):
            for i, (lab, ref) in enumerate(rows):
                s = r9['ladder_emb'][key][ref] if src == 'r9' else r8get(ref[0], ref[1], key)
                if not s: continue
                mx = s.get('mixed') or {}
                lo, hi = mx.get('lo', np.nan), mx.get('hi', np.nan)
                y = len(rows) - 1 - i + off
                ax_.plot([lo, hi], [y, y], color=colr, lw=1.2)
                ax_.plot([s['mean']], [y], marker='o' if key == 'first' else 's', color=colr, ms=3.5,
                         label=('first attempts' if key == 'first' else 'all predictions') if i == 0 else None)
        ax_.axvline(0, color='0.5', lw=0.6)
        ax_.set_yticks(range(len(rows))); ax_.set_yticklabels([r[0] for r in rows][::-1])
        ax_.set_title(title, loc='left', fontsize=8); ax_.set_xlabel('AUC difference')
        ax_.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); ax_.tick_params(axis='x', labelsize=6.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc='lower center', ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f'{OUT}/fig4_ladder.png', dpi=300); plt.close(fig)
print('figures written to', OUT)
