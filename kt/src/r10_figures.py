"""Figure 4 of release 3.7.0: terms of the null-graph ladder with 95% mixed-model intervals, as in
r9_figures.py, with the frozen-random-table rung ER(R) - U and the
random-vector rung without a graph RV - U (results_r10/r10_results.json) added to the probe panel next to ER - U.
Reads results_r9/r9_results.json and results_r8/r8_results.json; writes results_r10/fig4_ladder.png.
"""
import os, json, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); KT = os.path.dirname(HERE)
OUT = f'{KT}/results_r10'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})

# ---------------- Fig. 4
r9p = f'{KT}/results_r9/r9_results.json'; r8p = f'{KT}/results_r8/r8_results.json'
if os.path.exists(r9p) and os.path.exists(r8p):
    r9 = json.load(open(r9p)); r8 = json.load(open(r8p))
    r10 = json.load(open(f'{OUT}/r10_results.json'))['controls_without_graph']
    def r8get(a, b, key):
        for c in r8['comparisons']:
            if c['a'] == a and c['b'] == b and c['key'] == key: return c
    panels = [('Probe (FKG-KT)', [('U − ∅', 'U - empty (mixing)'), ('ER − U', 'ER - U (sparse mixing)'), ('ER(R) − U', 'FR:ER(frozen R) - U'), ('RV − U', 'NG:rvec_emb - uniform_emb'), ('DP − ER', 'DP - ER (degrees)'),
                                  ('G − DP', 'G - DP (edges)'), ('G − ∅', 'G - empty (total)')], 'r9'),
              ('SAKT-E', [('U − ∅', ('sakte_uniform', 'sakte_empty')), ('ER − U', ('sakte_er', 'sakte_uniform')),
                          ('G − ER', ('sakte_support', 'sakte_er')), ('G − ∅', ('sakte_support', 'sakte_empty'))], 'r8'),
              ('GKT', [('ER − ∅', ('gkt_er', 'gkt_empty')), ('DP − ER', ('gkt_rewired', 'gkt_er')), ('G − DP', ('gkt_support', 'gkt_rewired')),
                       ('G − ∅', ('gkt_support', 'gkt_empty'))], 'r8')]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=False)
    for ax_, (title, rows, src) in zip(axs, panels):
        for j, (key, colr, off) in enumerate((('first', '#1f5fa8', -0.15), ('all', '#9a9a9a', 0.15))):
            for i, (lab, ref) in enumerate(rows):
                s = ((r9['frozen_random_table'][key][ref[3:]] if ref.startswith('FR:') else
                      r10[key][ref[3:]] if ref.startswith('NG:') else r9['ladder_emb'][key][ref])
                     if src == 'r9' else r8get(ref[0], ref[1], key))
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
print('figure written to', OUT)
