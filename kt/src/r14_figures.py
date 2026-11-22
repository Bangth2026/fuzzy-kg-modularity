"""Release 3.11.0 (adapted from r13_figures.py): Figs. 2-4 of the revised TLT article. Fig. 2: the 100% update-budget point averages three training seeds (runs_r14). Fig. 3: panel (d) is the complete GKT ladder on a CPU. Fig. 4: effect sizes on one scale.
Original note: Fig. 2 of the TLT article, the learning curve of r9_figures.py (Fig. 3 there) with the update-budget
runs at 25% and 100% of runs_r13 added to those of runs_r6 at 5% and 10%. Writes kt/results_r13/fig2_learning_curve.png."""
import os, json, glob, numpy as np, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); KT = os.path.dirname(HERE); OUT = f'{KT}/results_r14'; os.makedirs(OUT, exist_ok=True)
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
R13 = pd.DataFrame([json.load(open(f)) for f in glob.glob(f'{KT}/runs_r13/*.json')] + [json.load(open(f)) for f in glob.glob(f'{KT}/runs_r14/*.json')])
R13 = R13[R13.cond.isin(['support', 'empty']) & (R13.frac > 0.2) & ((R13.lr - 0.002).abs() < 1e-9) & (R13.maxlen.fillna(200) == 200)]
R13 = R13[(R13.rep == 0) | (R13.frac == 1.0)]
R13['sub'] = R13['sub'] + 10 * R13['rep']   # at 100%, the three seeds enter as three units of the corpus
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



# ---------------- Fig. 3 of the TLT article: the ladder for every entry point
# Panels: probe with the graph in the embedding refinement (as results_r10/fig4_ladder.png), probe with the graph in the
# state-routing channel (runs_r13), SAKT-E (results_r8), GKT (results_r8 on a GPU, plus the CPU rows with the complete graph
# from runs_r13). Writes kt/results_r13/fig3_ladder.png.
import numpy as np, matplotlib.ticker
r9 = json.load(open(f'{KT}/results_r9/r9_results.json')); r8 = json.load(open(f'{KT}/results_r8/r8_results.json'))
r10 = json.load(open(f'{KT}/results_r10/r10_results.json'))['controls_without_graph']
r13 = json.load(open(f'{KT}/results_r13/r13_results.json')); r14 = json.load(open(f'{KT}/results_r14/r14_results.json'))
def r8get(a, b, key):
    for c in r8['comparisons']:
        if c['a'] == a and c['b'] == b and c['key'] == key: return c
def get(src, ref, key):
    if src == 'emb':
        if ref.startswith('FR:'): return r9['frozen_random_table'][key][ref[3:]]
        if ref.startswith('NG:'): return r10[key][ref[3:]]
        return r9['ladder_emb'][key][ref]
    if src == 'state': return r13['state_channel'][ref][key]
    if src == 'cpu': return r14['gkt_cpu_ladder'][ref][key]
    return r8get(ref[0], ref[1], key)
panels = [('(a) Probe, embedding refinement', [('U − ∅', 'emb', 'U - empty (mixing)'), ('ER − U', 'emb', 'ER - U (sparse mixing)'),
                                              ('ER(R) − U', 'emb', 'FR:ER(frozen R) - U'), ('RV − U', 'emb', 'NG:rvec_emb - uniform_emb'),
                                              ('DP − ER', 'emb', 'DP - ER (degrees)'), ('G − DP', 'emb', 'G - DP (edges)'),
                                              ('G − ∅', 'emb', 'G - empty (total)')]),
          ('(b) Probe, state channel', [('U − ∅', 'state', 'st_uniform - st_empty'), ('ER − U', 'state', 'st_er - st_uniform'),
                                        ('DP − ER', 'state', 'st_dp - st_er'), ('G − DP', 'state', 'st_support - st_dp'),
                                        ('G − ∅', 'state', 'st_support - st_empty')]),
          ('(c) SAKT-E', [('U − ∅', 'r8', ('sakte_uniform', 'sakte_empty')), ('ER − U', 'r8', ('sakte_er', 'sakte_uniform')),
                          ('G − ER', 'r8', ('sakte_support', 'sakte_er')), ('G − ∅', 'r8', ('sakte_support', 'sakte_empty'))]),
          ('(d) GKT, one device (CPU)', [('U − ∅', 'cpu', 'gkt_uniform - gkt_empty'), ('ER − ∅', 'cpu', 'gkt_er - gkt_empty'),
                       ('DP − ER', 'cpu', 'gkt_rewired - gkt_er'), ('G − DP', 'cpu', 'gkt_support - gkt_rewired'),
                       ('G − ∅', 'cpu', 'gkt_support - gkt_empty'), ('G − U', 'cpu', 'gkt_support - gkt_uniform')])]
fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.2))
for ax_, (title, rows) in zip(axs.ravel(), panels):
    for key, colr, off in (('first', '#1f5fa8', -0.15), ('all', '#9a9a9a', 0.15)):
        for i, (lab, src, ref) in enumerate(rows):
            s = get(src, ref, key)
            if not s: continue
            mx = s.get('mixed') or {}
            y = len(rows) - 1 - i + off
            ax_.plot([mx.get('lo', np.nan), mx.get('hi', np.nan)], [y, y], color=colr, lw=1.2)
            ax_.plot([s['mean']], [y], marker='o' if key == 'first' else 's', color=colr, ms=3.5,
                     label=('first attempts' if key == 'first' else 'all predictions') if i == 0 else None)
    ax_.axvline(0, color='0.5', lw=0.6)
    ax_.set_yticks(range(len(rows))); ax_.set_yticklabels([r[0] for r in rows][::-1])
    ax_.set_title(title, loc='left', fontsize=8); ax_.set_xlabel('AUC difference')
    ax_.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5)); ax_.tick_params(axis='x', labelsize=6.5)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, loc='lower center', ncol=2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{OUT}/fig3_ladder.png', dpi=300); plt.close(fig)

# ---------------- Fig. 4: effect sizes on one scale (first attempts unless marked)
fx = r14['effect_scale_first']
rows = [('architecture: probe − DKT†', 'architecture: empty-graph probe - DKT (all predictions, sample 0, epoch budget; Table S7.3)'),
        ('budget in updates, 4 small corpora', 'budget counted in updates, four small corpora (control)'),
        ('budget in updates, 6 larger corpora', 'budget counted in updates, six larger corpora (control)'),
        ('trainable second KC table', 'trainable second KC table'), ('twice the width (d = 128)', 'twice the width (d = 128)'),
        ('random code per KC (RV − ∅)', 'random code per KC (RV - empty)'),
        ('graph, both entry points', 'graph, both entry points (G - empty)'), ('graph, embedding refinement', 'graph, embedding refinement (G - empty)'),
        ('graph, state channel', 'state channel (G - empty)'), ('edges beyond DP, embedding', 'edges beyond rewiring, embedding (G - DP)'),
        ('edges beyond DP, state channel', 'edges beyond rewiring, state channel (G - DP)')]
rep = r14['replication']
fig, ax = plt.subplots(figsize=(3.45, 3.5))
n = len(rows)
for i, (lab, key) in enumerate(rows):
    v = fx[key]; y = n - 1 - i
    if v.get('lo') is not None: ax.plot([v['lo'], v['hi']], [y, y], color='#1f5fa8', lw=1.2)
    ax.plot([v['mean']], [y], 'o', color='#1f5fa8' if '†' not in lab else '#9a9a9a', ms=3.5)
for lab, k, y in (('replication: G − ∅', 'H1 support - empty', None),):
    pass
sd = fx['seed SD of G - empty (first attempts)']['mean']
ax.axvspan(-sd, sd, color='#c4581b', alpha=0.12, lw=0, label='± seed SD of the graph gain')
ax.axvline(0, color='0.5', lw=0.6)
ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.5)
ax.set_xlabel('AUC difference', fontsize=7); ax.tick_params(axis='x', labelsize=6.5)
ax.legend(frameon=False, fontsize=6, loc='lower center', bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_effect_scale.png', dpi=300); plt.close(fig)
print('Figs. 2-4 written to', OUT)
