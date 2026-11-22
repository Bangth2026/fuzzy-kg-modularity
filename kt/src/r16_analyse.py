"""Release 3.13.0: analyses of the runs added for the fourth version of the TLT manuscript.

1. Balanced subsamples (M3 of the third review): the support graph at both entry points against the empty-graph
   control at 10% under the update budget, with five subsamples on every corpus (runs_r6 for subsamples 0 to 4 of the
   four small corpora and 0 to 1 of the others, runs_r16 for subsamples 2 to 4 of the six larger corpora), 50 units.
   Mean of corpus means, mixed model (random intercept per corpus), corpus-then-subsample bootstrap, Wilcoxon over
   corpora, leave-one-corpus-out range, and the same with subsamples 0 and 1 only (20 units) for reference.
2. Other graph constructions (M4): the DKT conditional-influence graph and the co-performance graph (r16_controls.py)
   on the 32 units of the ladders, in the embedding refinement and in the state channel, against the empty graph,
   the Erdos-Renyi graph of the same size, the log-built support graph and (DKT graph) its own degree-preserving
   rewiring. Archived conditions as in r15_analyse.py.
3. Junyi with five subsamples: the published prerequisite map against the log-built graph and the empty graph, in
   the embedding refinement and the state channel (subsamples 0 and 1 from runs_r9, runs_r13, runs_r14; 2 to 4 from
   runs_r16).
Every term: classes at delta = 0.001, 0.002, 0.003 from the 90% mixed-model interval (B+/B-: beyond delta with its
sign; N: nonzero, not beyond; S: nonzero inside +-delta; E: equivalent to zero; U: unresolved).
Writes results_r16/r16_results.json.
"""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import norm, wilcoxon, t as tdist
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
OUT = f'{REPO}/results_r16'; os.makedirs(OUT, exist_ok=True)
SMALL = ['statics11', 'algebra05', 'bridge06', 'assist17']
CORPORA = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
LARGE = [c for c in CORPORA if c not in SMALL]
U10 = [(c, s) for c in SMALL for s in range(5)] + [(c, s) for c in LARGE for s in range(2)]
U50 = [(c, s) for c in CORPORA for s in range(5)]
DELTAS = (0.001, 0.002, 0.003)
rng = np.random.default_rng(2026)


def load(paths):
    R = {}
    for path in paths:
        for f in glob.glob(f'{path}/*.json'):
            r = json.load(open(f))
            if r.get('budget') != 'steps' or r.get('l2', 0) not in (0, 0.0, None): continue
            if abs(float(r.get('lr', 0.002)) - 0.002) > 1e-9 or int(r.get('rep', 0) or 0) != 0: continue
            if round(float(r.get('frac', 0.1)), 2) != 0.1 or int(r.get('maxlen', 200) or 200) != 200: continue
            R.setdefault((r['data'], int(r['sub']), r['cond']), r)
    return R


# archived ladders, exactly as r15_analyse.py
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
N14 = load([f'{REPO}/runs_r14'])
for k, v in N14.items():
    if k[2] in ('map_emb', 'st_map'): ALL.setdefault(k, v)
N16 = load([f'{REPO}/runs_r16'])
for k, v in N16.items():
    if k[2] in ('dkt_emb', 'dpdkt_emb', 'corr_emb', 'st_dkt', 'st_dpdkt', 'st_corr'): ALL[k] = v
    if k[0] == 'junyi' and k[1] >= 2 and k[2] in ('support_emb', 'map_emb', 'st_empty', 'st_support', 'st_map', 'empty'): ALL[k] = v
# both entry points (primary comparison under the update budget): runs_r6, plus runs_r16 for subsamples 2-4 of the larger corpora
R6 = load([f'{REPO}/runs_r6'])
for k, v in R6.items():
    if k[2] in ('support', 'empty'): ALL[(k[0], k[1], 'r6_' + k[2])] = v
for k, v in N16.items():
    if k[2] in ('support', 'empty') and k[0] in LARGE and k[1] >= 2: ALL[(k[0], k[1], 'r6_' + k[2])] = v


def diff(a, b, col, units):
    rows = []
    for c, s in units:
        if (c, s, a) in ALL and (c, s, b) in ALL:
            rows.append(dict(data=c, sub=s, d=ALL[(c, s, a)][col] - ALL[(c, s, b)][col]))
    return pd.DataFrame(rows)


def mixed(df):
    import statsmodels.formula.api as smf
    if df.data.nunique() < 2:
        se = df.d.std(ddof=1) / np.sqrt(len(df)); return float(df.d.mean()), float(se), len(df) - 1
    r = smf.mixedlm('d ~ 1', df, groups=df['data']).fit(reml=True)
    est = float(r.params['Intercept']); se = float(r.bse['Intercept'])
    if not np.isfinite(se) or se <= 0:
        r = smf.ols('d ~ 1', df).fit(); est = float(r.params['Intercept']); se = float(r.bse['Intercept'])
    return est, se, None


def boot(df, B=10000):
    g = {c: v.d.values for c, v in df.groupby('data')}; cs = list(g); out = np.empty(B)
    for i in range(B):
        pick = rng.choice(len(cs), len(cs))
        out[i] = np.mean([rng.choice(g[cs[j]], len(g[cs[j]])).mean() for j in pick])
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def klass(lo, hi, dl):
    if lo > dl: return 'B+'
    if hi < -dl: return 'B-'
    if -dl < lo and hi < dl: return 'S' if (lo > 0 or hi < 0) else 'E'
    if lo > 0 or hi < 0: return 'N'
    return 'U'


def describe(df):
    if len(df) == 0: return dict(units=0)
    per = df.groupby('data').d.mean(); est, se, dof = mixed(df)
    z95, z90 = (1.96, 1.645) if dof is None else (tdist.ppf(.975, dof), tdist.ppf(.95, dof))
    o = dict(units=int(len(df)), corpora=int(len(per)), mean=float(per.mean()), pos=int((per > 0).sum()),
             mixed=est, se=se, ci95=[est - z95 * se, est + z95 * se], ci90=[est - z90 * se, est + z90 * se],
             mde80=2.80 * se, per_corpus={k: float(v) for k, v in per.items()})
    if len(per) >= 5:
        o['boot95'] = boot(df); o['p_wilcoxon'] = float(wilcoxon(per.values).pvalue)
        o['loco'] = [float(per.drop(c).mean()) for c in per.index]
    else:
        o['per_unit'] = df.d.round(5).tolist(); o['units_pos'] = int((df.d > 0).sum())
    o['class'] = '/'.join(klass(o['ci90'][0], o['ci90'][1], dl) for dl in DELTAS)
    return o


def both(a, b, units):
    return {k: describe(diff(a, b, col, units)) for k, col in (('first', 'test_auc_first'), ('all', 'test_auc'))}


res = {}
# 1. balanced subsamples
res['balanced_both_entry_points'] = {'50 units (five subsamples per corpus)': both('r6_support', 'r6_empty', U50),
                                     '32 units (as Table II)': both('r6_support', 'r6_empty', U10),
                                     '20 units (subsamples 0 and 1)': both('r6_support', 'r6_empty', [(c, s) for c in CORPORA for s in (0, 1)]),
                                     '30 units, larger corpora only, five subsamples': both('r6_support', 'r6_empty', [(c, s) for c in LARGE for s in range(5)])}
# 2. other constructions
OTHER = {
    'emb DKT - empty': ('dkt_emb', 'empty'), 'emb DKT - ER': ('dkt_emb', 'er_emb'), 'emb DKT - DP(DKT)': ('dkt_emb', 'dpdkt_emb'),
    'emb DP(DKT) - ER': ('dpdkt_emb', 'er_emb'), 'emb DKT - G': ('dkt_emb', 'support_emb'),
    'emb corr - empty': ('corr_emb', 'empty'), 'emb corr - ER': ('corr_emb', 'er_emb'), 'emb corr - G': ('corr_emb', 'support_emb'),
    'emb G - empty': ('support_emb', 'empty'), 'emb G - ER': ('support_emb', 'er_emb'), 'emb G - DP': ('support_emb', 'dp_emb'),
    'state DKT - empty': ('st_dkt', 'st_empty'), 'state DKT - ER': ('st_dkt', 'st_er'), 'state DKT - DP(DKT)': ('st_dkt', 'st_dpdkt'),
    'state DP(DKT) - ER': ('st_dpdkt', 'st_er'), 'state DKT - G': ('st_dkt', 'st_support'),
    'state corr - empty': ('st_corr', 'st_empty'), 'state corr - ER': ('st_corr', 'st_er'), 'state corr - G': ('st_corr', 'st_support'),
    'state G - empty': ('st_support', 'st_empty'), 'state G - ER': ('st_support', 'st_er'), 'state G - DP': ('st_support', 'st_dp'),
}
res['other_constructions'] = {lab: both(a, b, U10) for lab, (a, b) in OTHER.items()}
gi = {}
for cond in ('dkt_emb', 'corr_emb'):
    v = [ALL[(c, s, cond)] for c, s in U10 if (c, s, cond) in ALL]
    gi[cond] = dict(runs=len(v), overlap_directed=float(np.mean([x['overlap_with_G_directed'] for x in v])),
                    overlap_undirected=float(np.mean([x['overlap_with_G_undirected'] for x in v])),
                    overlap_undirected_by_corpus=pd.DataFrame([dict(c=x['data'], o=x['overlap_with_G_undirected']) for x in v]).groupby('c').o.mean().round(3).to_dict())
    if cond == 'dkt_emb':
        gi[cond]['dkt_test_auc_mean'] = float(np.mean([x['dkt_test_auc'] for x in v]))
        gi[cond]['dkt_minus_probe_control_test_auc'] = float(np.mean([x['dkt_test_auc'] - ALL[(x['data'], x['sub'], 'empty')]['test_auc'] for x in v]))
res['other_constructions_graphs'] = gi
# 3. Junyi, five subsamples
JU = [('junyi', s) for s in range(5)]
res['junyi_five'] = {lab: both(a, b, JU) for lab, (a, b) in {
    'emb map - empty': ('map_emb', 'empty'), 'emb G - empty': ('support_emb', 'empty'), 'emb map - G': ('map_emb', 'support_emb'),
    'state map - empty': ('st_map', 'st_empty'), 'state G - empty': ('st_support', 'st_empty'), 'state map - G': ('st_map', 'st_support')}.items()}
res['n_runs_r16'] = len(glob.glob(f'{REPO}/runs_r16/*.json')) + len(glob.glob(f'{REPO}/runs_r16/repro/*.json'))
json.dump(res, open(f'{OUT}/r16_results.json', 'w'), indent=1)


def show(lab, o):
    f, a = o['first'], o['all']
    extra = f" boot {f['boot95'][0]:+.4f},{f['boot95'][1]:+.4f} W p={f['p_wilcoxon']:.3f}" if 'boot95' in f else f" units+ {f.get('units_pos')}/{f['units']} {f.get('per_unit')}"
    print(f"{lab:48s} first {f['mean']:+.4f} ({f['pos']}/{f['corpora']}) [{f['ci95'][0]:+.4f},{f['ci95'][1]:+.4f}] {f['class']}{extra} | "
          f"all {a['mean']:+.4f} ({a['pos']}) [{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}] {a['class']} n={f['units']}")


for blk in ('balanced_both_entry_points', 'other_constructions', 'junyi_five'):
    print('==', blk)
    for lab, o in res[blk].items():
        if o['first']['units'] and o['first']['corpora'] >= 1: show(lab, o)
print(json.dumps(gi, indent=0)[:800]); print('runs', res['n_runs_r16'])
