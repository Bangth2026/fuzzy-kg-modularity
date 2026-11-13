"""Release 3.7.0: first attempts split by a previously practised graph neighbour, for the
frozen-random-table control, and the same split with the position in the sequence held fixed.

Reads only the per-prediction files of runs_r9 (no training, no data files). For every unit
(corpus x subsample, 10% of the training learners, update budget, training seed 0) on which the
empty-graph control, er_emb, erR_emb and support_emb exist:

1. Split of the first-attempt predictions by nb (an in- or out-neighbour of the target KC, in the
   Erdos-Renyi graph `er` or in the support graph `sup`, occurs earlier in the learner's sequence).
   For each condition c in {support_emb, er_emb, erR_emb}: AUC(c) - AUC(empty) on the 'seen' and the
   'unseen' first attempts, and seen - unseen within unit. A group enters a unit only with more than
   20 first attempts and both classes of response, as in r9_analyse.py. The er_emb and support_emb rows repeat
   section 4 of r9_analyse.py.
2. The contrast that separates transfer from a fixed random component: er_emb - erR_emb (learned
   against frozen neighbour table, same graph) on seen and unseen first attempts, and seen - unseen.
3. Difficulty of the two groups for the empty-graph control: AUC(empty) on seen and on unseen first
   attempts, base rate of correct answers, and the median position in the sequence of each group.
4. Position held fixed: first attempts of a unit are cut into position strata (terciles of the position
   of first attempts in that unit). Within each stratum with at least 30 seen and 30 unseen first
   attempts (and both classes of response in each group), the gain is computed on seen and on unseen;
   the stratum differences are averaged with weights equal to the number of first attempts.

5. Training length: Spearman correlation, over units, of the first-attempt gain of erR_emb over the
   empty-graph control with the update at which the control's selected checkpoint was taken (best_step),
   over all units and after centring both within corpus.

6. Run sets of Section 5.4 (Supplementary Table S6.10): run files of runs_r8, runs_r9 and runs_r10 grouped by
   release, model, training seed and device, with the corpora and subsamples of each group.

7. Controls without a graph (runs_r10, kt/src/r10_controls.py), paired with empty, er_emb and erR_emb of runs_r9 and
   uniform_emb of runs_r7 on the same units: rvec_emb (a fixed random vector per KC with the row variance that the
   frozen table receives through the Erdos-Renyi graph, no graph), rvec1_emb (unit variance, no graph) and erE2_emb
   (Erdos-Renyi mixing of a second, trainable table). Differences on first attempts and on all predictions, and the
   reproduction of runs_r9 by the reruns of the empty-graph control in runs_r10/repro (subsample 0 of every corpus).
   A corpus whose rerun does not reproduce runs_r9 exactly is compared within its rebuilt data, with empty, uniform_emb,
   er_emb and erR_emb rerun by r9_controls.py into runs_r10/base.

Every quantity is summarised as in r9_analyse.py: mean over corpora of subsample means, number of
corpora with a positive value, Wilcoxon p over corpora (unadjusted), 95% mixed-model interval (random
intercept per corpus) and corpus-then-subsample bootstrap interval.
Output: results_r10/r10_results.json and a printed summary.
"""
import os, sys, json, glob, collections, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
R9 = os.environ.get('R9_RUNS', f'{REPO}/runs_r9')
OUT = os.environ.get('R10_OUT', f'{REPO}/results_r10'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
CONDS = ('empty', 'support_emb', 'er_emb', 'erR_emb')
MIN_STRATUM = 30


def mixed(df):
    warnings.simplefilter('ignore')
    try:
        import statsmodels.formula.api as smf
        if df.data.nunique() < 3: return None
        r = smf.mixedlm('d ~ 1', df, groups=df['data']).fit(reml=True)
        lo, hi = r.conf_int().loc['Intercept']
        if not (np.isfinite(lo) and np.isfinite(hi)):
            r = smf.ols('d ~ 1', df).fit(); lo, hi = r.conf_int().loc['Intercept']
            return dict(est=float(r.params['Intercept']), lo=float(lo), hi=float(hi), singular=True)
        return dict(est=float(r.params['Intercept']), lo=float(lo), hi=float(hi))
    except Exception as e:
        return dict(error=str(e)[:100])


def boot(df, B=4000):
    g = {c: v.d.values for c, v in df.groupby('data')}; cs = list(g); out = []
    for _ in range(B):
        pick = rng.choice(len(cs), len(cs))
        out.append(np.mean([rng.choice(g[cs[i]], len(g[cs[i]])).mean() for i in pick]))
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def summ(rows):
    df = pd.DataFrame(rows).dropna(subset=['d'])
    if df.empty: return None
    per = df.groupby('data').d.mean()
    o = dict(units=int(len(df)), n=int(len(per)), mean=float(per.mean()), pos=int((per > 0).sum()),
             per_corpus={k: float(v) for k, v in per.items()})
    if len(per) >= 4: o['p_wilcoxon'] = float(wilcoxon(per.values).pvalue)
    o['mixed'] = mixed(df); o['boot'] = boot(df) if len(per) >= 3 else None
    return o


def auc(y, p, min_n=1):
    return float(roc_auc_score(y, p)) if len(y) > min_n and 0 < y.mean() < 1 else np.nan


MIN_GROUP = 20   # as in section 4 of r9_analyse.py: a group enters a unit only with more than 20 first attempts


units = collections.defaultdict(dict)
for f in glob.glob(f'{R9}/*__steps.json'):
    r = json.load(open(f))
    if r.get('rep', 0) or r.get('family') != 'probe' or r['cond'] not in CONDS: continue
    units[(r['data'], r['sub'])][r['cond']] = r['tag']

S = collections.defaultdict(list)
for (c, s), tags in sorted(units.items()):
    if set(CONDS) - set(tags): continue
    P = {k: np.load(f'{R9}/{tags[k]}_pred.npz') for k in CONDS}
    for k in CONDS[1:]:
        assert (P[k]['kc'] == P['empty']['kc']).all() and (P[k]['y'] == P['empty']['y']).all()
    new = P['empty']['new'].astype(bool); y = P['empty']['y'].astype(int); pos = P['empty']['pos'].astype(int)
    for gname in ('er', 'sup'):
        nb = P['empty'][f'nb_{gname}'].astype(bool)
        grp = {'seen': new & nb, 'unseen': new & ~nb}
        S[f'{gname}|share_seen'].append(dict(data=c, sub=s, d=float(nb[new].mean())))
        # 3. difficulty and position of the two groups
        for lab, sel in grp.items():
            S[f'{gname}|empty_auc|{lab}'].append(dict(data=c, sub=s, d=auc(y[sel], P['empty']['p'][sel], MIN_GROUP), n=int(sel.sum())))
            S[f'{gname}|base_rate|{lab}'].append(dict(data=c, sub=s, d=float(y[sel].mean()) if sel.any() else np.nan))
            S[f'{gname}|median_pos|{lab}'].append(dict(data=c, sub=s, d=float(np.median(pos[sel])) if sel.any() else np.nan))
        S[f'{gname}|empty_auc|seen-unseen'].append(dict(data=c, sub=s, d=S[f'{gname}|empty_auc|seen'][-1]['d'] - S[f'{gname}|empty_auc|unseen'][-1]['d']))
        # 1-2. gains on the two groups
        pairs = [(k, 'empty') for k in CONDS[1:]] + [('er_emb', 'erR_emb')]
        for a, b in pairs:
            g = {lab: auc(y[sel], P[a]['p'][sel], MIN_GROUP) - auc(y[sel], P[b]['p'][sel], MIN_GROUP) for lab, sel in grp.items()}
            key = f'{gname}|{a}-{b}'
            for lab in g: S[f'{key}|{lab}'].append(dict(data=c, sub=s, d=g[lab]))
            S[f'{key}|seen-unseen'].append(dict(data=c, sub=s, d=g['seen'] - g['unseen']))
        # 4. position held fixed
        fp = pos[new]; cuts = np.quantile(fp, [1 / 3, 2 / 3])
        stratum = np.digitize(pos, cuts)
        for a, b in pairs:
            num = 0.0; den = 0
            for t in range(3):
                se = grp['seen'] & (stratum == t); un = grp['unseen'] & (stratum == t)
                if se.sum() < MIN_STRATUM or un.sum() < MIN_STRATUM: continue
                ds = auc(y[se], P[a]['p'][se]) - auc(y[se], P[b]['p'][se])
                du = auc(y[un], P[a]['p'][un]) - auc(y[un], P[b]['p'][un])
                if not (np.isfinite(ds) and np.isfinite(du)): continue
                w = int(se.sum() + un.sum()); num += w * (ds - du); den += w
            S[f'{gname}|{a}-{b}|seen-unseen|pos_strata'].append(dict(data=c, sub=s, d=num / den if den else np.nan))

# 5. training length ---------------------------------------------------------------------------
from scipy.stats import spearmanr
rows = []
for (c, s), tags in sorted(units.items()):
    if {'empty', 'erR_emb'} - set(tags): continue
    e = json.load(open(f'{R9}/{tags["empty"]}.json')); r = json.load(open(f'{R9}/{tags["erR_emb"]}.json'))
    rows.append(dict(data=c, sub=s, g=r['test_auc_first'] - e['test_auc_first'], best=e['best_step']))
TL = pd.DataFrame(rows)
for col in ('g', 'best'): TL[col + '_c'] = TL[col] - TL.groupby('data')[col].transform('mean')
rho_all = spearmanr(TL.g, TL.best); rho_w = spearmanr(TL.g_c, TL.best_c)
training_length = dict(units=int(len(TL)), best_step_min=float(TL.best.min()), best_step_median=float(TL.best.median()),
                       best_step_max=float(TL.best.max()), spearman=float(rho_all.statistic), p=float(rho_all.pvalue),
                       spearman_within=float(rho_w.statistic), p_within=float(rho_w.pvalue))

# 6. run sets --------------------------------------------------------------------------------
R8 = os.environ.get('R8_RUNS', f'{REPO}/runs_r8')
rs = []
R10i = os.environ.get('R10_RUNS', f'{REPO}/runs_r10')
for rel, path in (('3.5.0', R8), ('3.6.0', R9), ('3.7.0', R10i), ('3.7.0 rerun', f'{R10i}/repro'), ('3.7.0 rerun', f'{R10i}/base')):
    for f in glob.glob(f'{path}/*.json'):
        r = json.load(open(f))
        rs.append(dict(release=rel, family=r['family'], cond=r['cond'], data=r['data'], sub=int(r['sub']),
                       rep=int(r.get('rep', 0) or 0), device=r.get('device', 'cpu')))
RS = pd.DataFrame(rs)
run_sets = []
for (rel, fam, rep_any, dev), g in RS.assign(rep_any=RS.rep > 0).groupby(['release', 'family', 'rep_any', 'device']):
    run_sets.append(dict(release=rel, family=fam, seeds=sorted(int(v) for v in g.rep.unique()), device=dev,
                         conditions=sorted(g.cond.unique().tolist()), runs=int(len(g)),
                         subsamples={c: sorted(int(v) for v in gg['sub'].unique()) for c, gg in g.groupby('data')}))

# 7. controls without a graph -----------------------------------------------------------------
R10 = os.environ.get('R10_RUNS', f'{REPO}/runs_r10'); R7 = os.environ.get('R7_RUNS', f'{REPO}/runs_r7')
def load_runs(path, conds):
    out = []
    for f in glob.glob(f'{path}/*.json'):
        r = json.load(open(f))
        if r.get('budget') != 'steps' or abs(r.get('frac', 0.1) - 0.1) > 1e-9 or r.get('rep', 0): continue
        if r.get('lr', 0.002) != 0.002 or r.get('l2', 0) not in (0, 0.0, None): continue
        if r['cond'] in conds: out.append(r)
    return out
BASE = {'empty', 'er_emb', 'erR_emb', 'uniform_emb'}
RP = []
for f in glob.glob(f'{R10}/repro/*.json'):
    b = json.load(open(f)); a = json.load(open(f'{R9}/' + os.path.basename(f)))
    RP.append(dict(data=b['data'], same_all=a['test_auc'] == b['test_auc'], same_first=a['test_auc_first'] == b['test_auc_first'],
                   abs_diff=abs(a['test_auc'] - b['test_auc'])))
# a corpus whose rebuilt data do not reproduce the archived control exactly is paired within the rebuilt data:
# its reference conditions come from runs_r10/repro and runs_r10/base instead of runs_r9 / runs_r7
REBUILT = sorted({r['data'] for r in RP if not (r['same_all'] and r['same_first'])})
arch = [r for r in load_runs(R9, {'empty', 'er_emb', 'erR_emb'}) + load_runs(R7, {'uniform_emb'}) if r['data'] not in REBUILT]
reb = [r for r in load_runs(f'{R10}/base', BASE) + load_runs(f'{R10}/repro', BASE) if r['data'] in REBUILT]
NG = pd.DataFrame(arch + reb + load_runs(R10, {'rvec_emb', 'rvec1_emb', 'erE2_emb'}))
nograph = {'units': {}, 'paired_within_rebuilt_data': REBUILT}
if len(NG) and 'rvec_emb' in set(NG.cond):
    NG = NG.drop_duplicates(['data', 'sub', 'cond'])
    for key, col in (('first', 'test_auc_first'), ('all', 'test_auc')):
        T = NG.pivot_table(index=['data', 'sub'], columns='cond', values=col, aggfunc='first')
        out = {}
        for a_, b_ in (('rvec_emb', 'empty'), ('rvec_emb', 'uniform_emb'), ('erR_emb', 'rvec_emb'), ('er_emb', 'rvec_emb'),
                       ('rvec1_emb', 'empty'), ('rvec1_emb', 'uniform_emb'), ('rvec_emb', 'rvec1_emb'),
                       ('erE2_emb', 'empty'), ('erE2_emb', 'uniform_emb'), ('erE2_emb', 'erR_emb'), ('er_emb', 'erE2_emb'),
                       ('erR_emb', 'uniform_emb'), ('er_emb', 'uniform_emb')):
            if a_ in T and b_ in T:
                j = (T[a_] - T[b_]).dropna().rename('d').reset_index()
                out[f'{a_} - {b_}'] = summ(j.to_dict('records'))
        nograph[key] = out
        nograph['units'][key] = {c: int(T[c].notna().sum()) for c in T.columns}
    nograph['reproduction'] = dict(n=len(RP), all_identical=bool(all(r['same_all'] and r['same_first'] for r in RP)),
                                   max_abs_diff=max([r['abs_diff'] for r in RP], default=None), corpora=sorted(r['data'] for r in RP))

res = {'n_units': int(sum(1 for t in units.values() if not set(CONDS) - set(t))),
       'min_stratum': MIN_STRATUM, 'summary': {k: summ(v) for k, v in S.items()},
       'training_length_erR': training_length, 'controls_without_graph': nograph, 'run_sets': run_sets,
       'run_totals': {'all': int(len(RS)), 'by_device': {k: int(v) for k, v in RS.device.value_counts().items()}}}
json.dump(res, open(f'{OUT}/r10_results.json', 'w'), indent=1)


def fmt(v):
    if not v: return 'n/a'
    m = v.get('mixed') or {}
    return (f"{v['mean']:+.4f}  [{m.get('lo', np.nan):+.4f}, {m.get('hi', np.nan):+.4f}]  "
            f"{v['pos']}/{v['n']}  p={v.get('p_wilcoxon', np.nan):.3f}  units={v['units']}")


for key in ('first', 'all'):
    for k, v in nograph.get(key, {}).items(): print(f'NOGRAPH {key:5s} {k:28s}', fmt(v))
if 'reproduction' in nograph: print('REPRO', nograph['reproduction'])
print('units', res['n_units'], 'training length', {k: round(v, 3) for k, v in training_length.items()})
for k in sorted(res['summary']): print(f'{k:55s}', fmt(res['summary'][k]))
