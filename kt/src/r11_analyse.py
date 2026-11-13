"""Release 3.8.0: the random-vector control with frozen maps, the graph against the controls without a graph, first
attempts by training data on the target KC, and diagnostics of the neighbour split.

Reads run and per-prediction files only (runs_r7, runs_r9, runs_r10, runs_r11) and, for section 3, the rebuilt corpora
(to count training interactions per KC). Every unit is a corpus x subsample at 10% of the training learners under the
update budget, training seed 0, CPU with one thread, graph read in the embedding refinement only.

1. Graph against the controls without a graph: support_emb - rvec_emb and support_emb - rvec1_emb, on first attempts and
   on all predictions. On EdNet, whose rebuilt data differ from the archived data in under 0.5% of the test interactions,
   every condition is taken from the rebuilt data (runs_r10/repro, runs_r10/base, runs_r11/base); elsewhere the
   archived runs of runs_r9 (empty, support_emb, er_emb, erR_emb) and runs_r7 (uniform_emb) are used, which the reruns
   reproduce exactly.
2. Frozen maps: rvecF_emb and rvec1F_emb (r11_controls.py) against empty, uniform_emb and their trainable counterparts.
3. First attempts by training interactions n_k on the target KC: n_k = 0 against n_k > 0, for rvec_emb, rvecF_emb,
   erR_emb and support_emb against empty, and rvec_emb - rvecF_emb (a group enters a unit with more than 20 first
   attempts and both classes of response).
4. KCs without an Erdos-Renyi neighbour (row variance 0 in rvec_emb): share of KCs without an in-neighbour, without an
   out-neighbour and without either, and the share of first attempts at KCs without either.
5. Neighbour split (r10_analyse.py, sections 1 and 4) per unit: which units enter the analysis by position terciles and
   why the others drop out, the number of first attempts with a practised Erdos-Renyi neighbour in each tercile, and every
   summary with Junyi left out.

Summaries as in r10_analyse.py: mean over corpora of subsample means, corpora with a positive value, unadjusted Wilcoxon
p over corpora, 95% mixed-model interval (random intercept per corpus) and corpus-then-subsample bootstrap interval.
Output: results_r11/r11_results.json and a printed summary.
"""
import os, sys, json, glob, collections, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
R7, R9, R10, R11 = (os.environ.get(k, f'{REPO}/{d}') for k, d in
                    (('R7_RUNS', 'runs_r7'), ('R9_RUNS', 'runs_r9'), ('R10_RUNS', 'runs_r10'), ('R11_RUNS', 'runs_r11')))
OUT = os.environ.get('R11_OUT', f'{REPO}/results_r11'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
MIN_GROUP = 20; MIN_STRATUM = 30


def mixed(df):
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


def auc(y, p, min_n=MIN_GROUP):
    return float(roc_auc_score(y, p)) if len(y) > min_n and 0 < y.mean() < 1 else np.nan


def runs(path, conds):
    out = {}
    for f in glob.glob(f'{path}/*.json'):
        r = json.load(open(f))
        if r.get('budget') != 'steps' or abs(r.get('frac', 0.1) - 0.1) > 1e-9 or r.get('rep', 0): continue
        if r.get('lr', 0.002) != 0.002 or r.get('l2', 0) not in (0, 0.0, None): continue
        if r['cond'] in conds: out[(r['data'], r['sub'], r['cond'])] = (r, f[:-5])
    return out


REBUILT = ['ednet']
REF = {'empty', 'support_emb', 'er_emb', 'erR_emb', 'uniform_emb'}
NEW = {'rvec_emb', 'rvec1_emb', 'erE2_emb', 'rvecF_emb', 'rvec1F_emb'}
R = {}
for path, conds in ((R9, REF), (R7, {'uniform_emb'})):
    R.update({k: v for k, v in runs(path, conds).items() if k[0] not in REBUILT})
for path in (f'{R10}/repro', f'{R10}/base', f'{R11}/base'):
    R.update({k: v for k, v in runs(path, REF).items() if k[0] in REBUILT})
R.update(runs(R10, NEW)); R.update(runs(R11, NEW))
UNITS = sorted({(c, s) for c, s, k in R if k == 'rvec_emb'})

# 1-2. paired differences -------------------------------------------------------------------------------------------
PAIRS = [('support_emb', 'rvec_emb'), ('support_emb', 'rvec1_emb'), ('support_emb', 'empty'), ('support_emb', 'uniform_emb'),
         ('rvec_emb', 'uniform_emb'), ('rvec1_emb', 'uniform_emb'),
         ('rvecF_emb', 'empty'), ('rvecF_emb', 'uniform_emb'), ('rvec_emb', 'rvecF_emb'),
         ('rvec1F_emb', 'empty'), ('rvec1F_emb', 'uniform_emb'), ('rvec1_emb', 'rvec1F_emb')]
paired = {}
for key, col in (('first', 'test_auc_first'), ('all', 'test_auc')):
    paired[key] = {}
    for a, b in PAIRS:
        rows = [dict(data=c, sub=s, d=R[(c, s, a)][0][col] - R[(c, s, b)][0][col]) for c, s in UNITS
                if (c, s, a) in R and (c, s, b) in R]
        paired[key][f'{a} - {b}'] = summ(rows) if rows else None

# 3-4. first attempts by training interactions on the target KC; KCs without an Erdos-Renyi neighbour ------------
from r5_controls import load, transition_support
from r6_controls import erdos_renyi
NK = collections.defaultdict(list); ISO = []
for c, s in UNITS:
    df = load(c, 0.1, s); K = int(df.kc.max()) + 1; tr = df[df.split == 'train']
    nk = np.bincount(tr.kc.values, minlength=K)
    g = np.load(f'{REPO}/graphs_r5/{c}__f10__s{s}.npz') if os.path.exists(f'{REPO}/graphs_r5/{c}__f10__s{s}.npz') else None
    sup = g['support'] if g is not None else (transition_support(tr, K) > 0).astype(np.float32)
    er = erdos_renyi(K, int(sup.sum()), seed=2000 + s)
    no_in = er.sum(0) == 0; no_out = er.sum(1) == 0
    P = {k: np.load(R[(c, s, k)][1] + '_pred.npz') for k in ('empty', 'rvec_emb', 'rvecF_emb', 'erR_emb', 'support_emb')
         if (c, s, k) in R and os.path.exists(R[(c, s, k)][1] + '_pred.npz')}
    e = P['empty']; new = e['new'].astype(bool); y = e['y'].astype(int); kc = e['kc'].astype(int)
    for k in P: assert (P[k]['kc'] == kc).all() and (P[k]['y'] == y).all()
    ISO.append(dict(data=c, sub=s, K=K, no_in=float(no_in.mean()), no_out=float(no_out.mean()),
                    no_either=float((no_in & no_out).mean()), first_at_no_either=float((no_in & no_out)[kc[new]].mean()),
                    kc_without_training=float((nk == 0).mean()), first_at_nk0=float((nk[kc[new]] == 0).mean())))
    grp = {'nk0': new & (nk[kc] == 0), 'nk_pos': new & (nk[kc] > 0)}
    for a, b in (('rvec_emb', 'empty'), ('rvecF_emb', 'empty'), ('rvec_emb', 'rvecF_emb'), ('erR_emb', 'empty'),
                 ('support_emb', 'empty')):
        if a not in P or b not in P: continue
        d = {lab: auc(y[sel], P[a]['p'][sel]) - auc(y[sel], P[b]['p'][sel]) for lab, sel in grp.items()}
        for lab in d: NK[f'{a} - {b}|{lab}'].append(dict(data=c, sub=s, d=d[lab], n=int(grp[lab].sum())))
        NK[f'{a} - {b}|nk0-nk_pos'].append(dict(data=c, sub=s, d=d['nk0'] - d['nk_pos']))
ISO = pd.DataFrame(ISO)
iso = {col: dict(mean_over_corpora=float(ISO.groupby('data')[col].mean().mean()),
                 per_corpus={k: float(v) for k, v in ISO.groupby('data')[col].mean().items()})
       for col in ('no_in', 'no_out', 'no_either', 'first_at_no_either', 'kc_without_training', 'first_at_nk0')}

# 5. neighbour split: units, strata, Junyi left out (archived data, runs_r9, as in r10_analyse.py) ----------------
units9 = collections.defaultdict(dict)
for f in glob.glob(f'{R9}/*__steps.json'):
    r = json.load(open(f))
    if r.get('rep', 0) or r.get('family') != 'probe' or r['cond'] not in ('empty', 'er_emb', 'erR_emb'): continue
    units9[(r['data'], r['sub'])][r['cond']] = r['tag']
SP = []; STR = []
for (c, s), tags in sorted(units9.items()):
    if len(tags) < 3: continue
    P = {k: np.load(f'{R9}/{tags[k]}_pred.npz') for k in tags}
    e = P['empty']; new = e['new'].astype(bool); y = e['y'].astype(int); pos = e['pos'].astype(int); nb = e['nb_er'].astype(bool)
    seen = new & nb; unseen = new & ~nb
    row = dict(data=c, sub=s, n_first=int(new.sum()), n_seen=int(seen.sum()), n_unseen=int(unseen.sum()))
    for a, b in (('er_emb', 'empty'), ('er_emb', 'erR_emb')):
        ds = auc(y[seen], P[a]['p'][seen]) - auc(y[seen], P[b]['p'][seen])
        du = auc(y[unseen], P[a]['p'][unseen]) - auc(y[unseen], P[b]['p'][unseen])
        row[f'{a}-{b}'] = ds - du
    fp = pos[new]; cuts = np.quantile(fp, [1 / 3, 2 / 3]); st = np.digitize(pos, cuts)
    kept = []
    for t in range(3):
        se = seen & (st == t); un = unseen & (st == t)
        STR.append(dict(data=c, sub=s, tercile=t, n_seen=int(se.sum()), n_unseen=int(un.sum()),
                        enters=bool(se.sum() >= MIN_STRATUM and un.sum() >= MIN_STRATUM)))
        if STR[-1]['enters']: kept.append(t)
    row['strata_entering'] = kept
    for a, b in (('er_emb', 'empty'), ('er_emb', 'erR_emb')):
        num = 0.0; den = 0
        for t in kept:
            se = seen & (st == t); un = unseen & (st == t)
            dd = (auc(y[se], P[a]['p'][se], 1) - auc(y[se], P[b]['p'][se], 1)) - (auc(y[un], P[a]['p'][un], 1) - auc(y[un], P[b]['p'][un], 1))
            if np.isfinite(dd): w = int(se.sum() + un.sum()); num += w * dd; den += w
        row[f'{a}-{b}|strata'] = num / den if den else np.nan
    SP.append(row)
SP = pd.DataFrame(SP); STR = pd.DataFrame(STR)
split = {'units': SP.to_dict('records'), 'strata': STR.to_dict('records'),
         'units_entering_strata': int(SP['er_emb-empty|strata'].notna().sum()),
         'units_dropped': SP.loc[SP['er_emb-empty|strata'].isna(), ['data', 'sub', 'n_seen']].to_dict('records')}
for col in ('er_emb-empty', 'er_emb-erR_emb', 'er_emb-empty|strata', 'er_emb-erR_emb|strata'):
    rows = SP[['data', 'sub', col]].rename(columns={col: 'd'})
    split[col] = summ(rows.to_dict('records'))
    split[col + '|without_junyi'] = summ(rows[rows.data != 'junyi'].to_dict('records'))

res = dict(units=[list(u) for u in UNITS], paired=paired, first_by_nk={k: summ(v) for k, v in NK.items()},
           first_by_nk_counts={k: int(np.median([r['n'] for r in v])) for k, v in NK.items() if 'n' in v[0]},
           kcs_without_er_neighbour=iso, neighbour_split=split, rebuilt=REBUILT)
json.dump(res, open(f'{OUT}/r11_results.json', 'w'), indent=1, default=float)


def fmt(v):
    if not v: return 'n/a'
    m = v.get('mixed') or {}
    return (f"{v['mean']:+.4f}  [{m.get('lo', np.nan):+.4f}, {m.get('hi', np.nan):+.4f}]  "
            f"{v['pos']}/{v['n']}  p={v.get('p_wilcoxon', np.nan):.3f}  units={v['units']}")


for key in paired:
    for k, v in paired[key].items(): print(f'{key:5s} {k:28s}', fmt(v))
for k, v in res['first_by_nk'].items(): print(f'NK {k:40s}', fmt(v), res['first_by_nk_counts'].get(k, ''))
print('ISO', {k: round(v['mean_over_corpora'], 3) for k, v in iso.items()})
for k in ('er_emb-empty', 'er_emb-erR_emb', 'er_emb-empty|strata', 'er_emb-erR_emb|strata'):
    print('SPLIT', f'{k:25s}', fmt(split[k]), '| no Junyi', fmt(split[k + '|without_junyi']))
print('dropped', split['units_dropped'])
