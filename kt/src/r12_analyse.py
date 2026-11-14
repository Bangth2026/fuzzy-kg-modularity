"""Release 3.9.0: the first-attempt contrasts of the random-vector controls by training interactions on the target KC.

No training. Reads the run and per-prediction files of runs_r7, runs_r9, runs_r10 and runs_r11 (the same units and the
same choice of runs per corpus as r11_analyse.py) and the rebuilt corpora, from which n_k, the number of training
interactions on KC k in the subsample, is counted. Before any number is computed, the test targets and responses of each
rebuilt corpus are checked against the per-prediction files of the empty-graph control (same KC and same response at every
test prediction); a unit that fails the check is reported and left out.

1. First attempts at KCs with n_k = 0 against first attempts at KCs with n_k > 0, for rvec_emb - empty,
   rvecF_emb - empty, rvec_emb - rvecF_emb, rvec1_emb - empty, rvec1F_emb - empty, rvec1_emb - rvec1F_emb,
   erR_emb - empty and support_emb - empty (as in r11_analyse.py, section 3; a group enters a unit with more than
   20 first attempts and both classes of response).
2. Pooled quintiles of log10(n_k + 1) over the first attempts of all units, as in r9_analyse.py (section 5): the AUC
   difference in each quintile of each unit (more than 20 first attempts and both classes), summarised per quintile, and
   the slope per decade of n_k from a mixed model d ~ x with a random intercept per corpus, x the mean of
   log10(n_k + 1) of the first attempts of that unit in that quintile. Quintile cuts are computed once, on the pooled first
   attempts of the empty-graph control, so every contrast uses the same cuts.

3. Inventory of the run sets of Section 5.4: the sets of results_r10/r10_results.json (releases 3.5.0 to 3.7.0, 653 runs)
   and the sixth set (release 3.8.0: runs_r11, runs_r11/base, runs_r11/repro, 68 runs), in the same format, written also as
   results_r12/run_sets_section54.csv (the rows of Supplementary Table S6.10).

Output: results_r12/r12_results.json, results_r12/run_sets_section54.csv and a printed summary.
"""
import os, sys, json, glob, collections, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
R7, R9, R10, R11 = (os.environ.get(k, f'{REPO}/{d}') for k, d in
                    (('R7_RUNS', 'runs_r7'), ('R9_RUNS', 'runs_r9'), ('R10_RUNS', 'runs_r10'), ('R11_RUNS', 'runs_r11')))
OUT = os.environ.get('R12_OUT', f'{REPO}/results_r12'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
MIN_GROUP = 20


def mixed(df, formula='d ~ 1', term='Intercept'):
    try:
        import statsmodels.formula.api as smf
        if df.data.nunique() < 3: return None
        r = smf.mixedlm(formula, df, groups=df['data']).fit(reml=True)
        lo, hi = r.conf_int().loc[term]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            r = smf.ols(formula, df).fit(); lo, hi = r.conf_int().loc[term]
            return dict(est=float(r.params[term]), lo=float(lo), hi=float(hi), singular=True)
        return dict(est=float(r.params[term]), lo=float(lo), hi=float(hi))
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


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(y) > MIN_GROUP and 0 < y.mean() < 1 else np.nan


def runs(path, conds):
    out = {}
    for f in glob.glob(f'{path}/*.json'):
        r = json.load(open(f))
        if r.get('budget') != 'steps' or abs(r.get('frac', 0.1) - 0.1) > 1e-9 or r.get('rep', 0): continue
        if r.get('lr', 0.002) != 0.002 or r.get('l2', 0) not in (0, 0.0, None): continue
        if r['cond'] in conds: out[(r['data'], r['sub'], r['cond'])] = (r, f[:-5])
    return out


# runs: the same choice as r11_analyse.py (EdNet from the rebuilt data, elsewhere the archived runs) -----------------
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
CONTRASTS = [('rvec_emb', 'empty'), ('rvecF_emb', 'empty'), ('rvec_emb', 'rvecF_emb'),
             ('rvec1_emb', 'empty'), ('rvec1F_emb', 'empty'), ('rvec1_emb', 'rvec1F_emb'),
             ('erR_emb', 'empty'), ('support_emb', 'empty')]
CONDS = sorted({x for ab in CONTRASTS for x in ab})

from r5_controls import load
from train import seqs
NK = collections.defaultdict(list); POOL = []; CHECK = []
for c, s in UNITS:
    df = load(c, 0.1, s); K = int(df.kc.max()) + 1; tr = df[df.split == 'train']
    nk = np.bincount(tr.kc.values, minlength=K)
    P = {k: np.load(R[(c, s, k)][1] + '_pred.npz') for k in CONDS
         if (c, s, k) in R and os.path.exists(R[(c, s, k)][1] + '_pred.npz')}
    e = P['empty']; new = e['new'].astype(bool); y = e['y'].astype(int); kc = e['kc'].astype(int)
    q, r, _ = seqs(df, K, 'test'); qn = q[:, 1:].numpy(); rn = r[:, 1:].numpy(); mask = qn < K
    ok = bool(mask.sum() == len(kc) and (qn[mask] == kc).all() and (rn[mask] == y).all())
    for k in P: ok = ok and bool((P[k]['kc'] == kc).all() and (P[k]['y'] == y).all())
    CHECK.append(dict(data=c, sub=s, K=K, n_pred=int(len(kc)), n_first=int(new.sum()), aligned=ok))
    if not ok: continue
    grp = {'nk0': new & (nk[kc] == 0), 'nk_pos': new & (nk[kc] > 0)}
    for a, b in CONTRASTS:
        if a not in P or b not in P: continue
        d = {lab: auc(y[sel], P[a]['p'][sel]) - auc(y[sel], P[b]['p'][sel]) for lab, sel in grp.items()}
        for lab in d: NK[f'{a} - {b}|{lab}'].append(dict(data=c, sub=s, d=d[lab], n=int(grp[lab].sum())))
        NK[f'{a} - {b}|nk0-nk_pos'].append(dict(data=c, sub=s, d=d['nk0'] - d['nk_pos']))
    POOL.append(dict(data=c, sub=s, ln=np.log10(nk[kc[new]] + 1), y=y[new], p={k: P[k]['p'][new] for k in P}))

CHECK = pd.DataFrame(CHECK)
qs = np.quantile(np.concatenate([u['ln'] for u in POOL]), [.2, .4, .6, .8])
PQ = {}
for a, b in CONTRASTS:
    bins = collections.defaultdict(list); pts = []
    for u in POOL:
        if a not in u['p'] or b not in u['p']: continue
        band = np.digitize(u['ln'], qs, right=True)
        for k in range(5):
            sel = band == k
            dd = auc(u['y'][sel], u['p'][a][sel]) - auc(u['y'][sel], u['p'][b][sel])
            if not np.isfinite(dd): continue
            bins[k].append(dict(data=u['data'], sub=u['sub'], d=dd))
            pts.append(dict(data=u['data'], sub=u['sub'], x=float(np.mean(u['ln'][sel])), d=dd, w=int(sel.sum())))
    o = {f'Q{k + 1}': summ(v) for k, v in sorted(bins.items())}
    D = pd.DataFrame(pts)
    o['slope_per_log10_nk'] = dict(mixed(D, 'd ~ x', 'x') or {}, n_points=int(len(D)), units=int(D[['data', 'sub']].drop_duplicates().shape[0]))
    PQ[f'{a} - {b}'] = o

# 3. run inventory of Section 5.4 ------------------------------------------------------------------------------------
prev = json.load(open(f'{REPO}/results_r10/r10_results.json'))['run_sets']
rs = []
for sub_, lab in (('', '3.8.0'), ('/base', '3.8.0 EdNet rebuilt'), ('/repro', '3.8.0 rerun')):
    for f in glob.glob(f'{R11}{sub_}/*.json'):
        r = json.load(open(f))
        rs.append(dict(release=lab, family=r.get('family', 'probe'), cond=r['cond'], data=r['data'], sub=int(r['sub']),
                       rep=int(r.get('rep', 0) or 0), device=r.get('device', 'cpu')))
RS = pd.DataFrame(rs); sixth = []
for (rel, fam, dev), g in RS.groupby(['release', 'family', 'device']):
    sixth.append(dict(release=rel, family=fam, seeds=sorted(int(v) for v in g.rep.unique()), device=dev,
                      conditions=sorted(g.cond.unique().tolist()), runs=int(len(g)),
                      subsamples={c: sorted(int(v) for v in gg['sub'].unique()) for c, gg in g.groupby('data')}))
run_sets = prev + sixth
pd.DataFrame([dict(release=x['release'], model=x['family'], device=x['device'], seeds=' '.join(map(str, x['seeds'])),
                   conditions=' '.join(x['conditions']), corpora=len(x['subsamples']),
                   subsamples='; '.join(f"{c}: {' '.join(map(str, v))}" for c, v in x['subsamples'].items()), runs=x['runs'])
              for x in run_sets]).to_csv(f'{OUT}/run_sets_section54.csv', index=False)
dev = collections.Counter()
for x in run_sets: dev[x['device']] += x['runs']

res = dict(run_sets=run_sets, run_totals=dict(all=int(sum(x['runs'] for x in run_sets)), by_device=dict(dev)), units=[list(u) for u in UNITS], alignment=CHECK.to_dict('records'),
           units_aligned=int(CHECK.aligned.sum()),
           first_by_nk={k: summ(v) for k, v in NK.items()},
           first_by_nk_counts={k: int(np.median([r['n'] for r in v])) for k, v in NK.items() if 'n' in v[0]},
           nk_pooled_quintiles=dict(cuts_log10_nk_plus1=[float(x) for x in qs], **PQ), rebuilt=REBUILT)
json.dump(res, open(f'{OUT}/r12_results.json', 'w'), indent=1, default=float)


def fmt(v):
    if not v: return 'n/a'
    m = v.get('mixed') or {}
    return (f"{v['mean']:+.4f}  [{m.get('lo', np.nan):+.4f}, {m.get('hi', np.nan):+.4f}]  "
            f"{v['pos']}/{v['n']}  p={v.get('p_wilcoxon', np.nan):.3f}  units={v['units']}")


print('run totals', res['run_totals'])
print('aligned units', res['units_aligned'], 'of', len(UNITS))
print(CHECK[~CHECK.aligned])
for k, v in res['first_by_nk'].items(): print(f'NK {k:40s}', fmt(v), res['first_by_nk_counts'].get(k, ''))
print('cuts', [round(x, 3) for x in qs])
for k, v in PQ.items():
    sl = v['slope_per_log10_nk']
    print(f'Q {k:28s} slope {sl.get("est", np.nan):+.4f} [{sl.get("lo", np.nan):+.4f}, {sl.get("hi", np.nan):+.4f}] pts={sl["n_points"]}',
          ' | '.join(f"{q}:{v[q]['mean']:+.4f}" for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5') if v.get(q)))
