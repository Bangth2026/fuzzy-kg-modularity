"""Release 3.4.0 analysis of the runs in runs_r7 (update budget, 10%) -> results_r7/r7_results.json.

1. Uniform mixing: uniform_emb (complete graph in the embedding refinement only) against the
   empty-graph control, and against support_emb / er_emb of release 3.3.0 (runs_r6, same subsamples,
   same code), and against the full support graph.
2. Tuned shrinkage of the embedding table: for each corpus, lambda in {0, 1e-4, 1e-3, 1e-2, 1e-1}
   (on the six larger corpora {0, 1e-3, 1e-2, 1e-1}, subsample 0 only) chosen by the mean validation
   AUC over the subsamples on which the grid was run, separately for the control (empty_l2) and
   for the support graph (support_l2); comparisons of the tuned control with the untuned and the
   tuned support graph.
3. Learning rate: for each corpus, lr in {0.002, 0.005, 0.01} (six larger corpora: subsample 0 only)
   chosen by mean validation AUC,
   separately for the control and the support graph; the support-versus-control gain at each rate and
   at the selected rates.
4. First attempts by data per KC: n_k = training interactions on KC k in the subsample; KCs split into
   terciles of n_k within corpus; first-attempt AUC of support and control within each tercile.
5. First-attempt AUC decomposed: within-KC AUC (mean over KCs with both outcomes, weighted by the
   number of correct-incorrect pairs) and between-KC AUC (each prediction replaced by the mean
   prediction of its KC in that run); and the first-attempt AUC of a trivial predictor, the
   training-set correct rate of the target KC (global rate for KCs unseen in training).
Comparisons: mean over corpora of subsample means, Wilcoxon over corpora, 95% mixed-model intervals.
"""
import json, glob, os, collections, warnings, numpy as np, pandas as pd, sys
warnings.filterwarnings('ignore')
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt'); OUT = f'{ROOT}/results_r7'; os.makedirs(OUT, exist_ok=True)
CORP = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
LAMS = [1e-4, 1e-3, 1e-2, 1e-1]; LRS = [0.002, 0.005, 0.01]


def wil(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.all(x == 0): return 1.0
    return float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)


def mixed_ci(rows):
    import statsmodels.formula.api as smf
    d = pd.DataFrame(rows, columns=['corpus', 'g'])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = smf.mixedlm('g ~ 1', d, groups=d['corpus']).fit(reml=True)
        b = float(f.params['Intercept']); se = float(f.bse['Intercept'])
        return dict(lo=b - 1.96 * se, hi=b + 1.96 * se, p=float(f.pvalues['Intercept']))
    except Exception as e:
        return dict(error=repr(e))


def summ(per):
    cs = [c for c in CORP if c in per and len(per[c])]
    m = np.array([np.mean(per[c]) for c in cs])
    return dict(mean=float(m.mean()), pos=int((m > 0).sum()), n=len(cs), p=wil(m), mixed=mixed_ci([(c, g) for c in cs for g in per[c]]),
                per_corpus={c: float(np.mean(per[c])) for c in cs}, units=int(sum(len(per[c]) for c in cs)))


R7 = [json.load(open(f)) for f in glob.glob(f'{ROOT}/runs_r7/*.json')]
def key(r):
    k = r['cond']
    if k.endswith('_l2'): k += f"|{r['l2']:g}"
    if abs(r['lr'] - 0.002) > 1e-9: k += f"|lr{r['lr']:g}"
    return k
U = collections.defaultdict(dict)
for r in R7: U[(r['data'], r['sub'])][key(r)] = r
for r in (json.load(open(f)) for f in glob.glob(f'{ROOT}/runs_r6/*__steps.json')):
    if round(r['frac'], 2) == 0.1 and r['cond'] in ('support_emb', 'er_emb', 'er', 'rewired') and (r['data'], r['sub']) in U:
        U[(r['data'], r['sub'])].setdefault('r6_' + r['cond'], r)
M = {'all': 'test_auc', 'first': 'test_auc_first'}


def diff(a, b, metric):
    per = collections.defaultdict(list)
    for (c, s), v in U.items():
        if a in v and b in v and v[a].get(metric) is not None and v[b].get(metric) is not None:
            per[c].append(v[a][metric] - v[b][metric])
    return per


res = {'n_runs': len(R7)}
# reproduction check against release 3.1.0 (same code, same subsample, same seed)
chk = []
for f in glob.glob(f'{ROOT}/runs_r6/*__f10__*__empty__steps.json') + glob.glob(f'{ROOT}/runs_r6/*__f10__*__support__steps.json'):
    r6 = json.load(open(f)); v = U.get((r6['data'], r6['sub']), {}).get(r6['cond'])
    if v: chk.append(abs(v['test_auc'] - r6['test_auc']))
res['reproduction_max_abs_diff'] = float(max(chk)) if chk else None; res['reproduction_n'] = len(chk)

# 1. uniform mixing
uni = []
for a, b in (('uniform_emb', 'empty'), ('r6_er_emb', 'empty'), ('r6_support_emb', 'empty'), ('support', 'empty'),
             ('r6_support_emb', 'uniform_emb'), ('r6_er_emb', 'uniform_emb'), ('support', 'uniform_emb')):
    for k, metric in M.items():
        uni.append(dict(a=a, b=b, key=k, **summ(diff(a, b, metric))))
res['uniform'] = uni

# 2. tuned shrinkage
def select(prefix, base, cands, field):
    """per corpus: candidate maximising mean validation AUC over subsamples (base = no penalty / default lr)."""
    sel = {}
    for c in CORP:
        units = [v for (cc, s), v in U.items() if cc == c and sum(l in v for l in cands) > 1]   # subsamples where the grid was run
        if not units: continue
        score = {}
        for lab in cands:
            vals = [u[lab]['valid_auc'] for u in units if lab in u]
            if vals and len(vals) == len(units): score[lab] = float(np.mean(vals))
        if score: sel[c] = max(score, key=score.get)
    return sel
l2c = ['empty'] + [f'empty_l2|{l:g}' for l in LAMS]; l2s = ['support'] + [f'support_l2|{l:g}' for l in LAMS]
selC = select('empty', 'empty', l2c, 'l2'); selS = select('support', 'support', l2s, 'l2')
def diff_sel(selA, selB, metric, fixedA=None, fixedB=None):
    per = collections.defaultdict(list)
    for (c, s), v in U.items():
        a = fixedA or selA.get(c); b = fixedB or selB.get(c)
        if a in v and b in v and v[a].get(metric) is not None and v[b].get(metric) is not None:
            per[c].append(v[a][metric] - v[b][metric])
    return per
shr = []
for k, metric in M.items():
    shr.append(dict(cmp='support - tuned control', key=k, **summ(diff_sel(None, selC, metric, fixedA='support'))))
    shr.append(dict(cmp='tuned support - tuned control', key=k, **summ(diff_sel(selS, selC, metric))))
    shr.append(dict(cmp='tuned control - control', key=k, **summ(diff_sel(selC, None, metric, fixedB='empty'))))
    for l in LAMS:
        shr.append(dict(cmp=f'control_l2 {l:g} - control', key=k, **summ(diff(f'empty_l2|{l:g}', 'empty', metric))))
res['shrinkage'] = shr; res['shrinkage_selected'] = dict(control=selC, support=selS)

# 3. learning rate
lrc = ['empty'] + [f'empty|lr{l:g}' for l in LRS[1:]]; lrs_ = ['support'] + [f'support|lr{l:g}' for l in LRS[1:]]
selCl = select('empty', 'empty', lrc, 'lr'); selSl = select('support', 'support', lrs_, 'lr')
lr = []
for k, metric in M.items():
    for l in LRS:
        a = 'support' if l == 0.002 else f'support|lr{l:g}'; b = 'empty' if l == 0.002 else f'empty|lr{l:g}'
        lr.append(dict(cmp=f'support - control at lr {l:g}', key=k, **summ(diff(a, b, metric))))
    lr.append(dict(cmp='support - control at selected lr', key=k, **summ(diff_sel(selSl, selCl, metric))))
    lr.append(dict(cmp='control at selected lr - control at 0.002', key=k, **summ(diff_sel(selCl, None, metric, fixedB='empty'))))
res['learning_rate'] = lr; res['lr_selected'] = dict(control=selCl, support=selSl)

# 4-5. analyses on per-prediction files
from r5_controls import load
def pred(r): return np.load(f"{ROOT}/runs_r7/{r['tag']}_pred.npz")
def within_between(p, y, kc):
    num = 0.0; den = 0.0
    for k in np.unique(kc):
        s = kc == k; yy = y[s]
        npos = yy.sum(); nneg = len(yy) - npos
        if npos and nneg: w = npos * nneg; num += w * roc_auc_score(yy, p[s]); den += w
    within = num / den if den else np.nan
    mk = pd.Series(p).groupby(kc).transform('mean').values
    between = roc_auc_score(y, mk) if len(set(y)) > 1 else np.nan
    return within, between
terc = collections.defaultdict(lambda: collections.defaultdict(list)); dec = collections.defaultdict(lambda: collections.defaultdict(list))
triv = collections.defaultdict(list); trivgap = collections.defaultdict(list)
for (c, s), v in sorted(U.items()):
    if 'support' not in v or 'empty' not in v: continue
    df = load(c, 0.1, s); tr = df[df.split == 'train']
    nk = tr.groupby('kc').size(); rate = tr.groupby('kc').correct.mean(); g = tr.correct.mean()
    P = {k: pred(v[k]) for k in ('support', 'empty')}
    new = P['empty']['new'].astype(bool); kc = P['empty']['kc'][new]; y = P['empty']['y'][new].astype(int)
    assert (P['support']['kc'][new] == kc).all()
    n_of = nk.reindex(kc).fillna(0).values
    cuts = np.quantile(nk.reindex(range(int(df.kc.max()) + 1)).fillna(0).values, [1 / 3, 2 / 3])
    band = np.digitize(n_of, cuts, right=True)
    for b in range(3):
        sel = band == b
        if sel.sum() > 20 and 0 < y[sel].mean() < 1:
            terc[b][c].append(roc_auc_score(y[sel], P['support']['p'][new][sel]) - roc_auc_score(y[sel], P['empty']['p'][new][sel]))
    ws, bs = within_between(P['support']['p'][new], y, kc); we, be = within_between(P['empty']['p'][new], y, kc)
    dec['within'][c].append(ws - we); dec['between'][c].append(bs - be)
    tp = rate.reindex(kc).fillna(g).values; ta = roc_auc_score(y, tp)
    triv[c].append(ta); trivgap[c].append(v['empty']['test_auc_first'] - ta)
    dec['within_level_empty'][c].append(we); dec['between_level_empty'][c].append(be)
res['first_by_nk_tercile'] = {f'T{b + 1}': summ(terc[b]) for b in range(3)}
res['first_decomposition'] = {k: summ(v) for k, v in dec.items()}
res['trivial_first_auc'] = {c: float(np.mean(v)) for c, v in triv.items()}
res['control_minus_trivial_first'] = summ(trivgap)
json.dump(res, open(f'{OUT}/r7_results.json', 'w'), indent=1)
f4 = lambda x: f'{x:+.4f}'
print('runs', len(R7), 'repro max diff', res['reproduction_max_abs_diff'], 'n', res['reproduction_n'])
for name in ('uniform', 'shrinkage', 'learning_rate'):
    for r in res[name]:
        mx = r['mixed']; print(name, r.get('a', r.get('cmp')), '-', r.get('b', ''), r['key'], f4(r['mean']), f"{r['pos']}/{r['n']}", f"p={r['p']:.3f}", f"[{f4(mx.get('lo', 0))},{f4(mx.get('hi', 0))}]")
print('selected', res['shrinkage_selected'], res['lr_selected'])
for k, r in res['first_by_nk_tercile'].items(): print('tercile', k, f4(r['mean']), f"{r['pos']}/{r['n']}", f"[{f4(r['mixed'].get('lo',0))},{f4(r['mixed'].get('hi',0))}]")
for k, r in res['first_decomposition'].items(): print('decomp', k, f4(r['mean']), f"{r['pos']}/{r['n']}", f"[{f4(r['mixed'].get('lo',0))},{f4(r['mixed'].get('hi',0))}]")
print('trivial', {c: round(v, 4) for c, v in res['trivial_first_auc'].items()}, 'control-trivial', f4(res['control_minus_trivial_first']['mean']))
