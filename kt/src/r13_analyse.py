"""Release 3.10.0: analyses of the TLT revision.

Reads run and per-prediction files only: runs_r13 (new runs of r13_controls.py and, in runs_r13/gkt, of r9_controls.py)
and the archived CPU runs of runs_r6, runs_r7, runs_r8, runs_r9, runs_r10 and runs_r11 that they pair with. Every
comparison pairs runs on the same corpus x subsample unit, on a CPU with one thread.

 1. Update budget along the learning curve: support - empty (both entry points) at 10% (archived runs_r6), 25% (three
    subsamples) and 100% (one subsample), on all predictions and on first attempts; Holm over these six tests. The same for
    support_emb - empty, rvec_emb - empty and support - rvec_emb at 25% and 100%.
 2. Capacity controls at 10% (32 units): empty_d128 and tvec_emb against empty and rvec_emb.
 3. Learning rate selected per corpus on validation AUC from {0.002, 0.005, 0.01}, for empty and rvec_emb.
 4. SAKT-E on a CPU (24 units): ladder and the random-vector controls.
 5. State-routing channel (st_*): ladder empty -> U -> ER -> DP -> G inside the probe.
 6. GKT on a CPU (16 units): U - empty, G - empty, G - U.
 7. The reordered ladder empty -> RV(F) -> RV -> ER -> DP -> G at 10% (archived runs, as in r11_analyse.py).
 8. Decision analysis (25% and 100%): agreement of the KC a simple rule would present next, between conditions, from the
    candidate scores of r13_controls.py --dump; and the share of first attempts whose 0.5-threshold decision flips.
 9. Weighting by first attempts and leaving out the four small corpora, for the main first-attempt comparisons.
Output: results_r13/r13_results.json
"""
import os, sys, json, glob, warnings, itertools
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
OUT = f'{REPO}/results_r13'; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
SMALL = ['statics11', 'algebra05', 'bridge06', 'assist17']
CORPORA = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']


def mixed(df):
    try:
        import statsmodels.formula.api as smf
        if df.data.nunique() < 3: return None
        if df.groupby('data').size().max() == 1:
            r = smf.ols('d ~ 1', df).fit(); lo, hi = r.conf_int().loc['Intercept']
            return dict(est=float(r.params['Intercept']), lo=float(lo), hi=float(hi), ols=True)
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
    if not rows: return None
    df = pd.DataFrame(rows).dropna(subset=['d'])
    if df.empty: return None
    per = df.groupby('data').d.mean()
    o = dict(units=int(len(df)), n=int(len(per)), mean=float(per.mean()), pos=int((per > 0).sum()),
             per_corpus={k: float(v) for k, v in per.items()})
    if len(per) >= 4: o['p_wilcoxon'] = float(wilcoxon(per.values).pvalue)
    o['mixed'] = mixed(df); o['boot'] = boot(df) if len(per) >= 3 else None
    return o


def holm(ps):
    k = [i for i, p in enumerate(ps) if p is not None]; order = sorted(k, key=lambda i: ps[i]); out = [None] * len(ps); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(k) - r) * ps[i])); out[i] = run
    return out


def load_runs(paths, pred_ok=False):
    """{(data, frac, sub, cond, rep, lr): (json, stem)} for update-budget runs."""
    R = {}
    for path in paths:
        for f in glob.glob(f'{path}/*.json'):
            r = json.load(open(f))
            if r.get('budget') != 'steps' or r.get('l2', 0) not in (0, 0.0, None): continue
            key = (r['data'], round(float(r.get('frac', 0.1)), 2), int(r['sub']), r['cond'], int(r.get('rep', 0) or 0),
                   round(float(r.get('lr', 0.002)), 4))
            R[key] = (r, f[:-5])
    return R


def pairs(R, units, a, b, frac=0.1, col='test_auc_first', rep_a=0, rep_b=0, lr=0.002):
    rows = []
    for c, s in units:
        ka = (c, frac, s, a, rep_a, lr); kb = (c, frac, s, b, rep_b, lr)
        if ka in R and kb in R and R[ka][0][col] is not None and R[kb][0][col] is not None:
            rows.append(dict(data=c, sub=s, d=R[ka][0][col] - R[kb][0][col]))
    return rows


def both(R, units, a, b, **kw):
    return {'first': summ(pairs(R, units, a, b, col='test_auc_first', **kw)), 'all': summ(pairs(R, units, a, b, col='test_auc', **kw))}


res = {}
N13 = load_runs([f'{REPO}/runs_r13'])
U10 = [(c, s) for c in SMALL for s in range(5)] + [(c, s) for c in CORPORA if c not in SMALL for s in range(2)]
U25 = [(c, s) for c in CORPORA for s in range(3)]; U100 = [(c, 0) for c in CORPORA]
SIX = [(c, s) for c in SMALL for s in range(5)] + [(c, s) for c in ('assist09', 'assist15') for s in range(2)]

# 1. update budget along the learning curve ------------------------------------------------------------------------
R6 = load_runs([f'{REPO}/runs_r6'])
lc = {}
lc['10'] = both(R6, U10 if False else [(c, s) for c in CORPORA for s in range(5)], 'support', 'empty')
lc['25'] = both(N13, U25, 'support', 'empty', frac=0.25)
lc['100'] = both(N13, U100, 'support', 'empty', frac=1.0)
tests = [(f, k) for f in ('10', '25', '100') for k in ('all', 'first')]
ps = [(lc[f][k] or {}).get('p_wilcoxon') for f, k in tests]
for (f, k), h in zip(tests, holm(ps)):
    if lc[f][k]: lc[f][k]['holm6'] = h
res['update_curve_support_empty'] = lc
res['update_curve_other'] = {f: {f'{a} - {b}': both(N13, U, a, b, frac=fr)
                                 for a, b in (('support_emb', 'empty'), ('rvec_emb', 'empty'), ('support', 'rvec_emb'),
                                              ('support_emb', 'rvec_emb'))}
                             for f, U, fr in (('25', U25, 0.25), ('100', U100, 1.0))}
res['update_curve_steps'] = {f: {cond: dict(median_steps=float(np.median([N13[k][0]['steps'] for k in N13 if k[1] == fr and k[3] == cond and k[4] == 0])) if any(k[1] == fr and k[3] == cond for k in N13) else None,
                                            hit_max=int(sum(N13[k][0]['hit_max'] for k in N13 if k[1] == fr and k[3] == cond)))
                                 for cond in ('empty', 'support', 'rvec_emb')} for f, fr in (('25', 0.25), ('100', 1.0))}

# reference runs at 10% (archived; EdNet from the rebuilt data of this release when available) --------------------
ARCH = {}
for path, conds in ((f'{REPO}/runs_r9', {'empty', 'support_emb', 'er_emb', 'erR_emb', 'dp_emb'}), (f'{REPO}/runs_r7', {'uniform_emb', 'empty', 'support'}),
                    (f'{REPO}/runs_r10', {'rvec_emb', 'rvec1_emb', 'erE2_emb'}), (f'{REPO}/runs_r11', {'rvecF_emb', 'rvec1F_emb'})):
    for k, v in load_runs([path]).items():
        if k[3] in conds and k[0] != 'ednet' and k[1] == 0.1: ARCH.setdefault(k, v)
for path in (f'{REPO}/runs_r10/repro', f'{REPO}/runs_r10/base', f'{REPO}/runs_r11/base', f'{REPO}/runs_r11', f'{REPO}/runs_r10'):
    for k, v in load_runs([path]).items():
        if k[0] == 'ednet' and k[1] == 0.1: ARCH.setdefault(k, v)
R7lr = {k: v for k, v in load_runs([f'{REPO}/runs_r7']).items() if k[0] != 'ednet'}
REF = dict(ARCH); REF.update({k: v for k, v in R7lr.items() if k[5] != 0.002})
# EdNet: rerun conditions of this release replace the archived ones when present
ed_new = {k: v for k, v in N13.items() if k[0] == 'ednet' and k[1] == 0.1}
repro = {}
for k, v in ed_new.items():
    if k in REF: repro['|'.join(map(str, k))] = dict(archived=REF[k][0]['test_auc'], rerun=v[0]['test_auc'])
res['ednet_rerun_check'] = repro
REF.update(ed_new)
REF.update({k: v for k, v in N13.items() if k[1] == 0.1})

# 2. capacity controls --------------------------------------------------------------------------------------------
res['capacity'] = {f'{a} - {b}': both(REF, U10, a, b) for a, b in
                   (('empty_d128', 'empty'), ('tvec_emb', 'empty'), ('rvec_emb', 'empty'), ('rvec_emb', 'empty_d128'),
                    ('rvec_emb', 'tvec_emb'), ('support_emb', 'empty_d128'), ('support_emb', 'tvec_emb'))}
res['capacity_params'] = {cond: int(np.median([v[0]['n_trainable'] for k, v in N13.items() if k[3] == cond and k[1] == 0.1]))
                          for cond in ('empty_d128', 'tvec_emb') if any(k[3] == cond for k in N13)}

# 3. learning rate selected per corpus ----------------------------------------------------------------------------
def selected(cond):
    out = {}
    for c in CORPORA:
        us = [s for cc, s in U10 if cc == c]
        best = None
        for lr in (0.002, 0.005, 0.01):
            ks = [(c, 0.1, s, cond, 0, lr) for s in us]
            if all(k in REF for k in ks):
                va = np.mean([REF[k][0]['valid_auc'] for k in ks])
                if best is None or va > best[0]: best = (va, lr)
        if best: out[c] = best[1]
    return out
sel = {cond: selected(cond) for cond in ('empty', 'rvec_emb', 'support')}
res['lr_selected'] = sel
rows = {'first': [], 'all': []}
for c, s in U10:
    if c in sel['empty'] and c in sel['rvec_emb']:
        ka = (c, 0.1, s, 'rvec_emb', 0, sel['rvec_emb'][c]); kb = (c, 0.1, s, 'empty', 0, sel['empty'][c])
        if ka in REF and kb in REF:
            for key, col in (('first', 'test_auc_first'), ('all', 'test_auc')):
                rows[key].append(dict(data=c, sub=s, d=REF[ka][0][col] - REF[kb][0][col]))
res['lr_rvec_minus_empty'] = {k: summ(v) for k, v in rows.items()}

# 4. SAKT-E on a CPU ----------------------------------------------------------------------------------------------
res['sakte_cpu'] = {f'{a} - {b}': both(N13, SIX, a, b) for a, b in
                    (('sakte_uniform', 'sakte_empty'), ('sakte_er', 'sakte_uniform'), ('sakte_support', 'sakte_er'),
                     ('sakte_support', 'sakte_empty'), ('sakte_rvec', 'sakte_empty'), ('sakte_rvec', 'sakte_uniform'),
                     ('sakte_rvecF', 'sakte_empty'), ('sakte_rvec', 'sakte_rvecF'), ('sakte_support', 'sakte_rvec'),
                     ('sakte_er', 'sakte_empty'))}

# 5. state-routing channel -----------------------------------------------------------------------------------------
res['state_channel'] = {f'{a} - {b}': both(N13, U10, a, b) for a, b in
                        (('st_uniform', 'st_empty'), ('st_er', 'st_uniform'), ('st_dp', 'st_er'), ('st_support', 'st_dp'),
                         ('st_support', 'st_empty'), ('st_er', 'st_empty'), ('st_support', 'st_uniform'), ('st_support', 'st_er'))}
# the channel against the probe's own empty-graph control (different initial weights; descriptive)
res['state_vs_probe_empty'] = {cond: both({**REF, **{(k[0], k[1], k[2], k[3], k[4], k[5]): v for k, v in N13.items()}}, U10, cond, 'empty')
                               for cond in ('st_empty', 'st_support')}

# 6. GKT on a CPU -------------------------------------------------------------------------------------------------
G = {}
for path in (f'{REPO}/runs_r8', f'{REPO}/runs_r9', f'{REPO}/runs_r13/gkt'):
    for k, v in load_runs([path]).items():
        if v[0].get('family') == 'gkt' and str(v[0].get('device', 'cpu')) == 'cpu' and k[4] == 0: G[k] = v
GU = [(c, s) for c in SMALL for s in range(3)] + [(c, s) for c in ('assist09', 'assist15') for s in range(2)]
res['gkt_cpu'] = {f'{a} - {b}': both(G, GU, a, b) for a, b in
                  (('gkt_uniform', 'gkt_empty'), ('gkt_support', 'gkt_empty'), ('gkt_support', 'gkt_uniform'))}
res['gkt_cpu_units'] = sorted({f'{k[0]}:{k[2]}' for k in G if k[3] == 'gkt_uniform'})
res['gkt_training'] = {c: dict(median_steps=float(np.median([v[0]['steps'] for k, v in G.items() if k[0] == c])),
                               min_best_step=int(min(v[0]['best_step'] for k, v in G.items() if k[0] == c)),
                               hit_max=int(sum(v[0]['hit_max'] for k, v in G.items() if k[0] == c)))
                       for c in sorted({k[0] for k in G})}

# 7. reordered ladder at 10% --------------------------------------------------------------------------------------
lad = [('rvecF_emb', 'empty'), ('rvec_emb', 'rvecF_emb'), ('er_emb', 'rvec_emb'), ('dp_emb', 'er_emb'), ('support_emb', 'dp_emb'),
       ('support_emb', 'empty'), ('uniform_emb', 'empty'), ('er_emb', 'uniform_emb'), ('rvec_emb', 'uniform_emb')]
ARCH_ED = dict(ARCH)  # archived only, as in Table III
res['ladder_rv'] = {f'{a} - {b}': both(ARCH_ED, U10, a, b) for a, b in lad}

# 8. decision analysis --------------------------------------------------------------------------------------------
def decisions(sa, sb, seen, target=0.65, k=5):
    ok = ~np.isnan(sa[..., 0].astype(np.float32))
    agreeA = []; agreeB = []; jac = []; rho = []
    for i, t in zip(*np.nonzero(ok)):
        cand = ~seen[i, t]
        if cand.sum() < 2: continue
        a = sa[i, t, cand].astype(np.float32); b = sb[i, t, cand].astype(np.float32)
        agreeA.append(np.argmin(np.abs(a - target)) == np.argmin(np.abs(b - target)))
        agreeB.append(np.argmax(a) == np.argmax(b))
        kk = min(k, len(a)); ta = set(np.argsort(-a)[:kk]); tb = set(np.argsort(-b)[:kk]); jac.append(len(ta & tb) / len(ta | tb))
        if len(a) > 2: rho.append(spearmanr(a, b).correlation)
    return dict(n=len(agreeA), target=float(np.mean(agreeA)) if agreeA else None, easiest=float(np.mean(agreeB)) if agreeB else None,
                top5_jaccard=float(np.mean(jac)) if jac else None, spearman=float(np.nanmean(rho)) if rho else None)

dec = {}
for fr, U in ((0.25, U25), (1.0, U100)):
    out = {}
    for (a, ra), (b, rb) in ((('support', 0), ('empty', 0)), (('rvec_emb', 0), ('empty', 0)), (('support', 0), ('rvec_emb', 0)),
                             (('support_emb', 0), ('empty', 0)), (('empty', 1), ('empty', 0))):
        rows = []
        for c, s in U:
            ka = (c, fr, s, a, ra, 0.002); kb = (c, fr, s, b, rb, 0.002)
            if ka in N13 and kb in N13 and os.path.exists(N13[ka][1] + '_cand.npz') and os.path.exists(N13[kb][1] + '_cand.npz'):
                A = np.load(N13[ka][1] + '_cand.npz'); B = np.load(N13[kb][1] + '_cand.npz')
                assert (A['seen'] == B['seen']).all()
                d = decisions(A['score'], B['score'], A['seen']); d.update(data=c, sub=s); rows.append(d)
        if rows:
            df = pd.DataFrame(rows); per = df.groupby('data')[['target', 'easiest', 'top5_jaccard', 'spearman']].mean()
            out[f'{a}{"_rep" + str(ra) if ra else ""} vs {b}{"_rep" + str(rb) if rb else ""}'] = dict(
                units=len(df), corpora=int(len(per)), decisions=int(df.n.sum()),
                **{f'{m}_mean': float(per[m].mean()) for m in per.columns},
                **{f'{m}_range': [float(per[m].min()), float(per[m].max())] for m in per.columns},
                per_corpus=per.round(4).to_dict(orient='index'))
    dec[str(fr)] = out
res['decisions'] = dec

def flip(stem_a, stem_b, thr=0.5):
    A = np.load(stem_a + '_pred.npz'); B = np.load(stem_b + '_pred.npz'); new = A['new'].astype(bool)
    assert (A['y'] == B['y']).all()
    return float(((A['p'][new] >= thr) != (B['p'][new] >= thr)).mean())
fl = {}
for lab, Rset, U, fr, a, b in (('10%: support_emb vs empty', REF, U10, 0.1, 'support_emb', 'empty'),
                               ('10%: rvec_emb vs empty', REF, U10, 0.1, 'rvec_emb', 'empty'),
                               ('25%: support vs empty', N13, U25, 0.25, 'support', 'empty'),
                               ('25%: rvec_emb vs empty', N13, U25, 0.25, 'rvec_emb', 'empty'),
                               ('100%: support vs empty', N13, U100, 1.0, 'support', 'empty')):
    rows = []
    for c, s in U:
        ka = (c, fr, s, a, 0, 0.002); kb = (c, fr, s, b, 0, 0.002)
        if ka in Rset and kb in Rset and os.path.exists(Rset[ka][1] + '_pred.npz') and os.path.exists(Rset[kb][1] + '_pred.npz'):
            try: rows.append(dict(data=c, sub=s, d=flip(Rset[ka][1], Rset[kb][1])))
            except AssertionError: pass
    if rows:
        df = pd.DataFrame(rows); per = df.groupby('data').d.mean(); fl[lab] = dict(mean=float(per.mean()), min=float(per.min()), max=float(per.max()), n=int(len(per)))
rows = []
for c in CORPORA:
    ka = (c, 0.25, 0, 'empty', 1, 0.002); kb = (c, 0.25, 0, 'empty', 0, 0.002)
    if ka in N13 and kb in N13: rows.append(flip(N13[ka][1], N13[kb][1]))
if rows: fl['25%: empty seed 100 vs seed 0 (subsample 0)'] = dict(mean=float(np.mean(rows)), min=float(np.min(rows)), max=float(np.max(rows)), n=len(rows))
res['threshold_flips_first'] = fl

# 9. weighting by first attempts and leaving out the small corpora ------------------------------------------------
r34 = json.load(open(f'{REPO}/results_r6/r34_results.json'))
nfirst = {c: v['n_first'] for c, v in r34['test_counts_10'].items()}
wt = {}
for lab, per in [(f"{r['label']} | {r['budget']} | {r['frac']} | {r['key']}", r['per_corpus']) for r in r34['robustness'][:6]]:
    v = pd.Series(per); w = pd.Series(nfirst)[v.index]
    big = v.drop(SMALL)
    wt[lab] = dict(unweighted=float(v.mean()), weighted_first_attempts=float((v * w).sum() / w.sum()),
                   without_small_mean=float(big.mean()), without_small_pos=int((big > 0).sum()),
                   without_small_p=float(wilcoxon(big.values).pvalue))
for f in ('25',):
    s = lc[f]['first']
    if s:
        v = pd.Series(s['per_corpus']); w = pd.Series(nfirst)[v.index]; big = v.drop([c for c in SMALL if c in v.index])
        wt[f'support-empty | steps | 0.25 | first'] = dict(unweighted=float(v.mean()), weighted_first_attempts=float((v * w).sum() / w.sum()),
                                                            without_small_mean=float(big.mean()), without_small_pos=int((big > 0).sum()),
                                                            without_small_p=float(wilcoxon(big.values).pvalue) if len(big) >= 4 else None)
res['weighting'] = wt
res['n_runs_r13'] = dict(probe_and_sakte=len(N13), gkt=len(glob.glob(f'{REPO}/runs_r13/gkt/*.json')))
json.dump(res, open(f'{OUT}/r13_results.json', 'w'), indent=1, default=float)
print('written', OUT, res['n_runs_r13'])
