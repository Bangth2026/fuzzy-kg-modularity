"""Release 3.11.0: analyses of the second revision for IEEE Transactions on Learning Technologies.

Reads run, per-prediction and candidate files only (runs_r7 to runs_r14). Writes results_r14/r14_results.json.

 1. Equivalence: every term of the null-graph ladders (Tables III and IV) and of the graph-free controls with a 95% and
    a 90% mixed-model interval, two one-sided tests against DELTA = 0.002 AUC, the minimal detectable effect at 80%
    power, and a corpus-then-subsample bootstrap interval.
 2. Next-KC decisions at 25% with seeds crossed: (G, seed j) against (empty, seed k) for j != k, against
    (empty, seed j) against (empty, seed k); rules closest-to-0.65, easiest, top-5 overlap, band of +/- 0.05 around 0.65.
 3. Decision regret under simulated learners: the mean of three 100% models (empty-graph control, or support graph) plays
    the learner; the regret of a rule's choice made with a 25% model is the oracle's distance from the target (or from
    the oracle's easiest KC) at the chosen KC minus the smallest such distance.
 4. All training learners with three seeds: support - empty, rvec_emb - empty, support - rvec_emb.
 5. Confirmatory replication (prereg/replication_plan_r14.md) on XES3G5M and Eedi tasks 1-2.
 6. Curriculum-order null UR(G) and authored groupings, embedding refinement and state channel.
 7. First attempts with a cap of 500 interactions per learner.
 8. GKT on a CPU: the complete ladder empty, U, ER, DP, G on 16 units, and the authored grouping.
 9. Values for the effect-size figure (Fig. 4).
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r14_common import *
OUT = f'{REPO}/results_r14'; os.makedirs(OUT, exist_ok=True)
res = {'delta': DELTA}
ARCH, N13, REF = reference_runs(); LAD = ladder_runs()
N14 = load_runs([f'{REPO}/runs_r14']); G14 = load_runs([f'{REPO}/runs_r14/gkt'])
SIX = [(c, s) for c in SMALL for s in range(5)] + [(c, s) for c in ('assist09', 'assist15') for s in range(2)]

# 1. equivalence for every ladder term ----------------------------------------------------------------------------
eq = {}
eq['probe_E'] = {f'{a} - {b}': both(LAD, U10, a, b) for a, b in
                 (('uniform_emb', 'empty'), ('er_emb', 'uniform_emb'), ('dp_emb', 'er_emb'), ('support_emb', 'dp_emb'),
                  ('support_emb', 'empty'), ('erR_emb', 'uniform_emb'), ('er_emb', 'erR_emb'))}
eq['probe_E_rv'] = {f'{a} - {b}': both(ARCH, U10, a, b) for a, b in
                    (('rvec_emb', 'uniform_emb'), ('rvec1_emb', 'uniform_emb'), ('rvecF_emb', 'empty'), ('rvec_emb', 'rvecF_emb'),
                     ('er_emb', 'rvec_emb'), ('support_emb', 'rvec_emb'))}
eq['capacity'] = {f'{a} - {b}': both(REF, U10, a, b) for a, b in
                  (('empty_d128', 'empty'), ('rvec_emb', 'empty_d128'), ('support_emb', 'empty_d128'), ('tvec_emb', 'empty'))}
eq['state'] = {f'{a} - {b}': both(N13, U10, a, b) for a, b in
               (('st_uniform', 'st_empty'), ('st_er', 'st_uniform'), ('st_dp', 'st_er'), ('st_support', 'st_dp'), ('st_support', 'st_empty'),
                ('st_er', 'st_empty'))}
R8 = load_runs([f'{REPO}/runs_r8'])
eq['sakte_gpu'] = {f'{a} - {b}': both(R8, SIX, a, b) for a, b in
                   (('sakte_uniform', 'sakte_empty'), ('sakte_er', 'sakte_uniform'), ('sakte_support', 'sakte_er'), ('sakte_support', 'sakte_empty'))}
res['equivalence'] = eq

# 2./3. next-KC decisions ------------------------------------------------------------------------------------------
TGT = 0.65


def choose(s, cand):
    a = np.where(cand, s, np.nan)
    return dict(target=int(np.nanargmin(np.abs(a - TGT))), easiest=int(np.nanargmax(a)))


def agree(sa, sb, seen):
    ok = ~np.isnan(sa[..., 0].astype(np.float32)); rows = []
    for i, t in zip(*np.nonzero(ok)):
        cand = ~seen[i, t]
        if cand.sum() < 2 or np.isnan(sb[i, t, 0]): continue
        a = sa[i, t].astype(np.float32); b = sb[i, t].astype(np.float32)
        ca, cb = choose(a, cand), choose(b, cand); idx = np.flatnonzero(cand); aa, bb = a[idx], b[idx]
        kk = min(5, len(idx))
        t5 = len(set(np.argsort(-aa)[:kk]) & set(np.argsort(-bb)[:kk])) / kk
        g5 = len(set(np.argsort(np.abs(aa - TGT))[:kk]) & set(np.argsort(np.abs(bb - TGT))[:kk])) / kk
        Ba = set(np.flatnonzero(np.abs(aa - TGT) <= .05)); Bb = set(np.flatnonzero(np.abs(bb - TGT) <= .05))
        band = len(Ba & Bb) / len(Ba | Bb) if (Ba | Bb) else np.nan
        rows.append(dict(target=ca['target'] == cb['target'], easiest=ca['easiest'] == cb['easiest'], top5_easiest=t5, top5_target=g5,
                         band=band, spearman=spearmanr(aa, bb).correlation if len(idx) > 2 else np.nan))
    return pd.DataFrame(rows)


def cand(R, key):
    if key not in R or not os.path.exists(R[key][1] + '_cand.npz'): return None
    return np.load(R[key][1] + '_cand.npz')


C25 = {**N13, **N14}
crossed = {'G_j vs E_k (j != k)': [], 'E_j vs E_k (j != k)': [], 'G_j vs E_j': [], 'G_j vs G_k (j != k)': []}
for c in CORPORA:
    S = {(m, r): cand(C25, (c, 0.25, 0, m, r, 0.002)) for m in ('support', 'empty') for r in (0, 1, 2)}
    if any(v is None for v in S.values()): continue
    for (lab, ma, mb, same) in (('G_j vs E_k (j != k)', 'support', 'empty', False), ('E_j vs E_k (j != k)', 'empty', 'empty', False),
                                ('G_j vs E_j', 'support', 'empty', True), ('G_j vs G_k (j != k)', 'support', 'support', False)):
        for j in range(3):
            for k in range(3):
                if (j == k) != same or (ma == mb and j >= k): continue
                A, B = S[(ma, j)], S[(mb, k)]
                if A['score'].shape != B['score'].shape or not (A['seen'] == B['seen']).all(): continue
                d = agree(A['score'], B['score'], A['seen']).mean(numeric_only=True).to_dict(); d.update(data=c, pair=f'{j}-{k}')
                crossed[lab].append(d)
dec = {}
for lab, rows in crossed.items():
    if not rows: continue
    df = pd.DataFrame(rows); per = df.groupby('data').mean(numeric_only=True)
    dec[lab] = dict(corpora=int(len(per)), pairs=int(len(df)), **{m: float(per[m].mean()) for m in per.columns},
                    per_corpus=per.round(4).to_dict(orient='index'))
# paired comparison of the two kinds of pair, per corpus
if dec.get('G_j vs E_k (j != k)') and dec.get('E_j vs E_k (j != k)'):
    a = pd.DataFrame(crossed['G_j vs E_k (j != k)']).groupby('data').mean(numeric_only=True)
    b = pd.DataFrame(crossed['E_j vs E_k (j != k)']).groupby('data').mean(numeric_only=True)
    dd = (a - b).dropna()
    dec['difference (G vs E crossed) - (E vs E crossed)'] = {m: dict(mean=float(dd[m].mean()), pos=int((dd[m] > 0).sum()), n=int(len(dd)),
                                                                     p_wilcoxon=float(wilcoxon(dd[m].values).pvalue) if len(dd) >= 6 and dd[m].abs().sum() > 0 else None)
                                                             for m in dd.columns}
res['decisions_crossed_25'] = dec

# flips at 0.5 on first attempts, crossed seeds
def flip(sa, sb):
    A = np.load(sa + '_pred.npz'); B = np.load(sb + '_pred.npz'); new = A['new'].astype(bool)
    if not (A['y'] == B['y']).all(): return None
    return float(((A['p'][new] >= .5) != (B['p'][new] >= .5)).mean())
fl = {'G_j vs E_k (j != k)': [], 'E_j vs E_k (j != k)': [], 'G_j vs E_j': []}
for c in CORPORA:
    for j in range(3):
        for k in range(3):
            ka = (c, 0.25, 0, 'support', j, 0.002); kb = (c, 0.25, 0, 'empty', k, 0.002); ke = (c, 0.25, 0, 'empty', j, 0.002)
            if ka in C25 and kb in C25:
                v = flip(C25[ka][1], C25[kb][1])
                if v is not None: fl['G_j vs E_j' if j == k else 'G_j vs E_k (j != k)'].append(dict(data=c, v=v))
            if j < k and ke in C25 and kb in C25:
                v = flip(C25[ke][1], C25[kb][1])
                if v is not None: fl['E_j vs E_k (j != k)'].append(dict(data=c, v=v))
res['flips_first_crossed_25'] = {k: dict(mean=float(pd.DataFrame(v).groupby('data').v.mean().mean()), corpora=int(pd.DataFrame(v).data.nunique()))
                                 for k, v in fl.items() if v}

# 3. regret under simulated learners
reg_rows = []
for c in CORPORA:
    O = {}
    for om in ('empty', 'support'):
        arrs = [cand(C25, (c, 1.0, 0, om, r, 0.002)) for r in (0, 1, 2)]
        if any(a is None for a in arrs): continue
        O[om] = (np.mean([a['score'].astype(np.float32) for a in arrs], 0), arrs[0]['seen'])
    if not O: continue
    models = {(m, r): cand(C25, (c, 0.25, 0, m, r, 0.002)) for m in ('empty', 'support') for r in (0, 1, 2)}
    models[('rvec_emb', 0)] = cand(C25, (c, 0.25, 0, 'rvec_emb', 0, 0.002)); models[('support_emb', 0)] = cand(C25, (c, 0.25, 0, 'support_emb', 0, 0.002))
    for om, (Os, Oseen) in O.items():
        for (m, r), A in models.items():
            if A is None: continue
            Kc = min(A['score'].shape[2], Os.shape[2])   # a KC absent from the 25% subsample has no column there
            if A['score'].shape[:2] != Os.shape[:2] or not (A['seen'][..., :Kc] == Oseen[..., :Kc]).all(): continue
            sa = A['score'][..., :Kc].astype(np.float32); Os_ = Os[..., :Kc]; ok = ~np.isnan(sa[..., 0]) & ~np.isnan(Os_[..., 0]); rt = []; re = []; rr = []
            rng = np.random.default_rng(0)
            for i, t in zip(*np.nonzero(ok)):
                cd = ~A['seen'][i, t, :Kc]
                if cd.sum() < 2: continue
                ch = choose(sa[i, t], cd); o = np.where(cd, Os_[i, t], np.nan)
                dist = np.abs(o - TGT); rt.append(dist[ch['target']] - np.nanmin(dist)); re.append(np.nanmax(o) - o[ch['easiest']])
                rr.append(dist[rng.choice(np.flatnonzero(cd))] - np.nanmin(dist))
            reg_rows.append(dict(data=c, oracle=om, model=m, rep=r, regret_target=float(np.mean(rt)), regret_easiest=float(np.mean(re)),
                                 regret_target_random=float(np.mean(rr)), n=len(rt)))
REG = pd.DataFrame(reg_rows); regret = {}
if len(REG):
    for om, g in REG.groupby('oracle'):
        tab = g.groupby(['model', 'rep'])[['regret_target', 'regret_easiest', 'regret_target_random']].mean()
        out = dict(by_model=tab.round(5).reset_index().to_dict(orient='records'), corpora=int(g.data.nunique()))
        for lab, (ma, ra, mb, rb_set) in (('support - empty (same seed)', ('support', None, 'empty', None)),):
            pass
        # paired by corpus: support rep j - empty rep j (mean over j); empty rep j - empty rep k (seed spread)
        def per(model, rep): return g[(g.model == model) & (g.rep == rep)].set_index('data')
        for m in ('regret_target', 'regret_easiest'):
            dG = pd.concat([per('support', j)[m] - per('empty', j)[m] for j in range(3)], axis=1).mean(1).dropna()
            dS = pd.concat([(per('empty', j)[m] - per('empty', k)[m]).abs() for j in range(3) for k in range(j + 1, 3)], axis=1).mean(1).dropna()
            dGabs = pd.concat([(per('support', j)[m] - per('empty', j)[m]).abs() for j in range(3)], axis=1).mean(1).dropna()
            out[m] = dict(support_minus_empty=float(dG.mean()), pos=int((dG > 0).sum()), n=int(len(dG)),
                          p_wilcoxon=float(wilcoxon(dG.values).pvalue) if len(dG) >= 6 else None,
                          abs_support_minus_empty=float(dGabs.mean()), abs_seed_spread_empty=float(dS.mean()),
                          per_corpus=dG.round(5).to_dict())
        regret[om] = out
res['regret_25'] = regret

# 4. all training learners, three seeds ----------------------------------------------------------------------------
A100 = {**N13, **N14}
def seeds100(a, b, col):
    rows = []
    for c in CORPORA:
        for r in (0, 1, 2):
            ka = (c, 1.0, 0, a, r, 0.002); kb = (c, 1.0, 0, b, r, 0.002)
            if ka in A100 and kb in A100 and A100[ka][0][col] is not None and A100[kb][0][col] is not None:
                rows.append(dict(data=c, sub=r, d=A100[ka][0][col] - A100[kb][0][col]))
    return rows
f100 = {}
for a, b in (('support', 'empty'), ('rvec_emb', 'empty'), ('support', 'rvec_emb')):
    f100[f'{a} - {b}'] = {k: summ(seeds100(a, b, col)) for k, col in (('first', 'test_auc_first'), ('all', 'test_auc'))}
    df = pd.DataFrame(seeds100(a, b, 'test_auc_first'))
    if len(df): f100[f'{a} - {b}']['seed_sd_first'] = float(df.groupby('data').d.std(ddof=1).mean())
    if len(df): f100[f'{a} - {b}']['seeds_per_corpus'] = {k: int(v) for k, v in df.groupby('data').size().items()}
ps = [f100[k][kk]['p_wilcoxon'] if f100[k][kk] and 'p_wilcoxon' in f100[k][kk] else None for k in ('support - empty',) for kk in ('all', 'first')]
for kk, h in zip(('all', 'first'), holm(ps)):
    if f100['support - empty'][kk]: f100['support - empty'][kk]['holm2'] = h
res['all_learners_3seeds'] = f100

# 5. confirmatory replication --------------------------------------------------------------------------------------
REP = ['xes3g5m', 'eedi12']; UR = [(c, s) for c in REP for s in range(5)]
def strat_boot(rows, B=10000, seed=2026):
    df = pd.DataFrame(rows); rng = np.random.default_rng(seed); g = {c: v.d.values for c, v in df.groupby('data')}; out = []
    for _ in range(B): out.append(np.mean([rng.choice(v, len(v)).mean() for v in g.values()]))
    return np.array(out)
rep = {}
for lab, a, b, kind in (('H1 support - empty', 'support', 'empty', 'pos'), ('H2a rvec_emb - empty', 'rvec_emb', 'empty', 'pos'),
                        ('H2b support_emb - rvec_emb', 'support_emb', 'rvec_emb', 'equiv'), ('H3 st_support - st_dp', 'st_support', 'st_dp', 'pos'),
                        ('H4 st_er - st_empty', 'st_er', 'st_empty', 'below'), ('support_emb - empty', 'support_emb', 'empty', 'desc'),
                        ('st_support - st_empty', 'st_support', 'st_empty', 'desc'), ('st_dp - st_er', 'st_dp', 'st_er', 'desc')):
    o = {}
    for k, col in (('first', 'test_auc_first'), ('all', 'test_auc')):
        rows = pairs(N14, UR, a, b, col=col)
        if not rows: continue
        df = pd.DataFrame(rows); per = df.groupby('data').d.mean(); bt = strat_boot(rows)
        q = lambda p: float(np.quantile(bt, p))
        e = dict(units=int(len(df)), mean=float(per.mean()), per_corpus={c: float(v) for c, v in per.items()},
                 units_pos=int((df.d > 0).sum()), ci95=[q(.025), q(.975)], ci90=[q(.05), q(.95)])
        if kind == 'pos': e['verdict'] = 'supported' if q(.025) > 0 else ('contradicted' if q(.975) < 0 else 'not resolved')
        if kind == 'below': e['verdict'] = 'supported' if q(.95) < DELTA else ('contradicted' if q(.05) > DELTA else 'not resolved')
        if kind == 'equiv': e['verdict'] = 'supported' if (q(.05) > -DELTA and q(.95) < DELTA) else ('contradicted' if (q(.05) > DELTA or q(.95) < -DELTA) else 'not resolved')
        o[k] = e
    rep[lab] = o
res['replication'] = rep
# exploratory: the design corpora (archived runs, Tables III-IV) pooled with the two replication corpora
P12 = {**ARCH, **N13, **N14}
pool = {}
for lab, a, b, R in (('support_emb - empty', 'support_emb', 'empty', ARCH), ('rvec_emb - empty', 'rvec_emb', 'empty', ARCH),
                     ('support_emb - rvec_emb', 'support_emb', 'rvec_emb', ARCH), ('st_support - st_dp', 'st_support', 'st_dp', N13),
                     ('st_er - st_empty', 'st_er', 'st_empty', N13), ('support - empty (update budget)', 'support', 'empty', None)):
    o = {}
    for k, col in (('first', 'test_auc_first'), ('all', 'test_auc')):
        if R is None:
            rows = pairs(load_runs([f'{REPO}/runs_r6']), [(c, s) for c in CORPORA for s in range(5)], a, b, col=col)
        else:
            rows = pairs(R, U10, a, b, col=col)
        rows += pairs(N14, UR, a, b, col=col)
        o[k] = summ(rows)
    pool[lab] = o
res['pooled_12'] = pool
res['replication_runs'] = int(sum(1 for k in N14 if k[0] in REP))
res['replication_plan_sha256'] = open(f'{REPO}/prereg/replication_plan_r14.sha256').read().split()[0]

# 6. curriculum-order null and authored groupings -----------------------------------------------------------------
UN = [(c, s) for c in ('statics11', 'algebra05', 'bridge06') for s in range(5)] + [(c, s) for c in ('junyi', 'eedi', 'ednet') for s in range(2)]
CU = {**LAD, **{k: v for k, v in N13.items() if k[1] == 0.1}, **N14}
# EdNet: the probe ladder of Table III is on the data of release 3.6.0; the new runs are on the rebuilt data, so EdNet
# enters only comparisons among runs of the rebuilt data (runs_r13, runs_r14).
CU_ed = {k: v for k, v in CU.items() if not (k[0] == 'ednet' and k[3] in ('empty', 'er_emb', 'dp_emb', 'support_emb', 'uniform_emb')
                                             and v[1].find('runs_r9') >= 0)}
for k, v in load_runs([f'{REPO}/runs_r10/base', f'{REPO}/runs_r11/base', f'{REPO}/runs_r10/repro']).items():
    if k[0] == 'ednet': CU_ed[k] = v
cur = {}
cur['E'] = {f'{a} - {b}': both(CU_ed, UN, a, b) for a, b in
            (('unit_emb', 'er_emb'), ('support_emb', 'unit_emb'), ('unit_emb', 'empty'), ('clique_emb', 'empty'), ('support_emb', 'clique_emb'),
             ('support_emb', 'er_emb'))}
cur['S'] = {f'{a} - {b}': both(CU_ed, UN, a, b) for a, b in
            (('st_unit', 'st_er'), ('st_support', 'st_unit'), ('st_unit', 'st_empty'), ('st_clique', 'st_empty'), ('st_support', 'st_clique'),
             ('st_support', 'st_er'))}
JU = [('junyi', 0), ('junyi', 1)]
cur['junyi_map'] = {f'{a} - {b}': {k: [dict(sub=s, d=CU_ed[(c, .1, s, a, 0, .002)][0][col] - CU_ed[(c, .1, s, b, 0, .002)][0][col])
                                       for c, s in JU if (c, .1, s, a, 0, .002) in CU_ed and (c, .1, s, b, 0, .002) in CU_ed]
                                   for k, col in (('first', 'test_auc_first'), ('all', 'test_auc'))}
                    for a, b in (('map_emb', 'empty'), ('map_emb', 'support_emb'), ('st_map', 'st_empty'), ('st_map', 'st_support'))}
# share of within-unit edges and edge counts of the graphs
gs = {}
for c, s in UN:
    k = (c, .1, s, 'unit_emb', 0, .002)
    if k in CU_ed: gs.setdefault(c, []).append(dict(support=CU_ed[k][0]['edges'], unit=CU_ed[k][0]['edges_cond']))
    k = (c, .1, s, 'clique_emb', 0, .002)
    if k in CU_ed: gs.setdefault(c, [])[-1].update(clique=CU_ed[k][0]['edges_cond'])
cur['edges'] = gs
res['curriculum'] = cur

# 7. cap of 500 -----------------------------------------------------------------------------------------------------
L500 = load_len([f'{REPO}/runs_r14'], 500); UC = [(c, s) for c in ('statics11', 'algebra05') for s in range(5)]
cap = {}
for a, b in (('support', 'empty'), ('support_emb', 'empty')):
    for lab, R in (('len500', L500), ('len200', {**REF, **N14})):
        cap[f'{a} - {b} | {lab}'] = {k: summ(pairs(R, UC, a, b, col=col), do_boot=False) for k, col in (('first', 'test_auc_first'), ('all', 'test_auc'))}
cap['n_first'] = {lab: {c: int(np.mean([R[(c, .1, s, 'empty', 0, .002)][0]['n_first'] for s in range(5) if (c, .1, s, 'empty', 0, .002) in R] or [0]))
                        for c in ('statics11', 'algebra05')} for lab, R in (('len500', L500), ('len200', REF))}
res['cap500'] = cap

# 8. GKT on a CPU ---------------------------------------------------------------------------------------------------
G = {}
for path in (f'{REPO}/runs_r8', f'{REPO}/runs_r9', f'{REPO}/runs_r13/gkt', f'{REPO}/runs_r14/gkt'):
    for k, v in load_runs([path]).items():
        if v[0].get('family') == 'gkt' and str(v[0].get('device', 'cpu')) == 'cpu' and k[4] == 0: G[k] = v
GU = [(c, s) for c in SMALL for s in range(3)] + [(c, s) for c in ('assist09', 'assist15') for s in range(2)]
res['gkt_cpu_ladder'] = {f'{a} - {b}': both(G, GU, a, b) for a, b in
                         (('gkt_uniform', 'gkt_empty'), ('gkt_er', 'gkt_empty'), ('gkt_er', 'gkt_uniform'), ('gkt_rewired', 'gkt_er'),
                          ('gkt_support', 'gkt_rewired'), ('gkt_support', 'gkt_empty'), ('gkt_support', 'gkt_uniform'))}
res['gkt_cpu_units_complete'] = sorted({f'{c}:{s}' for c, s in GU if all((c, .1, s, k, 0, .002) in G for k in
                                         ('gkt_empty', 'gkt_uniform', 'gkt_er', 'gkt_rewired', 'gkt_support'))})
GC = [(c, s) for c in ('statics11', 'algebra05', 'bridge06') for s in range(3)]
res['gkt_clique'] = {f'{a} - {b}': both(G, GC, a, b) for a, b in
                     (('gkt_clique', 'gkt_empty'), ('gkt_clique', 'gkt_support'), ('gkt_clique', 'gkt_uniform'))}

# 9. effect sizes on one scale (Fig. 4) ----------------------------------------------------------------------------
r13 = json.load(open(f'{REPO}/results_r13/r13_results.json'))
def ci(o):
    if not o: return None
    m = o.get('mixed') or {}; return dict(mean=o['mean'], lo=m.get('lo'), hi=m.get('hi'))
fx = {}
fx['graph, both entry points (G - empty)'] = ci(summ(pairs(load_runs([f'{REPO}/runs_r6']), [(c, s) for c in CORPORA for s in range(5)], 'support', 'empty'), do_boot=False))
fx['graph, embedding refinement (G - empty)'] = ci(eq['probe_E']['support_emb - empty']['first'])
fx['random code per KC (RV - empty)'] = ci(summ(pairs(ARCH, U10, 'rvec_emb', 'empty'), do_boot=False))
fx['twice the width (d = 128)'] = ci(eq['capacity']['empty_d128 - empty']['first'])
fx['trainable second KC table'] = ci(eq['capacity']['tvec_emb - empty']['first'])
fx['state channel (G - empty)'] = ci(eq['state']['st_support - st_empty']['first'])
fx['edges beyond rewiring, embedding (G - DP)'] = ci(eq['probe_E']['support_emb - dp_emb']['first'])
fx['edges beyond rewiring, state channel (G - DP)'] = ci(eq['state']['st_support - st_dp']['first'])
E5 = {}; S6 = {}
for p in ('runs_r5', 'runs_r6'):
    for f in glob.glob(f'{REPO}/{p}/*.json'):
        j = json.load(open(f))
        if j.get('cond') != 'empty' or j.get('frac') != 0.1 or (j.get('rep') or 0) != 0 or j.get('lr', 0.002) != 0.002: continue
        (S6 if j.get('budget', 'epochs') == 'steps' else E5).setdefault((j['data'], j['sub']), j)
rows = [dict(data=k[0], sub=k[1], d=S6[k]['test_auc_first'] - E5[k]['test_auc_first']) for k in S6 if k in E5 and k[0] in SMALL]
fx['budget counted in updates, four small corpora (control)'] = ci(summ(rows, do_boot=False))
rows = [dict(data=k[0], sub=k[1], d=S6[k]['test_auc_first'] - E5[k]['test_auc_first']) for k in S6 if k in E5 and k[0] not in SMALL]
fx['budget counted in updates, six larger corpora (control)'] = ci(summ(rows, do_boot=False))
fx['seed SD of G - empty (first attempts)'] = dict(mean=0.0041)
fx['SD of the control across subsamples (all predictions)'] = dict(mean=0.0044)
fx['architecture: empty-graph probe - DKT (all predictions, sample 0, epoch budget; Table S7.3)'] = dict(mean=0.018)
res['effect_scale_first'] = fx
# 10. minor points: Algebra 2005 and Statics 2011 in the primary update-budget comparison at 10%
R6 = load_runs([f'{REPO}/runs_r6']); U6 = [(c, s) for c in CORPORA for s in range(5)]
mn = {}
for k, col in (('first', 'test_auc_first'), ('all', 'test_auc')):
    rows = pairs(R6, U6, 'support', 'empty', col=col)
    mn[f'algebra05_{k}'] = [r['d'] for r in rows if r['data'] == 'algebra05']
    mn[f'without_statics11_{k}'] = summ([r for r in rows if r['data'] != 'statics11'], do_boot=False)
    mn[f'units_per_corpus'] = {c: sum(1 for r in rows if r['data'] == c) for c in CORPORA}
res['minor'] = mn
res['n_runs_r14'] = len(glob.glob(f'{REPO}/runs_r14/*.json')) + len(glob.glob(f'{REPO}/runs_r14/gkt/*.json'))
json.dump(res, open(f'{OUT}/r14_results.json', 'w'), indent=1, default=float)
print('written', OUT, 'runs_r14', len(N14), 'gkt', len(G14))
