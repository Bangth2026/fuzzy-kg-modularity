"""Table 4 recomputed within one code version: graph inputs from the release-2.0.0 runs (sample 0 of
the 10% regime, two or three seeds), empty-graph control from the R5 runs on the same sample (seeds
0, 1, 2), both produced by the same model code. Holm over the 24 tests of the table.
Also writes the per-seed AUCs of Algebra 2005 and Eedi (response to D1, D2)."""
import json, glob, os, collections, numpy as np
from scipy.stats import wilcoxon
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
REL = os.environ.get('REL200', '/home/claude/z2/fuzzy-kg-modularity-2.0.0/kt/results/runs')
C8 = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet']
ROWS = [('transition', 'support', 'order-preserving: support'), ('transition', 'permuted', 'order-preserving: permuted grades'),
        ('transition', 'graded', 'order-preserving: graded'), ('transition', 'rank', 'order-preserving: rank-coded'),
        ('transition', 'cut25', 'order-preserving: lower-quartile cut'), ('transition', 'cut50', 'order-preserving: median cut'),
        ('transition', 'cut75', 'order-preserving: upper-quartile cut'), ('transition', 'product3', 'order-preserving: product closure, 3 hops'),
        ('copractice', 'support', 'co-practice: support'), ('copractice', 'rank', 'co-practice: rank-coded'),
        ('copractice', 'permuted', 'co-practice: permuted grades'), ('dktinf', 'rank', 'neural: DKT influence')]
rel = collections.defaultdict(dict)
for f in glob.glob(f'{REL}/*_f10__fkgkt2__*.json'):
    d = json.load(open(f)); rel[(d['data'], d['construction'], d['graph'])][d['seed']] = d
ctl = collections.defaultdict(dict)
for f in glob.glob(f'{ROOT}/runs_r5/*__f10__s0__empty*.json'):
    d = json.load(open(f))
    if d['d'] == 64 and d.get('lr_scalar') is None: ctl[d['data']][d['seed']] = d
def wil(x): return float(wilcoxon(x, method='exact').pvalue)
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj
out = []; ps = []
for cons, g, label in ROWS:
    for key, rk in (('test_auc', 'test_auc'), ('test_auc_first', 'test_auc_newkc')):
        diffs = {}
        for c in C8:
            ins = [r[rk] for r in rel[(c, cons, g)].values()]
            cs = [r[key] for r in ctl[c].values()]
            if ins and cs: diffs[c] = float(np.mean(ins) - np.mean(cs))
        v = np.array(list(diffs.values()))
        out.append(dict(label=label, key=key, mean=float(v.mean()), pos=int((v > 0).sum()), n=len(v), p=wil(v), per_corpus=diffs,
                        seeds_inputs=int(np.median([len(rel[(c, cons, g)]) for c in C8]))))
        ps.append(out[-1]['p'])
for r, a in zip(out, holm(np.array(ps))): r['holm'] = float(a)
seeds = {c: {'control': {s: (round(r['test_auc'], 4), round(r['test_auc_first'], 4), r['epochs_run']) for s, r in sorted(ctl[c].items())},
             'support': {s: (round(r['test_auc'], 4), round(r['test_auc_newkc'], 4), r['epochs_run']) for s, r in sorted(rel[(c, 'transition', 'support')].items())}}
         for c in ('algebra05', 'eedi')}
# Algebra 2005: all release-2.0.0 FKG-KT runs at 10%, sample 0: AUC by whether the run reached the 30-epoch cap
alg = [r for k, v in rel.items() if k[0] == 'algebra05' for r in v.values()]
cap = [r['test_auc'] for r in alg if r['epochs_run'] >= 29]; early = [r['test_auc'] for r in alg if r['epochs_run'] <= 23]
eedi = [r['test_auc'] for k, v in rel.items() if k[0] == 'eedi' for r in v.values()]
jun = {'expert': float(np.mean([r['test_auc'] for r in rel[('junyi', 'transition', 'expert')].values()])),
       'support': float(np.mean([r['test_auc'] for r in rel[('junyi', 'transition', 'support')].values()])),
       'control': float(np.mean([r['test_auc'] for r in ctl['junyi'].values()])), 'n_expert': len(rel[('junyi', 'transition', 'expert')])}
res = dict(rows=out, seeds=seeds, alg05_cap=dict(n=len(cap), lo=min(cap), hi=max(cap)), alg05_early=dict(n=len(early), lo=min(early), hi=max(early),
           ep=[min(r['epochs_run'] for r in alg if r['epochs_run'] <= 23), max(r['epochs_run'] for r in alg if r['epochs_run'] <= 23)]),
           eedi_range=[min(eedi), max(eedi), len(eedi)], junyi=jun)
os.makedirs(f'{ROOT}/results_r5', exist_ok=True); json.dump(res, open(f'{ROOT}/results_r5/table4_r5.json', 'w'), indent=1)
for r in out: print(f"{r['label']:42s} {r['key']:15s} {r['mean']:+.4f} {r['pos']}/{r['n']} holm={r['holm']:.3f}")
print(json.dumps({k: res[k] for k in ('seeds', 'alg05_cap', 'alg05_early', 'eedi_range', 'junyi')}, indent=0))
