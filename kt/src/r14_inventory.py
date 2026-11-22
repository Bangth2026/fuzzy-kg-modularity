"""Release 3.11.0: inventory of the run sets added in the second revision (Supplementary Table S6.10, lowest block).
Groups the run files of runs_r14 and runs_r14/gkt by set, model, training fraction, training seed, learning rate and
conditions, with the subsamples by corpus and the number of runs. Output: results_r13/run_sets_r14.csv"""
import os, json, glob, collections, csv
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
OUT = f'{REPO}/results_r14'; os.makedirs(OUT, exist_ok=True)


def setname(r):
    c = r['cond']
    if int(r.get('maxlen', 200) or 200) != 200: return 'cap of 500 interactions'
    if r['data'] in ('xes3g5m', 'eedi12'): return 'replication (held-out corpora)'
    if any(x in c for x in ('unit', 'clique', 'map')): return 'curriculum null and authored graphs'
    if c.startswith('gkt'): return 'GKT on a CPU'
    if r['frac'] > 0.99: return 'all learners, seeds 100 and 200'
    if r['frac'] > 0.2: return 'decisions with crossed seeds, 25%'
    if c.startswith('st_'): return 'state channel'
    if c.startswith('sakte'): return 'SAKT-E on a CPU'
    if c.startswith('gkt'): return 'GKT on a CPU'
    if c in ('empty_d128', 'tvec_emb'): return 'capacity'
    if abs(r.get('lr', 0.002) - 0.002) > 1e-9: return 'learning rate'
    if r['frac'] > 0.11: return 'update budget, larger samples'
    return 'EdNet reruns'


groups = collections.defaultdict(lambda: dict(conds=set(), subs=collections.defaultdict(set), n=0))
for f in glob.glob(f'{REPO}/runs_r14/*.json') + glob.glob(f'{REPO}/runs_r14/gkt/*.json'):
    r = json.load(open(f))
    fam = {'probe': 'FKG-KT (probe)', 'probe_state': 'FKG-KT (probe)', 'sakte': 'SAKT-E', 'gkt': 'GKT'}[r['family']]
    key = (setname(r), fam, f"{int(round(r['frac'] * 100))}%", int(r.get('rep', 0) or 0), f"{r.get('lr', 0.002):g}")
    g = groups[key]; g['conds'].add(r['cond']); g['subs'][r['data']].add(r['sub']); g['n'] += 1
rows = []
for (s, fam, fr, seed, lr), g in sorted(groups.items()):
    rows.append(dict(set=s, model=fam, device='cpu', fraction=fr, seeds=('subsample + 100 x %d' % seed) if seed else 'subsample', lr=lr,
                     conditions=' '.join(sorted(g['conds'])),
                     subsamples='; '.join(f"{c}: {' '.join(map(str, sorted(v)))}" for c, v in sorted(g['subs'].items())), runs=g['n']))
with open(f'{OUT}/run_sets_r14.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(sum(r['runs'] for r in rows), 'runs in', len(rows), 'sets')
