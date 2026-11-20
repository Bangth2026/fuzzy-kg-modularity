"""Release 3.10.0: what the support graph encodes, and why EdNet has many first attempts (no training runs).

1. For every corpus, the support graph of all training learners (Table I) and of each 10% subsample of
   Section VI-D: the share of its edges whose two KCs lie in the same curriculum unit (Algebra 2005,
   Bridge to Algebra: unit of the problem hierarchy; Statics 2011: module; Eedi: level-2 subject; EdNet:
   TOEIC part; Junyi: topic), against the same share for an Erdos-Renyi graph with as many edges
   (mean over 20 draws). A ratio well above 1 means the edges follow the curriculum grouping.
2. Junyi: the share of support edges a -> b that are an edge of the published prerequisite map
   (a is listed as a prerequisite of b), its reverse (b -> a), or neither; the same for the Erdos-Renyi graph.
3. ASSISTments 2009 and 2017: the share of support edges a -> b for which the most frequent transition
   between runs of a and b in the training logs happens inside one assignment (2009: assignment_id;
   2017: assignmentId), i.e. edges that follow the order in which an assignment presents its skills.
4. Per corpus at 10% (subsample 0): KCs per learner, the share of a learner's interactions that start a
   new KC, mean run length, and the share of first attempts among test predictions.
Output: results_r13/graph_audit.json
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_controls import load, transition_support
from r6_controls import erdos_renyi
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt'); RAW = os.environ.get('KT_RAW', '/home/claude/raw')
OUT = f'{ROOT}/results_r13'; os.makedirs(OUT, exist_ok=True)
CORPORA = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
GROUPED = ['algebra05', 'bridge06', 'statics11', 'eedi', 'ednet', 'junyi']


def kc_names(df):
    return df.drop_duplicates('kc').set_index('kc').kc_name.to_dict()


def same_group_share(A, grp):
    s, d = np.nonzero(A)
    ok = [(grp.get(a) is not None and grp.get(a) == grp.get(b)) for a, b in zip(s, d)]
    return float(np.mean(ok)) if ok else None


def main():
    res = {'same_unit': {}, 'junyi_map': {}, 'assignment': {}, 'first_attempts': {}}
    for c in CORPORA:
        full = load(c, 1.0, 0); K = int(full.kc.max()) + 1; names = kc_names(full)
        tr = full[full.split == 'train']; sup = (transition_support(tr, K) > 0).astype(np.float32)
        if c in GROUPED:
            g = json.load(open(f'{ROOT}/data/{c}_groups.json')); grp = {k: g.get(str(names.get(k))) for k in range(K)}
            row = {'edges': int(sup.sum()), 'support': same_group_share(sup, grp)}
            row['er'] = float(np.mean([same_group_share(erdos_renyi(K, int(sup.sum()), seed=9000 + i), grp) for i in range(20)]))
            subs = []
            for s in range(5 if c in ('algebra05', 'bridge06', 'statics11') else 2):
                d = load(c, 0.1, s); t = d[d.split == 'train']; A = (transition_support(t, K) > 0).astype(np.float32)
                if A.sum() > 0: subs.append(same_group_share(A, grp))
            row['support_10pct_mean'] = float(np.mean(subs)); row['ratio'] = row['support'] / row['er']
            res['same_unit'][c] = row
        if c == 'junyi':
            ex = pd.read_csv(f'{RAW}/junyi/junyi_Exercise_table.csv'); ex.columns = [x.strip('﻿') for x in ex.columns]
            pre = set()
            for b, p in zip(ex['name'], ex['prerequisites']):
                if isinstance(p, str):
                    for a in p.split(','):
                        if a.strip(): pre.add((a.strip(), b))
            def audit(A):
                s, d = np.nonzero(A); f = r = 0
                for a, b in zip(s, d):
                    na, nb = names.get(a), names.get(b)
                    f += (na, nb) in pre; r += (nb, na) in pre
                n = len(s); return {'edges': n, 'forward': f / n, 'reverse': r / n, 'neither': 1 - (f + r) / n}
            res['junyi_map'] = {'map_edges': len(pre), 'support': audit(sup),
                                'er': {k: float(np.mean([audit(erdos_renyi(K, int(sup.sum()), seed=9000 + i))[k] for i in range(5)]))
                                       for k in ('forward', 'reverse', 'neither')}}
        if c in ('assist09', 'assist17'):
            if c == 'assist09':
                raw = pd.read_csv(f'{RAW}/a09/skill_builder_data_corrected.csv', encoding='latin-1', low_memory=False,
                                  usecols=['order_id', 'user_id', 'skill_id', 'original', 'assignment_id'])
                raw = raw[raw.original == 1].dropna(subset=['skill_id']).drop_duplicates(['user_id', 'order_id'])
                amap = dict(zip(raw.order_id, raw.assignment_id)); key = 'order'
            else:
                raw = pd.read_csv(f'{RAW}/a17/anonymized_full_release_competition_dataset.csv', low_memory=False,
                                  usecols=['studentId', 'skill', 'original', 'startTime', 'action_num', 'assignmentId'])
                raw = raw[raw.original == 1].dropna(subset=['skill'])
                amap = dict(zip(raw.startTime * 1e4 + raw.action_num % 10000, raw.assignmentId)); key = 'order'
            within = {}; total = {}
            for _, gdf in tr.groupby('user', sort=False):
                q = gdf.kc.values; asg = gdf[key].map(amap).values
                st = np.r_[True, q[1:] != q[:-1]]; idx = np.flatnonzero(st)
                for i in range(len(idx) - 1):
                    a, b = q[idx[i]], q[idx[i + 1]]; same = asg[idx[i + 1] - 1] == asg[idx[i + 1]]
                    total[(a, b)] = total.get((a, b), 0) + 1; within[(a, b)] = within.get((a, b), 0) + same
            s, d = np.nonzero(sup)
            sh = [within.get((a, b), 0) / total[(a, b)] for a, b in zip(s, d) if total.get((a, b))]
            allp = [within[k] / total[k] for k in total]
            res['assignment'][c] = {'edges': int(len(s)), 'mean_share_within_assignment_edges': float(np.mean(sh)),
                                    'share_edges_majority_within': float(np.mean(np.array(sh) > 0.5)),
                                    'mean_share_within_assignment_all_transitions': float(np.mean(allp))}
        d = load(c, 0.1, 0); te = d[d.split == 'test']
        rows = []
        for _, gdf in te.groupby('user', sort=False):
            q = gdf.kc.values
            if len(q) < 2: continue
            runs = np.r_[True, q[1:] != q[:-1]].sum()
            seen = set([q[0]]); nf = 0
            for v in q[1:]:
                nf += v not in seen; seen.add(v)
            rows.append((len(set(q)), len(q), runs, nf, len(q) - 1))
        r = np.array(rows, float)
        res['first_attempts'][c] = {'kcs_per_learner': float(np.median(r[:, 0])), 'interactions_per_learner': float(np.median(r[:, 1])),
                                    'mean_run_length': float(r[:, 1].sum() / r[:, 2].sum()), 'share_first': float(r[:, 3].sum() / r[:, 4].sum())}
        print(c, json.dumps({k: res[k].get(c) for k in ('same_unit', 'assignment', 'first_attempts')}), flush=True)
    json.dump(res, open(f'{OUT}/graph_audit.json', 'w'), indent=1)
    print(json.dumps(res['junyi_map'], indent=1))


if __name__ == '__main__': main()
