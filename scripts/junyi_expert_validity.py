#!/usr/bin/env python3
"""Validity of the memberships of the Junyi graphs against the expert information
released with the corpus (Section 8.1): the crowd-annotated pairwise prerequisite
scores (1-9) and the knowledge-map parent links.

The annotation file gives, for a pair (Exercise_A, Exercise_B), the mean score of
"Exercise_B is a prerequisite of Exercise_A"; the direction is read off the data:
pairs whose knowledge-map parent is B score 6.1 on average, pairs whose parent is
A score 5.2 and unrelated pairs 4.0.  Reported per graph: Spearman correlation of the
score with mu(B -> A), with the directed difference mu(B -> A) - mu(A -> B) and with
the undirected max; ROC AUC of the knowledge-map parent links against random
non-linked pairs (undirected and directed); and the share of linked pairs present.

Usage: python scripts/junyi_expert_validity.py [graph names ...]
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loaders import load_junyi_expert  # noqa: E402


def map_edges(ex):
    mp = set()
    for n, p in zip(ex.name, ex.prerequisites):
        if isinstance(p, str):
            for q in p.split(","):
                if q.strip():
                    mp.add((q.strip(), n))
    return mp


def boot_rho(x, y, rng, B=1000):
    n = len(x); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        v.append(spearmanr(x[i], y[i]).statistic)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main(names):
    ex, ann = load_junyi_expert()
    mp = map_edges(ex)
    rng = np.random.default_rng(0)
    res = {"n_annotated": int(len(ann)), "n_map_edges": len(mp),
           "score_by_relation": {"parent_is_B": float(ann[[ (b, a) in mp for a, b in zip(ann.Exercise_A, ann.Exercise_B)]].Prerequisite_avg.mean()),
                                 "parent_is_A": float(ann[[ (a, b) in mp for a, b in zip(ann.Exercise_A, ann.Exercise_B)]].Prerequisite_avg.mean()),
                                 "unrelated": float(ann[[ (a, b) not in mp and (b, a) not in mp for a, b in zip(ann.Exercise_A, ann.Exercise_B)]].Prerequisite_avg.mean())},
           "graphs": {}}
    for name in names:
        e = pd.read_csv(f"data/graphs/{name}_edges.csv"); nd = pd.read_csv(f"data/graphs/{name}_nodes.csv")
        lab = dict(zip(nd.id, nd.label)); mu = {(lab[s], lab[d]): m for s, d, m in zip(e.src, e.dst, e.mu)}
        nodes = set(lab.values())
        sel = ann[ann.Exercise_A.isin(nodes) & ann.Exercise_B.isin(nodes)]
        y = sel.Prerequisite_avg.to_numpy()
        ba = np.array([mu.get((b, a), 0.0) for a, b in zip(sel.Exercise_A, sel.Exercise_B)])
        ab = np.array([mu.get((a, b), 0.0) for a, b in zip(sel.Exercise_A, sel.Exercise_B)])
        r = {"n_pairs": int(len(sel)), "coverage": float(((ba > 0) | (ab > 0)).mean()),
             "rho_directed": float(spearmanr(ba, y).statistic), "rho_directed_ci": boot_rho(ba, y, rng),
             "rho_difference": float(spearmanr(ba - ab, y).statistic), "rho_difference_ci": boot_rho(ba - ab, y, rng),
             "rho_undirected": float(spearmanr(np.maximum(ba, ab), y).statistic), "rho_undirected_ci": boot_rho(np.maximum(ba, ab), y, rng)}
        nl = sorted(nodes)
        pos = [(x, z) for x, z in mp if x in nodes and z in nodes]
        neg = []
        while len(neg) < 5 * len(pos):
            x, z = rng.choice(nl, 2, replace=False)
            if (x, z) not in mp and (z, x) not in mp:
                neg.append((x, z))
        und = lambda p: max(mu.get(p, 0.0), mu.get((p[1], p[0]), 0.0))  # noqa: E731
        r["map_links"] = len(pos)
        r["map_auc_undirected"] = float(roc_auc_score([1] * len(pos) + [0] * len(neg), [und(p) for p in pos] + [und(p) for p in neg]))
        r["map_auc_directed"] = float(roc_auc_score([1] * len(pos) + [0] * len(pos), [mu.get(p, 0.0) for p in pos] + [mu.get((p[1], p[0]), 0.0) for p in pos]))
        r["map_links_present"] = float(np.mean([und(p) > 0 for p in pos]))
        both = [p for p in pos if und(p) > 0]
        r["map_direction_accuracy"] = float(np.mean([mu.get(p, 0.0) > mu.get((p[1], p[0]), 0.0) for p in both])) if both else None
        res["graphs"][name] = r
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump(res, open("results/junyi_expert_validity.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["junyi_copractice", "junyi_transition"])
