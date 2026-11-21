#!/usr/bin/env python3
"""Version 4.7.0. Membership-free link-prediction heuristics on the hidden MOOCCubeX expert edges.

Reproduces the five folds and the random non-edges of
scripts/v9_grade_preserving_baselines.py (mooccubex(): rng 7, the same draw order), so the
heuristics are scored on exactly the pairs on which Supplementary Table S13 scores the GCN, the
UKGE-style model and the path operators. On the training graph of each fold (the remaining
expert edges, memberships ignored) every hidden edge (a, b) and every non-edge is scored by

  common neighbours   |N(a) & N(b)|, N the undirected neighbourhood
  Adamic-Adar         sum over common neighbours w of 1 / log(deg w)
  resource allocation sum over common neighbours w of 1 / deg w
  preferential attachment  out-degree(a) * in-degree(b) (directed)
  Jaccard             |N(a) & N(b)| / |N(a) | N(b)|

and the AUC against the non-edges is reported with the bootstrap of the v9 script (500
resamples of hidden edges and of non-edges). Writes results/v10/link_heuristics.json.
"""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

OUT = "results/v10/link_heuristics.json"


def main(folds=5, neg_per_pos=20):
    e = pd.read_csv("data/graphs/mooccubex_cs_full_edges.csv"); nd = pd.read_csv("data/graphs/mooccubex_cs_full_nodes.csv")
    n = len(nd); src = e.src.to_numpy(); dst = e.dst.to_numpy(); m = len(src)
    edges = set(zip(src.tolist(), dst.tolist()))
    rng = np.random.default_rng(7); perm = rng.permutation(m); parts = np.array_split(perm, folds)
    keys = ["common_neighbours", "adamic_adar", "resource_allocation", "preferential_attachment", "jaccard"]
    P = {k: [] for k in keys}; N = {k: [] for k in keys}
    for f in range(folds):
        test = parts[f]; train = np.setdiff1d(perm, test)
        neg = []
        while len(neg) < neg_per_pos * len(test):
            a, b = rng.integers(0, n, 2)
            if a != b and (a, b) not in edges and (b, a) not in edges:
                neg.append((a, b))
        neg = np.array(neg)
        nb = [set() for _ in range(n)]; dout = np.zeros(n); din = np.zeros(n)
        for a, b in zip(src[train], dst[train]):
            nb[a].add(b); nb[b].add(a); dout[a] += 1; din[b] += 1
        deg = np.array([len(s) for s in nb], float)

        def score(a, b):
            c = nb[a] & nb[b]; u = nb[a] | nb[b]
            return {"common_neighbours": len(c),
                    "adamic_adar": sum(1 / np.log(deg[w]) for w in c if deg[w] > 1),
                    "resource_allocation": sum(1 / deg[w] for w in c),
                    "preferential_attachment": dout[a] * din[b],
                    "jaccard": len(c) / len(u) if u else 0.0}
        for a, b in zip(src[test], dst[test]):
            s = score(a, b)
            for k in keys:
                P[k].append(s[k])
        for a, b in neg:
            s = score(a, b)
            for k in keys:
                N[k].append(s[k])
    out = {"graph": "mooccubex_cs_full", "n_nodes": n, "n_edges": m, "folds": folds, "neg_per_pos": neg_per_pos,
           "protocol": "folds and non-edges of scripts/v9_grade_preserving_baselines.py (rng 7)", "scores": {}}
    brng = np.random.default_rng(11)
    for k in keys:
        p = np.asarray(P[k], float); q = np.asarray(N[k], float); lab = np.r_[np.ones(len(p)), np.zeros(len(q))]
        bt = []
        for _ in range(500):
            i = brng.integers(0, len(p), len(p)); j = brng.integers(0, len(q), len(q))
            bt.append(roc_auc_score(np.r_[np.ones(len(i)), np.zeros(len(j))], np.r_[p[i], q[j]]))
        out["scores"][k] = {"auc": float(roc_auc_score(lab, np.r_[p, q])),
                            "auc_ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                            "hidden_scored_positive": float((p > 0).mean())}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print({k: (round(v["auc"], 3), [round(x, 3) for x in v["auc_ci"]], round(v["hidden_scored_positive"], 2)) for k, v in out["scores"].items()})


if __name__ == "__main__":
    main()
