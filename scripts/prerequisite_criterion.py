#!/usr/bin/env python3
"""External criteria for path composition that match the prerequisite semantics of
Section 6 (Section 8.5, Table 14).

(a) Junyi Academy.  On a fuzzy graph built from the logs (co-practice or transition
    construction), every composition operator scores the pair (B -> A) of each
    crowd-annotated pair by its best-path strength; the criterion is the Spearman
    correlation with the expert prerequisite score.  Separately, the knowledge-map
    parent links (parent -> child) are scored against random unlinked ordered pairs
    (ROC AUC).  Paired bootstrap over pairs for each operator minus the direct edge.

(b) MOOCCubeX-CS (full expert graph, 725 edges, acyclic).  Five-fold hidden-edge
    test: a fifth of the expert edges is removed, every operator scores the removed
    edges and 20 random non-edges per removed edge on the remaining graph, and the
    criterion is the ROC AUC of the removed edges against the non-edges, pooled over
    folds.

Operators: direct edge, max-min, product, Lukasiewicz, Yager p in {2, 5, 10},
Schweizer-Sklar p in {-8, -2, 0.5}; hop limits: none, 3 and 2 (with tau_path = 0.3
when a limit is set, the configuration of Equation (14)).

Usage: python scripts/prerequisite_criterion.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.paths_fast import strengths  # noqa: E402
from data.loaders import load_junyi_expert, load_mooccubex_cs  # noqa: E402

OPS = {"min": ("min", None), "product": ("product", None), "lukasiewicz": ("lukasiewicz", None),
       "yager_2": ("yager", 2.0), "yager_5": ("yager", 5.0), "yager_10": ("yager", 10.0),
       "ss_-8": ("schweizer_sklar", -8.0), "ss_-2": ("schweizer_sklar", -2.0), "ss_0.5": ("schweizer_sklar", 0.5)}
CONFIGS = {"unlimited": (None, 0.0), "L3_tau0.3": (3, 0.3), "L2_tau0.3": (2, 0.3)}


def load(name):
    e = pd.read_csv(f"data/graphs/{name}_edges.csv"); nd = pd.read_csv(f"data/graphs/{name}_nodes.csv")
    G = FuzzyGraph.from_edges(zip(e.src, e.dst, e.mu), n=len(nd))
    return G, nd.label.astype(str).tolist()


def score_mats(G, max_hops, tau):
    W = np.zeros((G.n, G.n)); keep = G.mu >= tau
    W[G.src[keep], G.dst[keep]] = G.mu[keep]
    mats = {"direct_edge": W}
    for k, (op, p) in OPS.items():
        mats[k] = np.maximum(strengths(G, op, p, max_hops, tau)[0], 0.0)
    return mats


def boot_diff_rho(x, x0, y, rng, B=1000):
    n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(spearmanr(x[i], y[i]).statistic - spearmanr(x0[i], y[i]).statistic)
    d = np.asarray(d)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def junyi(name, rng):
    G, labels = load(name)
    idx = {l: i for i, l in enumerate(labels)}
    ex, ann = load_junyi_expert()
    sel = ann[ann.Exercise_A.isin(idx) & ann.Exercise_B.isin(idx)]
    ia = sel.Exercise_A.map(idx).to_numpy(); ib = sel.Exercise_B.map(idx).to_numpy(); y = sel.Prerequisite_avg.to_numpy()
    mp = set()
    for n, p in zip(ex.name, ex.prerequisites):
        if isinstance(p, str):
            for q in p.split(","):
                if q.strip() in idx and n in idx:
                    mp.add((idx[q.strip()], idx[n]))
    pos = np.array(sorted(mp))
    neg = []
    while len(neg) < 5 * len(pos):
        a, b = rng.integers(0, G.n, 2)
        if a != b and (a, b) not in mp and (b, a) not in mp:
            neg.append((a, b))
    neg = np.array(neg)
    out = {"n_pairs": int(len(sel)), "n_map_links": int(len(pos)), "configs": {}}
    for cname, (L, tau) in CONFIGS.items():
        mats = score_mats(G, L, tau)
        res = {}
        x0 = mats["direct_edge"][ib, ia]
        for k, M in mats.items():
            x = M[ib, ia]
            r = {"rho": float(spearmanr(x, y).statistic), "coverage": float((x > 0).mean()),
                 "map_auc": float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                                                np.r_[M[pos[:, 0], pos[:, 1]], M[neg[:, 0], neg[:, 1]]])),
                 "map_direction_auc": float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(pos))],
                                                          np.r_[M[pos[:, 0], pos[:, 1]], M[pos[:, 1], pos[:, 0]]]))}
            if k != "direct_edge":
                r["rho_minus_direct_ci"] = boot_diff_rho(x, x0, y, rng)
            res[k] = r
        out["configs"][cname] = res
        print(name, cname, {k: (round(v["rho"], 3), round(v["map_auc"], 3)) for k, v in res.items()}, flush=True)
    return out


def mooccubex(rng, folds=5, neg_per_pos=20):
    df, _ = load_mooccubex_cs(full=True)
    labels = sorted(set(df.src) | set(df.dst)); idx = {l: i for i, l in enumerate(labels)}
    src = df.src.map(idx).to_numpy(); dst = df.dst.map(idx).to_numpy(); mu = df.mu.to_numpy()
    n = len(labels); m = len(src)
    edges = set(zip(src.tolist(), dst.tolist()))
    perm = rng.permutation(m); parts = np.array_split(perm, folds)
    scores = {c: {k: ([], []) for k in ["direct_edge"] + list(OPS)} for c in CONFIGS}
    for f in range(folds):
        test = parts[f]; train = np.setdiff1d(perm, test)
        G = FuzzyGraph(n, src[train], dst[train], mu[train])
        neg = []
        while len(neg) < neg_per_pos * len(test):
            a, b = rng.integers(0, n, 2)
            if a != b and (a, b) not in edges and (b, a) not in edges:
                neg.append((a, b))
        neg = np.array(neg)
        for c, (L, tau) in CONFIGS.items():
            mats = score_mats(G, L, tau)
            for k, M in mats.items():
                scores[c][k][0].extend(M[src[test], dst[test]].tolist())
                scores[c][k][1].extend(M[neg[:, 0], neg[:, 1]].tolist())
    out = {"n_nodes": n, "n_edges": m, "folds": folds, "configs": {}}
    for c in CONFIGS:
        res = {}
        for k, (p, q) in scores[c].items():
            y = np.r_[np.ones(len(p)), np.zeros(len(q))]; s = np.r_[p, q]
            boots = []
            for _ in range(500):
                i = rng.integers(0, len(p), len(p)); j = rng.integers(0, len(q), len(q))
                boots.append(roc_auc_score(np.r_[np.ones(len(i)), np.zeros(len(j))], np.r_[np.asarray(p)[i], np.asarray(q)[j]]))
            res[k] = {"auc": float(roc_auc_score(y, s)), "auc_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                      "hidden_reached": float(np.mean(np.asarray(p) > 0))}
        out["configs"][c] = res
        print("mooccubex", c, {k: round(v["auc"], 3) for k, v in res.items()}, flush=True)
    return out


def main():
    rng = np.random.default_rng(0)
    res = {"junyi_transition": junyi("junyi_transition", rng), "junyi_copractice": junyi("junyi_copractice", rng),
           "mooccubex_full_hidden_edges": mooccubex(rng)}
    json.dump(res, open("results/prerequisite_criterion.json", "w"), indent=1)


if __name__ == "__main__":
    main()
