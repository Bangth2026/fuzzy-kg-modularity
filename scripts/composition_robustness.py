#!/usr/bin/env python3
"""What the t-norm contributes beyond reachability, and how much of the best
operator's advantage is selection (Section 8.4).

Junyi (transition, transition without suggested attempts, co-practice):
* coverage of the annotated pairs (B -> A reached) for every operator;
* Spearman correlation with the expert score and knowledge-map AUC on the
  *common support*: the pairs (or links and negatives) reached by every
  composition operator, where all operators score the same set and differ only
  in ranking;
* the same on the pairs with a direct edge B -> A, where the direct edge is
  defined and composition can only re-rank;
* a coverage-only baseline: 1 / (1 + unweighted shortest hop distance), 0 when
  unreachable, which knows reachability and path length but no membership;
* split-half selection: in each of 500 random halvings of the annotated pairs, the
  operator with the highest correlation on one half is evaluated on the other,
  next to the pre-specified product and max-min;
* the density mechanism on the co-practice graph: the share of annotated pairs
  whose max-min best path is stronger than their direct edge, the mean number of
  hops of those paths and the spread of the max-min strengths.
MOOCCubeX: the hop baseline for the hidden-edge criterion.

Usage: python scripts/composition_robustness.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.paths_fast import strengths  # noqa: E402
from data.loaders import load_junyi_expert, load_mooccubex_cs  # noqa: E402
from prerequisite_criterion import OPS, load  # noqa: E402

COMP = list(OPS)


def hop_score(G):
    A = csr_matrix((np.ones(G.m), (G.src, G.dst)), shape=(G.n, G.n))
    D = shortest_path(A, directed=True, unweighted=True)
    S = np.where(np.isfinite(D), 1.0 / (1.0 + D), 0.0)
    np.fill_diagonal(S, 0.0)
    return S, D


def mats(G):
    W = np.zeros((G.n, G.n)); W[G.src, G.dst] = G.mu
    M = {"direct_edge": W}
    H = {}
    for k, (op, p) in OPS.items():
        S, Hk = strengths(G, op, p)
        M[k] = np.maximum(S, 0.0); H[k] = Hk
    M["hop_baseline"], D = hop_score(G)
    return M, H


def junyi(name, rng, B=500):
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
    pos = np.array(sorted(mp)); neg = []
    while len(neg) < 5 * len(pos):
        a, b = rng.integers(0, G.n, 2)
        if a != b and (a, b) not in mp and (b, a) not in mp:
            neg.append((a, b))
    neg = np.array(neg)
    M, H = mats(G)
    X = {k: v[ib, ia] for k, v in M.items()}
    reach_all = np.all([X[k] > 0 for k in COMP], axis=0)
    direct = X["direct_edge"] > 0
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    Spos = {k: v[pos[:, 0], pos[:, 1]] for k, v in M.items()}; Sneg = {k: v[neg[:, 0], neg[:, 1]] for k, v in M.items()}
    reach_map = np.all([np.r_[Spos[k], Sneg[k]] > 0 for k in COMP], axis=0)
    out = {"n_pairs": int(len(y)), "common_support_pairs": int(reach_all.sum()), "direct_pairs": int(direct.sum()),
           "common_support_map": {"links": int(reach_map[:len(pos)].sum()), "negatives": int(reach_map[len(pos):].sum())}, "operators": {}}
    for k in ["direct_edge", "hop_baseline"] + COMP:
        x = X[k]; s = np.r_[Spos[k], Sneg[k]]
        r = {"coverage": float((x > 0).mean()), "rho_all": float(spearmanr(x, y).statistic),
             "rho_common": float(spearmanr(x[reach_all], y[reach_all]).statistic) if reach_all.sum() > 10 else None,
             "rho_direct_pairs": float(spearmanr(x[direct], y[direct]).statistic) if direct.sum() > 10 else None,
             "map_auc_all": float(roc_auc_score(lab, s)),
             "map_auc_common": float(roc_auc_score(lab[reach_map], s[reach_map])) if 0 < lab[reach_map].sum() < reach_map.sum() else None}
        out["operators"][k] = r
    # bootstrap CI of rho_common difference to max-min and to the hop baseline
    def bci(a, b, mask):
        v = []
        ii = np.flatnonzero(mask)
        for _ in range(1000):
            j = rng.choice(ii, ii.size)
            v.append(spearmanr(X[a][j], y[j]).statistic - spearmanr(X[b][j], y[j]).statistic)
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    if reach_all.sum() > 10:
        out["common_ci"] = {f"{k}-hop_baseline": bci(k, "hop_baseline", reach_all) for k in ["product", "min", "yager_5", "lukasiewicz"]}
        out["common_ci"].update({f"{k}-min": bci(k, "min", reach_all) for k in ["product", "yager_5"]})
    if direct.sum() > 10:
        out["direct_ci"] = {f"{k}-direct_edge": bci(k, "direct_edge", direct) for k in ["product", "min", "yager_5"]}
    # split-half selection
    sel_test, prod_test, min_test, chosen = [], [], [], []
    n = len(y)
    for _ in range(B):
        perm = rng.permutation(n); a, b = perm[: n // 2], perm[n // 2:]
        best = max(COMP, key=lambda k: spearmanr(X[k][a], y[a]).statistic)
        chosen.append(best)
        sel_test.append(spearmanr(X[best][b], y[b]).statistic)
        prod_test.append(spearmanr(X["product"][b], y[b]).statistic); min_test.append(spearmanr(X["min"][b], y[b]).statistic)
    q = lambda v: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out["split_half"] = {"selected_test": q(sel_test), "product_test": q(prod_test), "min_test": q(min_test),
                         "selected_minus_product": q(np.array(sel_test) - np.array(prod_test)),
                         "chosen_counts": {k: chosen.count(k) for k in set(chosen)}}
    # density mechanism (max-min)
    mm, d = X["min"], X["direct_edge"]
    hm = H["min"][ib, ia]
    stronger = mm > d + 1e-12
    out["mechanism"] = {"share_pairs_maxmin_stronger_than_direct": float(stronger.mean()),
                        "mean_hops_maxmin_reached": float(np.mean(hm[mm > 0])) if (mm > 0).any() else None,
                        "sd_maxmin_strength": float(np.std(mm[mm > 0])) if (mm > 0).any() else None,
                        "sd_direct_strength": float(np.std(d[d > 0])) if (d > 0).any() else None,
                        "distinct_maxmin_values_share": float(np.unique(np.round(mm, 6)).size / max(1, mm.size))}
    print(name, json.dumps({k: (round(v["coverage"], 2), round(v["rho_all"], 3), v["rho_common"] and round(v["rho_common"], 3),
                                v["rho_direct_pairs"] and round(v["rho_direct_pairs"], 3), round(v["map_auc_all"], 3), v["map_auc_common"] and round(v["map_auc_common"], 3))
                            for k, v in out["operators"].items()}), json.dumps(out["split_half"]), json.dumps(out["mechanism"]), flush=True)
    return out


def mooccubex_hop(rng, folds=5, neg_per_pos=20):
    df, _ = load_mooccubex_cs(full=True)
    labels = sorted(set(df.src) | set(df.dst)); idx = {l: i for i, l in enumerate(labels)}
    src = df.src.map(idx).to_numpy(); dst = df.dst.map(idx).to_numpy(); mu = df.mu.to_numpy()
    n = len(labels); m = len(src); edges = set(zip(src.tolist(), dst.tolist()))
    perm = rng.permutation(m); parts = np.array_split(perm, folds)
    p, q = [], []
    for f in range(folds):
        test = parts[f]; train = np.setdiff1d(perm, test)
        G = FuzzyGraph(n, src[train], dst[train], mu[train])
        S, _ = hop_score(G)
        neg = []
        while len(neg) < neg_per_pos * len(test):
            a, b = rng.integers(0, n, 2)
            if a != b and (a, b) not in edges and (b, a) not in edges:
                neg.append((a, b))
        neg = np.array(neg)
        p += S[src[test], dst[test]].tolist(); q += S[neg[:, 0], neg[:, 1]].tolist()
    y = np.r_[np.ones(len(p)), np.zeros(len(q))]
    return {"hop_baseline_auc": float(roc_auc_score(y, np.r_[p, q])), "reached": float(np.mean(np.asarray(p) > 0))}


def main():
    rng = np.random.default_rng(7)
    res = {g: junyi(g, rng) for g in ("junyi_transition", "junyi_transition_unsuggested", "junyi_copractice")}
    res["mooccubex_full"] = mooccubex_hop(np.random.default_rng(0))
    print(res["mooccubex_full"])
    json.dump(res, open("results/composition_robustness.json", "w"), indent=1)


if __name__ == "__main__":
    main()
