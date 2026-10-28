#!/usr/bin/env python3
"""Binarisations of equal size that are not nested (Section 8.1).

On the planted sweep (shift 0 to 1, five graph seeds), four rules keep the same
number of edges, half or a quarter of them: a global threshold (the quantile
cut), a per-source threshold (the top half or quarter of each node's out-edges
by membership), the disparity filter ranked by significance and cut at the same
size, and uniform random thinning. Within one graph and one size the four
binarised graphs have the same number of edges and the same mean out-degree, so
the free proxies cannot separate them; the cost index, the Q_F drop and the
retained mass are compared with the recovery shortfall 1 - ARI(cut, truth) by
the within-graph concordance over the six pairs of rules. The script also
reports, on the cuts of the main sweep, the partial Spearman correlation of the
index and of the Q_F drop with the shortfall given the retained mass, and given
the mean out-degree.

Writes results/nonnested_binarisation.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden, fuzzy_modularity  # noqa: E402
from fuzzykg.multislice import multislice_leiden, level_grid, qbar  # noqa: E402
from fuzzykg.synthetic import planted_partition_fuzzy_graph  # noqa: E402
from fuzzykg.backbone import disparity_pvalues  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

RULES = ("global", "per_source", "disparity", "random")


def keep_mask(G, rule, frac, rng):
    m = G.m; k = int(round(frac * m))
    if rule == "global":
        order = np.argsort(-G.mu, kind="stable")
    elif rule == "disparity":
        ps, pd = disparity_pvalues(G); order = np.argsort(np.minimum(ps, pd) - 1e-12 * G.mu, kind="stable")
    elif rule == "random":
        order = rng.permutation(m)
    else:  # per source: rank within each source, then take the best-ranked overall
        rank = np.empty(m)
        for u in np.unique(G.src):
            idx = np.flatnonzero(G.src == u)
            r = np.argsort(np.argsort(-G.mu[idx], kind="stable"), kind="stable")
            rank[idx] = (r + 0.5) / idx.size
        order = np.lexsort((-G.mu, rank))
    mask = np.zeros(m, bool); mask[order[:k]] = True
    return mask


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def res(a):
        A = np.c_[np.ones_like(rz), rz]
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def main(seeds=5, lseeds=(0, 1, 2)):
    rng = np.random.default_rng(17); rows = []
    for shift in np.round(np.arange(0, 1.01, 0.1), 1):
        for seed in range(seeds):
            G, truth = planted_partition_fuzzy_graph(1000, 20, 8.0, 0.5, 0.5, 0.15, float(shift), seed=seed)
            alphas = level_grid(G)
            ms = max((multislice_leiden(G, alphas, seed=s) for s in lseeds), key=lambda r: r.Qbar)
            f = max((fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in lseeds), key=lambda r: r.Q_F)
            for frac in (0.5, 0.25):
                cells = {}
                for rule in RULES:
                    mask = keep_mask(G, rule, frac, rng)
                    Gb = FuzzyGraph(G.n, G.src[mask], G.dst[mask], np.ones(int(mask.sum())))
                    b = max((fuzzy_leiden(Gb, 1.0, seed=s, n_iterations=0) for s in lseeds), key=lambda r: r.Q_F)
                    cells[rule] = {"shortfall": 1 - ari(truth, b.membership), "index": ms.Qbar - qbar(G, b.membership, alphas),
                                   "qf_drop": f.Q_F - fuzzy_modularity(G, b.membership, 1.0),
                                   "retained_mass": float(G.mu[mask].sum() / G.W),
                                   "singleton_share": float((np.bincount(b.membership) == 1).sum() / G.n)}
                rows.append({"shift": float(shift), "seed": seed, "frac": frac, "cells": cells})
            print(shift, seed, {r: round(c["shortfall"], 2) for r, c in rows[-2]["cells"].items()}, flush=True)
    def conc(key, sign=1):
        v = []
        for r in rows:
            c = list(r["cells"].values()); s = n = 0
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    dl = c[i]["shortfall"] - c[j]["shortfall"]
                    if abs(dl) < 0.01:
                        continue
                    n += 1; d = sign * (c[i][key] - c[j][key])
                    s += 1 if dl * d > 0 else (0.5 if d == 0 else 0)
            if n:
                v.append(s / n)
        v = np.array(v); b = [np.mean(rng.choice(v, v.size)) for _ in range(1000)]
        return {"concordance": float(v.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "graphs": int(v.size)}
    # a lower retained mass predicts a larger shortfall, so its sign is reversed
    summary = {k: conc(k, -1 if k == "retained_mass" else 1) for k in ("index", "qf_drop", "retained_mass", "singleton_share")}
    summary["mean_shortfall_by_rule"] = {rule: {str(fr): float(np.mean([r["cells"][rule]["shortfall"] for r in rows if r["frac"] == fr])) for fr in (0.5, 0.25)} for rule in RULES}
    # partial correlations on the main sweep
    L = json.load(open("results/levelset_shift.json")); X = []
    for r in L:
        for c in ("q0.50", "q0.75", "a0.6"):
            cc = r["cuts"].get(c)
            if cc:
                X.append((cc["cost_index"], cc["cost_fuzzy"], 1 - cc["ari"], cc["retained_mass"], cc["mean_out_degree_cut"]))
    X = np.array(X)
    summary["partial_main_sweep"] = {"index_given_mass": partial(X[:, 0], X[:, 2], X[:, 3]), "qf_drop_given_mass": partial(X[:, 1], X[:, 2], X[:, 3]),
                                     "index_given_degree": partial(X[:, 0], X[:, 2], X[:, 4]), "qf_drop_given_degree": partial(X[:, 1], X[:, 2], X[:, 4])}
    print(json.dumps(summary, indent=1))
    json.dump({"rows": rows, "summary": summary}, open("results/nonnested_binarisation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
