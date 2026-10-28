#!/usr/bin/env python3
"""Pruning threshold of the Junyi transition graph on one fixed set of pairs
(Section 8.4).

The transition graph is rebuilt with pruning thresholds tau_E in {0 (no pruning
beyond a positive membership), 0.05, 0.1, 0.2, 0.3}. Every threshold is
evaluated on the same 1,954 annotated pairs: a pair whose exercises are not both
nodes of the graph scores 0 under every score, like a pair that no path
reaches. Reported per threshold and score: share of pairs reached, Spearman rho
and Kendall tau-b with two-way cluster-bootstrap intervals over exercises (500
resamples, a pair weighted by the product of the multiplicities of its two
exercises), and cluster intervals of the differences of every operator to the
hop distance and to the direct edge. The difference of the hop-distance
correlation between thresholds is bootstrapped on the same resamples.

Writes results/junyi_pruning_fixed_pairs.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.stats import spearmanr, kendalltau
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg.construction import ConstructionConfig, TransitionConfig, build_transition_graph, preprocess  # noqa: E402
from data.loaders import load_junyi, load_junyi_expert  # noqa: E402
from composition_robustness import mats  # noqa: E402

OPS = ["direct_edge", "hop_baseline", "min", "product", "lukasiewicz", "yager_5"]
TAUS = (1e-9, 0.05, 0.1, 0.2, 0.3)


def wstat(x, y, w, f):
    k = w > 0
    return f(np.repeat(x[k], w[k]), np.repeat(y[k], w[k])).statistic


def main(B=500):
    rng = np.random.default_rng(31)
    ex, ann = load_junyi_expert()
    A = ann.Exercise_A.astype(str).to_numpy(); Bn = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    exs = np.unique(np.r_[A, Bn]); pos = {x: i for i, x in enumerate(exs)}
    pa = np.array([pos[x] for x in A]); pb = np.array([pos[x] for x in Bn])
    W = [None] * B
    for b in range(B):
        m = rng.multinomial(exs.size, np.full(exs.size, 1 / exs.size)); W[b] = m[pa] * m[pb]
    df = load_junyi(); d = preprocess(df, ConstructionConfig())
    res = {"n_pairs": int(len(y)), "n_exercises": int(exs.size), "thresholds": {}}
    boot_hop = {}
    for tau in TAUS:
        G, _ = build_transition_graph(d, TransitionConfig(tau_E=tau), preprocessed=True)
        idx = {str(l): i for i, l in enumerate(G.labels)}
        ok = np.array([a in idx and b in idx for a, b in zip(A, Bn)])
        ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in Bn])
        M, _ = mats(G)
        X = {k: np.where(ok, M[k][ib, ia], 0.0) for k in OPS}
        r = {"n": int(G.n), "m": int(G.m), "pairs_in_graph": float(ok.mean()), "scores": {}, "diffs": {}}
        bs = {k: np.array([wstat(X[k], y, W[b], spearmanr) for b in range(B)]) for k in OPS}
        bt = {k: np.array([wstat(X[k], y, W[b], kendalltau) for b in range(B)]) for k in OPS}
        for k in OPS:
            r["scores"][k] = {"coverage": float((X[k] > 0).mean()), "rho": float(spearmanr(X[k], y).statistic),
                              "rho_ci": [float(np.nanpercentile(bs[k], 2.5)), float(np.nanpercentile(bs[k], 97.5))],
                              "tau_b": float(kendalltau(X[k], y).statistic),
                              "tau_b_ci": [float(np.nanpercentile(bt[k], 2.5)), float(np.nanpercentile(bt[k], 97.5))]}
        for k in OPS:
            for ref in ("hop_baseline", "direct_edge"):
                if k == ref:
                    continue
                v = bs[k] - bs[ref]
                r["diffs"][f"{k}-{ref}"] = {"diff": r["scores"][k]["rho"] - r["scores"][ref]["rho"],
                                            "ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
        boot_hop[tau] = bs["hop_baseline"]
        key = "none" if tau < 1e-6 else f"{tau:g}"
        res["thresholds"][key] = r
        print(key, G.summary(), {k: (round(v["coverage"], 2), round(v["rho"], 3)) for k, v in r["scores"].items()},
              {k: (round(v["diff"], 3), [round(c, 3) for c in v["ci"]]) for k, v in r["diffs"].items() if k.endswith("hop_baseline")}, flush=True)
    res["hop_threshold_diffs"] = {}
    for t1 in TAUS:
        for t2 in TAUS:
            if t1 < t2:
                v = boot_hop[t1] - boot_hop[t2]
                k1 = "none" if t1 < 1e-6 else f"{t1:g}"; k2 = f"{t2:g}"
                res["hop_threshold_diffs"][f"{k1}-{k2}"] = {"diff": res["thresholds"][k1]["scores"]["hop_baseline"]["rho"] - res["thresholds"][k2]["scores"]["hop_baseline"]["rho"],
                                                           "ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
    json.dump(res, open("results/junyi_pruning_fixed_pairs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
