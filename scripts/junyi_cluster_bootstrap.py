#!/usr/bin/env python3
"""Junyi expert criterion with dependence between pairs taken into account
(Section 8.4).

Annotated pairs share exercises, so a bootstrap over pairs understates the
uncertainty. Here every interval comes from a two-way cluster bootstrap: the
exercises are resampled with replacement and every annotated pair enters with
weight m(A) m(B), the product of the multiplicities of its two exercises.

Reported for the transition graph built with pruning thresholds tau_E in
{0.1, 0.2, 0.3} (0.2 is the paper's), for the graph without the platform's
suggested attempts, and for the co-practice graph: coverage, Spearman rho and
Kendall tau_b over all pairs, rho over the pairs each score reaches, and
cluster-bootstrap intervals of the differences product - hop distance,
Yager(5) - hop distance, max-min - hop distance and hop distance - direct edge.

Usage: python scripts/junyi_cluster_bootstrap.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.stats import spearmanr, kendalltau
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph  # noqa
from fuzzykg.construction import ConstructionConfig, TransitionConfig, build_transition_graph, preprocess  # noqa
from data.loaders import load_junyi, load_junyi_expert  # noqa
from composition_robustness import mats  # noqa
from prerequisite_criterion import load  # noqa

OPS = ["direct_edge", "hop_baseline", "min", "product", "lukasiewicz", "yager_5"]
DIFFS = [("product", "hop_baseline"), ("yager_5", "hop_baseline"), ("min", "hop_baseline"), ("hop_baseline", "direct_edge"), ("product", "direct_edge")]


def wspearman(x, y, w):
    keep = w > 0
    return spearmanr(np.repeat(x[keep], w[keep]), np.repeat(y[keep], w[keep])).statistic


def evaluate(G, labels, ann, rng, B=500):
    idx = {l: i for i, l in enumerate(labels)}
    sel = ann[ann.Exercise_A.isin(idx) & ann.Exercise_B.isin(idx)]
    ia = sel.Exercise_A.map(idx).to_numpy(); ib = sel.Exercise_B.map(idx).to_numpy(); y = sel.Prerequisite_avg.to_numpy()
    M, _ = mats(G); X = {k: M[k][ib, ia] for k in OPS}
    ex = np.unique(np.r_[ia, ib]); pos = {e: i for i, e in enumerate(ex)}
    pa = np.array([pos[e] for e in ia]); pb = np.array([pos[e] for e in ib])
    out = {"n_pairs": int(len(y)), "n_exercises": int(ex.size), "scores": {}, "diffs": {}}
    for k in OPS:
        x = X[k]; r = x > 0
        out["scores"][k] = {"coverage": float(r.mean()), "rho": float(spearmanr(x, y).statistic),
                            "tau_b": float(kendalltau(x, y).statistic),
                            "rho_reached": float(spearmanr(x[r], y[r]).statistic) if r.sum() > 10 else None}
    boots = {f"{a}-{b}": [] for a, b in DIFFS}; rb = {k: [] for k in OPS}
    for _ in range(B):
        m = rng.multinomial(ex.size, np.full(ex.size, 1.0 / ex.size))
        w = m[pa] * m[pb]
        r = {k: wspearman(X[k], y, w) for k in OPS}
        for k in OPS:
            rb[k].append(r[k])
        for a, b in DIFFS:
            boots[f"{a}-{b}"].append(r[a] - r[b])
    for k in OPS:
        out["scores"][k]["rho_ci_cluster"] = [float(np.nanpercentile(rb[k], 2.5)), float(np.nanpercentile(rb[k], 97.5))]
    for a, b in DIFFS:
        v = np.array(boots[f"{a}-{b}"])
        out["diffs"][f"{a}-{b}"] = {"diff": out["scores"][a]["rho"] - out["scores"][b]["rho"],
                                    "ci_cluster": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
    return out


def main():
    rng = np.random.default_rng(23)
    ex, ann = load_junyi_expert()
    res = {}
    for name in ("junyi_transition", "junyi_transition_unsuggested", "junyi_copractice"):
        G, labels = load(name)
        res[name] = evaluate(G, labels, ann, rng)
        print(name, json.dumps({k: (round(v["rho"], 3), round(v["tau_b"], 3)) for k, v in res[name]["scores"].items()}),
              json.dumps({k: (round(v["diff"], 3), [round(c, 3) for c in v["ci_cluster"]]) for k, v in res[name]["diffs"].items()}), flush=True)
    df = load_junyi(); d = preprocess(df, ConstructionConfig())
    for tau in (0.1, 0.3):
        G, _ = build_transition_graph(d, TransitionConfig(tau_E=tau), preprocessed=True)
        res[f"junyi_transition_tauE{tau}"] = r = evaluate(G, list(G.labels), ann, rng)
        r["n"] = int(G.n); r["m"] = int(G.m)
        print(tau, G.summary(), json.dumps({k: (round(v["coverage"], 2), round(v["rho"], 3)) for k, v in r["scores"].items()}),
              json.dumps({k: (round(v["diff"], 3), [round(c, 3) for c in v["ci_cluster"]]) for k, v in r["diffs"].items()}), flush=True)
    json.dump(res, open("results/junyi_cluster_bootstrap.json", "w"), indent=1)


if __name__ == "__main__":
    main()
