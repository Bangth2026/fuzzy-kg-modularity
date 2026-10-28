#!/usr/bin/env python3
"""Level-set modularity on the real graphs (Section 8.3, Table 9): the
multislice partition, its agreement with the Fuzzy-Leiden partition, and the
cost index cost(alpha_0) = Qbar(Pi*) - Qbar(Pi_alpha_0) of the single-cut
partitions at the quantile and absolute levels used in Table 9, next to the
persistence reading of the same cuts.

Usage: python scripts/levelset_real.py [--out results/levelset_real.json] [names ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden  # noqa: E402
from fuzzykg.multislice import multislice_leiden, level_grid, qbar, persistence, persistence_cost  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

NAMES = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2009_item", "assist2015_kc",
         "assist2017_kc", "assist2017_item", "junyi_transition", "junyi_copractice", "assist2009_kc_transition",
         "assist2015_kc_transition", "assist2017_kc_transition", "assist2017_item_transition"]


def truth_for(name, n):
    """Ground-truth partition where the corpus has one: the topic of each Junyi exercise."""
    if not name.startswith("junyi"):
        return None
    ex = pd.read_csv("data/raw/junyi/junyi/junyi_Exercise_table.csv", encoding="utf-8-sig")
    t = dict(zip(ex.name, ex.topic))
    labels = pd.read_csv(f"data/graphs/{name}_nodes.csv").label.astype(str).tolist()
    return [str(t.get(l, "NA")) for l in labels]


def load(name):
    df = pd.read_csv(f"data/graphs/{name}_edges.csv")
    n = len(pd.read_csv(f"data/graphs/{name}_nodes.csv"))
    return FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=NAMES)
    ap.add_argument("--out", default="results/levelset_real.json")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    rows = []
    for name in args.names:
        G = load(name)
        alphas = level_grid(G)
        f = max((fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda r: r.Q_F)
        ms = max((multislice_leiden(G, alphas, seed=s) for s in seeds), key=lambda r: r.Qbar)
        me = max((multislice_leiden(G, alphas, seed=s, weighting="equal") for s in seeds), key=lambda r: r.Qbar)
        tracks = persistence(G, alphas, seed=0)
        truth = truth_for(name, G.n)
        row = {"graph": name, "summary": G.summary(), "alphas": alphas.tolist(),
               "fuzzy": {"Q_F": f.Q_F, "k": f.n_communities, "Qbar": qbar(G, f.membership, alphas),
                         "Qbar_equal": qbar(G, f.membership, alphas, weighting="equal")},
               "multislice": {"Qbar": ms.Qbar, "k": ms.n_communities, "Q_F": ms.Q_F, "profile": ms.profile.tolist(),
                              "ari_vs_fuzzy": ari(f.membership, ms.membership)},
               "multislice_equal": {"Qbar": me.Qbar, "k": me.n_communities, "Q_F": me.Q_F,
                                    "ari_vs_fuzzy": ari(f.membership, me.membership)},
               "n_tracks": len(tracks), "cuts": {}}
        if truth is not None:
            row["fuzzy"]["ari_truth"] = ari(truth, f.membership)
            row["multislice"]["ari_truth"] = ari(truth, ms.membership)
            row["multislice_equal"]["ari_truth"] = ari(truth, me.membership)
        cuts = {"a0.2": 0.2, "a0.4": 0.4, "a0.6": 0.6, "a0.8": 0.8,
                "q0.25": float(np.quantile(G.mu, 0.25)), "q0.50": float(np.quantile(G.mu, 0.5)), "q0.75": float(np.quantile(G.mu, 0.75))}
        for cname, a in cuts.items():
            try:
                Ga = G.alpha_cut(a)
            except ValueError:
                row["cuts"][cname] = None
                continue
            best = max((fuzzy_leiden(Ga, 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda r: r.Q_F)
            row["cuts"][cname] = {"alpha": a, "k": best.n_communities,
                                  "retained_mass": float(G.mu[G.mu >= a].sum() / G.W),
                                  "cost_index": ms.Qbar - qbar(G, best.membership, alphas),
                                  "cost_index_equal": me.Qbar - qbar(G, best.membership, alphas, weighting="equal"),
                                  "ari_vs_multislice": ari(ms.membership, best.membership),
                                  "ari_vs_fuzzy": ari(f.membership, best.membership),
                                  "persistence_cost": persistence_cost(tracks, a, G.W, 0.05),
                                  "mean_out_degree_cut": float((G.mu >= a).sum() / G.n),
                                  "singleton_share": float((np.bincount(best.membership) == 1).sum() / G.n),
                                  "crisp_modularity": float(best.Q_F),
                                  "ari_truth": (ari(truth, best.membership) if truth is not None else None)}
        print(name, json.dumps({c: (None if v is None else {k: round(v[k], 3) for k in ("cost_index", "ari_vs_fuzzy", "retained_mass")}) for c, v in row["cuts"].items()}), flush=True)
        rows.append(row)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
