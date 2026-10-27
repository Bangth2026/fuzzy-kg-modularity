"""Algorithm 1 with its null-model test and 'no structure' outcome (Section 5.3).

Runs the revised procedure on (a) planted-partition graphs with structure,
(b) the same topology with memberships shuffled and communities destroyed
(p_in = 1/k, i.e. no planted structure), and (c) any real edge lists given on
the command line, and reports gamma_1, the z-score of Q_F against the
degree-preserving null model and the outcome.  Also reports the general
bipartition threshold theta(B) of the revised Proposition 1 for the spectral
bisection and for every 'community vs rest' split of the accepted partition,
next to the balanced-cut value 2c/W.

Usage: python scripts/null_model_check.py [--out results/null_model.json] [edges.csv ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden  # noqa: E402
from fuzzykg.modularity import bipartition_threshold_general, community_vs_rest_candidates  # noqa: E402
from fuzzykg.resolution import adaptive_resolution_with_null, spectral_bisection  # noqa: E402
from fuzzykg.synthetic import planted_partition_fuzzy_graph  # noqa: E402


def analyse(name, G, n_perm=20):
    r = adaptive_resolution_with_null(G, z_min=3.0, dq_min=0.10, reference_gamma=1.0, n_perm=n_perm)
    part = fuzzy_leiden(G, r["gamma1"], seed=0, n_iterations=0)
    cands = [spectral_bisection(G)] + community_vs_rest_candidates(part.membership)
    th = bipartition_threshold_general(G, cands)
    out = {"graph": name, "summary": G.summary(), "gamma1": r["gamma1"], "doublings": r["doublings"],
           "k_accepted": part.n_communities, "Q_F_accepted": part.Q_F,
           "null": r["null"], "outcome": r["outcome"],
           "theta_balanced_2c_over_W": th["thresholds"][0]["two_c_over_W"],
           "theta_spectral_bisection": th["thresholds"][0]["theta"],
           "theta_min_over_community_splits": th["min_theta"]}
    print(json.dumps(out, indent=1), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges", nargs="*")
    ap.add_argument("--out", default="results/null_model.json")
    ap.add_argument("--n-perm", type=int, default=20)
    args = ap.parse_args()
    rows = []
    G, _ = planted_partition_fuzzy_graph(600, 12, 8.0, 0.8, 0.6, 0.15, 0.3, seed=0)
    rows.append(analyse("planted, p_in=0.8", G, args.n_perm))
    G, _ = planted_partition_fuzzy_graph(600, 12, 8.0, 1 / 12, 0.6, 0.15, 0.0, seed=0)
    rows.append(analyse("planted, p_in=1/k (no structure)", G, args.n_perm))
    G, _ = planted_partition_fuzzy_graph(110, 4, 50.0, 0.3, 0.5, 0.2, 0.0, seed=0)
    rows.append(analyse("dense co-practice-like, n=110, 50 edges/node", G, args.n_perm))
    for path in args.edges:
        import pandas as pd
        df = pd.read_csv(path)
        rows.append(analyse(os.path.basename(path), FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu)), args.n_perm))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
