#!/usr/bin/env python3
"""Tables 2 and 4: Fuzzy Leiden on a saved fuzzy graph over a resolution grid,
with the Proposition 1 and 2(iii) checks, Proposition 3 / Corollary 1 numbers
for the best partition, and the Remark 3 threshold measured by Algorithm 1.

Usage:
  python scripts/run_leiden_grid.py data/graphs/mooccubex_cs_edges.csv --gammas 0.5 1.0 --seeds 10
  python scripts/run_leiden_grid.py data/graphs/assist2009_item_edges.csv --gammas 0.5 1 2 5 --seeds 5
  python scripts/run_leiden_grid.py <edges.csv> --algorithm1        # also run Algorithm 1
  python scripts/run_leiden_grid.py <edges.csv> --synthetic 59       # no file: a synthetic graph instead
"""

import argparse
import time

import numpy as np
import pandas as pd

from fuzzykg import FuzzyGraph
from fuzzykg.leiden import fuzzy_leiden
from fuzzykg.modularity import bipartition_threshold, correlation_gap, fuzzy_modularity
from fuzzykg.resolution import adaptive_resolution, spectral_bisection
from fuzzykg.synthetic import synthetic_fuzzy_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges", nargs="?", default=None)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--algorithm1", action="store_true")
    ap.add_argument("--synthetic", type=int, default=None)
    args = ap.parse_args()

    if args.edges:
        df = pd.read_csv(args.edges)
        G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
    else:
        G = synthetic_fuzzy_graph(args.synthetic or 59, 0.85, seed=0)
    print(G.summary())

    B = spectral_bisection(G)
    th = bipartition_threshold(G, B)
    print(f"Remark 3: balanced-cut c/W = {th['c_over_W']:.3f}, 2c/W = {th['two_c_over_W']:.3f}, "
          f"exact threshold {th['gamma_threshold']:.3f}; trivial partition has Q_F = 1 - gamma")

    print(f"\n{'gamma':>6} {'Q_F':>7} {'comms':>6} {'sizes (top 6)':>28} {'conn':>5} {'mono':>5} {'alpha-avg':>9} {'gap':>8} {'time(ms)':>9}")
    for gamma in args.gammas:
        t0 = time.perf_counter()
        runs = [fuzzy_leiden(G, gamma, seed=s, n_iterations=args.iters) for s in range(args.seeds)]
        dt = (time.perf_counter() - t0) / args.seeds * 1e3
        best = max(runs, key=lambda r: r.Q_F)
        conn = sum(r.connected_violations for r in runs)
        mono = sum(r.monotone_violations for r in runs)
        cg = correlation_gap(G, best.membership, gamma)
        print(f"{gamma:6.2f} {best.Q_F:7.3f} {best.n_communities:6d} {str(best.sizes[:6]):>28} "
              f"{'pass' if conn == 0 else conn:>5} {'pass' if mono == 0 else mono:>5} "
              f"{cg['alpha_average']:9.3f} {cg['gap']:8.4f} {dt:9.1f}")
        if best.n_communities == 1:
            print(f"       degenerate: trivial partition, Q_F = 1 - gamma = {1 - gamma:.3f}")

    if args.algorithm1:
        t0 = time.perf_counter()
        res = adaptive_resolution(G, seeds=range(args.seeds))
        print(f"\nAlgorithm 1: gamma1 = {res.gamma1:.3f}, gamma2 = {res.gamma2:.3f} after {res.doublings} doubling(s), "
              f"c/W = {res.c_over_W:.3f}, {time.perf_counter() - t0:.2f} s")
        for h in res.history:
            print(f"   gamma1 {h['gamma1']:.3f}: nontrivial {h['nontrivial']}, min ARI {h['min_ARI']:.3f}, k {h['k']}")


if __name__ == "__main__":
    main()
