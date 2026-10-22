#!/usr/bin/env python3
"""Tables 7 and 8: parametric t-norm families and compensatory alternatives on
a fuzzy graph.

For a sample of source nodes, every target reachable within --hops is scored
by the best path under each operator (exact bounded enumeration, since only
the minimum has optimal substructure).  Reports mean strength overall and by
hop count of the min-best path, erosion E = mean[(S_min - S_T)/S_min], and
Spearman rank agreement with the minimum.  Also times the Dijkstra widest-path
sweep against the enumeration (Section 8.3, third consequence) and counts
Fact 1 violations (must be zero for every t-norm).

Usage:
  python scripts/tnorm_sweep.py data/graphs/assist2009_item_edges.csv --sources 60 --hops 4
  python scripts/tnorm_sweep.py data/graphs/mooccubex_cs_edges.csv --sources all --hops 5 --standard-only
  python scripts/tnorm_sweep.py --synthetic 371     # offline demo on a synthetic graph
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fuzzykg import FuzzyGraph
from fuzzykg.synthetic import synthetic_fuzzy_graph
from fuzzykg.tnorms import (
    enumerate_best_paths,
    owa,
    schweizer_sklar,
    t_lukasiewicz,
    t_min,
    t_product,
    widest_path,
    yager,
)


def compensatory(name):
    if name == "geometric mean":
        return lambda a, b: np.sqrt(a * b), True
    if name == "arithmetic mean":
        return lambda a, b: 0.5 * (a + b), True
    raise ValueError


def best_paths_path_level(G, source, hops, agg):
    """For path-level aggregators (OWA over all edges of a path) we need the
    whole membership list; enumerate explicitly."""
    A = G.adjacency()
    best = {}
    frontier = [(source, (), (source,))]
    for hop in range(1, hops + 1):
        nxt = []
        for u, ms, seen in frontier:
            row = A.getrow(u)
            for v, w in zip(row.indices, row.data):
                if v in seen:
                    continue
                ms2 = ms + (float(w),)
                s = agg(ms2)
                if v not in best or s > best[v][0]:
                    best[v] = (s, hop)
                nxt.append((v, ms2, seen + (v,)))
        frontier = nxt
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges", nargs="?", default=None)
    ap.add_argument("--synthetic", type=int, default=None)
    ap.add_argument("--sources", default="60", help="number of sampled sources or 'all'")
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--standard-only", action="store_true", help="Table 8 style: min / product / Lukasiewicz by hop")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.edges:
        df = pd.read_csv(args.edges)
        G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
    else:
        G = synthetic_fuzzy_graph(args.synthetic or 371, 0.7, seed=args.seed)
        G.mu = np.clip(rng.beta(1.5, 2.0, size=G.m), 0.2, 1.0); G._adj = None    # wide spread like 2009-item
    print(G.summary())
    sources = np.arange(G.n) if args.sources == "all" else rng.choice(G.n, size=min(int(args.sources), G.n), replace=False)

    ops = [("min (default)", "t-norm", t_min), ("product", "t-norm", t_product), ("Lukasiewicz", "t-norm", t_lukasiewicz)]
    if not args.standard_only:
        ops += [(f"Yager p = {p}", "t-norm", yager(p)) for p in (1.5, 2, 3, 5, 10)]
        ops += [(f"Schweizer-Sklar p = {p}", "t-norm", schweizer_sklar(p)) for p in (-8, -4, -2, -1, 0, 0.5, 1)]

    # reference: min-best paths and hop counts, plus widest-path timing
    t0 = time.perf_counter()
    for s in sources:
        widest_path(G, int(s))
    t_dijkstra = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = {}
    expanded = 0
    for s in sources:
        b = enumerate_best_paths(G, int(s), args.hops, t_min)
        expanded += b.pop("__expanded__")[0]
        for v, (st, hop) in b.items():
            ref[(int(s), v)] = (st, hop)
    t_enum = time.perf_counter() - t0
    pairs = sorted(ref)
    print(f"{len(sources)} sources, {len(pairs)} reachable pairs within {args.hops} hops; "
          f"widest-path sweep {t_dijkstra*1e3:.1f} ms vs enumeration {t_enum:.2f} s ({expanded:.0f} partial paths expanded)")
    S_min = np.array([ref[p][0] for p in pairs])
    hops = np.array([ref[p][1] for p in pairs])

    hop_cols = sorted(set(hops.tolist()))
    print(f"\n{'Operator':24s} {'Class':13s} {'Mean S':>7} " + " ".join(f"{h}-hop" for h in hop_cols) + f" {'Erosion':>8} {'rho':>6} {'Fact1':>6}")

    def report(name, cls, S, viol):
        E = np.mean((S_min - S) / S_min)
        rho = spearmanr(S, S_min).correlation
        byhop = " ".join(f"{S[hops == h].mean():5.3f}" for h in hop_cols)
        print(f"{name:24s} {cls:13s} {S.mean():7.3f} {byhop} {E:8.3f} {rho:6.3f} {viol:>6}")

    for name, cls, T in ops:
        S = np.zeros(len(pairs)); viol = 0
        for s in sources:
            b = enumerate_best_paths(G, int(s), args.hops, T)
            viol += b.pop("__expanded__")[1]
            for i, p in enumerate(pairs):
                if p[0] == int(s) and p[1] in b:
                    S[i] = b[p[1]][0]
        report(name, cls, S, viol)

    if not args.standard_only:
        for name in ("geometric mean", "arithmetic mean"):
            agg, _ = compensatory(name)
            S = np.zeros(len(pairs))
            for s in sources:
                b = enumerate_best_paths(G, int(s), args.hops, agg)
                b.pop("__expanded__")
                for i, p in enumerate(pairs):
                    if p[0] == int(s) and p[1] in b:
                        S[i] = b[p[1]][0]
            report(name, "compensatory", S, "n/a")
        # path-level pessimistic OWA: weights increasing toward the smallest value
        def pess_owa(ms):
            n = len(ms); w = np.arange(1, n + 1, dtype=float); w /= w.sum()   # orness < 0.5
            return owa(ms, w)
        S = np.zeros(len(pairs))
        for s in sources:
            b = best_paths_path_level(G, int(s), args.hops, pess_owa)
            for i, p in enumerate(pairs):
                if p[0] == int(s) and p[1] in b:
                    S[i] = b[p[1]][0]
        report("path-level OWA (pessim.)", "compensatory", S, "n/a")


if __name__ == "__main__":
    main()
