#!/usr/bin/env python3
"""Release 5.0.0. Leiden partitions reused by every v15 script (cached, resumable).

For each of the 23 log-built and curated graphs and each resolution gamma of GAMMAS:
  support      crisp Leiden on the support graph (every membership 1)
  q0.25 ...    crisp Leiden on the quantile level sets (unit weights)
  F            Leiden on Q_F with the observed memberships
at 20 seeds for gamma = 1 and 10 seeds otherwise; at gamma = 1 also, for the value-of-grades
decomposition (Section 6), Leiden on Q_F after a global and after a row-wise permutation of the
memberships (10 draws each, best of three seeds by Q_F on the permuted weights).
Writes results/v15/partitions/<graph>.npz.
Usage: python scripts/v15_partitions.py [graph ...]
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden  # noqa: E402
from fuzzykg.nulltest import permute_global, permute_rows  # noqa: E402

OUT = "results/v15/partitions"
GRAPHS13 = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2009_item", "assist2015_kc", "assist2017_kc",
            "assist2017_item", "junyi_copractice", "assist2009_kc_transition", "assist2015_kc_transition",
            "assist2017_kc_transition", "assist2017_item_transition", "junyi_transition"]
GRAPHS_ADDED = ["assist2012_copractice", "assist2012_transition", "algebra2005_copractice", "algebra2005_transition",
                "bridge2006_copractice", "bridge2006_transition", "eedi_copractice", "eedi_transition",
                "ednet_copractice", "ednet_transition"]
GRAPHS = GRAPHS13 + GRAPHS_ADDED
GAMMAS = (0.25, 0.35, 0.5, 0.71, 1.0, 1.41, 2.0, 2.83, 4.0)
CUTS = (0.25, 0.50, 0.75)
PERM_DRAWS = 10


def load(name):
    df = pd.read_csv(f"data/graphs/{name}_edges.csv")
    return FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))


def main(names):
    os.makedirs(OUT, exist_ok=True)
    for name in names:
        path = f"{OUT}/{name}.npz"
        store = dict(np.load(path)) if os.path.exists(path) else {}
        G = load(name)
        Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
        H = {f"q{q:.2f}": G.alpha_cut(float(np.quantile(G.mu, q))) for q in CUTS}
        t0 = time.perf_counter()
        for g in GAMMAS:
            seeds = range(20) if g == 1.0 else range(10)
            for which, graph in [("support", Gs), *H.items(), ("F", G)]:
                for s in seeds:
                    k = f"{which}|{g:g}|{s}"
                    if k not in store:
                        store[k] = _leiden(graph, s, g).astype(np.int32)
            np.savez_compressed(path, **store)
        rng = np.random.default_rng(2026)
        for kind, fn in (("glob", permute_global), ("row", permute_rows)):
            for d in range(PERM_DRAWS):
                k = f"F{kind}{d}|1|best3"
                Gp = fn(G, rng)
                if k in store:
                    continue
                P = [_leiden(Gp, s, 1.0) for s in range(3)]
                store[k] = max(P, key=lambda p: fuzzy_modularity(Gp, p)).astype(np.int32)
                store[f"F{kind}{d}|1|mu"] = Gp.mu.astype(np.float64)
        np.savez_compressed(path, **store)
        print(name, G.n, G.m, f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or GRAPHS)
