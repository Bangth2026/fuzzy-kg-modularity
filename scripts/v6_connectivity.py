#!/usr/bin/env python3
"""Version 4.3.0. How many communities the Claim 1 check covers.

Leiden on Q_F on each of the thirteen graphs at gamma in {0.5, 1, 2}, seeds 0-4 (every run, not
only the best), and on every level set q = 0.25, 0.50, 0.75 at gamma = 1; each returned community
is checked for weak connectivity in the support graph of the graph it was computed on.
Writes results/v6/connectivity.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd, igraph as ig
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden  # noqa: E402

NAMES = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2009_kc_transition", "assist2009_item", "assist2015_kc",
         "assist2015_kc_transition", "assist2017_kc", "assist2017_kc_transition", "assist2017_item", "assist2017_item_transition",
         "junyi_copractice", "junyi_transition"]


def check(G, memb):
    g = ig.Graph(n=G.n, edges=list(zip(G.src.tolist(), G.dst.tolist())), directed=True)
    bad = 0; cnt = 0
    for c in np.unique(memb):
        nodes = np.flatnonzero(memb == c)
        cnt += 1
        if nodes.size > 1 and not g.induced_subgraph(nodes.tolist()).is_connected(mode="weak"):
            bad += 1
    return cnt, bad


def main():
    out = {}
    tot = bad = runs = 0
    for name in NAMES:
        df = pd.read_csv(f"data/graphs/{name}_edges.csv")
        G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
        r = {"runs": 0, "communities": 0, "disconnected": 0}
        graphs = [(G, g) for g in (0.5, 1.0, 2.0)] + [(G.alpha_cut(float(np.quantile(G.mu, q))), 1.0) for q in (0.25, 0.5, 0.75)]
        for H, gam in graphs:
            for s in range(5):
                m = np.asarray(fuzzy_leiden(H, gam, seed=s).membership)
                c, b = check(H, m)
                r["runs"] += 1; r["communities"] += c; r["disconnected"] += b
        out[name] = r; tot += r["communities"]; bad += r["disconnected"]; runs += r["runs"]
        print(name, r, flush=True)
    out["total"] = {"runs": runs, "communities": tot, "disconnected": bad}
    print(out["total"])
    json.dump(out, open("results/v6/connectivity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
