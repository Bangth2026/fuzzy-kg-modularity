"""Graph statistics table (Section 9.1): nodes, edges, density,
weak components, reciprocity, membership mean / sd / quartiles, and, when a
*_nodes.csv with a problem-set column exists next to the edge list, the NMI of
the Fuzzy-Leiden partition with the problem-set (or component) reference.

Usage: python scripts/graph_stats.py data/graphs/*_edges.csv [--gamma 0.5]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden
from sklearn.metrics import normalized_mutual_info_score as nmi
import scipy.sparse.csgraph as csg

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("edges", nargs="+"); ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--out", default="results/graph_stats.json")
    args = ap.parse_args(); rows = []
    for path in args.edges:
        df = pd.read_csv(path); G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
        ncomp, comp = csg.connected_components(G.symmetrised(), directed=False)
        part = fuzzy_leiden(G, args.gamma, seed=0, n_iterations=0)
        q = np.quantile(G.mu, [0.25, 0.5, 0.75])
        row = {"graph": os.path.basename(path), "n": G.n, "m": G.m, "density": G.m / G.n,
               "weak_components": int(ncomp), "largest_component": int(np.bincount(comp).max()),
               "reciprocity": G.reciprocity(), "mu_mean": float(G.mu.mean()), "mu_sd": float(G.mu.std()),
               "mu_q25_50_75": q.tolist(), "k_leiden": part.n_communities, "Q_F": part.Q_F,
               "NMI_leiden_vs_components": nmi(comp, part.membership)}
        nodes = path.replace("_edges.csv", "_nodes.csv")
        if os.path.exists(nodes):
            nd = pd.read_csv(nodes)
            for col in ("problem_set", "skill", "set_id"):
                if col in nd.columns and len(nd) == G.n:
                    row[f"NMI_leiden_vs_{col}"] = nmi(nd[col].astype(str), part.membership)
        print(json.dumps(row, indent=1)); rows.append(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(rows, open(args.out, "w"), indent=1)

if __name__ == "__main__":
    main()
