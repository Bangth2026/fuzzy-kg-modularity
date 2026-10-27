"""Does the binarisation conclusion survive strictly increasing rescalings of mu_E?
On planted partitions, apply random increasing piecewise-linear maps phi to the
memberships, re-run Fuzzy Leiden and the quantile-cut shortcut, and report the
recovery (ARI) and loss before and after (Section 8.2).  Usage: python scripts/scale_invariance_check.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden, fuzzy_modularity
from fuzzykg.synthetic import planted_partition_fuzzy_graph
from sklearn.metrics import adjusted_rand_score as ari

def best(G, Geval, seeds=(0, 1, 2), gammas=(0.25, 0.5, 1.0, 2.0, 4.0)):
    b = None
    for g in gammas:
        for s in seeds:
            r = fuzzy_leiden(G, g, seed=s, n_iterations=0); q = fuzzy_modularity(Geval, r.membership, 1.0)
            if b is None or q > b[0]:
                b = (q, r.membership)
    return b

def level(G, a):
    keep = G.mu >= a
    return FuzzyGraph(G.n, G.src[keep], G.dst[keep], np.ones(int(keep.sum())))

if __name__ == "__main__":
    rng = np.random.default_rng(0); rows = []
    for mean, sd, shift in [(0.3, 0.15, 0.0), (0.5, 0.15, 0.0), (0.5, 0.15, 0.5), (0.7, 0.25, 0.0), (0.7, 0.25, 0.5), (0.9, 0.15, 0.5)]:
        G, truth = planted_partition_fuzzy_graph(1000, 20, 8.0, 0.7, mean, sd, shift, seed=0)
        for m in range(4):
            if m == 0:
                Gx = G; label = "identity"
            else:
                grid = np.linspace(0, 1, 21); inc = rng.dirichlet(np.full(20, rng.uniform(0.3, 2.0)))
                vals = np.concatenate([[0.0], np.cumsum(inc)])
                Gx = FuzzyGraph(G.n, G.src, G.dst, np.clip(np.interp(G.mu, grid, vals), 1e-6, 1.0)); label = f"phi{m}"
            qf, pf = best(Gx, Gx)
            row = {"mean": mean, "sd": sd, "shift": shift, "map": label, "mu_mean": float(Gx.mu.mean()),
                   "fuzzy_ARI": ari(truth, pf), "cuts": {}}
            for q in (0.5, 0.75):
                a = float(np.quantile(Gx.mu, q)); Ga = level(Gx, a); qa, pa = best(Ga, Gx)
                row["cuts"][f"q={q}"] = {"ARI": ari(truth, pa), "loss": qf - qa, "mass_kept": float(Gx.mu[Gx.mu >= a].sum() / Gx.W)}
            # absolute cut 0.6 on the rescaled graph
            if (Gx.mu >= 0.6).sum() > 0:
                Ga = level(Gx, 0.6); qa, pa = best(Ga, Gx)
                row["cuts"]["alpha=0.6"] = {"ARI": ari(truth, pa), "loss": qf - qa, "mass_kept": float(Gx.mu[Gx.mu >= 0.6].sum() / Gx.W)}
            rows.append(row); print(row, flush=True)
    json.dump(rows, open("results/scale_invariance.json", "w"), indent=1)
