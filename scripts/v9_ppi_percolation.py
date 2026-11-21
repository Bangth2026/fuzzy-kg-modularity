#!/usr/bin/env python3
"""Version 4.6.0. Percolation of the alpha-cuts of PPI5k (Section 8.6): share of ordered pairs joined
in the cut (PPI5k is symmetric, so reachability is connected components), at quantile levels of
the memberships.  Needs data/graphs/ppi5k_edges.csv (scripts/v9_cross_domain.py --write-graphs).
Writes results/v9/ppi5k_percolation.json."""
import json
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

e = pd.read_csv("data/graphs/ppi5k_edges.csv"); n = int(max(e.src.max(), e.dst.max()) + 1); mu = e.mu.to_numpy()
out = []
for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
    a = np.quantile(mu, q); k = mu >= a
    A = csr_matrix((np.ones(k.sum()), (e.src[k], e.dst[k])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    sz = np.bincount(lab)
    out.append({"quantile": q, "alpha": float(a), "edges": int(k.sum()), "reachable_share": float((sz * (sz - 1)).sum() / (n * (n - 1))),
                "largest_component": int(sz.max())})
json.dump(out, open("results/v9/ppi5k_percolation.json", "w"), indent=1)
print(out)
