#!/usr/bin/env python3
"""Edge-level agreement of the Junyi memberships with the expert score
(Section 8.2), with two-way cluster-bootstrap intervals over exercises
(500 resamples; a pair enters with the product of the multiplicities of its two
exercises): rho of mu(B -> A), and of the directed difference
mu(B -> A) - mu(A -> B), with the crowd score that B is a prerequisite of A.
Writes results/junyi_expert_cluster.json."""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loaders import load_junyi_expert  # noqa: E402


def wrho(x, y, w):
    k = w > 0
    return spearmanr(np.repeat(x[k], w[k]), np.repeat(y[k], w[k])).statistic


def main(B=500):
    rng = np.random.default_rng(29); ex, ann = load_junyi_expert(); out = {}
    for name in ("junyi_copractice", "junyi_transition"):
        e = pd.read_csv(f"data/graphs/{name}_edges.csv"); nd = pd.read_csv(f"data/graphs/{name}_nodes.csv")
        lab = dict(zip(nd.id, nd.label.astype(str))); mu = {(lab[s], lab[d]): m for s, d, m in zip(e.src, e.dst, e.mu)}
        nodes = set(lab.values()); sel = ann[ann.Exercise_A.astype(str).isin(nodes) & ann.Exercise_B.astype(str).isin(nodes)]
        A = sel.Exercise_A.astype(str).to_numpy(); Bn = sel.Exercise_B.astype(str).to_numpy(); y = sel.Prerequisite_avg.to_numpy()
        ba = np.array([mu.get((b, a), 0.0) for a, b in zip(A, Bn)]); ab = np.array([mu.get((a, b), 0.0) for a, b in zip(A, Bn)])
        exs = np.unique(np.r_[A, Bn]); pos = {x: i for i, x in enumerate(exs)}
        pa = np.array([pos[x] for x in A]); pb = np.array([pos[x] for x in Bn])
        res = {}
        for key, x in (("directed", ba), ("difference", ba - ab)):
            v = []
            for _ in range(B):
                m = rng.multinomial(exs.size, np.full(exs.size, 1 / exs.size)); v.append(wrho(x, y, m[pa] * m[pb]))
            res[key] = {"rho": float(spearmanr(x, y).statistic), "ci_cluster": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
        out[name] = {"n_pairs": int(len(y)), **res}
        print(name, out[name], flush=True)
    json.dump(out, open("results/junyi_expert_cluster.json", "w"), indent=1)


if __name__ == "__main__":
    main()
