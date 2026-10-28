#!/usr/bin/env python3
"""Does the price of a cut depend on the unit of the membership scale? (Section 8.1)

On the planted sweep (shift 0 to 1, ten graph seeds), the three quantile cuts
(q = 0.25, 0.50, 0.75) keep the same edges under every strictly increasing
rescaling phi of mu_E, so their partitions and their recovery are unchanged. The
multislice index on the quantile grid is unchanged as well. The Q_F drop,
Q_F(Pi_fuzzy) - Q_F(Pi_cut), is recomputed on G^phi for phi in {identity, x^3,
x^(1/3), 1 - (1 - x)^3}; Leiden on Q_F is re-run on every G^phi. Reported:
Spearman correlation of each quantity with the recovery shortfall 1 - ARI(cut,
truth), pooled and within graph (pairwise concordance), and the rank agreement
of the Q_F drop under phi with the Q_F drop under the identity.

Usage: python scripts/rescaling_validity.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden, fuzzy_modularity  # noqa: E402
from fuzzykg.multislice import multislice_leiden, level_grid, qbar  # noqa: E402
from fuzzykg.synthetic import planted_partition_fuzzy_graph  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

PHI = {"identity": lambda x: x, "cube": lambda x: x ** 3, "cube_root": lambda x: x ** (1 / 3), "upper": lambda x: 1 - (1 - x) ** 3}
CUTS = (0.25, 0.50, 0.75)


def main(seeds=10, lseeds=(0, 1, 2)):
    rows = []
    for shift in np.round(np.arange(0, 1.01, 0.1), 1):
        for seed in range(seeds):
            G, truth = planted_partition_fuzzy_graph(1000, 20, 8.0, 0.5, 0.5, 0.15, float(shift), seed=seed)
            alphas = level_grid(G)
            ms = max((multislice_leiden(G, alphas, seed=s) for s in lseeds), key=lambda r: r.Qbar)
            cuts = {}
            for q in CUTS:
                Ga = G.alpha_cut(float(np.quantile(G.mu, q)))
                b = max((fuzzy_leiden(Ga, 1.0, seed=s, n_iterations=0) for s in lseeds), key=lambda r: r.Q_F)
                cuts[q] = b.membership
            rec = {"shift": float(shift), "seed": seed, "cuts": {}}
            drops = {}
            for name, phi in PHI.items():
                Gp = FuzzyGraph(G.n, G.src, G.dst, np.clip(phi(G.mu), 1e-9, 1.0))
                f = max((fuzzy_leiden(Gp, 1.0, seed=s, n_iterations=0) for s in lseeds), key=lambda r: r.Q_F)
                drops[name] = {q: f.Q_F - fuzzy_modularity(Gp, m, 1.0) for q, m in cuts.items()}
            for q, m in cuts.items():
                rec["cuts"][str(q)] = {"shortfall": 1 - ari(truth, m), "index": ms.Qbar - qbar(G, m, alphas),
                                       **{f"drop_{k}": v[q] for k, v in drops.items()}}
            rows.append(rec)
            print(shift, seed, {q: round(v["index"], 3) for q, v in rec["cuts"].items()}, flush=True)
    res = {"rows": rows, "summary": {}}
    cells = [c for r in rows for c in r["cuts"].values()]
    y = np.array([c["shortfall"] for c in cells])
    for k in ["index"] + [f"drop_{p}" for p in PHI]:
        x = np.array([c[k] for c in cells])
        conc = []
        for r in rows:
            v = list(r["cuts"].values()); s = n = 0
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    dl = v[i]["shortfall"] - v[j]["shortfall"]
                    if abs(dl) < 1e-9:
                        continue
                    n += 1; d = v[i][k] - v[j][k]
                    s += 1 if dl * d > 0 else (0.5 if d == 0 else 0)
            if n:
                conc.append(s / n)
        res["summary"][k] = {"rho_shortfall": float(spearmanr(x, y).statistic), "concordance": float(np.mean(conc)),
                             "rho_vs_identity_drop": float(spearmanr(x, [c["drop_identity"] for c in cells]).statistic)}
    print(json.dumps(res["summary"], indent=1))
    json.dump(res, open("results/rescaling_validity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
