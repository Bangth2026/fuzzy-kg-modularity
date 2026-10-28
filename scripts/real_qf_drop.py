#!/usr/bin/env python3
"""Q_F drop of the real-graph cuts, next to the cost index (Section 8.3).
Q_F(Pi_F) - Q_F(Pi_cut), best of five seeds for both partitions, for the support
partition and the three quantile cuts. Writes results/real_qf_drop.json."""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import fuzzy_leiden, fuzzy_modularity  # noqa
from levelset_real import load, NAMES  # noqa
cd = json.load(open("results/cost_decomposition.json"))
out = {}
for name in NAMES:
    G = load(name); seeds = range(5)
    f = max((fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda r: r.Q_F)
    row = {}
    for c, a in (("none", float(G.mu.min())), ("q0.25", float(np.quantile(G.mu, 0.25))), ("q0.50", float(np.quantile(G.mu, 0.5))), ("q0.75", float(np.quantile(G.mu, 0.75)))):
        b = max((fuzzy_leiden(G.alpha_cut(a), 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda r: r.Q_F)
        row[c] = {"qf_drop": f.Q_F - fuzzy_modularity(G, b.membership, 1.0),
                  "index": cd[name]["nocut"]["index_best5"] if c == "none" else cd[name]["cuts"][c]["index_best5"]}
    out[name] = row
    print(name, {c: (round(v["qf_drop"], 4), round(v["index"], 4)) for c, v in row.items()}, flush=True)
x = [v["qf_drop"] for r in out.values() for v in r.values()]; y = [v["index"] for r in out.values() for v in r.values()]
conc = []
for r in out.values():
    v = list(r.values()); s = n = 0
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            n += 1; s += 1 if (v[i]["qf_drop"] - v[j]["qf_drop"]) * (v[i]["index"] - v[j]["index"]) > 0 else 0
    conc.append(s / n)
summ = {"rho_pooled": float(spearmanr(x, y).statistic), "within_graph_concordance": float(np.mean(conc))}
print(summ); out["_summary"] = summ
json.dump(out, open("results/real_qf_drop.json", "w"), indent=1)
