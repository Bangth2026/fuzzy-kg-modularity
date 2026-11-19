#!/usr/bin/env python3
"""Version 4.0.0. Table 4 rows for the ten graphs of the added corpora: retained mass and mean
out-degree of the three quantile cuts, cost index over the no-cut index and over permuted
memberships (results/v4/cost_decomposition_new.json, ten seeds), graphs whose cut exceeds the
no-cut index by more than twice the larger seed SD, and the Q_F drop (best of five seeds, as in
scripts/real_qf_drop.py). Writes results/v4/table4_new.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import fuzzy_leiden, fuzzy_modularity  # noqa: E402
from levelset_real import load  # noqa: E402
cd = json.load(open("results/v4/cost_decomposition_new.json"))
out = {}
for name, r in cd.items():
    G = load(name); seeds = range(5)
    f = max((fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda x: x.Q_F)
    row = {"construction": "copractice" if name.endswith("copractice") else "transition"}
    for c, a in (("none", float(G.mu.min())), ("q0.25", None), ("q0.50", None), ("q0.75", None)):
        a = a if a is not None else float(np.quantile(G.mu, float(c[1:])))
        H = G.alpha_cut(a)
        b = max((fuzzy_leiden(H, 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda x: x.Q_F)
        v = r["nocut"] if c == "none" else r["cuts"][c]
        e = {"qf_drop": f.Q_F - fuzzy_modularity(G, b.membership, 1.0), "retained_mass": float(G.mu[G.mu >= a].sum() / G.W),
             "mean_out_degree": float(H.m / G.n), "index": v["index_best5"], "sd": v["sd"], "perm_mean": v["perm_mean"]}
        if c != "none":
            e["minus_nocut"] = v["index_best5"] - r["nocut"]["index_best5"]
            e["minus_perm"] = v["index_best5"] - v["perm_mean"]
            e["exceeds"] = bool(e["minus_nocut"] > 2 * max(v["sd"], r["nocut"]["sd"]))
        row[c] = e
    out[name] = row
    print(name, {c: (round(e["index"], 3), round(e["qf_drop"], 3), e.get("exceeds")) for c, e in row.items() if c != "construction"}, flush=True)
json.dump(out, open("results/v4/table4_new.json", "w"), indent=1)
