#!/usr/bin/env python3
"""Version 4.4.0. Seed dispersion of the Q_F drop on the real graphs (Algorithm 2, step 3).

Algorithm 2 flags a cut whose audit exceeds the no-cut value by more than twice the larger seed
SD.  The rule was read from the cost index (Table 4); here the same quantities are computed for
the Q_F drop itself: for the support partition (no cut) and the three quantile cuts of every
real graph, the Q_F drop Q_F(Pi_F) - Q_F(Pi_cut) is computed for ten Leiden seeds of the cut
(Pi_F fixed at the best of five seeds, as in scripts/real_qf_drop.py), and a cut is flagged
when its mean drop exceeds the no-cut mean by more than twice the larger of the two SDs.
Writes results/v7/qf_drop_seed_sd.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import fuzzy_leiden, fuzzy_modularity  # noqa: E402
from levelset_real import load, NAMES  # noqa: E402

out = {}
for name in NAMES:
    G = load(name)
    f = max((fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in range(5)), key=lambda r: r.Q_F)
    row = {}
    for c, a in (("none", float(G.mu.min())), ("q0.25", float(np.quantile(G.mu, 0.25))),
                 ("q0.50", float(np.quantile(G.mu, 0.5))), ("q0.75", float(np.quantile(G.mu, 0.75)))):
        H = G.alpha_cut(a)
        d = [f.Q_F - fuzzy_modularity(G, fuzzy_leiden(H, 1.0, seed=s, n_iterations=0).membership, 1.0) for s in range(10)]
        row[c] = {"mean": float(np.mean(d)), "sd": float(np.std(d, ddof=1)), "per_seed": [float(x) for x in d]}
    for c in ("q0.25", "q0.50", "q0.75"):
        row[c]["flag"] = bool(row[c]["mean"] - row["none"]["mean"] > 2 * max(row[c]["sd"], row["none"]["sd"]))
    out[name] = row
    print(name, {c: (round(v["mean"], 4), round(v["sd"], 4), v.get("flag")) for c, v in row.items()}, flush=True)
os.makedirs("results/v7", exist_ok=True)
json.dump(out, open("results/v7/qf_drop_seed_sd.json", "w"), indent=1)
