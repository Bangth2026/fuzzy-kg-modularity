#!/usr/bin/env python3
"""Paired bootstrap (over annotated pairs) of the Spearman correlation of each
t-norm minus that of the hop-distance score, all Junyi pairs, unlimited paths.
Writes results/hop_vs_tnorm.json."""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from composition_robustness import mats, COMP  # noqa
from prerequisite_criterion import load  # noqa
from data.loaders import load_junyi_expert  # noqa
rng = np.random.default_rng(5); out = {}
ex, ann = load_junyi_expert()
for g in ("junyi_transition", "junyi_transition_unsuggested", "junyi_copractice"):
    G, labels = load(g); idx = {l: i for i, l in enumerate(labels)}
    sel = ann[ann.Exercise_A.isin(idx) & ann.Exercise_B.isin(idx)]
    ia = sel.Exercise_A.map(idx).to_numpy(); ib = sel.Exercise_B.map(idx).to_numpy(); y = sel.Prerequisite_avg.to_numpy()
    M, _ = mats(G); X = {k: v[ib, ia] for k, v in M.items()}
    res = {}
    for k in COMP + ["direct_edge"]:
        d = []
        for _ in range(1000):
            j = rng.integers(0, len(y), len(y))
            d.append(spearmanr(X[k][j], y[j]).statistic - spearmanr(X["hop_baseline"][j], y[j]).statistic)
        res[k] = {"diff": float(spearmanr(X[k], y).statistic - spearmanr(X["hop_baseline"], y).statistic), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    out[g] = res
    print(g, {k: (round(v["diff"], 3), [round(c, 3) for c in v["ci"]]) for k, v in res.items()}, flush=True)
json.dump(out, open("results/hop_vs_tnorm.json", "w"), indent=1)
