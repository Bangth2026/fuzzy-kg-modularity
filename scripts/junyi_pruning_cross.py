#!/usr/bin/env python3
"""Graded composition without pruning against hop count after pruning
(Section 8.4).

On the 1,954 annotated Junyi pairs (absent exercises score 0), compares on the
same two-way cluster-bootstrap resamples over exercises (500, seed 31 as in
scripts/junyi_pruning_fixed_pairs.py) the expert correlation of
* the product and Yager (p = 5) best-path scores on the transition graph without
  pruning (tau_E -> 0) and with tau_E = 0.05,
* the hop distance at every pruning threshold, including the best one.
Writes results/junyi_pruning_cross.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg.construction import ConstructionConfig, TransitionConfig, build_transition_graph, preprocess  # noqa: E402
from data.loaders import load_junyi, load_junyi_expert  # noqa: E402
from composition_robustness import mats  # noqa: E402
from junyi_pruning_fixed_pairs import TAUS, wstat  # noqa: E402


def main(B=500):
    rng = np.random.default_rng(31)
    ex, ann = load_junyi_expert()
    A = ann.Exercise_A.astype(str).to_numpy(); Bn = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    exs = np.unique(np.r_[A, Bn]); pos = {x: i for i, x in enumerate(exs)}
    pa = np.array([pos[x] for x in A]); pb = np.array([pos[x] for x in Bn])
    W = []
    for b in range(B):
        m = rng.multinomial(exs.size, np.full(exs.size, 1 / exs.size)); W.append(m[pa] * m[pb])
    d = preprocess(load_junyi(), ConstructionConfig())
    boots, est = {}, {}
    for tau in TAUS:
        key = "none" if tau < 1e-6 else f"{tau:g}"
        G, _ = build_transition_graph(d, TransitionConfig(tau_E=tau), preprocessed=True)
        idx = {str(l): i for i, l in enumerate(G.labels)}
        ok = np.array([a in idx and b in idx for a, b in zip(A, Bn)])
        ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in Bn])
        M, _ = mats(G)
        for k in ("hop_baseline", "product", "yager_5"):
            x = np.where(ok, M[k][ib, ia], 0.0)
            est[f"{k}@{key}"] = float(spearmanr(x, y).statistic)
            boots[f"{k}@{key}"] = np.array([wstat(x, y, w, spearmanr) for w in W])
        print(key, {k: round(v, 3) for k, v in est.items() if k.endswith("@" + key)}, flush=True)
    best_hop = max((k for k in est if k.startswith("hop_baseline@")), key=est.get)
    out = {"estimates": est, "best_hop": best_hop, "diffs": {}}
    for k in ("product@none", "yager_5@none", "product@0.05", "product@0.1"):
        for ref in (best_hop, "hop_baseline@0.2", "product@0.2"):
            v = boots[k] - boots[ref]
            out["diffs"][f"{k} - {ref}"] = {"diff": est[k] - est[ref], "ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
    print(json.dumps(out["diffs"], indent=1))
    json.dump(out, open("results/junyi_pruning_cross.json", "w"), indent=1)


if __name__ == "__main__":
    main()
