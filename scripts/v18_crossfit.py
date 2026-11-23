#!/usr/bin/env python3
"""Release 5.3.0. Cross-fitted pruning thresholds on the Junyi transition graph with the scores of
Supplementary Tables S4b and S9c (composition_robustness.mats: hop score and product best-path
score), so that point values and held-out values come from one computation (Section 8.1,
Supplementary Table S4g).

200 random halvings of the 369 annotated exercises (seed 5302); a pair belongs to a half when both
of its exercises do. (a) both thresholds chosen on half A (hop count among the five thresholds,
product among the five) and evaluated on half B; (b) only the crisp threshold chosen, the product
fixed on the unpruned graph. The intervals are the 2.5 and 97.5 percentiles over the halvings,
which overlap; they describe the spread of the held-out difference, not a confidence interval.
Writes results/v18/crossfit.json.
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from data.loaders import load_junyi_expert  # noqa: E402
from composition_robustness import mats  # noqa: E402
from v18_reach import load_tau, TAGS  # noqa: E402


def main():
    rng = np.random.default_rng(5302)
    _, ann = load_junyi_expert()
    A = ann.Exercise_A.astype(str).to_numpy(); B = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    X = {}
    for t in TAGS:
        G, lab = load_tau(t); idx = {l: i for i, l in enumerate(lab)}
        ok = np.array([a in idx and b in idx for a, b in zip(A, B)])
        ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in B])
        M, _ = mats(G)
        X[("hop", t)] = np.where(ok, M["hop_baseline"][ib, ia], 0.0); X[("product", t)] = np.where(ok, M["product"][ib, ia], 0.0)
    rho = lambda x, m: float(spearmanr(x[m], y[m]).statistic)
    allm = np.ones(len(y), bool)
    point = {f"{k}|{t}": rho(X[(k, t)], allm) for (k, t) in X}
    exs = np.unique(np.r_[A, B]); pos = {e: i for i, e in enumerate(exs)}
    pa = np.array([pos[a] for a in A]); pb = np.array([pos[b] for b in B])
    d2, d1, ch = [], [], []
    for _ in range(200):
        h = rng.random(exs.size) < 0.5
        mA = h[pa] & h[pb]; mB = ~h[pa] & ~h[pb]
        tc = max(TAGS, key=lambda t: rho(X[("hop", t)], mA)); tg = max(TAGS, key=lambda t: rho(X[("product", t)], mA))
        d2.append(rho(X[("product", tg)], mB) - rho(X[("hop", tc)], mB))
        d1.append(rho(X[("product", "0")], mB) - rho(X[("hop", tc)], mB)); ch.append((tc, tg))
    s = lambda d: [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(np.array(d) > 0))]
    res = {"point": point, "both_chosen": s(d2), "crisp_chosen_product_unpruned": s(d1),
           "choices": {f"{a}|{b}": c / 200 for (a, b), c in Counter(ch).items()}}
    os.makedirs("results/v18", exist_ok=True)
    json.dump(res, open("results/v18/crossfit.json", "w"), indent=1)
    print({k: round(v, 3) for k, v in point.items()}); print(res["both_chosen"], res["crisp_chosen_product_unpruned"], res["choices"])


if __name__ == "__main__":
    main()
