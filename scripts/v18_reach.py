#!/usr/bin/env python3
"""Release 5.3.0. Reachability and ranking parts of the path-level Spearman correlation on the
Junyi transition graph (Section 8.1, Supplementary Table S9f).

Every annotated pair that no path reaches scores 0, so rho over the fixed 1,954 pairs mixes two
things: whether a pair is reached at all, and how the reached pairs are ordered. For each pruning
threshold tau_E (graphs data/graphs/junyi_transition_tau*) and each score this script reports
  coverage       share of the 1,954 pairs reached (score > 0);
  rho_all        Spearman rho over all pairs (unreached pairs tied at 0);
  rho_reach_ind  Spearman rho of the 0/1 reach indicator with the crowd score;
  auc_reach      probability that a reached pair has a higher crowd score than an unreached pair
                 (Mann-Whitney AUC of the crowd score for the reach indicator; ties count 1/2);
  rho_reached    Spearman rho over the pairs that the path scores reach (the reach set is the same
                 for the hop distance and every t-norm best-path score at one threshold);
  hop ties       share of reached pairs at hop distance 1 and 2, and the number of distinct hop
                 values among the reached pairs (a large tie block limits what the hop distance
                 can order).
Two-way cluster bootstrap over exercises (500 resamples) for rho_reached of the product and the
hop distance and for their difference at each threshold, and for the change of the coverage,
of the AUC and of rho_reached between thresholds 0.2 and 0.05.
Pairs reached at both thresholds 0.2 and 0.05 (the 837 pairs reached at 0.2) are scored at both
thresholds (same pairs, different graphs), and the pairs first reached at 0.05 are scored on their own.
The pairs that the Lukasiewicz best-path score reaches (Supplementary Table S9d: 522 pairs at 0.05)
are a subset dominated by adjacent exercises; the hop distance is nearly constant there.
Writes results/v18/reach.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph  # noqa: E402
from data.loaders import load_junyi_expert  # noqa: E402
from composition_robustness import mats  # noqa: E402

OUT = "results/v18/reach.json"
TAGS = ["0", "0.05", "0.1", "0.2", "0.3"]
OPS = ["direct_edge", "hop_baseline", "min", "product", "lukasiewicz", "yager_5"]


def load_tau(tag):
    e = pd.read_csv(f"data/graphs/junyi_transition_tau{tag}_edges.csv"); nd = pd.read_csv(f"data/graphs/junyi_transition_tau{tag}_nodes.csv")
    return FuzzyGraph(len(nd), e.src.to_numpy(), e.dst.to_numpy(), e.mu.to_numpy()), nd.label.astype(str).tolist()


def auc(y, r):
    """Mann-Whitney AUC of y for the binary label r (ties 1/2); weights by repetition."""
    r = r.astype(bool)
    if r.all() or (~r).all():
        return float("nan")
    rk = rankdata(y)
    n1 = r.sum(); n0 = (~r).sum()
    return float((rk[r].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rho(x, y):
    return float(spearmanr(x, y).statistic)


def main(B=500):
    os.makedirs("results/v18", exist_ok=True)
    rng = np.random.default_rng(5300)
    _, ann = load_junyi_expert()
    A = ann.Exercise_A.astype(str).to_numpy(); Bn = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    exs = np.unique(np.r_[A, Bn]); pos = {x: i for i, x in enumerate(exs)}
    pa = np.array([pos[x] for x in A]); pb = np.array([pos[x] for x in Bn])
    W = []
    for _ in range(B):
        m = rng.multinomial(exs.size, np.full(exs.size, 1 / exs.size)); W.append(m[pa] * m[pb])
    res = {"n_pairs": int(len(y)), "thresholds": {}}
    keep = {}
    for t in TAGS:
        G, lab = load_tau(t); idx = {l: i for i, l in enumerate(lab)}
        ok = np.array([a in idx and b in idx for a, b in zip(A, Bn)])
        ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in Bn])
        M, _ = mats(G)
        X = {k: np.where(ok, M[k][ib, ia], 0.0) for k in OPS}
        reach = X["hop_baseline"] > 0
        assert all(np.array_equal(reach, X[k] > 0) for k in ("min", "product")), "reach sets differ"
        d = np.where(reach, 1.0 / np.maximum(X["hop_baseline"], 1e-12) - 1.0, np.inf)
        r = {"n": int(G.n), "m": int(G.m), "coverage": float(reach.mean()), "n_reached": int(reach.sum()),
             "auc_reach": auc(y, reach), "rho_reach_ind": rho(reach.astype(float), y),
             "hop_dist_share": {str(k): float(np.mean(np.round(d[reach]) == k)) for k in (1, 2, 3)},
             "hop_distinct_values": int(np.unique(np.round(d[reach])).size),
             "direct_edge_coverage": float((X["direct_edge"] > 0).mean()), "scores": {}}
        for k in OPS:
            r["scores"][k] = {"rho_all": rho(X[k], y), "rho_reached": rho(X[k][reach], y[reach])}
        # cluster bootstrap on the reached pairs
        def wr(x, w, msk):
            k = (w > 0) & msk
            return spearmanr(np.repeat(x[k], w[k]), np.repeat(y[k], w[k])).statistic
        bp = np.array([wr(X["product"], w, reach) for w in W]); bh = np.array([wr(X["hop_baseline"], w, reach) for w in W])
        ba = np.array([auc(np.repeat(y, w), np.repeat(reach, w)) for w in W])
        bc = np.array([np.repeat(reach, w).mean() for w in W])
        ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        r["ci"] = {"product_reached": ci(bp), "hop_reached": ci(bh), "product_minus_hop_reached": ci(bp - bh),
                   "auc_reach": ci(ba), "coverage": ci(bc)}
        r["product_minus_hop_reached"] = r["scores"]["product"]["rho_reached"] - r["scores"]["hop_baseline"]["rho_reached"]
        keep[t] = (bp, bh, ba, bc, X, reach)
        luk = X["lukasiewicz"] > 0
        r["lukasiewicz_reached"] = {"n": int(luk.sum()), "hop_dist1_share": float(np.mean(np.round(d[luk]) == 1)),
                                    "rho": {k: rho(X[k][luk], y[luk]) for k in OPS}}
        res["thresholds"][t] = r
        print(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("scores", "ci")},
              {k: round(v["rho_reached"], 3) for k, v in r["scores"].items()}, flush=True)
    a, b = keep["0.05"], keep["0.2"]
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    T = res["thresholds"]
    res["change_0.2_to_0.05"] = {
        "coverage": [T["0.05"]["coverage"] - T["0.2"]["coverage"], *ci(a[3] - b[3])],
        "auc_reach": [T["0.05"]["auc_reach"] - T["0.2"]["auc_reach"], *ci(a[2] - b[2])],
        "hop_rho_reached": [T["0.05"]["scores"]["hop_baseline"]["rho_reached"] - T["0.2"]["scores"]["hop_baseline"]["rho_reached"], *ci(a[1] - b[1])],
        "product_rho_reached": [T["0.05"]["scores"]["product"]["rho_reached"] - T["0.2"]["scores"]["product"]["rho_reached"], *ci(a[0] - b[0])]}
    X5, r5 = a[4], a[5]; X2, r2 = b[4], b[5]; new = r5 & ~r2
    def bs(f):
        v = np.array([f(w) for w in W]); return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    def wr2(x, w, msk):
        k = (w > 0) & msk
        return spearmanr(np.repeat(x[k], w[k]), np.repeat(y[k], w[k])).statistic
    same = {}
    for nm, X in (("tau0.2", X2), ("tau0.05", X5)):
        for k in ("hop_baseline", "product"):
            same[f"{k}|{nm}"] = rho(X[k][r2], y[r2])
    same["hop_0.05_minus_0.2"] = [same["hop_baseline|tau0.05"] - same["hop_baseline|tau0.2"], *bs(lambda w: wr2(X5["hop_baseline"], w, r2) - wr2(X2["hop_baseline"], w, r2))]
    same["product_minus_hop_at_0.05"] = [same["product|tau0.05"] - same["hop_baseline|tau0.05"], *bs(lambda w: wr2(X5["product"], w, r2) - wr2(X5["hop_baseline"], w, r2))]
    res["pairs_reached_at_0.2"] = {"n": int(r2.sum()), **same}
    res["pairs_first_reached_at_0.05"] = {"n": int(new.sum()), "hop": rho(X5["hop_baseline"][new], y[new]), "product": rho(X5["product"][new], y[new]),
                                          "hop_ci": bs(lambda w: wr2(X5["hop_baseline"], w, new)),
                                          "mean_crowd": float(y[new].mean()), "mean_crowd_reached_0.2": float(y[r2].mean()),
                                          "mean_crowd_unreached_0.05": float(y[~r5].mean())}
    json.dump(res, open(OUT, "w"), indent=1)
    print(res["change_0.2_to_0.05"]); print(res["pairs_reached_at_0.2"]); print(res["pairs_first_reached_at_0.05"])


if __name__ == "__main__":
    main()
