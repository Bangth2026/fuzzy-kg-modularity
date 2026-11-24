#!/usr/bin/env python3
"""Release 5.4.0. Checks of the fourth revision on the Junyi transition graph (Section 8.1,
Supplementary Tables S4g, S9c, S9f).

1. tiers      A three-level reach score at tau_E = 0.05: 2 for pairs reached at the default 0.2,
              1 for pairs first reached at 0.05, 0 for pairs unreached at 0.05; Spearman rho on the
              1,954 pairs, against the hop score at 0.05 (0.567) and at 0.2 (0.425). Also the
              lexicographic score (tier, then hop score at 0.05 inside the tier), the mean hop
              distance at 0.05 of the old and of the new pairs, and a decomposition of the hop
              correlation at 0.05 into the part explained by the tier and the part inside tiers.
              Two-way cluster bootstrap over exercises (500 resamples, seed 5400).
2. crossfit   (a) the scheme described in Table S9c, recomputed: exercises split at random into
              two halves, each annotated pair assigned to the half of its candidate (prerequisite)
              exercise, crisp threshold chosen on one half among 0.05, 0.1, 0.2, 0.3 and evaluated
              on the other next to the product on the unpruned graph; 200 splits, both directions
              (400 folds), seed 5401. (b) the halvings of Table S4g (seed 5302): crisp threshold
              chosen on one half among the five thresholds and compared on the other with the
              default 0.2 (gain of choosing the threshold, held out).
Writes results/v19/misc.json.
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


def main(B=500):
    os.makedirs("results/v19", exist_ok=True)
    _, ann = load_junyi_expert()
    A = ann.Exercise_A.astype(str).to_numpy(); Bn = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    X = {}
    for t in TAGS:
        G, lab = load_tau(t); idx = {l: i for i, l in enumerate(lab)}
        ok = np.array([a in idx and b in idx for a, b in zip(A, Bn)])
        ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in Bn])
        M, _ = mats(G)
        X[("hop", t)] = np.where(ok, M["hop_baseline"][ib, ia], 0.0); X[("product", t)] = np.where(ok, M["product"][ib, ia], 0.0)
    rho = lambda x, m=None: float(spearmanr(x if m is None else x[m], y if m is None else y[m]).statistic)
    h5, h2 = X[("hop", "0.05")], X[("hop", "0.2")]
    r5, r2 = h5 > 0, h2 > 0
    tier = np.where(r2, 2.0, np.where(r5, 1.0, 0.0))
    lex = tier * 10 + h5  # hop score is in (0, 1]
    d5 = np.where(r5, 1.0 / np.maximum(h5, 1e-12) - 1.0, np.nan)
    new = r5 & ~r2
    # inside-tier part: Spearman of hop score after removing the tier means of the ranks
    from scipy.stats import rankdata
    rh, ry = rankdata(h5), rankdata(y)
    def within(rx, rt):
        out = rx.copy()
        for c in np.unique(tier):
            k = tier == c; out[k] = rx[k] - rx[k].mean()
        return out
    rw = within(rh, tier); yw = within(ry, tier)
    res = {"tiers": {"n_old": int(r2.sum()), "n_new": int(new.sum()), "n_unreached": int((~r5).sum()),
                     "rho_tier": rho(tier), "rho_lex_tier_then_hop": rho(lex), "rho_hop_0.05": rho(h5), "rho_hop_0.2": rho(h2),
                     "rho_reach_0.2": rho(r2.astype(float)), "rho_reach_0.05": rho(r5.astype(float)),
                     "mean_hop_old": float(np.nanmean(d5[r2])), "mean_hop_new": float(np.nanmean(d5[new])),
                     "hop_dist_old": {str(k): float(np.mean(np.round(d5[r2]) == k)) for k in (1, 2, 3, 4)},
                     "hop_dist_new": {str(k): float(np.mean(np.round(d5[new]) == k)) for k in (1, 2, 3, 4)},
                     "rho_hop_within_tiers": float(np.corrcoef(rw, yw)[0, 1]),
                     "mean_crowd": {"old": float(y[r2].mean()), "new": float(y[new].mean()), "unreached": float(y[~r5].mean())}}}
    rng = np.random.default_rng(5400)
    exs = np.unique(np.r_[A, Bn]); pos = {e: i for i, e in enumerate(exs)}
    pa = np.array([pos[a] for a in A]); pb = np.array([pos[b] for b in Bn])
    bt, bl, bh = [], [], []
    for _ in range(B):
        m = rng.multinomial(exs.size, np.full(exs.size, 1 / exs.size)); w = m[pa] * m[pb]; k = w > 0
        yy = np.repeat(y[k], w[k])
        bt.append(spearmanr(np.repeat(tier[k], w[k]), yy).statistic); bl.append(spearmanr(np.repeat(lex[k], w[k]), yy).statistic)
        bh.append(spearmanr(np.repeat(h5[k], w[k]), yy).statistic)
    bt, bl, bh = map(np.array, (bt, bl, bh)); ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res["tiers"]["ci"] = {"tier": ci(bt), "lex": ci(bl), "hop_0.05": ci(bh), "hop_minus_tier": ci(bh - bt), "lex_minus_hop": ci(bl - bh)}
    print(json.dumps(res["tiers"], indent=1), flush=True)
    # (a) S9c scheme
    rng = np.random.default_rng(5401); CAND = ["0.05", "0.1", "0.2", "0.3"]
    diffs, held, choice = [], [], []
    for _ in range(200):
        h = rng.random(exs.size) < 0.5
        side = h[pa]  # half of the candidate (prerequisite) exercise: Exercise_A
        for mA, mB in ((side, ~side), (~side, side)):
            t = max(CAND, key=lambda t: rho(X[("hop", t)], mA)); choice.append(t)
            hb = rho(X[("hop", t)], mB); pb_ = rho(X[("product", "0")], mB)
            held.append((hb, pb_)); diffs.append(pb_ - hb)
    held = np.array(held); diffs = np.array(diffs)
    res["s9c_scheme"] = {"choices": dict(Counter(choice)), "hop_heldout_mean": float(held[:, 0].mean()), "product_heldout_mean": float(held[:, 1].mean()),
                         "diff_mean": float(diffs.mean()), "diff_p2.5": float(np.percentile(diffs, 2.5)), "diff_p97.5": float(np.percentile(diffs, 97.5)),
                         "share_positive_splits": float(np.mean(diffs.reshape(200, 2).mean(1) > 0)), "share_positive_folds": float(np.mean(diffs > 0))}
    print(res["s9c_scheme"], flush=True)
    # (b) threshold choice vs default, S4g halvings
    rng = np.random.default_rng(5302); g = []; ch = []
    for _ in range(200):
        h = rng.random(exs.size) < 0.5
        mA = h[pa] & h[pb]; mB = ~h[pa] & ~h[pb]
        t = max(TAGS, key=lambda t: rho(X[("hop", t)], mA)); ch.append(t)
        g.append(rho(X[("hop", t)], mB) - rho(X[("hop", "0.2")], mB))
    g = np.array(g)
    res["threshold_vs_default"] = {"mean": float(g.mean()), "p2.5": float(np.percentile(g, 2.5)), "p97.5": float(np.percentile(g, 97.5)),
                                   "share_positive": float(np.mean(g > 0)), "choices": dict(Counter(ch))}
    print(res["threshold_vs_default"], flush=True)
    json.dump(res, open("results/v19/misc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
