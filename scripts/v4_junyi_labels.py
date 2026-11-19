#!/usr/bin/env python3
"""Version 4.0.0. Quality of the Junyi crowd prerequisite labels and what it does to the
expert criterion of Section 8.4 (reply to the request for inter-rater agreement and noise).

From the per-rater scores (Prerequisite_raw, 1 to 9) of the 1,954 annotated pairs:
  * one-way random-effects ICC(1) for a single rater and ICC(1,k) for the mean of k raters
    (unequal k, the ANOVA estimator with the adjusted mean group size);
  * Krippendorff's alpha (interval metric);
  * split-half reliability: raters of each pair split at random into two halves (pairs with at
    least four raters), Spearman correlation of the half means, Spearman-Brown corrected, over 200
    splits;
  * the ceiling this sets on a Spearman correlation with the pair mean (square root of the
    reliability), and the correlation of one half with the other as an empirical ceiling;
  * the expert criterion of Table 5 (direct edge, hop distance, max-min, product on the
    transition graph pruned at 0.2 and on the co-practice graph, fixed 1,954 pairs, pairs outside
    the graph score 0) on all pairs, on the half of the pairs with the lower rater SD, and with
    the score of each pair normalised within rater is not possible (raters are not identified),
    so pair-level z-scores are not computed.
Writes results/v4/junyi_labels.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from data.loaders import load_junyi_expert  # noqa: E402
from fuzzykg import FuzzyGraph  # noqa: E402
from composition_robustness import mats  # noqa: E402


def icc1(groups):
    k = np.array([len(g) for g in groups]); N = k.sum(); a = len(groups)
    allv = np.concatenate(groups); gm = allv.mean()
    ssb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups); ssw = sum(((g - np.mean(g)) ** 2).sum() for g in groups)
    msb = ssb / (a - 1); msw = ssw / (N - a)
    k0 = (N - (k ** 2).sum() / N) / (a - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    icck = (msb - msw) / msb
    return float(icc), float(icck), float(k0)


def kripp_interval(groups):
    # pairable values only (units with >= 2 values)
    groups = [g for g in groups if len(g) >= 2]
    n = sum(len(g) for g in groups)
    Do = sum(((g[:, None] - g[None, :]) ** 2).sum() / (len(g) - 1) for g in groups) / n
    allv = np.concatenate(groups)
    De = ((allv[:, None] - allv[None, :]) ** 2).sum() / (n * (n - 1))
    return float(1 - Do / De)


def main():
    ex, ann = load_junyi_expert()
    raw = [np.array([float(x) for x in str(s).split("_") if x != ""]) for s in ann.Prerequisite_raw]
    k = np.array([len(r) for r in raw]); y = ann.Prerequisite_avg.to_numpy(); sd = np.array([r.std(ddof=1) if len(r) > 1 else np.nan for r in raw])
    res = {"n_pairs": int(len(raw)), "raters_per_pair": {"min": int(k.min()), "max": int(k.max()), "mean": float(k.mean())},
           "mean_check_max_abs_diff": float(np.max(np.abs(np.array([r.mean() for r in raw]) - y)))}
    i1, ik, k0 = icc1(raw); res["icc1_single"] = i1; res["icc1_mean_of_k"] = ik; res["k0"] = k0
    res["krippendorff_alpha_interval"] = kripp_interval(raw)
    rng = np.random.default_rng(0); sb = []; half = []
    sel = [i for i, r in enumerate(raw) if len(r) >= 4]
    for _ in range(200):
        a, b = [], []
        for i in sel:
            p = rng.permutation(raw[i]); h = len(p) // 2
            a.append(p[:h].mean()); b.append(p[h:].mean())
        r = spearmanr(a, b).statistic; half.append(r); sb.append(2 * r / (1 + r))
    res["split_half"] = {"pairs": len(sel), "rho_half_mean": float(np.mean(half)), "rho_half_ci": [float(np.percentile(half, 2.5)), float(np.percentile(half, 97.5))],
                         "spearman_brown": float(np.mean(sb))}
    res["ceiling_sqrt_reliability"] = float(np.sqrt(np.mean(sb)))
    low = sd <= np.nanmedian(sd)
    res["low_sd_pairs"] = int(low.sum()); res["median_rater_sd"] = float(np.nanmedian(sd))
    res["criterion"] = {}
    A = ann.Exercise_A.astype(str).to_numpy(); B = ann.Exercise_B.astype(str).to_numpy()
    for name in ("junyi_transition", "junyi_copractice"):
        e = pd.read_csv(f"data/graphs/{name}_edges.csv"); lab = pd.read_csv(f"data/graphs/{name}_nodes.csv").label.astype(str).tolist()
        G = FuzzyGraph.from_edges(zip(e.src, e.dst, e.mu), n=len(lab), labels=lab)
        idx = {l: i for i, l in enumerate(lab)}
        ok = np.array([a in idx and b in idx for a, b in zip(A, B)])
        ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in B])
        M, _ = mats(G)
        out = {}
        for key in ("direct_edge", "hop_baseline", "min", "product"):
            x = np.where(ok, M[key][ib, ia], 0.0)
            out[key] = {"rho_all": float(spearmanr(x, y).statistic), "rho_low_sd": float(spearmanr(x[low], y[low]).statistic),
                        "rho_high_sd": float(spearmanr(x[~low], y[~low]).statistic)}
        res["criterion"][name] = out
        print(name, {k_: {kk: round(v, 3) for kk, v in d.items()} for k_, d in out.items()})
    os.makedirs("results/v4", exist_ok=True)
    json.dump(res, open("results/v4/junyi_labels.json", "w"), indent=1)
    print({k_: v for k_, v in res.items() if k_ != "criterion"})


if __name__ == "__main__":
    main()
