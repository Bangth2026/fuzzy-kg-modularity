#!/usr/bin/env python3
"""Follow-up analyses on Junyi (release 4.2.0).

(a) orientation   How are knowledge-map links oriented on the co-practice and the transition
                  graph, and by which signal? For every map link (parent -> child) with an edge in
                  either direction: share oriented correctly by mu, by precedence alone and by gain
                  alone (co-practice), by asym and by pref alone (transition); the distribution of
                  |mu(a,b) - mu(b,a)| over reciprocal pairs.
(b) bootstrap     Stability of that orientation under a bootstrap over learners (B draws; both
                  graphs rebuilt on every draw).
(c) crossfit      Hop distance with the pruning threshold chosen on one half of the exercises and
                  evaluated on the other (pairs assigned to the fold of their candidate
                  prerequisite B), against the product on the unpruned graph on the same pairs;
                  200 random splits.
(d) reached       Spearman rho on the pairs reached by every score, at each pruning threshold.
(e) exposure      Learners in the lowest quartile of the share of platform-suggested attempts
                  (and learners with no suggested attempt): transition graph rebuilt from them,
                  expert criterion recomputed.
Output: results/v5/junyi_followup.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from data.loaders import load_junyi_expert, load_junyi_with_suggested  # noqa: E402
from fuzzykg.construction import (ConstructionConfig, TransitionConfig, build_fuzzy_graph,  # noqa: E402
                                  build_transition_graph, preprocess)
from composition_robustness import mats  # noqa: E402

OPS = ["direct_edge", "hop_baseline", "min", "product", "lukasiewicz", "yager_5"]
TAUS = (0.05, 0.1, 0.2, 0.3)
OUT = os.path.join(ROOT, "results/v5/junyi_followup.json")
B_BOOT = int(os.environ.get("JF_BOOT", 20))


def map_links(ex):
    out = set()
    for n, p in zip(ex.name.astype(str), ex.prerequisites):
        if isinstance(p, str):
            for q in p.split(","):
                q = q.strip()
                if q and q != n:
                    out.add((q, n))
    return out


def orient_copractice(tab, links):
    t = tab.set_index(["src", "dst"])
    rows = []
    for a, b in links:
        f = (a, b) in t.index; r = (b, a) in t.index
        if not (f or r):
            continue
        mf = t.loc[(a, b)] if f else None; mr = t.loc[(b, a)] if r else None
        rows.append(dict(mu=(mf.mu if f else 0) - (mr.mu if r else 0),
                         prec=(mf.precedence if f else 0) - (mr.precedence if r else 0),
                         gain=(mf.gain if f else 0) - (mr.gain if r else 0), both=f and r))
    df = pd.DataFrame(rows)
    return df


def orient_transition(tab, links):
    t = tab.set_index(["src", "dst"])
    rows = []
    for a, b in links:
        f = (a, b) in t.index; r = (b, a) in t.index
        if not (f or r):
            continue
        rows.append(dict(mu=1 if f else -1))
    return pd.DataFrame(rows)


def summarise_orient(df, cols):
    out = {"links_with_edge": int(len(df))}
    for c in cols:
        out[f"correct_by_{c}"] = float((df[c] > 0).mean())
        out[f"ties_{c}"] = float((df[c] == 0).mean())
    return out


def build_both(d):
    G1, tab1 = build_fuzzy_graph(d, ConstructionConfig(), preprocessed=True)
    G2, tab2 = build_transition_graph(d, TransitionConfig(), preprocessed=True)
    return tab1, tab2


def criterion(G, A, Bn, y):
    idx = {str(l): i for i, l in enumerate(G.labels)}
    ok = np.array([a in idx and b in idx for a, b in zip(A, Bn)])
    ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in Bn])
    M, _ = mats(G)
    return {k: np.where(ok, M[k][ib, ia], 0.0) for k in OPS}


def main():
    t0 = time.time()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    ex, ann = load_junyi_expert()
    links = map_links(ex)
    raw = load_junyi_with_suggested()
    cfg = ConstructionConfig()
    d = preprocess(raw.drop(columns="suggested"), cfg)
    print("loaded", len(d), time.time() - t0, flush=True)

    # (a) orientation on the full log
    if "orientation" not in res:
        tab1, tab2 = build_both(d)
        o1 = orient_copractice(tab1, links); o2 = orient_transition(tab2, links)
        t = tab1.set_index(["src", "dst"]).mu
        rec = pd.DataFrame({"f": t}).join(t.rename_axis(["dst", "src"]).rename("r"), how="inner")
        gap = (rec.f - rec.r).abs().to_numpy()
        res["orientation"] = {
            "copractice": summarise_orient(o1, ["mu", "prec", "gain"]) | {
                "abs_gap_reciprocal_pairs_quantiles": {q: float(np.quantile(gap, q)) for q in (0.25, 0.5, 0.75, 0.9)},
                "abs_gap_map_links_quantiles": {q: float(np.quantile(o1.mu.abs(), q)) for q in (0.25, 0.5, 0.75)},
                "corr_mu_gap_with_prec_gap": float(np.corrcoef(o1.mu, o1.prec)[0, 1]),
                "corr_mu_gap_with_gain_gap": float(np.corrcoef(o1.mu, o1.gain)[0, 1]),
                "both_directions_share": float(o1.both.mean())},
            "transition": summarise_orient(o2, ["mu"])}
        json.dump(res, open(OUT, "w"), indent=1)
        print("orientation", res["orientation"], time.time() - t0, flush=True)

    # (c), (d) cross-fitted threshold and reached pairs
    A = ann.Exercise_A.astype(str).to_numpy(); Bn = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    if "crossfit" not in res:
        X = {}
        for tau in (1e-9,) + TAUS:
            G, _ = build_transition_graph(d, TransitionConfig(tau_E=tau), preprocessed=True)
            X[tau] = criterion(G, A, Bn, y)
        reached = {}
        for tau in TAUS:
            ra = np.all([X[tau][k] > 0 for k in OPS if k != "direct_edge"], axis=0)
            reached[str(tau)] = {"n_pairs": int(ra.sum()),
                                 **{k: float(spearmanr(X[tau][k][ra], y[ra]).statistic) for k in OPS}}
        exs = np.unique(np.r_[A, Bn]); rng = np.random.default_rng(42)
        diffs, tuned, prod, chosen = [], [], [], []
        for _ in range(200):
            fold = dict(zip(exs, rng.integers(0, 2, exs.size)))
            f = np.array([fold[b] for b in Bn])
            for k in (0, 1):
                tr, te = f == k, f != k
                best = max(TAUS, key=lambda t: spearmanr(X[t]["hop_baseline"][tr], y[tr]).statistic)
                h = spearmanr(X[best]["hop_baseline"][te], y[te]).statistic
                p = spearmanr(X[1e-9]["product"][te], y[te]).statistic
                tuned.append(h); prod.append(p); diffs.append(p - h); chosen.append(best)
        diffs = np.array(diffs)
        res["crossfit"] = {"splits": 200, "folds": 2,
                           "hop_tuned_heldout_rho_mean": float(np.mean(tuned)),
                           "product_unpruned_heldout_rho_mean": float(np.mean(prod)),
                           "diff_mean": float(diffs.mean()),
                           "diff_2.5_97.5": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                           "share_diff_positive": float((diffs > 0).mean()),
                           "chosen_tau_counts": {str(t): int(sum(c == t for c in chosen)) for t in TAUS}}
        res["reached_by_all"] = reached
        json.dump(res, open(OUT, "w"), indent=1)
        print("crossfit", res["crossfit"], reached, time.time() - t0, flush=True)

    # (e) learners least exposed to suggestions
    if "exposure" not in res:
        share = raw.groupby("user_id").suggested.mean()
        out = {}
        for label, users in (("lowest_quartile", share.index[share <= share.quantile(0.25)]),
                             ("no_suggested", share.index[share == 0])):
            dd = preprocess(raw[raw.user_id.isin(users)].drop(columns="suggested"), cfg)
            r = {"learners": int(dd.user_id.nunique()), "interactions": int(len(dd)),
                 "max_suggested_share": float(share.loc[users].max())}
            for tau in (0.2, 1e-9):
                G, _ = build_transition_graph(dd, TransitionConfig(tau_E=tau), preprocessed=True)
                X = criterion(G, A, Bn, y)
                r[f"tau_{tau:g}"] = {"n": int(G.n), "m": int(G.m),
                                     **{k: float(spearmanr(X[k], y).statistic) for k in OPS},
                                     "coverage_hop": float((X["hop_baseline"] > 0).mean())}
            out[label] = r
        res["exposure"] = out
        json.dump(res, open(OUT, "w"), indent=1)
        print("exposure", out, time.time() - t0, flush=True)

    # (b) learner bootstrap of the orientation
    if "orientation_bootstrap" not in res:
        users = d.user_id.unique(); rng = np.random.default_rng(7); vals = {"copractice": [], "transition": []}
        base1 = base2 = None
        for b in range(B_BOOT):
            pick = rng.choice(users, users.size, replace=True)
            cnt = pd.Series(pick).value_counts()
            parts = []
            for rep in range(1, int(cnt.max()) + 1):
                us = cnt.index[cnt >= rep]
                part = d[d.user_id.isin(us)].copy()
                part["user_id"] = part["user_id"].astype("int64") * 100 + rep
                parts.append(part)
            db = pd.concat(parts, ignore_index=True).sort_values(["user_id", "order"], kind="stable").reset_index(drop=True)
            tab1, tab2 = build_both(db)
            o1 = orient_copractice(tab1, links); o2 = orient_transition(tab2, links)
            vals["copractice"].append(float((o1.mu > 0).mean())); vals["transition"].append(float((o2.mu > 0).mean()))
            print("boot", b, vals["copractice"][-1], vals["transition"][-1], time.time() - t0, flush=True)
        res["orientation_bootstrap"] = {k: {"draws": len(v), "mean": float(np.mean(v)), "min": float(np.min(v)),
                                            "max": float(np.max(v)), "2.5_97.5": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
                                        for k, v in vals.items()}
        json.dump(res, open(OUT, "w"), indent=1)
        print("bootstrap", res["orientation_bootstrap"], flush=True)


if __name__ == "__main__":
    main()
