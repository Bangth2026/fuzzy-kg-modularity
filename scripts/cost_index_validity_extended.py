#!/usr/bin/env python3
"""Validity of the cost index, extended (Section 8.1, Table 2).

Adds to scripts/cost_index_validity.py:
* a loss that shares no partition with either objective: the recovery
  shortfall 1 - ARI(cut, truth), next to the loss relative to Leiden on Q_F;
* the cheap competitor that needs no multiplex optimiser: the drop in Q_F from
  the Leiden-on-Q_F partition to the cut partition, Q_F(Pi_fuzzy) - Q_F(Pi_cut);
* the index relative to the reference value, cost / Qbar(Pi*);
* within-graph agreement: for every benchmark graph (setting x seed) the three
  cuts are compared pairwise, and the share of cut pairs that a quantity orders as
  the loss does (concordance; ties count one half) is averaged over graphs;
* 95% bootstrap intervals over benchmark graphs for every pooled Spearman
  correlation and every concordance.

The rank-coded Q_F drop of scripts/rank_drop_audit.py (Lemma 2) is merged by graph
key when its results exist; graphs it does not cover are then left out.

Usage: python scripts/cost_index_validity_extended.py
"""
from __future__ import annotations
import json
import os
import numpy as np
from scipy.stats import spearmanr

CUTS = ("q0.50", "q0.75", "a0.6")


def _key(r):
    return tuple(r.get(k) for k in ("n", "mixing", "shift", "seed"))


def graphs(path, rank_path=None):
    rows = json.load(open(path)); out = []
    rank = {}
    if rank_path and os.path.exists(rank_path):
        rank = {_key(r): r for r in json.load(open(rank_path))}
    for r in rows:
        g = []
        rr = rank.get(_key(r))
        for c in CUTS:
            cc = r["cuts"].get(c)
            if cc is None:
                continue
            if rank and (rr is None or rr["cuts"].get(c) is None):
                continue
            g.append({"loss_fuzzy": r["fuzzy"]["ari"] - cc["ari"], "shortfall": 1.0 - cc["ari"],
                      "cost index": cc["cost_index"], "relative index": cc["cost_index"] / max(r["multislice"]["Qbar"], 1e-12),
                      "equal-weighted index": cc["cost_index_equal"], "Q_F drop": cc["cost_fuzzy"],
                      "retained mass": -cc["retained_mass"], "mean out-degree": -cc["mean_out_degree_cut"],
                      "rank-coded Q_F drop": rr["cuts"][c]["rank_drop"] if rr else float("nan")})
        if g:
            out.append(g)
    return out


FEATS = ["cost index", "relative index", "equal-weighted index", "Q_F drop", "rank-coded Q_F drop", "retained mass", "mean out-degree"]


def pooled(gs, feat, loss):
    x = np.array([c[feat] for g in gs for c in g]); y = np.array([c[loss] for g in gs for c in g])
    return float(spearmanr(x, y).statistic)


def concord(gs, feat, loss):
    vals = []
    for g in gs:
        s, n = 0.0, 0
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                dl = g[i][loss] - g[j][loss]; df = g[i][feat] - g[j][feat]
                if abs(dl) < 1e-9:
                    continue
                n += 1
                s += 1.0 if dl * df > 0 else (0.5 if abs(df) < 1e-12 else 0.0)
        if n:
            vals.append(s / n)
    return vals


def analyse(gs, rng, B=1000):
    out = {"n_graphs": len(gs), "n_cells": sum(len(g) for g in gs), "features": {}}
    idx = np.arange(len(gs))
    for f in FEATS:
        r = {}
        for loss in ("loss_fuzzy", "shortfall"):
            est = pooled(gs, f, loss)
            boots = [pooled([gs[i] for i in rng.choice(idx, idx.size)], f, loss) for _ in range(B)]
            cv = np.array(concord(gs, f, loss))
            cb = [np.mean(rng.choice(cv, cv.size)) for _ in range(B)]
            r[loss] = {"rho": est, "rho_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                       "concordance": float(cv.mean()), "concordance_ci": [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))],
                       "graphs_with_pairs": int(cv.size)}
        out["features"][f] = r
        print(f, {k: (round(v["rho"], 2), [round(x, 2) for x in v["rho_ci"]], round(v["concordance"], 2)) for k, v in r.items()}, flush=True)
    return out


def main():
    rng = np.random.default_rng(11)
    res = {"planted_sweep": analyse(graphs("results/levelset_shift.json", "results/rank_drop_shift.json"), rng),
           "lfr": analyse(graphs("results/levelset_lfr.json", "results/rank_drop_lfr.json"), rng)}
    json.dump(res, open("results/cost_index_validity_extended.json", "w"), indent=1)


if __name__ == "__main__":
    main()
