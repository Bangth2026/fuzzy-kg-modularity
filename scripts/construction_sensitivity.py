#!/usr/bin/env python3
"""Sensitivity of the co-practice construction to its horizon h and minimum support
(Supplementary Table S6), and a bootstrap over learners of the memberships
(Section 8.1).

Part A: for h in {3, 5, 10, 20} and support in {5, 10, 20, 50}, on the 2009 and 2017
KC-level logs: density, mean and SD of mu_E, the optimised Q_F at gamma = 1, the
weights-only (global) excess with 50 draws, and the held-out MRR of the direct edge,
the bigram and the best composition operator (graph rebuilt on training learners).

Part B: 50 bootstrap resamples of the learners of each KC-level log at the default
construction; for every edge of the default graph the SD of its membership across
resamples (an edge absent from a resample counts as mu = 0), reported as the mean and
median over edges next to the SD of the memberships themselves.

Usage: python scripts/construction_sensitivity.py --part grid|bootstrap
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loaders import load_assistments  # noqa: E402
from fuzzykg.construction import ConstructionConfig, build_fuzzy_graph, preprocess  # noqa: E402
from fuzzykg.nulltest import best_qf, weights_only_test  # noqa: E402

LOGS = [("2009", "kc"), ("2017", "kc")]


def grid(out):
    res = json.load(open(out)) if os.path.exists(out) else {}
    for rel, lev in LOGS:
        df = load_assistments(rel, lev)
        for h in (3, 5, 10, 20):
            for sup in (5, 10, 20, 50):
                key = f"{rel}_{lev}_h{h}_s{sup}"
                if key in res:
                    continue
                cfg = ConstructionConfig(horizon=h, min_support=sup)
                G, _ = build_fuzzy_graph(df, cfg)
                obs = best_qf(G, 1.0)
                w = weights_only_test(G, "global", 50, 1.0, obs=obs.Q_F)
                hp = f"/tmp/heldout_{key}.json"
                subprocess.run([sys.executable, "scripts/heldout_transitions.py", "--release", rel, "--level", lev,
                                "--horizon", str(h), "--min-support", str(sup), "--no-grid", "--out", hp], check=True,
                               stdout=subprocess.DEVNULL)
                ho = json.load(open(hp))
                best = ho["best_operator"]
                res[key] = {"release": rel, "level": lev, "horizon": h, "min_support": sup, "n": G.n, "m": G.m,
                            "density": G.m / (G.n * (G.n - 1)), "reciprocity": G.reciprocity(), "mu_mean": float(G.mu.mean()), "mu_sd": float(G.mu.std()),
                            "Q_F": obs.Q_F, "k": obs.n_communities, "weights_dQ": w["delta_Q"], "weights_z": w["z"],
                            "mrr_direct": ho["baselines"]["direct_edge"]["mrr_learner_mean"], "mrr_bigram": ho["baselines"]["bigram"]["mrr_learner_mean"],
                            "mrr_best_op": ho["operators"][best]["mrr_learner_mean"], "best_op": best,
                            "mrr_maxmin": ho["operators"]["min"]["mrr_learner_mean"]}
                print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[key].items()}, flush=True)
                json.dump(res, open(out, "w"), indent=1)


def bootstrap(out, B=50):
    res = {}
    for rel, lev in LOGS + [("2015", "kc")]:
        cfg = ConstructionConfig()
        d = preprocess(load_assistments(rel, lev), cfg)
        G0, tab0 = build_fuzzy_graph(d, cfg, preprocessed=True)
        key0 = {(a, b): i for i, (a, b) in enumerate(zip(tab0.src.astype(str), tab0.dst.astype(str)))}
        vals = np.zeros((B, len(key0)))
        users = d.user_id.unique()
        rng = np.random.default_rng(0)
        for bidx in range(B):
            pick = rng.choice(users, size=users.size, replace=True)
            cnt = pd.Series(pick).value_counts()
            parts = []
            for rep in range(1, cnt.max() + 1):
                us = cnt.index[cnt >= rep]
                sub = d[d.user_id.isin(us)].copy()
                sub["user_id"] = sub["user_id"].astype(str) + f"#{rep}"
                parts.append(sub)
            db = pd.concat(parts).sort_values(["user_id", "order"], kind="stable")
            _, tab = build_fuzzy_graph(db, cfg, preprocessed=True)
            for a, b, m in zip(tab.src.astype(str), tab.dst.astype(str), tab.mu):
                j = key0.get((a, b))
                if j is not None:
                    vals[bidx, j] = m
        sd = vals.std(axis=0, ddof=1)
        res[f"{rel}_{lev}"] = {"edges": len(key0), "mu_sd_across_edges": float(tab0.mu.std()),
                               "bootstrap_sd_mean": float(sd.mean()), "bootstrap_sd_median": float(np.median(sd)),
                               "bootstrap_sd_q90": float(np.percentile(sd, 90)), "share_edges_dropped_some_resample": float((vals == 0).any(axis=0).mean())}
        print(rel, lev, res[f"{rel}_{lev}"], flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", choices=["grid", "bootstrap"], required=True)
    a = ap.parse_args()
    if a.part == "grid":
        grid("results/construction_sensitivity.json")
    else:
        bootstrap("results/membership_bootstrap.json")
