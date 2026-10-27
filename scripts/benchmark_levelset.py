#!/usr/bin/env python3
"""Level-set multislice modularity against Fuzzy Leiden and the single-cut
shortcut (Section 8.2, Tables 8 and S1).

Stage A (``--stage shift``): directed planted partitions, n = 1000, k = 20,
out-degree 8, p_in = 0.5, memberships Beta(0.5, 0.15) with the community
shift swept from 0 (no information in the memberships) to 1 in steps of 0.1,
ten graph seeds.  For every graph: Fuzzy Leiden at gamma = 1, the level-set
multislice optimiser (eight quantile levels), and crisp Leiden at gamma = 1 on
one level set (median quantile, upper quartile, absolute 0.6; the practitioner
protocol).  Reported per method: ARI against the planted partition (mean and
95% bootstrap CI over seeds), and for each single cut the cost index
cost(alpha_0) = Qbar(Pi*) - Qbar(Pi_alpha_0) and the persistence reading.

Stage B (``--stage lfr``): the LFR-style directed benchmark of
``fuzzykg.synthetic.lfr_style_fuzzy_graph`` with n in {500, 2000, 10000},
mixing in {0.1, ..., 0.6}, shift in {0, 0.5}, ten seeds at n <= 2000 and five at
n = 10000; the same methods with two optimiser seeds and no persistence tracking.

Usage: python scripts/benchmark_levelset.py --stage shift|lfr [--seeds 10] [--out results/levelset_<stage>.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden, fuzzy_modularity  # noqa: E402
from fuzzykg.multislice import multislice_leiden, level_grid, qbar, persistence, persistence_cost  # noqa: E402
from fuzzykg.resolution import adaptive_resolution  # noqa: E402
from fuzzykg.synthetic import planted_partition_fuzzy_graph, lfr_style_fuzzy_graph  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402


def single_cut(G: FuzzyGraph, alpha: float, seeds):
    """Practitioner protocol: crisp Leiden at gamma = 1 on E_alpha, best level-set modularity."""
    try:
        Ga = G.alpha_cut(alpha)
    except ValueError:
        return None
    best = max((fuzzy_leiden(Ga, 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda r: r.Q_F)
    return best.membership


def evaluate(G: FuzzyGraph, truth, seeds, with_alg1=False, with_persistence=True):
    alphas = level_grid(G)
    rec = {"alphas": alphas.tolist(), "mean_out_degree": G.m / G.n}
    t0 = time.perf_counter()
    f = max((fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda r: r.Q_F)
    rec["fuzzy"] = {"ari": ari(truth, f.membership), "k": f.n_communities, "Q_F": f.Q_F,
                    "Qbar": qbar(G, f.membership, alphas), "time": time.perf_counter() - t0}
    t0 = time.perf_counter()
    ms = max((multislice_leiden(G, alphas, seed=s) for s in seeds), key=lambda r: r.Qbar)
    rec["multislice"] = {"ari": ari(truth, ms.membership), "k": ms.n_communities, "Qbar": ms.Qbar,
                         "Q_F": ms.Q_F, "profile": ms.profile.tolist(), "time": time.perf_counter() - t0}
    me = max((multislice_leiden(G, alphas, seed=s, weighting="equal") for s in seeds), key=lambda r: r.Qbar)
    rec["multislice_equal"] = {"ari": ari(truth, me.membership), "k": me.n_communities, "Qbar": me.Qbar,
                               "Q_F": me.Q_F}
    tracks = persistence(G, alphas, seed=seeds[0]) if with_persistence else None
    cuts = {"q0.50": float(np.quantile(G.mu, 0.5)), "q0.75": float(np.quantile(G.mu, 0.75)), "a0.6": 0.6,
            "a0.4": 0.4}
    rec["cuts"] = {}
    for name, a in cuts.items():
        memb = single_cut(G, a, seeds)
        if memb is None:
            rec["cuts"][name] = None
            continue
        retained = float(G.mu[G.mu >= a].sum() / G.W)
        rec["cuts"][name] = {
            "alpha": a, "ari": ari(truth, memb), "k": int(np.unique(memb).size), "retained_mass": retained,
            "mean_out_degree_cut": float((G.mu >= a).sum() / G.n),
            "cost_index": ms.Qbar - qbar(G, memb, alphas),
            "cost_index_equal": me.Qbar - qbar(G, memb, alphas, weighting="equal"),
            "cost_fuzzy": f.Q_F - fuzzy_modularity(G, memb, 1.0),
            "ari_vs_multislice": ari(ms.membership, memb),
            "persistence_cost": persistence_cost(tracks, a, G.W, 0.05) if tracks is not None else None,
        }
    if with_alg1:
        t0 = time.perf_counter()
        r = adaptive_resolution(G, seeds=seeds)
        a1 = max((fuzzy_leiden(G, r.gamma1, seed=s, n_iterations=0) for s in seeds), key=lambda r_: r_.Q_F)
        rec["algorithm1"] = {"ari": ari(truth, a1.membership), "k": a1.n_communities, "gamma1": r.gamma1,
                             "doublings": r.doublings, "time": time.perf_counter() - t0}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["shift", "lfr"], required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--leiden-seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shifts", type=float, nargs="*", default=None, help="subset of shifts for the shift stage")
    args = ap.parse_args()
    out = args.out or f"results/levelset_{args.stage}.json"
    lseeds = list(range(args.leiden_seeds))
    rows = []
    if args.stage == "shift":
        for shift in (np.round(np.arange(0, 1.01, 0.1), 1) if args.shifts is None else args.shifts):
            for seed in range(args.seeds):
                G, truth = planted_partition_fuzzy_graph(1000, 20, 8.0, 0.5, 0.5, 0.15, float(shift), seed=seed)
                rec = evaluate(G, truth, lseeds, with_alg1=True)
                rec.update({"shift": float(shift), "seed": seed})
                rows.append(rec)
                print(json.dumps({k: rec[k] for k in ("shift", "seed")}),
                      f"fuzzy {rec['fuzzy']['ari']:.3f} multislice {rec['multislice']['ari']:.3f} "
                      f"q0.5 {rec['cuts']['q0.50']['ari']:.3f} a0.6 {(rec['cuts']['a0.6'] or {}).get('ari', float('nan')):.3f}",
                      flush=True)
                json.dump(rows, open(out, "w"), indent=1)
    else:
        for n in (500, 2000, 10000):
            for mixing in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                for shift in (0.0, 0.5):
                    for seed in range(args.seeds if n <= 2000 else max(5, args.seeds // 2)):
                        G, truth = lfr_style_fuzzy_graph(n, mixing, mu_mean=0.5, mu_sd=0.15, shift=shift, seed=seed)
                        rec = evaluate(G, truth, lseeds[:2], with_alg1=False, with_persistence=False)
                        rec.update({"n": n, "mixing": mixing, "shift": shift, "seed": seed,
                                    "k_true": int(np.unique(truth).size)})
                        rows.append(rec)
                        print(n, mixing, shift, seed, f"fuzzy {rec['fuzzy']['ari']:.3f} multislice {rec['multislice']['ari']:.3f} "
                              f"q0.5 {rec['cuts']['q0.50']['ari']:.3f}", flush=True)
                        json.dump(rows, open(out, "w"), indent=1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
