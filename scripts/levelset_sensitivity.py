#!/usr/bin/env python3
"""Sensitivity of the level-set objective and the cost index (Section 8.2,
Section 8.3), and the disparity-filter backbone as a comparison cut.

Part A (``--part shift``), on the planted sweep of Table 8 (eleven community
shifts, five graph seeds by default): for grids of L in {4, 8, 16} quantile levels the
level-set partition (mass weights, three optimiser seeds), the cost index of
the three single cuts of Table 8 on each grid, and the rank agreement of the
cost index across grids; the dispersion of Qbar(Pi*) over ten optimiser seeds
at L = 8; and the crisp partition of the disparity-filter backbone at
significance 0.05 with its recovery, retained edge share and cost index.

Part B (``--part real``), on the seven real graphs: the same grid comparison
and the disparity-filter backbone next to the cuts of Table 9.

Part C (``--part correlations``): Spearman correlations between the cost index
and the recovery loss on the planted sweep and on the LFR-style benchmark, all
cells and the cells with loss above 0.02, with the retained mass as the
free proxy, read from the recorded results.

Part D (``--part lfr_seeds``): optimiser-seed dispersion of Qbar and of the
recovery on the LFR-style benchmark (sizes 500 and 2000, mixing 0.3 and 0.6,
both shifts, three graph seeds, five optimiser seeds), so that the two-seed
protocol of Table S1 can be read against the seed noise.

Usage: python scripts/levelset_sensitivity.py --part shift|real|correlations|lfr_seeds
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden  # noqa: E402
from fuzzykg.backbone import disparity_filter  # noqa: E402
from fuzzykg.multislice import multislice_leiden, level_grid, qbar  # noqa: E402
from fuzzykg.synthetic import planted_partition_fuzzy_graph  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

GRIDS = (4, 8, 16)
CUTS = ("q0.50", "q0.75", "a0.6")


def crisp_best(Gc, seeds):
    return max((fuzzy_leiden(Gc, 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda r: r.Q_F).membership


def cut_graph(G, name):
    a = {"q0.50": float(np.quantile(G.mu, 0.5)), "q0.75": float(np.quantile(G.mu, 0.75)),
         "a0.2": 0.2, "a0.4": 0.4, "a0.6": 0.6, "a0.8": 0.8}[name]
    try:
        return G.alpha_cut(a), a
    except ValueError:
        return None, a


def analyse(G, seeds, truth=None, cuts=CUTS, n_disp_seeds=10):
    rec = {"grids": {}, "disparity": {}}
    memb_cuts = {}
    for c in cuts:
        Gc, a = cut_graph(G, c)
        memb_cuts[c] = None if Gc is None else crisp_best(Gc, seeds)
    for L in GRIDS:
        alphas = level_grid(G, n_levels=L)
        ms = max((multislice_leiden(G, alphas, seed=s) for s in seeds), key=lambda r: r.Qbar)
        g = {"L": int(len(alphas)), "Qbar": ms.Qbar, "k": ms.n_communities,
             "ari_truth": ari(truth, ms.membership) if truth is not None else None, "cost": {}}
        for c, memb in memb_cuts.items():
            g["cost"][c] = None if memb is None else ms.Qbar - qbar(G, memb, alphas)
        rec["grids"][str(L)] = g
    alphas = level_grid(G, n_levels=8)
    qs = [multislice_leiden(G, alphas, seed=s).Qbar for s in range(n_disp_seeds)]
    rec["Qbar_seed_sd"] = float(np.std(qs, ddof=1)); rec["Qbar_seed_range"] = float(max(qs) - min(qs))
    rec["Qbar_seed_best"] = float(max(qs))
    for sig in (0.05, 0.20):
        keep = disparity_filter(G, sig)
        d = {"alpha_sig": sig, "kept_edges": int(keep.sum()), "kept_share": float(keep.mean()),
             "retained_mass": float(G.mu[keep].sum() / G.W)}
        if keep.sum() > 0:
            Gb = FuzzyGraph(G.n, G.src[keep], G.dst[keep], np.ones(int(keep.sum())))
            memb = crisp_best(Gb, seeds)
            d.update({"k": int(np.unique(memb).size),
                      "ari_truth": ari(truth, memb) if truth is not None else None,
                      "cost_index": rec["grids"]["8"]["Qbar"] - qbar(G, memb, alphas),
                      "mean_out_degree": float(keep.sum() / G.n)})
        rec["disparity"][str(sig)] = d
    return rec


def part_shift(out, seeds=(0, 1, 2), graph_seeds=5):
    rows = json.load(open(out)) if os.path.exists(out) else []        # resume
    done = {(r["shift"], r["seed"]) for r in rows}
    for shift in np.round(np.arange(0, 1.01, 0.1), 1):
        for seed in range(graph_seeds):
            if (float(shift), seed) in done:
                continue
            G, truth = planted_partition_fuzzy_graph(1000, 20, 8.0, 0.5, 0.5, 0.15, float(shift), seed=seed)
            rec = analyse(G, list(seeds), truth)
            rec.update({"shift": float(shift), "seed": seed})
            rows.append(rec)
            print(shift, seed, {L: round(rec["grids"][str(L)]["ari_truth"], 3) for L in GRIDS},
                  "disp0.05", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec["disparity"]["0.05"].items()},
                  "sd", round(rec["Qbar_seed_sd"], 4), flush=True)
            json.dump(rows, open(out, "w"), indent=1)
    summarise_shift(rows)


def summarise_shift(rows):
    cost = {L: [] for L in GRIDS}; loss = []
    for r in rows:
        for c in CUTS:
            if any(r["grids"][str(L)]["cost"][c] is None for L in GRIDS):
                continue
            for L in GRIDS:
                cost[L].append(r["grids"][str(L)]["cost"][c])
    print("rank agreement of cost index across grids (Spearman):",
          {f"{a}-{b}": round(spearmanr(cost[a], cost[b]).statistic, 3) for a, b in ((4, 8), (8, 16), (4, 16))})
    print("Qbar seed sd: mean", np.mean([r["Qbar_seed_sd"] for r in rows]), "max", np.max([r["Qbar_seed_sd"] for r in rows]))


REAL13 = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2009_kc_transition", "assist2009_item", "assist2015_kc",
          "assist2015_kc_transition", "assist2017_kc", "assist2017_kc_transition", "assist2017_item", "assist2017_item_transition",
          "junyi_copractice", "junyi_transition"]


def part_real(out, seeds=(0, 1, 2, 3, 4), names=None):
    rows = json.load(open(out)) if os.path.exists(out) else []
    done = {r["graph"] for r in rows}
    for name in (names or REAL13):
        if name in done:
            continue
        df = pd.read_csv(f"data/graphs/{name}_edges.csv")
        G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
        rec = analyse(G, list(seeds), None, cuts=("a0.2", "a0.4", "a0.6", "q0.50", "q0.75"))
        f = max((fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda r: r.Q_F)
        for sig, d in rec["disparity"].items():
            if "k" in d:
                keep = disparity_filter(G, float(sig))
                Gb = FuzzyGraph(G.n, G.src[keep], G.dst[keep], np.ones(int(keep.sum())))
                memb = crisp_best(Gb, list(seeds))
                d["ari_vs_fuzzy"] = ari(f.membership, memb)
        rec["graph"] = name
        rows.append(rec)
        print(name, json.dumps({k: rec[k] for k in ("grids", "disparity", "Qbar_seed_sd")}, indent=None)[:1500], flush=True)
        json.dump(rows, open(out, "w"), indent=1)


def part_correlations(out):
    res = {}
    for label, path in (("planted_sweep", "results/levelset_shift.json"), ("lfr", "results/levelset_lfr.json")):
        rows = json.load(open(path))
        cost, loss, mass, shift = [], [], [], []
        for r in rows:
            for c in CUTS:
                cc = r["cuts"].get(c)
                if cc is None:
                    continue
                cost.append(cc["cost_index"]); loss.append(r["multislice"]["ari"] - cc["ari"])
                mass.append(cc["retained_mass"]); shift.append(r["shift"])
        cost, loss, mass, shift = map(np.asarray, (cost, loss, mass, shift))
        sel = loss > 0.02
        res[label] = {"n_cells": int(cost.size), "rho_cost_loss": float(spearmanr(cost, loss).statistic),
                      "rho_mass_loss": float(spearmanr(mass, loss).statistic),
                      "n_cells_loss_gt_0.02": int(sel.sum()),
                      "rho_cost_loss_gt_0.02": float(spearmanr(cost[sel], loss[sel]).statistic),
                      "rho_mass_loss_gt_0.02": float(spearmanr(mass[sel], loss[sel]).statistic),
                      "n_negative_cost": int((cost < -1e-9).sum()), "min_cost": float(cost.min())}
        if label == "planted_sweep":
            for lo, hi, key in ((0.0, 0.2, "shift_0_to_0.2"), (0.3, 1.0, "shift_0.3_to_1")):
                s = (shift >= lo - 1e-9) & (shift <= hi + 1e-9)
                res[label][key] = {"n": int(s.sum()), "rho_cost_loss": float(spearmanr(cost[s], loss[s]).statistic),
                                   "rho_mass_loss": float(spearmanr(mass[s], loss[s]).statistic)}
        print(label, json.dumps(res[label], indent=1))
    json.dump(res, open(out, "w"), indent=1)


def part_lfr_seeds(out):
    from fuzzykg.synthetic import lfr_style_fuzzy_graph
    rows = []
    for n in (500, 2000):
        for mixing in (0.3, 0.6):
            for shift in (0.0, 0.5):
                for gseed in range(3):
                    G, truth = lfr_style_fuzzy_graph(n, mixing, mu_mean=0.5, mu_sd=0.15, shift=shift, seed=gseed)
                    alphas = level_grid(G)
                    res = [multislice_leiden(G, alphas, seed=s) for s in range(5)]
                    qs = np.array([r.Qbar for r in res]); ar = np.array([ari(truth, r.membership) for r in res])
                    fz = [fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in range(5)]
                    fq = np.array([r.Q_F for r in fz]); fa = np.array([ari(truth, r.membership) for r in fz])
                    rows.append({"n": n, "mixing": mixing, "shift": shift, "graph_seed": gseed,
                                 "Qbar_sd": float(qs.std(ddof=1)), "Qbar_range": float(qs.max() - qs.min()),
                                 "Qbar_best2_minus_best5": float(qs[:2].max() - qs.max()),
                                 "ari_ms_sd": float(ar.std(ddof=1)), "ari_ms_best2_vs_best5": float(ar[np.argmax(qs[:2])] - ar[np.argmax(qs)]),
                                 "QF_sd": float(fq.std(ddof=1)), "ari_fuzzy_sd": float(fa.std(ddof=1)),
                                 "ari_ms_mean": float(ar.mean()), "ari_fuzzy_mean": float(fa.mean())})
                    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
                    json.dump(rows, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["shift", "real", "correlations", "lfr_seeds"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--graph-seeds", type=int, default=5, help="graph seeds per shift (part shift)")
    args = ap.parse_args()
    out = args.out or f"results/levelset_sensitivity_{args.part}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.part == "shift":
        part_shift(out, graph_seeds=args.graph_seeds); return
    if args.part == "real":
        part_real(out, names=args.names); return
    {"shift": part_shift, "real": part_real, "correlations": part_correlations, "lfr_seeds": part_lfr_seeds}[args.part](out)


if __name__ == "__main__":
    main()
