#!/usr/bin/env python3
"""Calibration of the effect-size floor of Algorithm 1 and a seed check on the
weights-only null (Section 8.1, Table 6 and the text around it).

Part A (``--part planted``).  Dense planted partitions at the density of the
KC-level co-practice graphs (110 nodes, 50 out-edges per node, four equal
communities) with the in-community edge fraction p_in swept from the
structureless value 1/k upward.  For every p_in: Fuzzy Leiden at gamma = 1
(best of five seeds), its adjusted Rand index against the planted truth, and
the excess of Q_F over the degree-preserving null (50 draws, same seeds).  The
sweep says which excess a *real* planted structure produces at that density,
so that the absolute floor Delta Q_min can be read in units of planted
in-community preference.

Part B (``--part seeds``).  The observed Q_F of every real graph at gamma = 1
is recomputed as the best of 50 optimiser seeds instead of 5 and compared
with the weights-only null mean already recorded (best of 5 seeds per draw).
A negative excess that survives ten times the optimiser effort on the observed
graph cannot be an artefact of a rougher optimisation surface on the observed
graph.

Part C (``--part floors``).  From the recorded null results, the outcome of the
degree-preserving test under floors Delta Q_min in {0.02, 0.05, 0.10} and the
relative excess Delta Q / mean null Q_F.

Usage: python scripts/null_floor_calibration.py --part planted|seeds|floors
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden  # noqa: E402
from fuzzykg.resolution import adjusted_rand_index  # noqa: E402
from fuzzykg.synthetic import planted_partition_fuzzy_graph  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from null_models_extended import null_draw  # noqa: E402

REAL = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2015_kc", "assist2017_kc",
        "assist2009_item", "assist2017_item"]


def best_q(G, seeds, gamma=1.0):
    return max((fuzzy_leiden(G, gamma, seed=s, n_iterations=0) for s in seeds), key=lambda r: r.Q_F)


def part_planted(out, draws=50, seeds=range(5)):
    rows = []
    for p_in in (0.25, 0.28, 0.30, 0.33, 0.36, 0.40, 0.45, 0.50, 0.60, 0.80):
        for gseed in range(3):
            G, truth = planted_partition_fuzzy_graph(110, 4, 50.0, p_in, 0.5, 0.2, 0.0, seed=gseed)
            obs = best_q(G, seeds)
            rng = np.random.default_rng(gseed)
            vals = [best_q(null_draw(G, "degree", rng), seeds).Q_F for _ in range(draws)]
            vals = np.asarray(vals)
            rows.append({"p_in": p_in, "graph_seed": gseed, "n": G.n, "m": G.m, "Q_F": obs.Q_F,
                         "k": obs.n_communities, "ari_truth": adjusted_rand_index(truth, obs.membership),
                         "null_mean": float(vals.mean()), "null_sd": float(vals.std(ddof=1)),
                         "delta_Q": float(obs.Q_F - vals.mean()),
                         "z": float((obs.Q_F - vals.mean()) / vals.std(ddof=1))})
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
            json.dump(rows, open(out, "w"), indent=1)


def part_seeds(out, n_seeds=50):
    recorded = {r["graph"]: r for r in json.load(open("results/null_models_extended.json"))}
    rows = []
    for name in REAL:
        df = pd.read_csv(f"data/graphs/{name}_edges.csv")
        G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
        res = [fuzzy_leiden(G, 1.0, seed=s, n_iterations=0).Q_F for s in range(n_seeds)]
        rec = recorded[f"{name}_edges.csv"]
        w = rec["nulls"]["weights"]
        rows.append({"graph": name, "Q_F_best5": rec["Q_F_obs"], "Q_F_best50": float(max(res)),
                     "Q_F_seed_sd": float(np.std(res, ddof=1)), "Q_F_seed_mean": float(np.mean(res)),
                     "weights_null_mean": w["mean"], "weights_null_sd": w["sd"],
                     "delta_Q_best5": w["delta_Q"], "delta_Q_best50": float(max(res) - w["mean"]),
                     "z_best50": float((max(res) - w["mean"]) / w["sd"]) if w["sd"] > 0 else None})
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
        json.dump(rows, open(out, "w"), indent=1)


def part_floors(out):
    rows = []
    for r in json.load(open("results/null_models_extended.json")):
        d = r["nulls"]["degree"]
        rows.append({"graph": r["graph"], "Q_F": r["Q_F_obs"], "null_mean": d["mean"], "delta_Q": d["delta_Q"],
                     "z": d["z"], "relative_excess": d["delta_Q"] / d["mean"] if d["mean"] else None,
                     "outcome": {str(f): ("structure" if (d["z"] >= 3 and d["delta_Q"] >= f) else "no structure detected")
                                 for f in (0.02, 0.05, 0.10)}})
        print(rows[-1]["graph"], round(rows[-1]["relative_excess"], 2), rows[-1]["outcome"])
    json.dump(rows, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["planted", "seeds", "floors"], required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/null_floor_{args.part}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    {"planted": part_planted, "seeds": part_seeds, "floors": part_floors}[args.part](out)


if __name__ == "__main__":
    main()
