#!/usr/bin/env python3
"""Version 4.0.0. Algorithm 1 with the structure-matched (degree-corrected) floor.

(a) --new: the full Algorithm 1 of release 1.4.3 (degree null, density-matched floor, both
    weights-only nulls, power calibration) on the ten graphs of the five added corpora;
    200 null draws, five seeds (three seeds on graphs with more than 10,000 edges).
(b) --dcfloor: the structure-matched floor of fuzzykg.nulltest.structure_matched_floor on every
    graph (the thirteen of Table 1 and the ten new ones), with the three-valued outcome under
    both floors; the degree-null excess and z are taken from (a) or from
    results/calibrated_null_test.json.
Writes results/v4/algorithm1_new.json and results/v4/dcfloor.json.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.nulltest import (best_qf, degree_test, weights_only_test, weights_only_power, density_matched_floor,  # noqa: E402
                              membership_degree_correlation, structure_matched_floor, outcome)

NEW = [f"{c}_{k}" for c in ("assist2012", "algebra2005", "bridge2006", "eedi", "ednet") for k in ("copractice", "transition")]
OLD = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2009_kc_transition", "assist2009_item", "assist2015_kc",
       "assist2015_kc_transition", "assist2017_kc", "assist2017_kc_transition", "assist2017_item", "assist2017_item_transition",
       "junyi_copractice", "junyi_transition"]


def load(name):
    df = pd.read_csv(f"data/graphs/{name}_edges.csv")
    return FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))


def run_new(names, out="results/v4/algorithm1_new.json", draws=200):
    res = json.load(open(out)) if os.path.exists(out) else {}
    for name in names:
        if name in res:
            continue
        t0 = time.perf_counter(); G = load(name)
        seeds = (0, 1, 2) if G.m > 10000 else (0, 1, 2, 3, 4)
        obs = best_qf(G, 1.0, seeds)
        r = {"summary": G.summary(), "n": G.n, "m": G.m, "Q_F": obs.Q_F, "k": obs.n_communities, "seeds": len(seeds),
             "mu_degree": membership_degree_correlation(G)}
        r["degree"] = degree_test(G, draws, 1.0, seeds, obs=obs.Q_F)
        r["floor"] = density_matched_floor(G, obs.n_communities, draws=20)
        r["weights_global"] = weights_only_test(G, "global", draws, 1.0, seeds, obs=obs.Q_F)
        r["weights_row"] = weights_only_test(G, "row", draws, 1.0, seeds, obs=obs.Q_F)
        r["power_global"] = weights_only_power(G, obs.membership, kind="global", draws=30)
        r["power_row"] = weights_only_power(G, obs.membership, kind="row", draws=30)
        r["membership"] = obs.membership.tolist()
        r["time_s"] = time.perf_counter() - t0
        res[name] = r; json.dump(res, open(out, "w"), indent=1)
        print(name, round(obs.Q_F, 3), obs.n_communities, "deg", round(r["degree"]["delta_Q"], 4), round(r["degree"]["z"], 1),
              "floor", r["floor"]["floor"], "wg", round(r["weights_global"]["z"], 1), "wr", round(r["weights_row"]["z"], 1),
              f"{r['time_s']:.0f}s", flush=True)
    return res


def run_dc(names, out="results/v4/dcfloor.json"):
    res = json.load(open(out)) if os.path.exists(out) else {}
    old = json.load(open("results/calibrated_null_test.json"))
    new = json.load(open("results/v4/algorithm1_new.json")) if os.path.exists("results/v4/algorithm1_new.json") else {}
    for name in names:
        if name in res:
            continue
        src = old.get(name) or new.get(name)
        if src is None:
            print("skip (no Algorithm 1 result)", name); continue
        t0 = time.perf_counter(); G = load(name)
        big = G.m > 50000
        memb = np.asarray(src["membership"]) if "membership" in src else best_qf(G, 1.0, (0, 1, 2) if big else (0, 1, 2, 3, 4)).membership
        f = structure_matched_floor(G, memb, graph_seeds=1 if big else 2, draws=10 if big else 20, seeds=(0, 1, 2))
        dq, z = src["degree"]["delta_Q"], src["degree"]["z"]
        r = {"delta_Q_deg": dq, "z_deg": z, "floor_density": src["floor"]["floor"], "floor_density_p_in": src["floor"]["floor_p_in"],
             "floor_dc": f["floor"], "floor_dc_p_in": f["floor_p_in"], "p_structureless_dc": f["p_structureless"], "by_p_in_dc": f["by_p_in"],
             "outcome_density": outcome(z, dq, src["floor"]["floor"]), "outcome_dc": outcome(z, dq, f["floor"]),
             "time_s": time.perf_counter() - t0}
        for thr in (0.3, 0.7):
            rec = [a for a in f["by_p_in"] if a["ari"] >= thr]
            r[f"floor_dc_ari{thr}"] = rec[0]["delta_Q"] if rec else None
            r[f"outcome_dc_ari{thr}"] = outcome(z, dq, rec[0]["delta_Q"] if rec else None)
        res[name] = r; json.dump(res, open(out, "w"), indent=1)
        print(name, r["outcome_density"], "->", r["outcome_dc"], "floors", r["floor_density"], r["floor_dc"], f"{r['time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--new", action="store_true"); ap.add_argument("--dcfloor", action="store_true")
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    if a.new:
        run_new(a.names or NEW)
    if a.dcfloor:
        run_dc(a.names or (OLD + NEW))
