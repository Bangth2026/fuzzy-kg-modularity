#!/usr/bin/env python3
"""Algorithm 1 (calibrated null test) on the real graphs (Section 8.1, Table 6).

For every graph: the optimised Q_F at gamma = 1 (best of five seeds); the
degree-preserving null; the density-matched floor read off planted graphs with the
same n, m, number of communities and membership distribution; the two weights-only
nulls (global and row permutation); the power of each weights-only null on the
observed topology (smallest detectable tilt and the corresponding point-biserial
correlation); and the Spearman correlation between memberships and the degrees of
the edge endpoints.

Usage: python scripts/calibrated_null_test.py [--draws 200] [--fast] names ...
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.nulltest import (best_qf, degree_test, weights_only_test, weights_only_power,  # noqa: E402
                              density_matched_floor, membership_degree_correlation)


def load(name):
    df = pd.read_csv(f"data/graphs/{name}_edges.csv")
    return FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--power-draws", type=int, default=30)
    ap.add_argument("--floor-draws", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-degree", action="store_true", help="reuse nothing; skip the degree null and the floor")
    ap.add_argument("--out", default="results/calibrated_null_test.json")
    args = ap.parse_args()
    rows = json.load(open(args.out)) if os.path.exists(args.out) else {}
    seeds = tuple(range(args.seeds))
    for name in args.names:
        if name in rows:
            continue
        t0 = time.perf_counter()
        G = load(name)
        obs = best_qf(G, 1.0, seeds)
        r = {"summary": G.summary(), "n": G.n, "m": G.m, "Q_F": obs.Q_F, "k": obs.n_communities,
             "mu_degree": membership_degree_correlation(G)}
        if not args.skip_degree:
            r["degree"] = degree_test(G, args.draws, 1.0, seeds, obs=obs.Q_F)
            r["floor"] = density_matched_floor(G, obs.n_communities, draws=args.floor_draws)
        r["weights_global"] = weights_only_test(G, "global", args.draws, 1.0, seeds, obs=obs.Q_F)
        r["weights_row"] = weights_only_test(G, "row", args.draws, 1.0, seeds, obs=obs.Q_F)
        r["power_global"] = weights_only_power(G, obs.membership, kind="global", draws=args.power_draws)
        r["power_row"] = weights_only_power(G, obs.membership, kind="row", draws=args.power_draws)
        r["time_s"] = time.perf_counter() - t0
        rows[name] = r
        print(name, json.dumps({"Q_F": round(obs.Q_F, 3), "k": obs.n_communities,
                                "deg_dQ": round(r.get("degree", {}).get("delta_Q", float("nan")), 4),
                                "floor": r.get("floor", {}).get("floor"),
                                "wg": (round(r["weights_global"]["delta_Q"], 4), round(r["weights_global"]["z"], 1)),
                                "wr": (round(r["weights_row"]["delta_Q"], 4), round(r["weights_row"]["z"], 1)),
                                "pow_g": (r["power_global"]["detectable_tilt"], r["power_global"]["detectable_r_pb"]),
                                "pow_r": (r["power_row"]["detectable_tilt"], r["power_row"]["detectable_r_pb"]),
                                "mu_deg": {k: round(v, 2) for k, v in r["mu_degree"].items()}, "t": round(r["time_s"])}), flush=True)
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
