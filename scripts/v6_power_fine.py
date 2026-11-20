#!/usr/bin/env python3
"""Version 4.3.0. Finer tilt grid for the power of the weights-only nulls (Algorithm 1, step 5).

On the graphs where the smallest tilt of the default grid, 0.25, already reaches z >= 3, the
detectable correlation r* was reported as an interval with lower end 0.  The tilts 0.05, 0.10,
0.15 and 0.20 are run here with the same procedure (30 draws, three seeds; 15 draws on the
Junyi co-practice graph, as in Table 3), so that r* is bracketed by two positive tilts.
Writes results/v6/power_fine.json.
"""
from __future__ import annotations
import json, os, sys, time
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.nulltest import best_qf, weights_only_power  # noqa: E402

OUT = "results/v6/power_fine.json"
GRAPHS = ["assist2017_kc", "assist2015_kc", "assist2009_kc", "assist2017_item", "junyi_copractice"]


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in (sys.argv[1:] or GRAPHS):
        if name in res:
            continue
        t0 = time.perf_counter()
        df = pd.read_csv(f"data/graphs/{name}_edges.csv")
        G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
        big = G.m > 100000
        obs = best_qf(G, 1.0, (0, 1, 2) if big else (0, 1, 2, 3, 4))
        r = {}
        for kind in ("global", "row"):
            p = weights_only_power(G, obs.membership, tilts=(0.05, 0.10, 0.15, 0.20), kind=kind, draws=15 if big else 30)
            r[kind] = {"rows": [{k: row[k] for k in ("tilt", "r_pb", "z", "delta_Q")} for row in p["rows"]],
                       "detectable_tilt": p["detectable_tilt"], "detectable_r_pb": p["detectable_r_pb"]}
        r["time_s"] = time.perf_counter() - t0
        res[name] = r
        json.dump(res, open(OUT, "w"), indent=1)
        print(name, {k: [(x["tilt"], round(x["r_pb"], 3), round(x["z"], 1)) for x in r[k]["rows"]] for k in ("global", "row")}, round(r["time_s"]), flush=True)


if __name__ == "__main__":
    main()
