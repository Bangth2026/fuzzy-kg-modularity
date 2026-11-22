#!/usr/bin/env python3
"""Version 4.8.1. Measured cost of the refined tilt grid of Table 3 on the Junyi co-practice graph.

Table 3 refines the power calibration of Algorithm 1 by the tilts 0.05, 0.10, 0.15 and 0.20 on the
co-practice graphs that the tilt 0.25 already detects.  Supplementary Table S15 timed only the
default grid.  This script times the four refined tilts under both weights-only nulls at the
settings of Table S15 (15 power draws, three seeds per draw, observed partition the best of three
seeds) on one core.  Writes results/v11/refined_tilt_cost.json.
"""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.nulltest import best_qf, weights_only_power  # noqa: E402

df = pd.read_csv("data/graphs/junyi_copractice_edges.csv")
G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
obs = best_qf(G, 1.0, (0, 1, 2))
out = {"graph": "junyi_copractice", "n": int(G.n), "m": int(G.m), "tilts": [0.05, 0.10, 0.15, 0.20], "draws": 15, "seeds_per_draw": 3}
for kind in ("global", "row"):
    t = time.perf_counter()
    r = weights_only_power(G, obs.membership, tilts=(0.05, 0.10, 0.15, 0.20), kind=kind, draws=15)
    out[kind] = {"seconds": time.perf_counter() - t, "rows": r["rows"], "detectable_r_pb": r["detectable_r_pb"]}
    print(kind, round(out[kind]["seconds"], 1), [(x["tilt"], round(x["r_pb"], 3), round(x["z"], 1)) for x in r["rows"]], flush=True)
os.makedirs("results/v11", exist_ok=True)
json.dump(out, open("results/v11/refined_tilt_cost.json", "w"), indent=1)
