#!/usr/bin/env python3
"""Weights-only nulls (global and row) with more draws for one graph.
Usage: python scripts/weights_null_more_draws.py NAME --draws 200 --seeds 3
Writes results/weights_null_<NAME>_<draws>.json"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from calibrated_null_test import load  # noqa
from fuzzykg.nulltest import best_qf, weights_only_test  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--draws", type=int, default=200); ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args(); seeds = tuple(range(a.seeds))
G = load(a.name); obs = best_qf(G, 1.0, seeds)
res = {"Q_F": obs.Q_F, "seeds": a.seeds,
       "weights_global": weights_only_test(G, "global", a.draws, 1.0, seeds, obs=obs.Q_F),
       "weights_row": weights_only_test(G, "row", a.draws, 1.0, seeds, obs=obs.Q_F)}
json.dump(res, open(f"results/weights_null_{a.name}_{a.draws}.json", "w"), indent=1)
print(json.dumps({k: (v["delta_Q"], v["z"], v["p_upper"], v["p_lower"]) for k, v in res.items() if k.startswith("weights")}))
