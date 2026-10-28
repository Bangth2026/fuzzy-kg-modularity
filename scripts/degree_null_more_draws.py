#!/usr/bin/env python3
"""Degree-preserving null of Algorithm 1 with more draws for one graph.
Usage: python scripts/degree_null_more_draws.py NAME --draws 200 --seeds 3
Writes results/degree_null_<NAME>_<draws>.json"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from calibrated_null_test import load  # noqa
from fuzzykg.nulltest import best_qf, degree_test  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--draws", type=int, default=200); ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args(); seeds = tuple(range(a.seeds))
G = load(a.name); obs = best_qf(G, 1.0, seeds)
res = {"Q_F": obs.Q_F, "seeds": a.seeds, "degree": degree_test(G, a.draws, 1.0, seeds, obs=obs.Q_F)}
json.dump(res, open(f"results/degree_null_{a.name}_{a.draws}.json", "w"), indent=1, default=float)
print(json.dumps(res["degree"], default=float))
