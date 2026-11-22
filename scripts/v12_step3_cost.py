#!/usr/bin/env python3
"""Version 4.9.0. Wall-clock cost of step 3 of Algorithm 2 as the algorithm now specifies it.

Step 3 computes, for the no-cut support partition and the three quartile cuts, the Q_F drop on
mu_E and the rank-coded drop over 20 optimiser seeds with Pi_F, Pi_rank and the cut partition
re-drawn for every seed (fuzzykg.audit.cut_audit, no rescalings). The same call with ten seeds is
timed in the same process, so that the row of Supplementary Table S15 measured in release 4.6.0
(ten seeds, 64.9 s) can be compared on this machine. Writes results/v12/step3_cost_<graph>.json.

Usage: python scripts/v12_step3_cost.py junyi_copractice --unit arbitrary
"""
from __future__ import annotations
import argparse, json, os, platform, resource, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.audit import cut_audit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--unit", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 20])
    a = ap.parse_args()
    e = pd.read_csv(f"data/graphs/{a.name}_edges.csv")
    G = FuzzyGraph.from_edges(zip(e.src, e.dst, e.mu))
    cuts = {f"q{q:.2f}": float(np.quantile(G.mu, q)) for q in (0.25, 0.5, 0.75)}
    rec = {"graph": a.name, "n": G.n, "m": G.m, "unit": a.unit, "machine": platform.processor() or platform.machine(),
           "python": platform.python_version(), "runs": {}}
    out = f"results/v12/step3_cost_{a.name}.json"; os.makedirs("results/v12", exist_ok=True)
    if os.path.exists(out):
        rec["runs"] = json.load(open(out)).get("runs", {})
    for k in a.seeds:
        t = time.perf_counter(); ca = cut_audit(G, cuts, seeds=range(k), unit=a.unit); dt = time.perf_counter() - t
        rec["runs"][f"seeds_{k}"] = {"seconds": round(dt, 1), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                                    "cuts": {c: {x: r.get(x) for x in ("qf_drop", "qf_drop_sd", "rank_drop", "rank_drop_sd", "flag_qf_drop", "flag_rank_drop", "decision")}
                                             for c, r in ca["cuts"].items()}, "baseline_warning": ca["baseline_warning"]}
        json.dump(rec, open(out, "w"), indent=1); print(k, round(dt, 1), flush=True)


if __name__ == "__main__":
    main()
