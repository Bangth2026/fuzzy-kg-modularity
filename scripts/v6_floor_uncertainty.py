#!/usr/bin/env python3
"""Version 4.3.0. Sampling uncertainty of the decision floor of Algorithm 1 (step 3).

The floor Delta Q* of results/calibrated_null_test.json rests on two planted graphs per
in-community fraction p_in.  Here the floor is recomputed with ten planted graphs per p_in
(same generator, grid, n, m, k and membership distribution; degree-preserving null with 20
draws and three seeds per planted graph), and a bootstrap over the planted graphs of every
p_in (2,000 resamples) re-reads both the threshold p_in (mean ARI >= 0.5) and the floor, so
that the interval includes the uncertainty of the threshold.  "borderline": the observed
excess Delta Q_deg lies inside the 95% interval of the floor.
Writes results/v6/floor_uncertainty.json.
Usage: python scripts/v6_floor_uncertainty.py [graph ...]
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.nulltest import density_matched_floor, outcome  # noqa: E402

OUT = "results/v6/floor_uncertainty.json"
GRAPHS = ["assist2017_item_transition", "assist2017_kc", "assist2015_kc", "assist2009_kc", "junyi_transition",
          "assist2015_kc_transition", "mooccubex_cs_full", "assist2009_item", "assist2017_item",
          "mooccubex_cs", "assist2009_kc_transition", "assist2017_kc_transition"]


def load(name):
    df = pd.read_csv(f"data/graphs/{name}_edges.csv")
    return FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))


def main():
    base = json.load(open("results/calibrated_null_test.json"))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in (sys.argv[1:] or GRAPHS):
        if name in res:
            continue
        t0 = time.perf_counter()
        b = base[name]
        p, k = b["floor"]["floor_p_in"], b["floor"]["k"]
        G = load(name)
        fl = density_matched_floor(G, k, graph_seeds=10, draws=20, rng_seed=4300)
        rows = fl["rows"]; ps = sorted({r_["p_in"] for r_ in rows})
        A = {p_: np.array([r_["ari"] for r_ in rows if r_["p_in"] == p_]) for p_ in ps}
        D = {p_: np.array([r_["delta_Q"] for r_ in rows if r_["p_in"] == p_]) for p_ in ps}
        rng = np.random.default_rng(4301)
        boot, bp = [], []
        for _ in range(2000):
            f = None
            for p_ in ps:
                i = rng.integers(0, A[p_].size, A[p_].size)
                if A[p_][i].mean() >= 0.5:
                    f = D[p_][i].mean(); bp.append(p_); break
            boot.append(np.nan if f is None else f)
        boot = np.array(boot)
        ok = boot[~np.isnan(boot)]
        lo, hi = (np.percentile(ok, [2.5, 97.5]) if ok.size else (np.nan, np.nan))
        obs = b["degree"]["delta_Q"]
        new = fl["floor"]
        res[name] = {"k": k, "floor_reported": b["floor"]["floor"], "p_in_reported": b["floor"]["floor_p_in"],
                     "floor_10": new, "p_in_10": fl["floor_p_in"], "by_p_in": fl["by_p_in"],
                     "sd_planted_at_threshold": (float(D[fl["floor_p_in"]].std(ddof=1)) if fl["floor_p_in"] is not None else None),
                     "ci95_floor": [float(lo), float(hi)], "share_boot_undefined": float(np.isnan(boot).mean()),
                     "boot_p_in_counts": {str(p_): int(np.sum(np.array(bp) == p_)) for p_ in ps if np.any(np.array(bp) == p_)},
                     "delta_Q_deg": obs, "z_deg": b["degree"]["z"],
                     "outcome_10": outcome(b["degree"]["z"], obs, new),
                     "borderline": bool(ok.size and lo <= obs <= hi), "time_s": time.perf_counter() - t0}
        json.dump(res, open(OUT, "w"), indent=1)
        r = res[name]
        print(name, round(obs, 3), round(r["floor_reported"], 3), r["p_in_reported"], r["floor_10"], r["p_in_10"],
              [round(x, 3) for x in r["ci95_floor"]], r["outcome_10"], r["borderline"], round(r["time_s"]), flush=True)


if __name__ == "__main__":
    main()
