#!/usr/bin/env python3
"""Version 4.4.0. Ten-graph decision floor for variants of the Junyi transition construction.

Table S6c reports the density-matched floor of each variant with two planted graphs per
in-community fraction.  For the variants whose outcome rests on a narrow margin or reads
"structure", the floor is recomputed as in scripts/v6_floor_uncertainty.py: ten planted graphs
per fraction and a bootstrap over them that re-reads the threshold fraction and the floor.
Writes results/v7/variant_floor.json.
Usage: python scripts/v7_variant_floor.py [variant ...]
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from data.loaders import RAW  # noqa: E402
from fuzzykg.nulltest import best_qf, density_matched_floor, outcome  # noqa: E402
from v6_transition_sensitivity import variant_graph, VARIANTS  # noqa: E402

OUT = os.environ.get("VF_OUT", "results/v7/variant_floor.json")


def main():
    T = pd.read_parquet(RAW / "cache" / "junyi_transition_counts.parquet")
    base = {}
    for f in ("transition_sensitivity", "transition_sensitivity_a", "transition_sensitivity_b"):
        base.update(json.load(open(f"results/v6/{f}.json")))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in sys.argv[1:]:
        if name in res:
            continue
        t0 = time.perf_counter()
        G, _ = variant_graph(T, *VARIANTS[name])
        b = base[name]
        k = b["k"]
        fl = density_matched_floor(G, k, graph_seeds=10, draws=20, rng_seed=4400)
        rows = fl["rows"]; ps = sorted({r["p_in"] for r in rows})
        A = {p: np.array([r["ari"] for r in rows if r["p_in"] == p]) for p in ps}
        D = {p: np.array([r["delta_Q"] for r in rows if r["p_in"] == p]) for p in ps}
        rng = np.random.default_rng(4401); boot = []
        for _ in range(2000):
            f = np.nan
            for p in ps:
                i = rng.integers(0, A[p].size, A[p].size)
                if A[p][i].mean() >= 0.5:
                    f = D[p][i].mean(); break
            boot.append(f)
        boot = np.array(boot); ok = boot[~np.isnan(boot)]
        lo, hi = (np.percentile(ok, [2.5, 97.5]) if ok.size else (np.nan, np.nan))
        obs = b["degree"]["delta_Q"]
        res[name] = {"k": k, "delta_Q_deg": obs, "z_deg": b["degree"]["z"], "floor_2": b["floor"]["floor"], "p_in_2": b["floor"]["floor_p_in"],
                     "outcome_2": b["outcome"], "floor_10": fl["floor"], "p_in_10": fl["floor_p_in"], "by_p_in": fl["by_p_in"],
                     "ci95_floor": [float(lo), float(hi)], "share_boot_undefined": float(np.isnan(boot).mean()),
                     "outcome_10": outcome(b["degree"]["z"], obs, fl["floor"]), "borderline": bool(ok.size and lo <= obs <= hi),
                     "time_s": time.perf_counter() - t0}
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1)
        r = res[name]
        print(name, round(obs, 3), r["floor_2"], r["floor_10"], [round(x, 3) for x in r["ci95_floor"]], r["outcome_2"], "->", r["outcome_10"], round(r["time_s"]), flush=True)


if __name__ == "__main__":
    main()
