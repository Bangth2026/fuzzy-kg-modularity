#!/usr/bin/env python3
"""Version 4.5.0. Decision floor of Algorithm 1 with ten planted graphs where it still rested on two.

Two sets of graphs still had a floor read from two planted graphs per in-community fraction:
the Junyi co-practice graph (Table 3, marked with a dagger) and the twelve variants of the Junyi
transition construction (Supplementary Table S6c).  This script recomputes the floor with ten
planted graphs per fraction and the bootstrap of scripts/v6_floor_uncertainty.py (2,000
resamples of the planted graphs within every fraction; the threshold fraction and the floor are
re-read on each resample).

For the Junyi co-practice graph (716 nodes, 191,962 edges) the fraction grid is cut at the first
fraction that two planted graphs already recovered at ARI 1.0 plus one step, i.e. p_in in
{1/3, 0.383, 0.433, 0.483}; the floor is defined by the smallest recovered fraction, so larger
fractions cannot change it.  The variants use the full grid of density_matched_floor.

Graph names: a file stem of data/graphs (e.g. junyi_copractice), or "junyi:<variant>" with a
variant of scripts/v6_transition_sensitivity.py (needs data/raw/cache/junyi_transition_counts.parquet).
Writes results/v8/floor_extra.json.
Usage: python scripts/v8_floor_extra.py junyi_copractice junyi:mean_sum_tau0.2 ...
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.nulltest import density_matched_floor, outcome, best_qf, degree_test  # noqa: E402

OUT = os.environ.get("FLOOR_OUT", "results/v8/floor_extra.json")
GRIDS = {"junyi_copractice": [1 / 3, 1 / 3 + 0.05, 1 / 3 + 0.10, 1 / 3 + 0.15]}


def load(name):
    if name.startswith("junyi:"):
        from data.loaders import RAW
        from v6_transition_sensitivity import VARIANTS, variant_graph
        T = pd.read_parquet(RAW / "cache" / "junyi_transition_counts.parquet")
        G, _ = variant_graph(T, *VARIANTS[name.split(":", 1)[1]])
        return G
    df = pd.read_csv(f"data/graphs/{name}_edges.csv")
    return FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))


def reference(name):
    """Observed excess, z and k as reported in the release (Table 3 / Table S6c)."""
    if name.startswith("junyi:"):
        v = name.split(":", 1)[1]
        for f in ("results/v6/transition_sensitivity.json", "results/v6/transition_sensitivity_a.json",
                  "results/v6/transition_sensitivity_b.json"):
            if os.path.exists(f):
                d = json.load(open(f))
                if v in d:
                    a = d[v]["algorithm1"] if "algorithm1" in d[v] else d[v]
                    return a
        raise KeyError(v)
    return json.load(open("results/calibrated_null_test.json"))[name]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in sys.argv[1:]:
        if name in res:
            continue
        t0 = time.perf_counter()
        b = reference(name)
        k = b["floor"].get("k", b.get("k"))
        G = load(name)
        fl = density_matched_floor(G, k, p_grid=GRIDS.get(name), graph_seeds=10, draws=20, rng_seed=4400)
        rows = fl["rows"]; ps = sorted({r_["p_in"] for r_ in rows})
        A = {p_: np.array([r_["ari"] for r_ in rows if r_["p_in"] == p_]) for p_ in ps}
        D = {p_: np.array([r_["delta_Q"] for r_ in rows if r_["p_in"] == p_]) for p_ in ps}
        rng = np.random.default_rng(4401)
        boot = []
        for _ in range(2000):
            f = None
            for p_ in ps:
                i = rng.integers(0, A[p_].size, A[p_].size)
                if A[p_][i].mean() >= 0.5:
                    f = D[p_][i].mean(); break
            boot.append(np.nan if f is None else f)
        boot = np.array(boot); ok = boot[~np.isnan(boot)]
        lo, hi = (np.percentile(ok, [2.5, 97.5]) if ok.size else (np.nan, np.nan))
        obs, z = b["degree"]["delta_Q"], b["degree"]["z"]
        res[name] = {"k": k, "n": int(G.n), "m": int(G.src.size), "floor_2": b["floor"]["floor"],
                     "p_in_2": b["floor"]["floor_p_in"], "floor_10": fl["floor"], "p_in_10": fl["floor_p_in"],
                     "by_p_in": fl["by_p_in"], "ci95_floor": [float(lo), float(hi)],
                     "share_boot_undefined": float(np.isnan(boot).mean()), "delta_Q_deg": obs, "z_deg": z,
                     "outcome_2": outcome(z, obs, b["floor"]["floor"]), "outcome_10": outcome(z, obs, fl["floor"]),
                     "borderline": bool(ok.size and lo <= obs <= hi), "time_s": time.perf_counter() - t0}
        json.dump(res, open(OUT, "w"), indent=1)
        r = res[name]
        print(name, round(obs, 3), r["floor_2"], r["floor_10"], r["p_in_10"], [round(x, 3) for x in r["ci95_floor"]],
              r["outcome_2"], "->", r["outcome_10"], r["borderline"], round(r["time_s"]), flush=True)


if __name__ == "__main__":
    main()
