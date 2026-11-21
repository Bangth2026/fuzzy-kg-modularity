#!/usr/bin/env python3
"""Version 4.6.0. Measured cost of the full pre-deployment audit (Algorithm 2) on one graph.

Every step of Algorithm 2 is run with the settings the paper uses for the graph, timed by
wall clock on one CPU core, and followed by the peak resident memory of the process so far
(ru_maxrss, so the value after a step is the peak up to and including that step).

  step 1   describe the graph and its quartile cuts
  step 2   Algorithm 1: observed partition; degree-preserving null (N draws); density-matched
           floor (graph_seeds planted graphs per in-community fraction, 20 null draws each);
           global and row weights-only nulls (N draws); power on the observed topology for
           both nulls (six tilts, Np draws each)
  step 3   cut audit: Q_F drop and rank-coded Q_F drop of the three quartile cuts and of the
           no-cut support partition, ten optimiser seeds
  step 5   best-path strengths between all ordered pairs for the nine t-norms of Section 7,
           and the hop distance on the support

Settings follow Section 7: graphs with more than 10,000 edges use three seeds per draw, and the
Junyi co-practice graph 15 power draws and two planted graphs per fraction (Table 3).
Writes results/v9/audit_cost_<name>.json.

Usage: python scripts/v9_audit_cost.py junyi_copractice [--draws 200] [--edges path.csv]
"""
from __future__ import annotations
import argparse, json, os, resource, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.audit import cut_audit, describe  # noqa: E402
from fuzzykg.leiden import fuzzy_leiden  # noqa: E402
from fuzzykg.nulltest import (best_qf, degree_test, density_matched_floor, weights_only_test,  # noqa: E402
                              weights_only_power, outcome)
from fuzzykg.paths_fast import strengths  # noqa: E402

OPS = [("min", None), ("product", None), ("lukasiewicz", None), ("yager", 2.0), ("yager", 5.0), ("yager", 10.0),
       ("schweizer_sklar", -8.0), ("schweizer_sklar", -2.0), ("schweizer_sklar", 0.5)]


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--edges"); ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--power-draws", type=int); ap.add_argument("--graph-seeds", type=int); ap.add_argument("--out")
    ap.add_argument("--floor-draws", type=int, default=20); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    path = a.edges or f"data/graphs/{a.name}_edges.csv"
    out = a.out or f"results/v9/audit_cost_{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rec = {"graph": a.name, "steps": {}}
    prev = json.load(open(out))["steps"] if a.resume and os.path.exists(out) else {}

    def log(step, t0, **kw):
        rec["steps"][step] = {"seconds": round(time.perf_counter() - t0, 2), "peak_rss_mb": round(peak_mb(), 1), **kw}
        json.dump(rec, open(out, "w"), indent=1)
        print(step, rec["steps"][step], flush=True)

    t = time.perf_counter()
    e = pd.read_csv(path)
    G = FuzzyGraph.from_edges(zip(e.src, e.dst, e.mu))
    big = G.m > 10000
    seeds = (0, 1, 2) if big else (0, 1, 2, 3, 4)
    pdraws = a.power_draws or (15 if a.name == "junyi_copractice" else 30)
    gseeds = a.graph_seeds or (2 if big else 10)
    rec.update({"n": G.n, "m": G.m, "seeds_per_draw": len(seeds), "null_draws": a.draws, "power_draws": pdraws,
                "planted_graphs_per_fraction": gseeds, "floor_null_draws": a.floor_draws})
    log("load", t)

    t = time.perf_counter()
    cuts = {f"q{q:.2f}": float(np.quantile(G.mu, q)) for q in (0.25, 0.5, 0.75)}
    d = {"graph": describe(G), **{c: describe(G.alpha_cut(x)) for c, x in cuts.items()}}
    log("step1_describe", t, describe=d)

    t = time.perf_counter(); times = []
    for s in range(5):
        t1 = time.perf_counter(); fuzzy_leiden(G, gamma=1.0, seed=s); times.append(time.perf_counter() - t1)
    log("one_leiden_run", t, per_run_seconds=[round(x, 3) for x in times])

    t = time.perf_counter(); obs = best_qf(G, 1.0, seeds); log("step2a_observed_partition", t, Q_F=obs.Q_F, k=obs.n_communities)
    if "step2b_degree_null" in prev:          # resumed: keep the measured step
        rec["steps"]["step2b_degree_null"] = prev["step2b_degree_null"]; dg = prev["step2b_degree_null"]
    else:
        t = time.perf_counter(); dg = degree_test(G, a.draws, 1.0, seeds, obs=obs.Q_F)
        log("step2b_degree_null", t, delta_Q=dg["delta_Q"], z=dg["z"])
    t = time.perf_counter(); fl = density_matched_floor(G, obs.n_communities, graph_seeds=gseeds, draws=a.floor_draws)
    log("step2c_floor", t, floor=fl["floor"], outcome=outcome(dg["z"], dg["delta_Q"], fl["floor"]))
    t = time.perf_counter(); wg = weights_only_test(G, "global", a.draws, 1.0, seeds, obs=obs.Q_F); log("step2d_global_null", t, z=wg["z"])
    t = time.perf_counter(); wr = weights_only_test(G, "row", a.draws, 1.0, seeds, obs=obs.Q_F); log("step2e_row_null", t, z=wr["z"])
    t = time.perf_counter(); pg = weights_only_power(G, obs.membership, kind="global", draws=pdraws)
    log("step2f_power_global", t, r=pg["detectable_r_pb"])
    t = time.perf_counter(); pr = weights_only_power(G, obs.membership, kind="row", draws=pdraws)
    log("step2g_power_row", t, r=pr["detectable_r_pb"])

    t = time.perf_counter(); ca = cut_audit(G, cuts, seeds=range(10))
    log("step3_cut_audit", t, cuts={c: {k: r.get(k) for k in ("qf_drop", "qf_drop_sd", "rank_drop", "rank_drop_sd")} for c, r in ca["cuts"].items()})

    t = time.perf_counter(); per = {}
    for op, p in OPS:
        t1 = time.perf_counter(); strengths(G, op, p); per[f"{op}{'' if p is None else '_' + str(p)}"] = round(time.perf_counter() - t1, 2)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    t1 = time.perf_counter()
    shortest_path(csr_matrix((np.ones(G.m), (G.src, G.dst)), shape=(G.n, G.n)), directed=True, unweighted=True)
    per["hop_distance"] = round(time.perf_counter() - t1, 2)
    log("step5_all_best_paths", t, per_operator_seconds=per)

    s = rec["steps"]
    alg1 = sum(v["seconds"] for k, v in s.items() if k.startswith("step2"))
    rec["total_seconds"] = round(sum(v["seconds"] for v in s.values()), 1)
    rec["algorithm1_seconds"] = round(alg1, 1)
    rec["audit_without_algorithm1_seconds"] = round(sum(s[k]["seconds"] for k in ("step1_describe", "step3_cut_audit", "step5_all_best_paths")), 1)
    rec["peak_rss_mb"] = round(peak_mb(), 1)
    import platform
    rec["machine"] = {"python": platform.python_version(), "processor": platform.processor() or platform.machine(), "cores_used": 1}
    json.dump(rec, open(out, "w"), indent=1)
    print("TOTAL", rec["total_seconds"], "alg1", rec["algorithm1_seconds"], "peak MB", rec["peak_rss_mb"])


if __name__ == "__main__":
    main()
