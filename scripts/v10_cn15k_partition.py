#!/usr/bin/env python3
"""Version 4.7.0. The partition half of the audit (Algorithm 2, steps 1 to 3) on CN15k.

CN15k is built as in scripts/v9_cross_domain.py: the three UKGE splits pooled and the relation
types of an ordered pair collapsed to their largest confidence (full graph, nothing held out).
Raw files: clone https://github.com/stasl0217/UKGE and copy data/cn15k to data/raw/ukge/cn15k.

Steps are run in the order of their cost and saved after each one, so that an interrupted run
keeps what it measured (--resume skips the saved steps):
  step1   describe the graph and its quartile cuts
  step3   cut audit: Q_F drop and rank-coded Q_F drop of the quartile cuts and of the no-cut
          support partition over ten optimiser seeds, with the flagging rule of Algorithm 2
  step2   Algorithm 1 with the settings used for PPI5k (Supplementary Table S14b): three seeds per
          draw, 50 null draws, two planted graphs per in-community fraction with 10 null draws
          each, 10 power draws per tilt
Every step is timed on one core. Writes results/v10/cn15k_partition.json.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.audit import cut_audit, describe  # noqa: E402
from fuzzykg.leiden import fuzzy_leiden  # noqa: E402
from fuzzykg.nulltest import (best_qf, degree_test, density_matched_floor, weights_only_test,  # noqa: E402
                              weights_only_power, outcome)

RAW = "data/raw/ukge/cn15k"
OUT = "results/v10/cn15k_partition.json"


def build():
    cols = ["h", "r", "t", "w"]
    df = pd.concat([pd.read_csv(f"{RAW}/{s}.tsv", sep="\t", header=None, names=cols) for s in ("train", "val", "test")])
    ent = sorted(set(df.h) | set(df.t)); ix = {e: i for i, e in enumerate(ent)}
    df["u"] = df.h.map(ix); df["v"] = df.t.map(ix)
    g = df[df.u != df.v].groupby(["u", "v"]).w.max().reset_index()
    return FuzzyGraph(len(ent), g.u.to_numpy(np.int64), g.v.to_numpy(np.int64), g.w.to_numpy(float))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--draws", type=int, default=50); ap.add_argument("--power-draws", type=int, default=10)
    ap.add_argument("--graph-seeds", type=int, default=2); ap.add_argument("--floor-draws", type=int, default=10)
    ap.add_argument("--only", default="1,3,2")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rec = json.load(open(OUT)) if a.resume and os.path.exists(OUT) else {"graph": "cn15k", "steps": {}}
    S = rec["steps"]

    def save():
        json.dump(rec, open(OUT, "w"), indent=1)

    G = build(); seeds = (0, 1, 2)
    rec.update({"n": G.n, "m": G.m, "seeds_per_draw": 3, "null_draws": a.draws, "power_draws": a.power_draws,
                "planted_graphs_per_fraction": a.graph_seeds, "floor_null_draws": a.floor_draws})
    cuts = {f"q{q:.2f}": float(np.quantile(G.mu, q)) for q in (0.25, 0.5, 0.75)}
    order = a.only.split(",")
    if "1" in order and "step1" not in S:
        t = time.perf_counter()
        d = {"graph": describe(G), **{c: describe(G.alpha_cut(x)) for c, x in cuts.items()}}
        times = []
        for s in range(3):
            t1 = time.perf_counter(); fuzzy_leiden(G, gamma=1.0, seed=s); times.append(round(time.perf_counter() - t1, 2))
        S["step1"] = {"seconds": round(time.perf_counter() - t, 1), "describe": d, "leiden_run_seconds": times}; save(); print("step1", S["step1"], flush=True)
    if "3" in order and "step3" not in S:
        t = time.perf_counter(); ca = cut_audit(G, cuts, seeds=range(10))
        b = ca["cuts"]["support"]
        for c, r in ca["cuts"].items():
            if c == "support" or r.get("empty"):
                continue
            for k in ("qf_drop", "rank_drop"):   # rule of Algorithm 2, step 3: 2 x larger seed SD and 0.01
                ex = r[k] - b[k]
                r[f"flag_{k}_alg2"] = bool(ex > 2 * max(r[k + "_sd"], b[k + "_sd"]) and ex > 0.01)
        S["step3"] = {"seconds": round(time.perf_counter() - t, 1), "cut_audit": ca}; save()
        print("step3", {c: (round(r.get("qf_drop", 0), 4), round(r.get("qf_drop_sd", 0), 4), r.get("flag_qf_drop_alg2"), round(r.get("mean_out_degree", 0), 2)) for c, r in ca["cuts"].items()}, flush=True)
    if "2" in order:
        if "step2a" not in S:
            t = time.perf_counter(); obs = best_qf(G, 1.0, seeds)
            S["step2a"] = {"seconds": round(time.perf_counter() - t, 1), "Q_F": obs.Q_F, "k": obs.n_communities}; save()
            np.save("results/v10/cn15k_partition_obs.npy", obs.membership)
        memb = np.load("results/v10/cn15k_partition_obs.npy"); qobs = S["step2a"]["Q_F"]; k = S["step2a"]["k"]
        if "step2b_degree" not in S:
            t = time.perf_counter(); dg = degree_test(G, a.draws, 1.0, seeds, obs=qobs)
            S["step2b_degree"] = {"seconds": round(time.perf_counter() - t, 1), **{x: dg[x] for x in dg if not isinstance(dg[x], (list, np.ndarray))}}; save(); print("deg", S["step2b_degree"], flush=True)
        if "step2d_global" not in S:
            t = time.perf_counter(); wg = weights_only_test(G, "global", a.draws, 1.0, seeds, obs=qobs)
            S["step2d_global"] = {"seconds": round(time.perf_counter() - t, 1), **{x: wg[x] for x in wg if not isinstance(wg[x], (list, np.ndarray))}}; save(); print("glob", S["step2d_global"], flush=True)
        if "step2e_row" not in S:
            t = time.perf_counter(); wr = weights_only_test(G, "row", a.draws, 1.0, seeds, obs=qobs)
            S["step2e_row"] = {"seconds": round(time.perf_counter() - t, 1), **{x: wr[x] for x in wr if not isinstance(wr[x], (list, np.ndarray))}}; save(); print("row", S["step2e_row"], flush=True)
        if "step2c_floor" not in S:
            t = time.perf_counter(); fl = density_matched_floor(G, k, graph_seeds=a.graph_seeds, draws=a.floor_draws)
            dgq = S["step2b_degree"]
            S["step2c_floor"] = {"seconds": round(time.perf_counter() - t, 1), "floor": fl["floor"], "p_in": fl.get("floor_p_in"),
                                 "outcome": outcome(dgq["z"], dgq["delta_Q"], fl["floor"])}; save(); print("floor", S["step2c_floor"], flush=True)
        for kind in ("global", "row"):
            key = f"step2_power_{kind}"
            if key not in S:
                t = time.perf_counter(); p = weights_only_power(G, memb, kind=kind, draws=a.power_draws)
                S[key] = {"seconds": round(time.perf_counter() - t, 1), "detectable_r": p["detectable_r_pb"]}; save(); print(key, S[key], flush=True)
    save()


if __name__ == "__main__":
    main()
