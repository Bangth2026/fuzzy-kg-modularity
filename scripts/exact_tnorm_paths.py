"""Exact best paths under Archimedean t-norms by additive generators (revision, M3).

1. Exactness: on a synthetic graph the generator-based Bellman-Ford (hop-limited)
   returns the same best strengths as the bounded enumeration of the earlier
   code, for every family, to floating-point precision.
2. Cost: Dijkstra on sum g(mu_e) is compared with the max-min widest-path sweep
   and with the bounded enumeration as the graph gets denser; enumeration grows
   with density and hop limit, the shortest-path solutions do not.
3. Ordinal invariance: under random strictly increasing rescalings of mu_E the
   max-min ranking of targets is unchanged (Kendall tau = 1), the Archimedean
   rankings are not.

Usage: python scripts/exact_tnorm_paths.py [--out results/tnorm_paths.json] [edges.csv]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.synthetic import synthetic_fuzzy_graph, planted_partition_fuzzy_graph  # noqa: E402
from fuzzykg.tnorms import (  # noqa: E402
    enumerate_best_paths, generator, generator_best_paths, widest_path,
)

FAMILIES = [("product", None), ("lukasiewicz", None), ("yager", 0.5), ("yager", 2.0),
            ("schweizer_sklar", -2.0), ("schweizer_sklar", 0.5)]


def load_edges(path):
    import pandas as pd
    df = pd.read_csv(path)
    return FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))


def exactness(G, n_sources=20, max_hops=4, seed=0):
    rng = np.random.default_rng(seed)
    srcs = rng.choice(G.n, min(n_sources, G.n), replace=False)
    out = {}
    for name, p in FAMILIES:
        gen = generator(name, p); T = gen.tnorm()
        maxdiff, n_targets = 0.0, 0
        for s in srcs:
            e = enumerate_best_paths(G, int(s), max_hops, T); e.pop("__expanded__", None)
            b = generator_best_paths(G, int(s), gen, max_hops=max_hops)
            assert set(e) == set(b), (name, p)
            n_targets += len(e)
            if e:
                maxdiff = max(maxdiff, max(abs(e[k][0] - b[k][0]) for k in e))
        out[gen.name] = {"kind": gen.kind, "targets_checked": n_targets, "max_abs_diff": maxdiff}
    return out


def cost(densities=(1.5, 4.0, 8.0), n=371, n_sources=10, max_hops=4):
    rows = []
    gen = generator("product")
    for d in densities:
        G = synthetic_fuzzy_graph(n, density=d, seed=0) if d < 3 else \
            planted_partition_fuzzy_graph(n, 8, d, 0.7, 0.6, 0.2, 0.3, seed=0)[0]
        srcs = np.random.default_rng(0).choice(G.n, n_sources, replace=False)
        t = time.perf_counter(); [widest_path(G, int(s)) for s in srcs]; t_mm = time.perf_counter() - t
        t = time.perf_counter(); [generator_best_paths(G, int(s), gen) for s in srcs]; t_dj = time.perf_counter() - t
        t = time.perf_counter(); [generator_best_paths(G, int(s), gen, max_hops=max_hops) for s in srcs]; t_bf = time.perf_counter() - t
        t = time.perf_counter(); exp = sum(enumerate_best_paths(G, int(s), max_hops, gen.tnorm())["__expanded__"][0] for s in srcs); t_en = time.perf_counter() - t
        rows.append({"n": G.n, "m": G.m, "density": G.m / G.n,
                     "maxmin_s": t_mm, "dijkstra_s": t_dj, "bellman_ford_4hop_s": t_bf,
                     "enumeration_4hop_s": t_en, "paths_expanded": exp,
                     "ratio_dijkstra_over_maxmin": t_dj / t_mm, "ratio_enum_over_maxmin": t_en / t_mm})
        print(rows[-1], flush=True)
    return rows


def ordinal_invariance(G, n_sources=20, n_maps=10, seed=0):
    """Random strictly increasing maps phi: [0,1]->[0,1] applied to mu_E; compare
    the ranking of targets by best-path strength before and after."""
    rng = np.random.default_rng(seed)
    srcs = rng.choice(G.n, min(n_sources, G.n), replace=False)

    def random_monotone_map():
        # random strictly increasing piecewise-linear map on [0,1] with phi(0)=0, phi(1)=1:
        # cumulative sum of positive Dirichlet increments on a 20-point grid
        grid = np.linspace(0, 1, 21)
        inc = rng.dirichlet(np.full(20, rng.uniform(0.3, 2.0)))
        vals = np.concatenate([[0.0], np.cumsum(inc)])
        return lambda x: np.interp(np.asarray(x, float), grid, vals)

    def ranking(Gx, s, fam):
        if fam == "min":
            st, _ = widest_path(Gx, int(s)); return st
        gen = generator(*fam)
        bp = generator_best_paths(Gx, int(s), gen)
        v = np.zeros(Gx.n)
        for t, (sv, _) in bp.items():
            v[t] = sv
        return v

    fams = ["min"] + FAMILIES
    taus = {str(f): [] for f in fams}
    for _ in range(n_maps):
        phi = random_monotone_map()
        Gx = FuzzyGraph(G.n, G.src, G.dst, np.clip(phi(G.mu), 1e-9, 1.0))
        for s in srcs:
            for f in fams:
                r0 = ranking(G, s, f); r1 = ranking(Gx, s, f)
                reach = r0 > 0
                if reach.sum() < 3:
                    continue
                taus[str(f)].append(kendalltau(r0[reach], r1[reach]).statistic)
    return {f: {"kendall_tau_mean": float(np.mean(v)), "kendall_tau_min": float(np.min(v)), "n": len(v)}
            for f, v in taus.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges", nargs="?", default=None)
    ap.add_argument("--out", default="results/tnorm_paths.json")
    args = ap.parse_args()
    G = load_edges(args.edges) if args.edges else synthetic_fuzzy_graph(371, seed=0)
    res = {"graph": G.summary(), "exactness": exactness(G), "cost": cost(),
           "ordinal_invariance": ordinal_invariance(G)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "cost"}, indent=1))


if __name__ == "__main__":
    main()
