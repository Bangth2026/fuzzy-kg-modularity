#!/usr/bin/env python3
"""Four null models for the optimised Q_F at the reference resolution (Table 6).

For every graph, Fuzzy Leiden (five seeds, best Q_F) at gamma = 1 is compared
with the same optimisation on N draws from each of

  degree     degree-preserving rewiring of the whole graph (in- and out-degree
             sequence kept), memberships permuted over the rewired edges;
  weights    the observed topology, memberships permuted over the edges;
  order      rewiring that keeps every edge's orientation with respect to a
             curriculum order (the topological order when the graph is acyclic,
             otherwise the order by in-strength minus out-strength), so that
             forward edges are swapped only with forward edges and backward
             edges only with backward edges; memberships permuted;
  component  degree-preserving rewiring inside each weakly connected component,
             so that the component structure is kept exactly; memberships permuted.

Reported per null: mean and standard deviation of the null Q_F, the excess
Delta Q of the observed value, the z-score and the empirical p-value.

Usage: python scripts/null_models_extended.py [--draws 200] [--planted-draws 50] [--out results/null_models_extended.json] edges.csv ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden  # noqa: E402
from fuzzykg.synthetic import planted_partition_fuzzy_graph  # noqa: E402


def curriculum_order(G: FuzzyGraph) -> np.ndarray:
    """Rank of every node: topological order if the support graph is acyclic,
    otherwise ascending in-strength minus out-strength (sources first)."""
    g = G.to_igraph()
    if g.is_dag():
        order = g.topological_sorting()
        rank = np.empty(G.n, dtype=np.int64)
        rank[np.asarray(order)] = np.arange(G.n)
        return rank
    score = G.in_strength() - G.out_strength()
    order = np.argsort(score, kind="stable")
    rank = np.empty(G.n, dtype=np.int64)
    rank[order] = np.arange(G.n)
    return rank


def _swap_rewire(src, dst, rng, n_attempts, allowed=None):
    """Degree-preserving double-edge swaps (u->v, x->y) -> (u->y, x->v),
    rejecting self-loops, multi-edges and, if ``allowed`` is given, swaps whose
    new edges fail allowed(u, y) or allowed(x, v)."""
    src = src.copy(); dst = dst.copy()
    m = src.size
    present = set(zip(src.tolist(), dst.tolist()))
    done = 0
    for _ in range(n_attempts):
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        u, v, x, y = int(src[i]), int(dst[i]), int(src[j]), int(dst[j])
        if u == y or x == v or (u, y) in present or (x, v) in present:
            continue
        if allowed is not None and not (allowed(u, y) and allowed(x, v)):
            continue
        present.discard((u, v)); present.discard((x, y))
        present.add((u, y)); present.add((x, v))
        dst[i], dst[j] = y, v
        done += 1
    return src, dst, done


def null_draw(G: FuzzyGraph, kind: str, rng, rank=None, comp=None):
    if kind == "weights":
        src, dst = G.src, G.dst
    elif kind == "degree":
        src, dst, _ = _swap_rewire(G.src, G.dst, rng, 10 * G.m)
    elif kind == "order":
        fwd = rank[G.src] < rank[G.dst]
        allowed_f = lambda a, b: rank[a] < rank[b]     # noqa: E731
        allowed_b = lambda a, b: rank[a] > rank[b]     # noqa: E731
        s1, d1, _ = _swap_rewire(G.src[fwd], G.dst[fwd], rng, 10 * int(fwd.sum()), allowed_f)
        if (~fwd).any():
            s2, d2, _ = _swap_rewire(G.src[~fwd], G.dst[~fwd], rng, 10 * int((~fwd).sum()), allowed_b)
            src, dst = np.concatenate([s1, s2]), np.concatenate([d1, d2])
        else:
            src, dst = s1, d1
    elif kind == "component":
        parts_s, parts_d = [], []
        for c in np.unique(comp):
            sel = comp[G.src] == c
            if sel.sum() == 0:
                continue
            s, d, _ = _swap_rewire(G.src[sel], G.dst[sel], rng, 10 * int(sel.sum()))
            parts_s.append(s); parts_d.append(d)
        src, dst = np.concatenate(parts_s), np.concatenate(parts_d)
    else:
        raise ValueError(kind)
    mu = rng.permutation(G.mu)
    return FuzzyGraph(G.n, src, dst, mu)


def analyse(name, G, draws, seeds, gamma=1.0, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    obs = max(fuzzy_leiden(G, gamma, seed=s, n_iterations=0).Q_F for s in seeds)
    rank = curriculum_order(G)
    _, comp = connected_components(G.symmetrised(), directed=False)
    fwd_share = float((rank[G.src] < rank[G.dst]).mean())
    out = {"graph": name, "summary": G.summary(), "Q_F_obs": float(obs), "gamma": gamma, "draws": draws,
           "acyclic": bool(G.to_igraph().is_dag()), "forward_share": fwd_share,
           "components": int(comp.max() + 1), "nulls": {}}
    for kind in ("degree", "weights", "order", "component"):
        t0 = time.perf_counter()
        vals = []
        for _ in range(draws):
            Gn = null_draw(G, kind, rng, rank, comp)
            vals.append(max(fuzzy_leiden(Gn, gamma, seed=s, n_iterations=0).Q_F for s in seeds))
        vals = np.asarray(vals)
        sd = float(vals.std(ddof=1))
        out["nulls"][kind] = {"mean": float(vals.mean()), "sd": sd, "delta_Q": float(obs - vals.mean()),
                              "z": float((obs - vals.mean()) / sd) if sd > 0 else float("inf"),
                              "p_empirical": float((np.sum(vals >= obs) + 1) / (draws + 1)),
                              "time": time.perf_counter() - t0}
        print(name, kind, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out["nulls"][kind].items()}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges", nargs="*")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--planted-draws", type=int, default=50, help="draws on the three synthetic reference graphs")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/null_models_extended.json")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    rows = json.load(open(args.out)) if os.path.exists(args.out) else []   # resume: graphs already done are kept
    done = {r["graph"] for r in rows}

    def run(name, G, draws):
        if name in done:
            return
        rows.append(analyse(name, G, draws, seeds))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=1)

    G, _ = planted_partition_fuzzy_graph(600, 12, 8.0, 0.8, 0.6, 0.15, 0.3, seed=0)
    run("planted, p_in=0.8", G, args.planted_draws)
    G, _ = planted_partition_fuzzy_graph(600, 12, 8.0, 1 / 12, 0.6, 0.15, 0.0, seed=0)
    run("planted, p_in=1/k (no structure)", G, args.planted_draws)
    G, _ = planted_partition_fuzzy_graph(110, 4, 50.0, 0.3, 0.5, 0.2, 0.0, seed=0)
    run("dense co-practice-like, n=110, 50 edges/node", G, args.planted_draws)
    import pandas as pd
    for path in args.edges:
        df = pd.read_csv(path)
        run(os.path.basename(path), FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu)), args.draws)


if __name__ == "__main__":
    main()
