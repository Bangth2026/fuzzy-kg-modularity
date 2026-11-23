#!/usr/bin/env python3
"""Release 5.2.0. The divergence check of Section 4.5 with a null reference.

The divergence D (Eq. 7) compares the graded partition with the support partition against optimiser
noise only. On a sparse graph any weighting moves the Leiden partition, so D also fires for grades
that carry nothing. The null-referenced form asks whether the observed grades move the partition
further than random grades on the same topology do:
    A_obs   = mean_s ARI(Pi_F,s, Pi_sup,s)                         (ten seeds)
    A_r     = mean_s ARI(Pi_row_r, Pi_sup,s)    for ten row permutations r of mu (best of three seeds)
    D_null  = mean_r A_r - A_obs ;  fires_null = A_obs < min_r A_r
Computed on every graph of the injected worlds (scripts/v16_injected.py: eta in {0, 0.5, 1, 2},
both constructions, 96 worlds) and on the 23 real graphs (cached partitions; the row permutations
are those of scripts/v15_partitions.py). The plain D of scripts/v16_divergence.py is recomputed
alongside so that both can be calibrated by density class.
Writes results/v17/divergence_null.json (resumable; PART=i/n splits the worlds).
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden  # noqa: E402
from fuzzykg.construction import build_fuzzy_graph, build_transition_graph  # noqa: E402
from fuzzykg.nulltest import permute_rows  # noqa: E402
from fuzzykg.semisynthetic import World, simulate  # noqa: E402
from fuzzykg.simplex import divergence  # noqa: E402
from v16_semisynthetic import relabel, GRID  # noqa: E402
from v16_injected import ETAS  # noqa: E402
from v15_partitions import GRAPHS, load  # noqa: E402

OUT = "results/v17/divergence_null.json"


def best3(H):
    P = [_leiden(H, k) for k in range(3)]
    return max(P, key=lambda p: fuzzy_modularity(H, p))


def stats(S, F, R):
    d = divergence(S, F)
    A_obs = d["a_FS"]
    A_r = [float(np.mean([ari(r, s) for s in S])) for r in R]
    return {**d, "A_obs": A_obs, "A_null_mean": float(np.mean(A_r)), "A_null_min": float(np.min(A_r)),
            "D_null": float(np.mean(A_r) - A_obs), "fires_null": bool(A_obs < np.min(A_r))}


def real(res):
    res["real"] = {}
    for name in GRAPHS:
        st = np.load(f"results/v15/partitions/{name}.npz")
        S = [st[f"support|1|{s}"] for s in range(20)]; F = [st[f"F|1|{s}"] for s in range(20)]
        R = [st[f"Frow{d}|1|best3"] for d in range(10)]
        res["real"][name] = stats(S, F, R)


def main():
    os.makedirs("results/v17", exist_ok=True)
    part = os.environ.get("PART")
    out = OUT if part is None else OUT.replace(".json", f"_part{part.split('/')[0]}.json")
    res = json.load(open(out)) if os.path.exists(out) else {"worlds": {}}
    if part is None or part.startswith("0/"):
        real(res); json.dump(res, open(out, "w"), indent=1)
    for gi, (pc, noise, il, sug, N, seed) in enumerate(GRID):
        key = f"p{pc}_n{noise}_i{il}_s{sug}_N{N}_seed{seed}"
        if part is not None and gi % int(part.split("/")[1]) != int(part.split("/")[0]):
            continue
        if key in res["worlds"]:
            continue
        t0 = time.perf_counter(); rng = np.random.default_rng(1000 + gi)
        w = World(p_comm=pc, noise=noise, interleave=il, suggest=sug, N=N, seed=seed)
        log, truth = simulate(w); comm = truth["community"]
        rec = {}
        for con, build in (("co-practice", build_fuzzy_graph), ("transition", build_transition_graph)):
            G0, _ = build(log); G = relabel(G0, G0.labels, w.K)
            inside = comm[G.src] == comm[G.dst]
            Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
            S = [_leiden(Gs, s) for s in range(10)]
            rec[con] = {}
            for eta in ETAS:
                mu = np.where(inside, G.mu ** (1.0 / (1.0 + eta)), G.mu ** (1.0 + eta))
                H = FuzzyGraph(G.n, G.src, G.dst, mu)
                F = [_leiden(H, s) for s in range(10)]
                R = [best3(permute_rows(H, rng)) for _ in range(10)]
                r = stats(S, F, R)
                r["V_partition"] = float(np.mean([ari(comm, p) for p in F]) - np.mean([ari(comm, p) for p in S]))
                rec[con][f"{eta:g}"] = r
        res["worlds"][key] = rec
        json.dump(res, open(out, "w"), indent=1)
        print(key, f"{time.perf_counter() - t0:.0f}s", {c: [(round(rec[c][e]["D"], 2), round(rec[c][e]["D_null"], 2)) for e in rec[c]] for c in rec}, flush=True)


if __name__ == "__main__":
    main()
