#!/usr/bin/env python3
"""Release 5.1.0. Divergence of the graded partition from the support partition (Section 4.5).

The loss of a cut against the support and against the graded partition Pi_F coincide when Pi_F and
Pi_sup agree. The divergence check compares their agreement with the seed-to-seed agreement of each:
    gap = min(mean ARI(Pi_sup,s, Pi_sup,s+1), mean ARI(Pi_F,s, Pi_F,s+1)) - mean ARI(Pi_F,s, Pi_sup,s)
over ten seeds (cyclic pairs). A large gap says that the grades move the partition beyond optimiser
noise, a necessary condition for a partition value of either sign.
Computed on the graphs of the injected worlds (scripts/v16_injected.py, eta in {0, 0.5, 1, 2}, both
constructions, 96 worlds) together with the partition value ARI(Pi_F) - ARI(Pi_sup) (seed means), and
on the 23 real graphs from the cached partitions (20 seeds).
Writes results/v16/divergence.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.audit import _leiden  # noqa: E402
from fuzzykg.construction import build_fuzzy_graph, build_transition_graph  # noqa: E402
from fuzzykg.semisynthetic import World, simulate  # noqa: E402
from v16_semisynthetic import relabel, GRID  # noqa: E402
from v16_injected import ETAS  # noqa: E402
from v15_partitions import GRAPHS  # noqa: E402

OUT = "results/v16/divergence.json"


def stats(S, F, comm=None):
    n = len(S)
    fs = float(np.mean([ari(F[i], S[i]) for i in range(n)]))
    ss = float(np.mean([ari(S[i], S[(i + 1) % n]) for i in range(n)]))
    ff = float(np.mean([ari(F[i], F[(i + 1) % n]) for i in range(n)]))
    r = {"F_S": fs, "S_S": ss, "F_F": ff, "gap": min(ss, ff) - fs}
    if comm is not None:
        r["V_partition"] = float(np.mean([ari(comm, p) for p in F]) - np.mean([ari(comm, p) for p in S]))
    return r


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"worlds": {}, "real": {}}
    for name in GRAPHS:
        st = np.load(f"results/v15/partitions/{name}.npz")
        res["real"][name] = stats([st[f"support|1|{s}"] for s in range(20)], [st[f"F|1|{s}"] for s in range(20)])
    json.dump(res, open(OUT, "w"), indent=1)
    for (pc, noise, il, sug, N, seed) in GRID:
        key = f"p{pc}_n{noise}_i{il}_s{sug}_N{N}_seed{seed}"
        if key in res["worlds"]:
            continue
        t0 = time.perf_counter()
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
                rec[con][f"{eta:g}"] = stats(S, [_leiden(H, s) for s in range(10)], comm)
        res["worlds"][key] = rec
        json.dump(res, open(OUT, "w"), indent=1)
        print(key, f"{time.perf_counter() - t0:.0f}s", {c: [round(rec[c][e]["gap"], 2) for e in rec[c]] for c in rec}, flush=True)


if __name__ == "__main__":
    main()
