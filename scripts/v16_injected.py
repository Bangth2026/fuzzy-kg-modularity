#!/usr/bin/env python3
"""Release 5.1.0. Graded graphs whose grades mark the communities by construction (Section 7.3).

Raising the strengths of within-community prerequisites in the generator (scripts/v16_semisynthetic.py,
delta up to 0.6) changes the learner logs but leaves the partition value of the log-built grades
near its value at delta = 0. To test the audit where the grades do carry community information,
the memberships of every graph of the 96 worlds of release 5.0.0 are transformed after
construction, with the true communities:
    mu'(e) = mu(e) ** (1 / (1 + eta))   for an edge inside a true community (raised),
    mu'(e) = mu(e) ** (1 + eta)         for an edge between communities (lowered),
eta in {0, 0.5, 1, 2}; eta = 0 is the original graph. The topology, and hence the support
partition, is unchanged; the quantile cuts, Pi_F and the audit change. On every graph:
ARI of the support, cut and graded partitions with the true communities (ten seeds); decision
and kappa* of the audit (minimum excess of release 5.0.0); losses of every cut against the support
and against Pi_F; agreement ARI(Pi_F, Pi_sup) (mean over seed pairs); the row-null gate (ten
draws). Writes results/v16/injected.json (resumable).
"""
from __future__ import annotations
import itertools, json, os, sys, time
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden  # noqa: E402
from fuzzykg.construction import build_fuzzy_graph, build_transition_graph  # noqa: E402
from fuzzykg.nulltest import permute_rows  # noqa: E402
from fuzzykg.semisynthetic import World, simulate  # noqa: E402
from fuzzykg.simplex import CutFamily, kappa_star  # noqa: E402
from v16_semisynthetic import relabel, loss_rule, best3, GRID, DSEL, CUTS, SEEDS  # noqa: E402

OUT = "results/v16/injected.json"
ETAS = (0.0, 0.5, 1.0, 2.0)


def record(G, comm, con, rng):
    K = G.n
    Gs = FuzzyGraph(K, G.src, G.dst, np.ones(G.m))
    parts = {"support": [_leiden(Gs, s) for s in SEEDS]}
    for q in CUTS:
        parts[f"q{q:.2f}"] = [_leiden(G.alpha_cut(float(np.quantile(G.mu, q))), s) for s in SEEDS]
    parts["F"] = [_leiden(G, s) for s in SEEDS]
    ar = {k: [ari(comm, p) for p in v] for k, v in parts.items()}
    agree = float(np.mean([ari(a, b) for a, b in zip(parts["F"], parts["support"])]))
    fam = CutFamily(G, {k: v for k, v in parts.items() if k != "F"})
    rec = {"ari": {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in ar.items()}, "agree_F_sup": agree, "cuts": {}}
    for q in CUTS:
        c = f"q{q:.2f}"
        k1 = kappa_star(fam, c, min_excess=DSEL[con], kmax=100.0)
        ls, ds = loss_rule(ar["support"], ar[c]); lf, df = loss_rule(ar["F"], ar[c])
        rec["cuts"][c] = {"decision_mu": k1["decision_mu"], "kappa_star": k1["kappa_star"], "margin_mu": k1["margin_mu"],
                          "loss_vs_support": ls, "loss_vs_support_value": ds, "loss_vs_F": lf, "loss_vs_F_value": df}
    pF, pS = best3(G), best3(Gs)
    g_obs = float(fuzzy_modularity(G, pF) - fuzzy_modularity(G, pS))
    null = []
    for _ in range(10):
        H = permute_rows(G, rng); null.append(float(fuzzy_modularity(H, best3(H)) - fuzzy_modularity(H, pS)))
    rec["gate"] = {"g_obs": g_obs, "null_max": float(np.max(null)), "null_mean": float(np.mean(null))}
    rec["V_partition"] = float(np.mean(ar["F"]) - np.mean(ar["support"]))
    return rec


def main():
    part = os.environ.get("PART")
    out = OUT if part is None else OUT.replace(".json", f"_part{part.split('/')[0]}.json")
    res = json.load(open(out)) if os.path.exists(out) else {"etas": list(ETAS), "worlds": {}}
    for gi, (pc, noise, il, sug, N, seed) in enumerate(GRID):
        key = f"p{pc}_n{noise}_i{il}_s{sug}_N{N}_seed{seed}"
        if part is not None and gi % int(part.split("/")[1]) != int(part.split("/")[0]):
            continue
        if key in res["worlds"]:
            continue
        t0 = time.perf_counter(); rng = np.random.default_rng(gi)
        w = World(p_comm=pc, noise=noise, interleave=il, suggest=sug, N=N, seed=seed)
        log, truth = simulate(w); comm = truth["community"]
        rec = {}
        for con, build in (("co-practice", build_fuzzy_graph), ("transition", build_transition_graph)):
            G0, _ = build(log); G = relabel(G0, G0.labels, w.K)
            inside = comm[G.src] == comm[G.dst]
            rec[con] = {}
            for eta in ETAS:
                mu = np.where(inside, G.mu ** (1.0 / (1.0 + eta)), G.mu ** (1.0 + eta))
                rec[con][f"{eta:g}"] = record(FuzzyGraph(G.n, G.src, G.dst, mu), comm, con, rng)
        rec["time_s"] = time.perf_counter() - t0
        res["worlds"][key] = rec
        json.dump(res, open(out, "w"), indent=1, default=float)
        print(key, f"{rec['time_s']:.0f}s", {c: [round(rec[c][e]["V_partition"], 3) for e in rec[c]] for c in rec if c != "time_s"}, flush=True)


if __name__ == "__main__":
    main()
