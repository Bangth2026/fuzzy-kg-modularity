#!/usr/bin/env python3
"""Release 5.0.0. Numerical checks and costs behind Sections 4 and 9 (Supplementary Section S17).

1. Proposition 1: Q_F from the closed form (6) against Q_F computed directly on the rescaled
   memberships phi_s(mu_E), for random slope vectors and partitions on four graphs.
2. The search for kappa*: on every cut of results/v15/simplex_real.json, 20,000 random slope
   vectors in B_{0.99 kappa*} (no reversal expected) and in B_{1.25 kappa*} (a reversal expected
   when kappa* is finite), at the minimum excess used for the decision.
3. Cost on Junyi co-practice at gamma = 1: the 80 Leiden runs of step 3 (20 seeds, support and
   three cuts), the closed-form evaluation of 10^6 rescalings, and the kappa* search.
Writes results/v15/checks.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden  # noqa: E402
from fuzzykg.simplex import CutFamily, kappa_star, segment_basis  # noqa: E402
from v15_partitions import load, GRAPHS  # noqa: E402

CUTS = ("q0.25", "q0.50", "q0.75")


def delta_for(name):
    s = json.load(open("results/v15/summary.json"))["delta"]
    return s["dense"] if not (name.endswith("transition") or name.startswith("mooccubex")) else s["sparse"]


def main():
    out = {}
    rng = np.random.default_rng(1)
    # 1. Proposition 1
    errs = []
    for name in ("junyi_transition", "assist2009_kc", "eedi_copractice", "mooccubex_cs_full"):
        G = load(name); knots, ell = segment_basis(G.mu)
        P = [_leiden(G, s) for s in range(3)] + [_leiden(G.alpha_cut(float(np.quantile(G.mu, 0.5))), 0)]
        fam = CutFamily(G, {"p": P})
        for _ in range(50):
            s = rng.gamma(0.5, 1.0, fam.J) + 1e-6
            w = ell @ s
            Gs = FuzzyGraph(G.n, G.src, G.dst, w)
            direct = np.array([fuzzy_modularity(Gs, p) for p in P])
            errs.append(float(np.max(np.abs(fam.q("p", s[None, :])[0] - direct))))
    out["proposition1_max_abs_error"] = float(max(errs)); out["proposition1_evaluations"] = len(errs)
    print("prop1", out["proposition1_max_abs_error"], flush=True)
    # 2. kappa* search check
    R = json.load(open("results/v15/simplex_real.json"))["graphs"]
    viol_in, rev_out, finite = 0, 0, 0
    for name in GRAPHS:
        store = dict(np.load(f"results/v15/partitions/{name}.npz")); G = load(name)
        fam = CutFamily(G, {w: [store[f"{w}|1|{s}"] for s in range(20)] for w in ("support", *CUTS)})
        d = delta_for(name)
        for c in CUTS:
            k = R[name]["gammas"]["1"][c]["by_delta"][f"{d:g}"]["kappa_star"]
            _, m0 = fam.margin(c, np.ones((1, fam.J)), d); fl = m0[0] > 0
            kk = min(k, 1e3)
            S = 1 + (0.99 * kk - 1) * rng.random((20000, fam.J))
            _, mg = fam.margin(c, S, d)
            viol_in += int(np.any((mg > 0) != fl))
            if np.isfinite(k):
                finite += 1
                S = 1 + (1.25 * k - 1) * rng.random((20000, fam.J)); S = np.vstack([S, np.array(np.meshgrid(*[[1, 1.25 * k]] * fam.J)).reshape(fam.J, -1).T])
                _, mg = fam.margin(c, S, d)
                rev_out += int(np.any((mg > 0) != fl))
    out["kappa_search"] = {"cuts": 3 * len(GRAPHS), "reversals_inside_0.99kappa": viol_in, "finite_kappa": finite,
                           "reversal_found_at_1.25kappa": rev_out}
    print(out["kappa_search"], flush=True)
    # 3. cost on Junyi co-practice
    G = load("junyi_copractice"); Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
    t = time.perf_counter(); parts = {"support": [_leiden(Gs, s) for s in range(20)]}
    for q in (0.25, 0.5, 0.75):
        parts[f"q{q:.2f}"] = [_leiden(G.alpha_cut(float(np.quantile(G.mu, q))), s) for s in range(20)]
    out["junyi_copractice_step3_partitions_s"] = time.perf_counter() - t
    t = time.perf_counter(); fam = CutFamily(G, parts); out["junyi_copractice_stats_s"] = time.perf_counter() - t
    S = 1 + 9 * rng.random((10 ** 6, fam.J))
    t = time.perf_counter()
    for i in range(0, len(S), 100000):
        fam.margin("q0.50", S[i:i + 100000])
    out["junyi_copractice_1e6_rescalings_s"] = time.perf_counter() - t
    t = time.perf_counter(); [kappa_star(fam, c, min_excess=delta_for("junyi_copractice"), kmax=1000.0) for c in CUTS]
    out["junyi_copractice_kappa_search_s"] = time.perf_counter() - t
    t = time.perf_counter(); _leiden(G, 0); out["junyi_copractice_one_leiden_on_QF_s"] = time.perf_counter() - t
    print(out, flush=True)
    json.dump(out, open("results/v15/checks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
