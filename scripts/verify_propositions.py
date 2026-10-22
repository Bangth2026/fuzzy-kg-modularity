#!/usr/bin/env python3
"""Numerical verification of Propositions 1-3, Corollary 1, Remark 3 and Fact 1.

Offline by default (synthetic graphs); pass --graph edges.csv to run the same
checks on a real fuzzy graph saved by the construction scripts (columns
src,dst,mu with integer node ids).

Checks
  Proposition 2(i)   -gamma <= Q_F <= 1 on random and optimised partitions
  Proposition 2(ii)  Q_F is unchanged by contracting a partition into super-nodes
  Proposition 2(iii) Q_F non-decreasing along the optimisation
  Proposition 2(v)   crisp memberships, gamma = 1 -> leidenalg's directed modularity
  Proposition 1      every community weakly connected in the support graph
  Proposition 3      Q_F >= alpha-average of crisp modularities; equality iff crisp
  Corollary 1        gap = gamma * sum Cov(f_k, g_k) <= Cauchy-Schwarz <= AM-GM bound
  Remark 3           a bipartition beats the trivial partition iff gamma > threshold
  Fact 1             no t-norm composition exceeds the weakest link
  Section 6.2        Yager / Schweizer-Sklar identities on a 100 x 100 grid
"""

import argparse

import numpy as np
import pandas as pd

from fuzzykg import FuzzyGraph
from fuzzykg.leiden import check_weak_connectivity, fuzzy_leiden
from fuzzykg.modularity import bipartition_threshold, correlation_gap, fuzzy_modularity
from fuzzykg.resolution import spectral_bisection
from fuzzykg.synthetic import synthetic_fuzzy_graph
from fuzzykg.tnorms import (
    STANDARD_TNORMS,
    compose_path,
    schweizer_sklar,
    t_lukasiewicz,
    t_min,
    t_product,
    yager,
)


def contracted(G: FuzzyGraph, memb):
    """G_F / Pi of Proposition 2(ii): communities become nodes with self-loops."""
    memb = np.asarray(memb)
    K = memb.max() + 1
    A = np.zeros((K, K))
    np.add.at(A, (memb[G.src], memb[G.dst]), G.mu)
    return A


def contracted_modularity(A, memb2, gamma):
    """Q_F on a contracted graph with self-loops, Eq. (8) evaluated directly."""
    W = A.sum()
    memb2 = np.asarray(memb2)
    K2 = memb2.max() + 1
    Wk = np.zeros(K2); So = np.zeros(K2); Si = np.zeros(K2)
    for i in range(A.shape[0]):
        So[memb2[i]] += A[i].sum(); Si[memb2[i]] += A[:, i].sum()
        for j in range(A.shape[0]):
            if memb2[i] == memb2[j]:
                Wk[memb2[i]] += A[i, j]
    return float(np.sum(Wk / W - gamma * So * Si / W**2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=str, default=None, help="edges.csv with src,dst,mu")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.graph:
        df = pd.read_csv(args.graph)
        G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
    else:
        G = synthetic_fuzzy_graph(500, 0.7, seed=args.seed)
    print(G.summary())
    gamma = args.gamma
    fails = 0

    def report(name, ok, detail=""):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")

    # Proposition 2(i): boundedness on random partitions
    worst_lo, worst_hi = 0.0, 0.0
    for K in (1, 2, 5, 20, G.n):
        for _ in range(20):
            memb = rng.integers(0, K, size=G.n)
            q = fuzzy_modularity(G, memb, gamma)
            worst_lo = min(worst_lo, q); worst_hi = max(worst_hi, q)
    report("Prop 2(i) boundedness on random partitions", worst_lo >= -gamma - 1e-12 and worst_hi <= 1 + 1e-12,
           f"range [{worst_lo:.3f}, {worst_hi:.3f}] within [-{gamma}, 1]")

    # Leiden
    r = fuzzy_leiden(G, gamma, seed=args.seed, n_iterations=12)
    report("Prop 1 weak connectivity of every community", r.connected_violations == 0,
           f"{r.n_communities} communities, Q_F = {r.Q_F:.4f}")
    report("Prop 2(iii) monotone Q_F along optimisation", r.monotone_violations == 0,
           f"min successive difference {np.diff(r.trajectory).min():.2e}")
    report("Prop 2(i) boundedness of optimised Q_F", -gamma <= r.Q_F <= 1)

    # Proposition 2(ii): aggregation-invariance
    A = contracted(G, r.membership)
    memb2 = rng.integers(0, 3, size=A.shape[0])
    induced = memb2[r.membership]
    q_orig = fuzzy_modularity(G, induced, gamma)
    q_contr = contracted_modularity(A, memb2, gamma)
    report("Prop 2(ii) aggregation-invariance", abs(q_orig - q_contr) < 1e-10, f"|diff| = {abs(q_orig - q_contr):.1e}")

    # Proposition 2(v): reduction to directed modularity for crisp memberships
    Gc = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
    import leidenalg as la
    g = Gc.to_igraph()
    part = la.RBConfigurationVertexPartition(g, weights="weight", resolution_parameter=1.0,
                                             initial_membership=r.membership.tolist())
    report("Prop 2(v) reduction to Leicht-Newman directed modularity (crisp, gamma=1)",
           abs(part.quality() / Gc.W - fuzzy_modularity(Gc, r.membership, 1.0)) < 1e-10)

    # Proposition 3 and Corollary 1
    cg = correlation_gap(G, r.membership, gamma)
    report("Prop 3 Q_F >= alpha-average", cg["gap"] >= -1e-12, f"Q_F {cg['Q_F']:.4f}, avg {cg['alpha_average']:.4f}, gap {cg['gap']:.2e}")
    report("Cor 1 gap == gamma * sum Cov", abs(cg["gap"] - cg["gamma_sum_cov"]) < 1e-9)
    report("Cor 1 gap <= Cauchy-Schwarz <= AM-GM", cg["gap"] <= cg["cauchy_schwarz_bound"] + 1e-12 <= cg["am_gm_bound"] + 1e-12,
           f"{cg['gap']:.3e} <= {cg['cauchy_schwarz_bound']:.3e} <= {cg['am_gm_bound']:.3e}")
    cgc = correlation_gap(Gc, r.membership, gamma)
    report("Prop 3 equality in the crisp case", abs(cgc["gap"]) < 1e-12, f"gap {cgc['gap']:.1e}")
    # fuzzy, non-crisp graph should have strictly positive gap on a non-trivial partition
    report("Prop 3 strict inequality on a non-crisp graph", cg["gap"] > 1e-9 or G.mu.min() == G.mu.max())

    # Remark 3
    B = spectral_bisection(G)
    th = bipartition_threshold(G, B)
    q_triv = fuzzy_modularity(G, np.zeros(G.n, dtype=int), 1.0)
    ok = True
    for gam in (th["gamma_threshold"] * 0.9, th["gamma_threshold"] * 1.1):
        q_bi = fuzzy_modularity(G, B, gam)
        improves = q_bi > 1 - gam
        ok &= improves == (gam > th["gamma_threshold"])
    report("Remark 3 bipartition improves iff gamma > threshold", ok,
           f"c/W = {th['c_over_W']:.3f}, threshold {th['gamma_threshold']:.3f}, trivial Q_F(gamma=1) = {q_triv:.3f}")

    # Fact 1
    paths = [rng.uniform(0.05, 1.0, size=rng.integers(2, 6)) for _ in range(2000)]
    T_all = dict(STANDARD_TNORMS)
    for p in (1.5, 2, 3, 5, 10):
        T_all[f"yager_{p}"] = yager(p)
    for p in (-8, -4, -2, -1, 0.5, 1):
        T_all[f"ss_{p}"] = schweizer_sklar(p)
    viol = {k: sum(compose_path(ms, T) > min(ms) + 1e-12 for ms in paths) for k, T in T_all.items()}
    report("Fact 1 no t-norm exceeds the weakest link", sum(viol.values()) == 0,
           f"{len(paths)} paths x {len(T_all)} operators, violations {sum(viol.values())}")

    # Section 6.2 identities
    gr = np.linspace(0.01, 1.0, 100); X, Y = np.meshgrid(gr, gr)
    d1 = np.abs(yager(1)(X, Y) - t_lukasiewicz(X, Y)).max()
    d2 = np.abs(schweizer_sklar(1)(X, Y) - t_lukasiewicz(X, Y)).max()
    d3 = np.abs(schweizer_sklar(1e-9)(X, Y) - t_product(X, Y)).max()
    d4 = np.abs(yager(200)(X, Y) - t_min(X, Y)).max()   # (1-a)^p underflows for very large p
    report("Sec 6.2 Yager(1) = Lukasiewicz", d1 < 1e-12, f"max dev {d1:.1e}")
    report("Sec 6.2 SS(1) = Lukasiewicz", d2 < 1e-12, f"max dev {d2:.1e}")
    report("Sec 6.2 SS(p->0) -> product", d3 < 1e-6, f"max dev at p=1e-9: {d3:.1e}")
    report("Sec 6.2 Yager(p->inf) -> min", d4 < 5e-2, f"max dev at p=200: {d4:.1e} (bounded by 2^(1/p)-1 plus underflow)")

    print(f"\n{fails} failure(s)")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
