#!/usr/bin/env python3
"""Numerical check of the finite identity of Lemma 2 (Section 5.4).

For a grid alpha_1 < ... < alpha_L with non-empty level sets E_l and weights
w_l proportional to c_l m_l (c_l > 0), the level-set modularity equals

    Qbar_A(Pi) = Q_F[mu_c](Pi) - gamma * sum_kappa Cov_p(x_kappa, y_kappa),

where mu_c(e) = sum_{l : alpha_l <= mu(e)} c_l, p_l = c_l m_l / sum_r c_r m_r,
and x_kappa(l), y_kappa(l) are the shares of the level-l out- and in-degree
that fall in kappa. Checked for (a) the full grid at the distinct membership
values with c_l = Delta_l, where mu_c = mu; (b) the quantile grid with the
mass weighting, c_l = 1, where mu_c is the grid rank N_A(e)/L up to a constant;
(c) the quantile grid with equal weights, c_l = 1/m_l.

Usage: python scripts/check_levelset_identity.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_modularity, fuzzy_leiden  # noqa: E402
from fuzzykg.multislice import level_grid, qbar  # noqa: E402
from fuzzykg.synthetic import planted_partition_fuzzy_graph  # noqa: E402


def cov_term(G, memb, alphas, c):
    memb = np.asarray(memb); K = memb.max() + 1
    m = np.array([(G.mu >= a).sum() for a in alphas], float)
    p = c * m / (c * m).sum()
    tot = 0.0
    for kap in range(K):
        x = np.array([((G.mu >= a) & (memb[G.src] == kap)).sum() for a in alphas]) / m
        y = np.array([((G.mu >= a) & (memb[G.dst] == kap)).sum() for a in alphas]) / m
        tot += (p * x * y).sum() - (p * x).sum() * (p * y).sum()
    return tot


def lhs_general(G, memb, alphas, c):
    m = np.array([(G.mu >= a).sum() for a in alphas], float)
    w = c * m / (c * m).sum()
    return float(sum(wl * fuzzy_modularity(G.alpha_cut(a), memb, 1.0) for wl, a in zip(w, alphas)))


def rhs(G, memb, alphas, c):
    muc = np.array([c[alphas <= v].sum() for v in G.mu])
    Gc = FuzzyGraph(G.n, G.src, G.dst, muc)
    return fuzzy_modularity(Gc, memb, 1.0) - cov_term(G, memb, alphas, c)


def main():
    out = {}
    for seed in range(3):
        G, truth = planted_partition_fuzzy_graph(300, 6, 6.0, 0.5, 0.5, 0.15, 0.5, seed=seed)
        G = FuzzyGraph(G.n, G.src, G.dst, np.round(G.mu, 2))  # few distinct values
        f = fuzzy_leiden(G, 1.0, seed=0, n_iterations=0).membership
        for name, memb in (("truth", np.asarray(truth)), ("leiden_qf", np.asarray(f))):
            v = np.unique(G.mu); d = np.diff(np.r_[0.0, v])
            full = abs(lhs_general(G, memb, v, d) - rhs(G, memb, v, d))
            full_qf = abs((fuzzy_modularity(G, memb, 1.0) - cov_term(G, memb, v, d)) - lhs_general(G, memb, v, d))
            a = level_grid(G); one = np.ones(a.size)
            mass = abs(qbar(G, memb, a) - rhs(G, memb, a, one))
            ma = np.array([(G.mu >= x).sum() for x in a], float)
            eq = abs(qbar(G, memb, a, weighting="equal") - rhs(G, memb, a, 1.0 / ma))
            out[f"seed{seed}_{name}"] = {"full_grid": full, "full_grid_vs_QF": full_qf, "quantile_mass": mass, "quantile_equal": eq,
                                         "gap_full": float(cov_term(G, memb, v, d))}
    print(json.dumps(out, indent=1))
    worst = max(max(r[k] for k in ("full_grid", "full_grid_vs_QF", "quantile_mass", "quantile_equal")) for r in out.values())
    print("largest absolute error", worst)
    os.makedirs("results", exist_ok=True)
    json.dump({"cases": out, "largest_abs_error": worst}, open("results/levelset_identity_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
