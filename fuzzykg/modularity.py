"""Fuzzy-graph modularity and its relation to the alpha-cuts (Section 4).

Q_F(Pi)   Eq. (7)/(8): directed form of Leicht-Newman with memberships as weights.
Q~(alpha) Eq. (9):     alpha-cut modularity under *global* normalisation by W.
Proposition 3:  Q_F >= int_0^1 Q~(alpha) d alpha, equality iff crisp (a.e.).
Corollary 1:    the gap equals gamma * sum_k Cov(f_k, g_k).
Remark 3:       a bipartition beats the trivial partition iff gamma > (c/W)/(a1 b2 + a2 b1).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .graph import FuzzyGraph


def _community_sums(G: FuzzyGraph, membership: Sequence[int]):
    memb = np.asarray(membership, dtype=np.int64)
    K = int(memb.max()) + 1
    internal = memb[G.src] == memb[G.dst]
    W_k = np.bincount(memb[G.src[internal]], weights=G.mu[internal], minlength=K)
    S_out = np.bincount(memb, weights=G.out_strength(), minlength=K)
    S_in = np.bincount(memb, weights=G.in_strength(), minlength=K)
    return W_k, S_out, S_in


def fuzzy_modularity(G: FuzzyGraph, membership: Sequence[int], gamma: float = 1.0) -> float:
    """Q_F of Eq. (8): sum_k [ W_k / W - gamma * S_k^out S_k^in / W^2 ]."""
    W = G.W
    W_k, S_out, S_in = _community_sums(G, membership)
    return float(np.sum(W_k / W - gamma * S_out * S_in / W**2))


def alpha_cut_modularity(G: FuzzyGraph, membership: Sequence[int], alpha: float, gamma: float = 1.0) -> float:
    """Q~(alpha; Pi) of Eq. (9): alpha-cut counts normalised by the *fuzzy* mass W."""
    W = G.W
    memb = np.asarray(membership, dtype=np.int64)
    K = int(memb.max()) + 1
    keep = G.mu >= alpha
    s, d = G.src[keep], G.dst[keep]
    internal = memb[s] == memb[d]
    E_in = np.bincount(memb[s[internal]], minlength=K).astype(float)
    deg_out = np.bincount(s, minlength=G.n).astype(float)
    deg_in = np.bincount(d, minlength=G.n).astype(float)
    S_out = np.bincount(memb, weights=deg_out, minlength=K)
    S_in = np.bincount(memb, weights=deg_in, minlength=K)
    return float(np.sum(E_in / W - gamma * S_out * S_in / W**2))


def _strength_profiles(G: FuzzyGraph, membership: Sequence[int], grid: np.ndarray):
    """f_k(alpha) = S_k^out(alpha)/W and g_k(alpha) = S_k^in(alpha)/W on a grid."""
    memb = np.asarray(membership, dtype=np.int64)
    K = int(memb.max()) + 1
    W = G.W
    f = np.zeros((K, grid.size))
    g = np.zeros((K, grid.size))
    for j, a in enumerate(grid):
        keep = G.mu >= a
        deg_out = np.bincount(G.src[keep], minlength=G.n).astype(float)
        deg_in = np.bincount(G.dst[keep], minlength=G.n).astype(float)
        f[:, j] = np.bincount(memb, weights=deg_out, minlength=K) / W
        g[:, j] = np.bincount(memb, weights=deg_in, minlength=K) / W
    return f, g


def alpha_average_modularity(G: FuzzyGraph, membership: Sequence[int], gamma: float = 1.0) -> float:
    """int_0^1 Q~(alpha; Pi) d alpha, computed *exactly*.

    Q~ is piecewise constant in alpha with breakpoints at the distinct
    memberships, so the integral is a finite sum over the sorted levels.
    """
    levels = np.unique(G.mu)
    # on (levels[i-1], levels[i]] the cut equals E_{levels[i]}
    prev = 0.0
    total = 0.0
    for a in levels:
        total += (a - prev) * alpha_cut_modularity(G, membership, a, gamma)
        prev = a
    return float(total)


def correlation_gap(G: FuzzyGraph, membership: Sequence[int], gamma: float = 1.0) -> Dict[str, float]:
    """Corollary 1: gap = gamma * sum_k Cov(f_k, g_k) with its two upper bounds.

    Returns Q_F, the exact alpha-average, their difference, the covariance
    expression (which must equal the difference), and the Cauchy-Schwarz and
    AM-GM bounds. Exact integration uses the step structure of the profiles.
    """
    levels = np.unique(G.mu)
    # Profiles are right-continuous step functions on (0,1] with breakpoints at
    # the levels; integrate exactly by the same partition as above.
    edges = np.concatenate([[0.0], levels])
    widths = np.diff(edges)
    f, g = _strength_profiles(G, membership, levels)
    Ef = f @ widths
    Eg = g @ widths
    Efg = (f * g) @ widths
    Eff = (f * f) @ widths
    Egg = (g * g) @ widths
    cov = Efg - Ef * Eg
    var_f = Eff - Ef**2
    var_g = Egg - Eg**2
    QF = fuzzy_modularity(G, membership, gamma)
    Qbar = alpha_average_modularity(G, membership, gamma)
    return {
        "Q_F": QF,
        "alpha_average": Qbar,
        "gap": QF - Qbar,
        "gamma_sum_cov": float(gamma * cov.sum()),
        "cauchy_schwarz_bound": float(gamma * np.sum(np.sqrt(np.clip(var_f, 0, None) * np.clip(var_g, 0, None)))),
        "am_gm_bound": float(gamma / 2 * np.sum(var_f + var_g)),
    }


def bipartition_threshold(G: FuzzyGraph, membership: Sequence[int]) -> Dict[str, float]:
    """Remark 3 for a given bipartition: c/W, a_i, b_i and the gamma threshold."""
    memb = np.asarray(membership, dtype=np.int64)
    if memb.max() != 1:
        raise ValueError("membership must describe a bipartition {0,1}")
    W = G.W
    cross = memb[G.src] != memb[G.dst]
    c = float(G.mu[cross].sum())
    _, S_out, S_in = _community_sums(G, memb)
    a1, a2 = S_out / W
    b1, b2 = S_in / W
    denom = a1 * b2 + a2 * b1
    return {
        "c_over_W": c / W,
        "two_c_over_W": 2 * c / W,
        "gamma_threshold": (c / W) / denom if denom > 0 else np.inf,
    }


def bipartition_threshold_general(G: FuzzyGraph, candidates: Sequence[Sequence[int]]) -> Dict[str, object]:
    """Revised Proposition 1: the onset of structure over a family of bipartitions.

    For a bipartition B with cross mass c(B) and out/in strength shares
    (a_1, b_1), (a_2, b_2), the bipartition beats the trivial partition iff
    gamma > theta(B) = (c(B)/W) / (a_1 b_2 + a_2 b_1).  For a strength-balanced
    split (a_i = b_i = 1/2) theta(B) = 2c(B)/W, which is therefore only the
    threshold of balanced cuts; an unbalanced, weakly attached block can have a
    much smaller theta.  Returns theta for every candidate, the minimising
    candidate, and the balanced-cut value for comparison.
    """
    rows = []
    for i, memb in enumerate(candidates):
        memb = np.asarray(memb, dtype=np.int64)
        if memb.min() != 0 or memb.max() != 1:
            continue
        t = bipartition_threshold(G, memb)
        rows.append({"candidate": i, "theta": t["gamma_threshold"],
                     "two_c_over_W": t["two_c_over_W"], "c_over_W": t["c_over_W"]})
    if not rows:
        raise ValueError("no valid bipartition among candidates")
    best = min(rows, key=lambda r: r["theta"])
    return {"thresholds": rows, "min_theta": best["theta"], "argmin": best["candidate"]}


def community_vs_rest_candidates(membership: Sequence[int]) -> list:
    """Bipartitions 'one community vs the rest' for every community of a partition."""
    memb = np.asarray(membership, dtype=np.int64)
    return [(memb == c).astype(np.int64) for c in np.unique(memb)]
