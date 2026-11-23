"""Unit-robust cut audit (release 5.0.0; refined knots and the divergence check added in 5.1.0; Section 4 of the manuscript).

A strictly increasing rescaling phi of the memberships that is piecewise linear with knots
t_0 = 0 < t_1 < ... < t_J (the minimum membership and the quantiles of mu_E at k/L) is fixed by its
slopes s_j > 0 on the J segments, and

    phi(mu(e)) = sum_j s_j * ell_j(e),   ell_j(e) = clip(mu(e) - t_{j-1}, 0, t_j - t_{j-1}),

so every quantity of Q_F that is linear in the weights is linear in s. For a fixed partition Pi,

    Q_F[phi o mu](Pi) = (s . a) / (s . m) - gamma * s' M s / (s . m)^2,

with a_j the ell_j-mass internal to Pi, m_j the total ell_j-mass and M = X'Y, X (Y) the
community-by-segment ell-mass of out- (in-) edges. s = 1 is mu_E itself; the scale of s is
irrelevant (Q_F is invariant to a constant factor); the vertex s = e_1 is the support graph and
the vertices of higher segments approach the alpha-cuts (Lemma 2 of the manuscript writes the
same family on the quantile grid).

The cut audit of step 3 compares the crisp partition of the support graph with the crisp
partition of a level set. Neither partition depends on the unit of mu_E, so its excess

    E(s) = mean_seeds Q_F[phi_s o mu](Pi_support) - mean_seeds Q_F[phi_s o mu](Pi_cut)

is a closed-form function of s, and the flag rule of release 4.x (excess above twice the larger
seed SD and above ``min_excess``) can be evaluated over a whole family of units at no further
optimisation cost. ``kappa_star`` is the largest slope ratio max(s)/min(s) under which the
decision taken on mu_E holds for every unit in the box {s : max s / min s <= kappa}.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, Sequence

import numpy as np

from .graph import FuzzyGraph


def segment_basis(mu: np.ndarray, L: int = 8):
    """Knots and the m x J matrix ell of segment lengths covered by each membership."""
    q = np.quantile(mu, np.arange(L) / L)
    knots = np.unique(np.concatenate([[0.0], q, [mu.max()]]))
    lo, hi = knots[:-1], knots[1:]
    ell = np.clip(mu[:, None] - lo[None, :], 0.0, (hi - lo)[None, :])
    return knots, ell


def partition_stats(ell: np.ndarray, src: np.ndarray, dst: np.ndarray, membership: np.ndarray):
    """(a, M) of one partition: internal segment mass and X'Y."""
    p = np.asarray(membership)
    k = int(p.max()) + 1
    internal = p[src] == p[dst]
    a = ell[internal].sum(0)
    X = np.zeros((k, ell.shape[1])); Y = np.zeros((k, ell.shape[1]))
    np.add.at(X, p[src], ell); np.add.at(Y, p[dst], ell)
    return a, X.T @ Y


def q_of(S: np.ndarray, a: np.ndarray, M: np.ndarray, m: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Q_F of a set of partitions (a: P x J, M: P x J x J) under a set of units S (n x J).
    Returns an n x P array."""
    S = np.atleast_2d(S)
    W = S @ m                                   # n
    lin = (S @ a.T) / W[:, None]                # n x P
    quad = np.einsum("ni,pij,nj->np", S, M, S) / (W ** 2)[:, None]
    return lin - gamma * quad


class CutFamily:
    """Precomputed statistics of the support and cut partitions of one graph at one resolution."""

    def __init__(self, G: FuzzyGraph, parts: Dict[str, Sequence[np.ndarray]], L: int = 8, gamma: float = 1.0):
        self.knots, self.ell = segment_basis(G.mu, L)
        self.m = self.ell.sum(0)
        self.gamma = gamma
        self.stats = {}
        for name, plist in parts.items():
            A, Ms = zip(*[partition_stats(self.ell, G.src, G.dst, p) for p in plist])
            self.stats[name] = (np.array(A), np.array(Ms))
        self.J = self.ell.shape[1]

    def q(self, name: str, S: np.ndarray) -> np.ndarray:
        a, M = self.stats[name]
        return q_of(S, a, M, self.m, self.gamma)

    def margin(self, cut: str, S: np.ndarray, min_excess: float = 0.01):
        """Excess of the cut over the support partition, and its margin over the flag threshold."""
        qs, qc = self.q("support", S), self.q(cut, S)
        ex = qs.mean(1) - qc.mean(1)
        sd = np.maximum(qs.std(1, ddof=1), qc.std(1, ddof=1))
        thr = np.maximum(2 * sd, min_excess)
        return ex, ex - thr


def _box_extreme(fam: CutFamily, cut: str, kappa: float, sign: int, min_excess: float, rng, n_rand: int = 400):
    """min (sign = +1) or max (sign = -1) of the margin over the box s in [1, kappa]^J."""
    from scipy.optimize import minimize
    J = fam.J
    V = np.array(list(product([1.0, kappa], repeat=J))) if J <= 12 else np.vstack([np.ones(J), kappa * np.ones(J)])  # 5.2.0: J <= 12 (refined family)
    R = 1.0 + (kappa - 1.0) * rng.random((n_rand, J))
    S = np.vstack([V, R, np.ones((1, J))])
    _, mg = fam.margin(cut, S, min_excess)
    vals = sign * mg
    best = float(vals.min()); arg = S[int(vals.argmin())]
    for i in np.argsort(vals)[:3]:
        f = lambda x: float(sign * fam.margin(cut, x[None, :], min_excess)[1][0])
        r = minimize(f, S[i], method="L-BFGS-B", bounds=[(1.0, kappa)] * J, options={"maxiter": 200})
        if r.fun < best:
            best, arg = float(r.fun), r.x
    return sign * best, arg


def kappa_star(fam: CutFamily, cut: str, min_excess: float = 0.01, kmax: float = 1e3, tol: float = 0.01, seed: int = 0):
    """Decision on mu_E and the largest slope ratio under which it holds for every unit."""
    rng = np.random.default_rng(seed)
    _, m0 = fam.margin(cut, np.ones((1, fam.J)), min_excess)
    flagged = bool(m0[0] > 0)
    sign = 1 if flagged else -1          # flagged: need min margin > 0; cut: need max margin <= 0

    def holds(k):
        v, _ = _box_extreme(fam, cut, k, sign, min_excess, rng)
        return v > 0 if flagged else v <= 0

    if holds(kmax):
        return {"decision_mu": "flagged" if flagged else "cut", "margin_mu": float(m0[0]), "kappa_star": float("inf")}
    lo, hi = 0.0, np.log(kmax)
    if not holds(np.exp(1e-6)):
        return {"decision_mu": "flagged" if flagged else "cut", "margin_mu": float(m0[0]), "kappa_star": 1.0}
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if holds(np.exp(mid)) else (lo, mid)
    return {"decision_mu": "flagged" if flagged else "cut", "margin_mu": float(m0[0]), "kappa_star": float(np.exp(lo))}


def simplex_extremes(fam: CutFamily, cut: str, min_excess: float = 0.01, n_rand: int = 3000, seed: int = 0):
    """Range of the excess and of the margin over every unit of the family (the open simplex),
    approached through the vertices and random Dirichlet points."""
    rng = np.random.default_rng(seed)
    J = fam.J
    eps = 1e-9
    V = np.eye(J) * (1 - eps * J) + eps
    S = np.vstack([V, rng.dirichlet(np.ones(J), n_rand), rng.dirichlet(0.2 * np.ones(J), n_rand)])
    ex, mg = fam.margin(cut, S, min_excess)
    return {"excess_min": float(ex.min()), "excess_max": float(ex.max()),
            "margin_min": float(mg.min()), "margin_max": float(mg.max()),
            "argmin_vertex": int(np.argmin(ex[:J])), "argmax_vertex": int(np.argmax(ex[:J])),
            "vertex_excess": ex[:J].tolist()}


def refined_knots(mu: np.ndarray, levels: Sequence[float], L: int = 8):
    """Knots of Section 4.2 plus, for each cut level, the largest membership below it (Corollary 1):
    the vertex of the segment ending at a cut level is then the cut's indicator."""
    knots, _ = segment_basis(mu, L)
    extra = [float(mu[mu < a].max()) for a in levels if (mu < a).any()]
    knots = np.unique(np.concatenate([knots, extra]))
    lo, hi = knots[:-1], knots[1:]
    return knots, np.clip(mu[:, None] - lo[None, :], 0.0, (hi - lo)[None, :])


def divergence(support_parts: Sequence[np.ndarray], graded_parts: Sequence[np.ndarray]) -> Dict[str, float]:
    """Divergence D of the graded partition from the support partition (Section 4.5, Eq. 7):
    min(seed-to-seed ARI of each) - mean ARI between them over matched seeds (cyclic seed pairs)."""
    from sklearn.metrics import adjusted_rand_score as ari
    S, F = list(support_parts), list(graded_parts); n = min(len(S), len(F))
    fs = float(np.mean([ari(F[i], S[i]) for i in range(n)]))
    ss = float(np.mean([ari(S[i], S[(i + 1) % n]) for i in range(n)]))
    ff = float(np.mean([ari(F[i], F[(i + 1) % n]) for i in range(n)]))
    return {"a_FS": fs, "a_SS": ss, "a_FF": ff, "D": min(ss, ff) - fs}
