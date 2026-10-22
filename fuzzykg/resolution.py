"""Algorithm 1: adaptive resolution selection from the normalised cut (Remark 3).

Input:  fuzzy graph G_F; multipliers rho1 >= 1, rho2 > 1 (defaults 1.25, 2);
        seeds S; grid factor g = 2; stability threshold theta = 0.8
1:  B <- strength-balanced spectral bisection of the symmetrised adjacency
2:  c/W <- inter-block fuzzy mass of B divided by the total fuzzy mass W
3:  gamma1 <- rho1 * (2c/W);  gamma2 <- rho2 * gamma1
4:  repeat
5:      run Fuzzy Leiden at gamma1 once per seed
6:      if every run is non-trivial and pairwise ARI >= theta: return
8:      gamma1 <- g*gamma1; gamma2 <- g*gamma2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.special import comb

from .graph import FuzzyGraph
from .leiden import fuzzy_leiden


def adjusted_rand_index(a: Sequence[int], b: Sequence[int]) -> float:
    a = np.asarray(a); b = np.asarray(b)
    ia, a_inv = np.unique(a, return_inverse=True)
    ib, b_inv = np.unique(b, return_inverse=True)
    cont = sp.coo_matrix((np.ones(a.size), (a_inv, b_inv)), shape=(ia.size, ib.size)).toarray()
    sum_comb = comb(cont, 2).sum()
    sum_a = comb(cont.sum(axis=1), 2).sum()
    sum_b = comb(cont.sum(axis=0), 2).sum()
    n_comb = comb(a.size, 2)
    expected = sum_a * sum_b / n_comb
    max_index = 0.5 * (sum_a + sum_b)
    if max_index == expected:
        return 1.0
    return float((sum_comb - expected) / (max_index - expected))


def spectral_bisection(G: FuzzyGraph) -> np.ndarray:
    """Strength-balanced bisection by the Fiedler vector of the symmetrised
    normalised Laplacian, split at the strength-weighted median."""
    A = G.symmetrised().astype(float)
    d = np.asarray(A.sum(axis=1)).ravel()
    d_safe = np.where(d > 0, d, 1.0)
    Dm = sp.diags(1.0 / np.sqrt(d_safe))
    L = sp.identity(G.n) - Dm @ A @ Dm
    k = min(3, G.n - 1)
    vals, vecs = eigsh(L, k=k, which="SM")
    order = np.argsort(vals)
    fiedler = Dm @ vecs[:, order[1]]
    # strength-balanced split: sort by Fiedler value, cut where cumulative
    # strength reaches half of the total
    idx = np.argsort(fiedler)
    cum = np.cumsum(d[idx])
    cut = int(np.searchsorted(cum, cum[-1] / 2.0))
    memb = np.zeros(G.n, dtype=np.int64)
    memb[idx[cut + 1:]] = 1
    if memb.min() == memb.max():            # degenerate; force a split
        memb[idx[G.n // 2:]] = 1
    return memb


@dataclass
class ResolutionResult:
    gamma1: float
    gamma2: float
    c_over_W: float
    doublings: int
    history: List[dict] = field(default_factory=list)


def adaptive_resolution(
    G: FuzzyGraph,
    rho1: float = 1.25,
    rho2: float = 2.0,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    grid_factor: float = 2.0,
    theta: float = 0.8,
    max_doublings: int = 12,
) -> ResolutionResult:
    B = spectral_bisection(G)
    cross = B[G.src] != B[G.dst]
    c_over_W = float(G.mu[cross].sum()) / G.W
    g1 = rho1 * 2.0 * c_over_W
    g2 = rho2 * g1
    history = []
    for k in range(max_doublings + 1):
        runs = [fuzzy_leiden(G, g1, seed=s, n_iterations=0) for s in seeds]
        nontrivial = all(r.n_communities > 1 for r in runs)
        aris = [adjusted_rand_index(x.membership, y.membership) for x, y in combinations(runs, 2)]
        min_ari = min(aris) if aris else 1.0
        history.append({"gamma1": g1, "nontrivial": nontrivial, "min_ARI": min_ari,
                        "Q_F": [r.Q_F for r in runs], "k": [r.n_communities for r in runs]})
        if nontrivial and min_ari >= theta:
            return ResolutionResult(g1, g2, c_over_W, k, history)
        g1 *= grid_factor
        g2 *= grid_factor
    raise RuntimeError("Algorithm 1 did not accept within max_doublings")
