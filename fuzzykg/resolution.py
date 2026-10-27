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


# ------------------------------------------- null model and 'no structure' (M2)
def null_model_zscore(
    G: FuzzyGraph,
    gamma: float,
    n_perm: int = 20,
    seeds: Sequence[int] = (0,),
    rng_seed: int = 0,
    rewire: bool = True,
) -> dict:
    """z-score of the optimised Q_F against a degree-preserving null model.

    Each null graph keeps the in- and out-degree sequence (degree-preserving
    edge swaps via igraph's rewiring when ``rewire`` is True, the observed
    topology otherwise) and reassigns the observed memberships to edges in
    random order, so the membership distribution is kept exactly.  Fuzzy
    Leiden is run on each null graph with the same gamma and seeds, and the
    best Q_F is recorded.  Returns the observed Q_F, the null mean and
    standard deviation, the z-score and the empirical one-sided p-value.
    """
    import igraph as ig

    rng = np.random.default_rng(rng_seed)
    obs = max(fuzzy_leiden(G, gamma, seed=s, n_iterations=0).Q_F for s in seeds)
    null = []
    base = G.to_igraph()
    for _ in range(n_perm):
        g = base.copy()
        if rewire:
            g.rewire(n=10 * G.m, mode="simple")
        src = np.array([e.source for e in g.es]); dst = np.array([e.target for e in g.es])
        mu = rng.permutation(G.mu)[: src.size]
        Gn = FuzzyGraph(G.n, src, dst, mu)
        null.append(max(fuzzy_leiden(Gn, gamma, seed=s, n_iterations=0).Q_F for s in seeds))
    null = np.asarray(null)
    sd = float(null.std(ddof=1)) if n_perm > 1 else float("nan")
    z = (obs - null.mean()) / sd if sd > 0 else float("inf")
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"Q_F_obs": float(obs), "null_mean": float(null.mean()), "null_sd": sd,
            "z": float(z), "p_empirical": p, "n_perm": n_perm}


def adaptive_resolution_with_null(
    G: FuzzyGraph,
    z_min: float = 3.0,
    dq_min: float = 0.10,
    reference_gamma: float = 1.0,
    n_perm: int = 20,
    **kwargs,
) -> dict:
    """Algorithm 1 (revised): run ``adaptive_resolution`` to seed (gamma1, gamma2),
    then test the graph against the degree-preserving null model at the
    reference resolution (gamma = 1 by default, the scale on which Q_F is
    comparable across graphs; the accepted gamma1 can be far above 1 on graphs
    without structure, where both the observed and the null partitions are
    near-singleton and a z-score is uninformative).  Outcome is 'structure'
    when z >= z_min AND the excess Q_F over the null mean is at least dq_min
    (an effect-size floor: on dense graphs the null standard deviation is of
    order 1e-3, so a z-score alone certifies differences of a few thousandths).
    """
    res = adaptive_resolution(G, **kwargs)
    seeds = kwargs.get("seeds", (0, 1, 2, 3, 4))
    nm = null_model_zscore(G, reference_gamma, n_perm=n_perm, seeds=seeds)
    nm["delta_Q"] = nm["Q_F_obs"] - nm["null_mean"]
    nm["reference_gamma"] = reference_gamma
    outcome = "structure" if (nm["z"] >= z_min and nm["delta_Q"] >= dq_min) else "no structure detected"
    return {"gamma1": res.gamma1, "gamma2": res.gamma2, "c_over_W": res.c_over_W,
            "doublings": res.doublings, "null": nm, "outcome": outcome}
