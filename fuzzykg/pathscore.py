"""Path scores used in Section 5 of the release 5.0.0 manuscript.

``connectivity_level``: the level alpha_c up to which the alpha-cuts keep (a share of) the pairs
that the support graph connects; by Proposition 3 the max-min score of every such pair is at least
alpha_c and depends only on the memberships at or above it, so ``usable_share`` = share of edges
with mu_E >= alpha_c is the part of the grades that max-min can read.

``hop_penalised``: best paths under the edge length lam + (1 - lam) * g(mu_E), g = -ln (the product
generator); lam = 0 is the product t-norm, lam = 1 the hop count, and every lam is solved exactly by
Dijkstra because every length is non-negative. For lam > 0 the composition is not a t-norm (an edge
of membership 1 still costs lam), so Fact 1 does not apply to it.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from .graph import FuzzyGraph
from .paths_fast import strengths


def widest(G: FuzzyGraph) -> np.ndarray:
    S, _ = strengths(G, "min")
    S = S.copy(); np.fill_diagonal(S, 0.0)
    return S


def connectivity_level(G: FuzzyGraph, share: float = 0.95, S: np.ndarray | None = None):
    """alpha_c(share): the highest level at which at least ``share`` of the ordered pairs that the
    support connects are still connected; share = 1 gives the hard level min mu_inf over connected
    pairs. Returns alpha_c, the usable share of edges (mu_E >= alpha_c) and the hard level."""
    S = widest(G) if S is None else S
    off = ~np.eye(G.n, dtype=bool)
    v = S[off & (S > 0)]
    if v.size == 0:
        return {"alpha_c": float("nan"), "usable_share": float("nan"), "alpha_hard": float("nan"), "reachable_pairs": 0}
    a = float(np.quantile(v, 1.0 - share, method="lower"))
    return {"alpha_c": a, "usable_share": float((G.mu >= a).mean()), "alpha_hard": float(v.min()),
            "usable_share_hard": float((G.mu >= v.min()).mean()), "reachable_pairs": int(v.size),
            "widest_sd": float(v.std())}


def hop_penalised(G: FuzzyGraph, lam: float, sources=None) -> np.ndarray:
    """Shortest-path lengths under lam + (1 - lam) * (-ln mu); inf where unreachable."""
    ln = lam + (1.0 - lam) * (-np.log(G.mu))
    ln = np.where(ln <= 0, 1e-12, ln)
    A = sp.csr_matrix((ln, (G.src, G.dst)), shape=(G.n, G.n))
    return dijkstra(A, directed=True, indices=sources)


def score_from_length(D: np.ndarray) -> np.ndarray:
    """A ranking score: -length, unreachable pairs below every reachable one."""
    out = -D.copy()
    fin = np.isfinite(D)
    low = (out[fin].min() - 1.0) if fin.any() else -1.0
    out[~fin] = low
    return out
