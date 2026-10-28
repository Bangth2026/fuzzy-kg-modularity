"""Null-model tests for the partition information carried by a fuzzy graph, with a
power calibration on the observed graph (Algorithm 1, Section 5.3).

Two questions are separated.

1. Does the graph carry community structure beyond its degree sequence?  The optimised
   Q_F at a reference resolution is compared with a degree-preserving null (directed
   double-edge swaps, memberships permuted).  Because the excess a real structure
   produces depends on the density, the decision floor is not a constant but is read
   off planted graphs matched to the observed graph: same number of nodes and edges,
   same number of communities as the observed partition, memberships resampled from
   the observed membership distribution, and the in-community edge fraction p_in swept
   upward from the structureless value 1/k.  The floor is the excess at the smallest
   p_in at which the planted communities are recovered (adjusted Rand index >= 0.5).

2. Do the memberships carry partition information beyond the topology?  The optimised
   Q_F is compared with two weights-only nulls on the fixed topology: a global
   permutation of the memberships over all edges, and a row permutation that permutes
   memberships only among the out-edges of each node, so that every out-strength is kept
   exactly and the null term of (7) changes only through the in-strengths.  The power
   of the test is calibrated on the observed topology itself: the memberships are
   re-assigned so that edges internal to the observed partition receive higher ranks,
   with a tilt s (score Z + s * 1[internal], Z standard normal, memberships assigned by
   the rank of the score, so the marginal distribution is the observed one exactly), and
   the smallest tilt the test detects is reported together with the point-biserial
   correlation between membership and internality it corresponds to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .graph import FuzzyGraph
from .leiden import fuzzy_leiden


@dataclass
class _Best:
    Q_F: float
    membership: np.ndarray
    n_communities: int


def best_qf(G: FuzzyGraph, gamma: float = 1.0, seeds: Sequence[int] = (0, 1, 2, 3, 4)):
    """Best Q_F over the seeds (Leiden until convergence), without the diagnostic checks of
    ``fuzzy_leiden``; same optimiser, same seeds, same result."""
    import igraph as ig
    import leidenalg as la
    g = ig.Graph(n=G.n, edges=list(zip(G.src.tolist(), G.dst.tolist())), directed=True)
    g.es["weight"] = G.mu.tolist()
    best = None
    for s in seeds:
        part = la.RBConfigurationVertexPartition(g, weights="weight", resolution_parameter=gamma)
        opt = la.Optimiser(); opt.set_rng_seed(int(s))
        opt.optimise_partition(part, n_iterations=-1)
        q = part.quality() / G.W
        if best is None or q > best.Q_F:
            m = np.asarray(part.membership, dtype=np.int64)
            best = _Best(float(q), m, int(m.max() + 1))
    return best


def permute_global(G: FuzzyGraph, rng) -> FuzzyGraph:
    return FuzzyGraph(G.n, G.src, G.dst, rng.permutation(G.mu))


def permute_rows(G: FuzzyGraph, rng) -> FuzzyGraph:
    """Permute memberships among the out-edges of each node (out-strengths kept)."""
    mu = G.mu.copy()
    order = np.argsort(G.src, kind="stable")
    src_sorted = G.src[order]
    bounds = np.flatnonzero(np.diff(src_sorted)) + 1
    for block in np.split(order, bounds):
        if block.size > 1:
            mu[block] = mu[rng.permutation(block)]
    return FuzzyGraph(G.n, G.src, G.dst, mu)


def weights_only_test(G: FuzzyGraph, kind: str = "global", draws: int = 200, gamma: float = 1.0,
                      seeds: Sequence[int] = (0, 1, 2, 3, 4), rng_seed: int = 0, obs: Optional[float] = None) -> Dict:
    rng = np.random.default_rng(rng_seed)
    if obs is None:
        obs = best_qf(G, gamma, seeds).Q_F
    f = permute_global if kind == "global" else permute_rows
    vals = np.array([best_qf(f(G, rng), gamma, seeds).Q_F for _ in range(draws)])
    sd = float(vals.std(ddof=1))
    return {"kind": kind, "Q_F_obs": float(obs), "null_mean": float(vals.mean()), "null_sd": sd,
            "null_q025": float(np.percentile(vals, 2.5)), "null_q975": float(np.percentile(vals, 97.5)),
            "delta_Q": float(obs - vals.mean()), "z": float((obs - vals.mean()) / sd) if sd > 0 else float("inf"),
            "p_upper": float((np.sum(vals >= obs) + 1) / (draws + 1)),
            "p_lower": float((np.sum(vals <= obs) + 1) / (draws + 1)), "draws": draws}


def tilt_memberships(G: FuzzyGraph, internal: np.ndarray, s: float, rng) -> np.ndarray:
    """Re-assign the observed memberships so that internal edges rank higher by tilt s;
    the marginal distribution is the observed one exactly."""
    score = rng.standard_normal(G.m) + s * internal
    mu_sorted = np.sort(G.mu)
    out = np.empty(G.m)
    out[np.argsort(score, kind="stable")] = mu_sorted
    return out


def weights_only_power(G: FuzzyGraph, membership: np.ndarray, tilts=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
                       kind: str = "global", draws: int = 30, gamma: float = 1.0,
                       seeds: Sequence[int] = (0, 1, 2), rng_seed: int = 0, z_min: float = 3.0) -> Dict:
    """Detection of a planted membership-community correlation by the weights-only test,
    on the observed topology with the observed partition as the planted one."""
    rng = np.random.default_rng(rng_seed)
    internal = (membership[G.src] == membership[G.dst]).astype(float)
    rows = []
    for s in tilts:
        mu = tilt_memberships(G, internal, s, rng)
        Gt = FuzzyGraph(G.n, G.src, G.dst, mu)
        r_pb = float(np.corrcoef(mu, internal)[0, 1]) if internal.std() > 0 else 0.0
        t = weights_only_test(Gt, kind, draws, gamma, seeds, rng_seed=int(rng.integers(1 << 30)))
        rows.append({"tilt": float(s), "r_pb": r_pb, **{k: t[k] for k in ("Q_F_obs", "null_mean", "null_sd", "delta_Q", "z")}})
    det = [r for r in rows if r["z"] >= z_min and r["delta_Q"] > 0]
    return {"kind": kind, "internal_share": float(internal.mean()), "rows": rows,
            "detectable_tilt": det[0]["tilt"] if det else None, "detectable_r_pb": det[0]["r_pb"] if det else None}


try:
    from numba import njit as _njit
except Exception:          # pragma: no cover
    def _njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@_njit(cache=True)
def _swap_kernel(src, dst, n, pairs):
    """Directed double-edge swaps (u->v, x->y) -> (u->y, x->v) over the given index pairs,
    rejecting self-loops and multi-edges; adjacency kept in a dense boolean matrix."""
    A = np.zeros((n, n), dtype=np.bool_)
    for k in range(src.size):
        A[src[k], dst[k]] = True
    for t in range(pairs.shape[0]):
        i = pairs[t, 0]; j = pairs[t, 1]
        if i == j:
            continue
        u = src[i]; v = dst[i]; x = src[j]; y = dst[j]
        if u == y or x == v or A[u, y] or A[x, v]:
            continue
        A[u, v] = False; A[x, y] = False; A[u, y] = True; A[x, v] = True
        dst[i] = y; dst[j] = v
    return dst


def degree_rewire(G: FuzzyGraph, rng, n_attempts: Optional[int] = None) -> FuzzyGraph:
    """Directed double-edge swaps keeping every in- and out-degree; memberships permuted."""
    m = G.src.size
    pairs = rng.integers(0, m, size=(n_attempts or 10 * m, 2)).astype(np.int64)
    dst = _swap_kernel(G.src.astype(np.int64).copy(), G.dst.astype(np.int64).copy(), G.n, pairs)
    return FuzzyGraph(G.n, G.src.copy(), dst, rng.permutation(G.mu))


def degree_test(G: FuzzyGraph, draws: int = 200, gamma: float = 1.0, seeds=(0, 1, 2, 3, 4), rng_seed: int = 0,
                obs: Optional[float] = None) -> Dict:
    rng = np.random.default_rng(rng_seed)
    if obs is None:
        obs = best_qf(G, gamma, seeds).Q_F
    vals = np.array([best_qf(degree_rewire(G, rng), gamma, seeds).Q_F for _ in range(draws)])
    sd = float(vals.std(ddof=1))
    return {"Q_F_obs": float(obs), "null_mean": float(vals.mean()), "null_sd": sd,
            "null_q025": float(np.percentile(vals, 2.5)), "null_q975": float(np.percentile(vals, 97.5)),
            "delta_Q": float(obs - vals.mean()), "z": float((obs - vals.mean()) / sd) if sd > 0 else float("inf"),
            "p_upper": float((np.sum(vals >= obs) + 1) / (draws + 1)), "draws": draws}


def planted_like(G: FuzzyGraph, k: int, p_in: float, rng) -> tuple:
    """Directed planted partition matched to G: same n and m, k equal communities, a
    fraction p_in of every node's out-edges inside its community, memberships resampled
    from the observed membership distribution (no community shift)."""
    n, m = G.n, G.m
    k = int(min(k, n))
    truth = np.repeat(np.arange(k), int(np.ceil(n / k)))[:n]
    if np.unique(truth).size < k:          # ceil-blocks leave communities empty when k is close to n
        truth = np.arange(n) % k
    rng.shuffle(truth)
    members = [np.flatnonzero(truth == c) for c in range(k)]
    edges = set()
    tries = 0
    while len(edges) < m and tries < 50 * m:
        tries += 1
        u = int(rng.integers(0, n))
        if rng.random() < p_in:
            pool = members[truth[u]]
        else:
            c2 = int(rng.integers(0, k - 1)); c2 += c2 >= truth[u]
            pool = members[c2]
        v = int(pool[rng.integers(0, pool.size)])
        if v != u:
            edges.add((u, v))
    e = np.array(sorted(edges))
    mu = rng.choice(G.mu, size=len(e), replace=True)
    return FuzzyGraph(n, e[:, 0], e[:, 1], mu), truth


def density_matched_floor(G: FuzzyGraph, k: int, p_grid: Optional[Sequence[float]] = None, graph_seeds: int = 2,
                          draws: int = 20, seeds=(0, 1, 2), rng_seed: int = 0, ari_min: float = 0.5) -> Dict:
    from sklearn.metrics import adjusted_rand_score as ari
    rng = np.random.default_rng(rng_seed)
    k = max(2, int(k))
    base = 1.0 / k
    if p_grid is None:
        p_grid = sorted({min(0.95, base + d) for d in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9)})
    rows = []
    for p in p_grid:
        p = min(p, 0.95)
        for gs in range(graph_seeds):
            Gp, truth = planted_like(G, k, p, rng)
            r = best_qf(Gp, 1.0, seeds)
            t = degree_test(Gp, draws, 1.0, seeds, rng_seed=int(rng.integers(1 << 30)), obs=r.Q_F)
            rows.append({"p_in": float(p), "graph_seed": gs, "Q_F": r.Q_F, "k_found": r.n_communities,
                         "ari": float(ari(truth, r.membership)), "delta_Q": t["delta_Q"], "z": t["z"]})
    ps = sorted({r["p_in"] for r in rows})
    agg = [{"p_in": p, "ari": float(np.mean([r["ari"] for r in rows if r["p_in"] == p])),
            "delta_Q": float(np.mean([r["delta_Q"] for r in rows if r["p_in"] == p]))} for p in ps]
    rec = [a for a in agg if a["ari"] >= ari_min]
    # if no planted structure at this density is recoverable, the floor is undefined
    # and the structure question cannot be answered by recovery; this is reported.
    return {"k": k, "rows": rows, "by_p_in": agg,
            "floor": rec[0]["delta_Q"] if rec else None, "floor_p_in": rec[0]["p_in"] if rec else None}


def membership_degree_correlation(G: FuzzyGraph) -> Dict:
    from scipy.stats import spearmanr
    kout = np.bincount(G.src, minlength=G.n); kin = np.bincount(G.dst, minlength=G.n)
    return {"rho_mu_outdeg_src": float(spearmanr(G.mu, kout[G.src]).statistic),
            "rho_mu_indeg_dst": float(spearmanr(G.mu, kin[G.dst]).statistic),
            "rho_mu_indeg_src": float(spearmanr(G.mu, kin[G.src]).statistic),
            "rho_mu_outdeg_dst": float(spearmanr(G.mu, kout[G.dst]).statistic)}
