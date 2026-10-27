"""Triangular norms, OWA and path composition (Sections 2.2, 2.3, 6).

Eq. (2)  T(a,b) <= min(a,b) for every t-norm         (Fact 1)
Eq. (3)  Yager family           T^Y_p,  p > 0;  T^Y_1 = Lukasiewicz, -> min as p -> inf
Eq. (4)  Schweizer-Sklar family T^SS_p, p != 0; -> product as p -> 0, T^SS_1 = Lukasiewicz,
                                                -> min as p -> -inf
Eq. (1)  strength of connectedness = max-min composition = widest path.
"""

from __future__ import annotations

import heapq
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .graph import FuzzyGraph

TNorm = Callable[[np.ndarray, np.ndarray], np.ndarray]


# -------------------------------------------------------------------- t-norms
def t_min(a, b):
    return np.minimum(a, b)


def t_product(a, b):
    return np.multiply(a, b)


def t_lukasiewicz(a, b):
    return np.maximum(0.0, np.add(a, b) - 1.0)


def yager(p: float) -> TNorm:
    if p <= 0:
        raise ValueError("Yager parameter must be > 0")

    def T(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        return np.maximum(0.0, 1.0 - ((1 - a) ** p + (1 - b) ** p) ** (1.0 / p))

    T.__name__ = f"yager_p{p}"
    return T


def schweizer_sklar(p: float) -> TNorm:
    if p == 0:
        return t_product            # limit p -> 0

    def T(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            if p > 0:
                inner = np.maximum(0.0, a ** p + b ** p - 1.0)
                return inner ** (1.0 / p)
            # p < 0: a^p is large for small a; max{0, .} never binds for a,b in (0,1]
            inner = a ** p + b ** p - 1.0
            out = inner ** (1.0 / p)
        out = np.where((a <= 0) | (b <= 0), 0.0, out)
        return out

    T.__name__ = f"schweizer_sklar_p{p}"
    return T


STANDARD_TNORMS: Dict[str, TNorm] = {
    "min": t_min,
    "product": t_product,
    "lukasiewicz": t_lukasiewicz,
}


# ------------------------------------------------------------------------ OWA
def owa(x: Sequence[float], w: Sequence[float]) -> float:
    b = np.sort(np.asarray(x, float))[::-1]
    w = np.asarray(w, float)
    return float(np.dot(w, b))


def orness(w: Sequence[float]) -> float:
    w = np.asarray(w, float)
    n = w.size
    return float(np.sum((n - np.arange(1, n + 1)) * w) / (n - 1))


def owa_weights_from_orness(target: float, n: int, tol: float = 1e-10) -> np.ndarray:
    """Yager RIM quantifier Q(r) = r^a; weights w_j = Q(j/n) - Q((j-1)/n), with
    the exponent a chosen so that orness(w) = target (bisection)."""
    def weights(a):
        r = np.arange(0, n + 1) / n
        Q = r ** a
        return np.diff(Q)

    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if orness(weights(mid)) > target:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1 < tol:
            break
    return weights(np.sqrt(lo * hi))


# ------------------------------------------------------------- path composition
def compose_path(memberships: Sequence[float], T: TNorm = t_min) -> float:
    """S_T(pi) = T-composition of edge memberships along a path."""
    s = float(memberships[0])
    for m in memberships[1:]:
        s = float(T(s, m))
    return s


def fact1_violations(paths: Iterable[Sequence[float]], T: TNorm, tol: float = 1e-12) -> int:
    """Count paths on which S_T exceeds the weakest link (must be 0 for a t-norm)."""
    bad = 0
    for ms in paths:
        if compose_path(ms, T) > min(ms) + tol:
            bad += 1
    return bad


def widest_path(G: FuzzyGraph, source: int) -> Tuple[np.ndarray, np.ndarray]:
    """Strength of connectedness mu_E^inf(source, .) of Eq. (1) by a Dijkstra-type
    sweep on the max-min semiring, O(|E| log |C|).  Returns (strength, predecessor)."""
    A = G.adjacency()
    strength = np.zeros(G.n)
    pred = np.full(G.n, -1, dtype=np.int64)
    strength[source] = 1.0
    heap = [(-1.0, source)]
    done = np.zeros(G.n, dtype=bool)
    while heap:
        negs, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        row = A.getrow(u)
        for v, w in zip(row.indices, row.data):
            cand = min(-negs, w)
            if cand > strength[v]:
                strength[v] = cand
                pred[v] = u
                heapq.heappush(heap, (-cand, v))
    strength[source] = 0.0 if G.n > 1 else 1.0   # strength to itself is not a path
    return strength, pred


def enumerate_best_paths(G: FuzzyGraph, source: int, max_hops: int, T: TNorm) -> Dict[int, Tuple[float, int]]:
    """Exact best path under an arbitrary t-norm by bounded enumeration (no
    optimal substructure for strict t-norms, Section 8.3).  Returns
    {target: (best strength, hop count of that path)}."""
    A = G.adjacency()
    best: Dict[int, Tuple[float, int]] = {}
    frontier = [(source, 1.0, 1.0, (source,))]
    expanded = 0
    violations = 0
    for hop in range(1, max_hops + 1):
        nxt = []
        for u, s, weakest, seen in frontier:
            row = A.getrow(u)
            for v, w in zip(row.indices, row.data):
                if v in seen:
                    continue
                expanded += 1
                s2 = float(T(s, w))
                wk = min(weakest, float(w))
                if s2 > wk + 1e-12:
                    violations += 1
                if s2 <= 0.0:
                    continue
                if v not in best or s2 > best[v][0]:
                    best[v] = (s2, hop)
                nxt.append((v, s2, wk, seen + (v,)))
        frontier = nxt
        if not frontier:
            break
    best["__expanded__"] = (float(expanded), violations)   # bookkeeping for Section 8.3
    return best


# ------------------------------------------- additive generators (Section 7.3)
# Every continuous Archimedean t-norm has an additive generator g : [0,1] -> [0, inf],
# continuous, strictly decreasing, g(1) = 0, with T(a, b) = g^(-1)(g(a) + g(b)) where
# g^(-1) is the pseudo-inverse (x -> g^{-1}(min(x, g(0)))).  Hence the T-composition
# of a path equals g^(-1)(sum_e g(mu_e)), and the best path under T is the path of
# minimum additive length sum_e g(mu_e): an ordinary shortest-path problem with
# non-negative edge lengths.  No enumeration is needed.
#
#   product:            g(x) = -ln x               (strict,     g(0) = inf)
#   Lukasiewicz:        g(x) = 1 - x               (nilpotent,  g(0) = 1)
#   Yager p > 0:        g(x) = (1 - x)^p           (nilpotent,  g(0) = 1)
#   Schweizer-Sklar p:  g(x) = (1 - x^p) / p       (p < 0 strict, p > 0 nilpotent,
#                                                   p -> 0 product)
# The minimum is not Archimedean; its best path is the widest path (max-min).

from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class AdditiveGenerator:
    name: str
    g: Callable[[np.ndarray], np.ndarray]          # generator
    g_inv: Callable[[np.ndarray], np.ndarray]      # ordinary inverse on [0, g(0)]
    g0: float                                       # g(0); inf for strict t-norms

    def pseudo_inverse(self, x):
        x = np.asarray(x, float)
        out = self.g_inv(np.minimum(x, self.g0)) if np.isfinite(self.g0) else self.g_inv(x)
        return np.where(x >= self.g0, 0.0, out) if np.isfinite(self.g0) else out

    def tnorm(self) -> TNorm:
        def T(a, b):
            a = np.asarray(a, float); b = np.asarray(b, float)
            return self.pseudo_inverse(self.g(a) + self.g(b))
        T.__name__ = f"gen_{self.name}"
        return T

    @property
    def kind(self) -> str:
        return "strict" if not np.isfinite(self.g0) else "nilpotent"


def _safe_log(x):
    x = np.asarray(x, float)
    with np.errstate(divide="ignore"):
        return np.where(x > 0, -np.log(np.where(x > 0, x, 1.0)), np.inf)


def generator(name: str, p: float | None = None) -> AdditiveGenerator:
    """Additive generator of a named family."""
    if name == "product":
        return AdditiveGenerator("product", _safe_log, lambda y: np.exp(-y), np.inf)
    if name == "lukasiewicz":
        return AdditiveGenerator("lukasiewicz", lambda x: 1.0 - np.asarray(x, float),
                                 lambda y: 1.0 - np.asarray(y, float), 1.0)
    if name == "yager":
        if p is None or p <= 0:
            raise ValueError("Yager needs p > 0")
        return AdditiveGenerator(f"yager_p{p}", lambda x: (1.0 - np.asarray(x, float)) ** p,
                                 lambda y: 1.0 - np.asarray(y, float) ** (1.0 / p), 1.0)
    if name == "schweizer_sklar":
        if p is None:
            raise ValueError("Schweizer-Sklar needs p")
        if p == 0:
            return generator("product")
        if p > 0:
            return AdditiveGenerator(f"schweizer_sklar_p{p}",
                                     lambda x: (1.0 - np.asarray(x, float) ** p) / p,
                                     lambda y: (1.0 - p * np.asarray(y, float)) ** (1.0 / p), 1.0 / p)

        def g(x):
            x = np.asarray(x, float)
            with np.errstate(divide="ignore"):
                return np.where(x > 0, (1.0 - np.where(x > 0, x, 1.0) ** p) / p, np.inf)

        return AdditiveGenerator(f"schweizer_sklar_p{p}", g,
                                 lambda y: (1.0 - p * np.asarray(y, float)) ** (1.0 / p), np.inf)
    raise ValueError(f"unknown family {name}")


def generator_best_paths(
    G: FuzzyGraph, source: int, gen: AdditiveGenerator, max_hops: int | None = None
) -> Dict[int, Tuple[float, int]]:
    """Exact best path under a continuous Archimedean t-norm.

    Without a hop limit: Dijkstra on the additive lengths g(mu_e), O(|E| log |C|).
    With a hop limit L: L rounds of Bellman-Ford relaxation.  Lengths are
    non-negative, so the minimum over walks of at most L hops is attained by a
    simple path and the hop-limited result is exact.

    Returns {target: (best strength, hop count of a best path)}; targets that are
    unreachable, or whose best strength is 0 under a nilpotent t-norm, are omitted
    (mirroring ``enumerate_best_paths``).
    """
    A = G.adjacency()
    length = gen.g(A.data)                         # edge lengths, non-negative
    n = G.n
    if max_hops is None:
        dist = np.full(n, np.inf); hops = np.zeros(n, dtype=np.int64)
        dist[source] = 0.0
        heap = [(0.0, source)]
        done = np.zeros(n, dtype=bool)
        indptr, indices = A.indptr, A.indices
        while heap:
            d, u = heapq.heappop(heap)
            if done[u]:
                continue
            done[u] = True
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                nd = d + length[k]
                if nd < dist[v]:
                    dist[v] = nd; hops[v] = hops[u] + 1
                    heapq.heappush(heap, (nd, v))
    else:
        dist = np.full(n, np.inf); hops = np.zeros(n, dtype=np.int64)
        dist[source] = 0.0
        src, dst = G.src, G.dst
        length_e = gen.g(G.mu)                     # lengths in edge-list order (not CSR order)
        for h in range(1, max_hops + 1):
            cand = dist[src] + length_e
            new = dist.copy(); newh = hops.copy()
            # vectorised relaxation: minimum over incoming edges
            order = np.argsort(cand)
            dst_sorted = dst[order]
            first = np.unique(dst_sorted, return_index=True)[1]
            best_v = dst_sorted[first]; best_c = cand[order][first]
            better = best_c < new[best_v]
            new[best_v[better]] = best_c[better]
            newh[best_v[better]] = h
            dist, hops = new, newh
    # back to the membership scale
    out: Dict[int, Tuple[float, int]] = {}
    for v in range(n):
        if v == source or not np.isfinite(dist[v]):
            continue
        s = float(gen.pseudo_inverse(dist[v]))
        if s <= 0.0:
            continue
        out[v] = (s, int(hops[v]))
    return out
