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
