"""All-pairs best-path strengths under max-min and the Archimedean t-norms (Section 6.3),
with an optional hop limit and an optional floor tau_path on the edges a path may use
(constraint (P2) of Section 4.3).

* Without a hop limit: max-min by a Dijkstra-type widest-path sweep from every source
  (compiled with numba); Archimedean t-norms by Dijkstra on the additive lengths g(mu)
  (scipy), mapped back through the pseudo-inverse.
* With a hop limit L: L rounds of the (max, min) or (min, +) matrix product, which is
  exact for the same reason as hop-limited Bellman-Ford (non-negative lengths).

Returns S[s, t] (best-path strength, 0 when t is unreachable from s, -1 on the
diagonal) and H[s, t] (hop count of a best path with the fewest hops; 0 when
unreachable).  ``tests/test_paths_fast.py`` checks agreement with
``fuzzykg.tnorms.widest_path`` and ``generator_best_paths``.
"""
from __future__ import annotations

import heapq

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from .graph import FuzzyGraph
from .tnorms import generator

try:
    from numba import njit
except Exception:          # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _widest_all(indptr, indices, data, n):
    S = np.zeros((n, n))
    H = np.zeros((n, n), dtype=np.int64)
    for s in range(n):
        best = np.zeros(n)
        hops = np.zeros(n, dtype=np.int64)
        done = np.zeros(n, dtype=np.bool_)
        best[s] = 2.0                      # sentinel above any membership
        # simple O(n^2) selection: dense graphs make a heap no faster
        for _ in range(n):
            u = -1
            bu = 0.0
            for v in range(n):
                if not done[v] and best[v] > bu:
                    bu = best[v]
                    u = v
            if u < 0:
                break
            done[u] = True
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                c = min(bu, data[k])
                if c > best[v] or (c == best[v] and c > 0 and not done[v] and hops[u] + 1 < hops[v]):
                    best[v] = c
                    hops[v] = hops[u] + 1
        for v in range(n):
            if v != s:
                S[s, v] = best[v]
                H[s, v] = hops[v] if best[v] > 0 else 0
    return S, H


@njit(cache=True)
def _maxmin_product(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            a = A[i, k]
            if a <= 0.0:
                continue
            for j in range(n):
                b = B[k, j]
                v = a if a < b else b
                if v > C[i, j]:
                    C[i, j] = v
    return C


@njit(cache=True)
def _minplus_product(A, B):
    n = A.shape[0]
    C = np.full((n, n), np.inf)
    for i in range(n):
        for k in range(n):
            a = A[i, k]
            if a == np.inf:
                continue
            for j in range(n):
                v = a + B[k, j]
                if v < C[i, j]:
                    C[i, j] = v
    return C


def _matrix(G: FuzzyGraph, tau_path: float):
    keep = G.mu >= tau_path
    W = np.zeros((G.n, G.n))
    W[G.src[keep], G.dst[keep]] = G.mu[keep]
    return W


def strengths(G: FuzzyGraph, op: str, p=None, max_hops: int | None = None, tau_path: float = 0.0):
    """op in {'min', 'product', 'lukasiewicz', 'yager', 'schweizer_sklar'}."""
    W = _matrix(G, tau_path)
    n = G.n
    if max_hops is None:
        A = sp.csr_matrix(W)
        if op == "min":
            S, H = _widest_all(A.indptr, A.indices, A.data, n)
        else:
            gen = generator(op, p)
            L = A.copy()
            L.data = gen.g(A.data)
            L.data = np.where(L.data <= 0, 1e-300, L.data)      # an edge of membership 1 has length 0
            dist, pred = dijkstra(L, directed=True, return_predecessors=True)
            S = np.where(np.isfinite(dist), gen.pseudo_inverse(np.where(np.isfinite(dist), dist, 0.0)), 0.0)
            S = np.where(np.isfinite(dist), S, 0.0)
            H = _hops_from_pred(pred)
            H = np.where(S > 0, H, 0)
        np.fill_diagonal(S, -1.0); np.fill_diagonal(H, 0)
        return S, H
    # hop-limited: L rounds of the semiring product
    if op == "min":
        S = W.copy(); H = (W > 0).astype(np.int64)
        P = W.copy()
        for h in range(2, max_hops + 1):
            P = _maxmin_product(P, W)
            better = P > S
            S = np.where(better, P, S); H = np.where(better, h, H)
    else:
        gen = generator(op, p)
        Lm = np.full((n, n), np.inf)
        mask = W > 0
        Lm[mask] = gen.g(W[mask])
        D = Lm.copy(); H = mask.astype(np.int64)
        P = Lm.copy()
        for h in range(2, max_hops + 1):
            P = _minplus_product(P, Lm)
            better = P < D - 1e-15
            D = np.where(better, P, D); H = np.where(better, h, H)
        fin = np.isfinite(D)
        S = np.zeros((n, n))
        S[fin] = gen.pseudo_inverse(D[fin])
        H = np.where(S > 0, H, 0)
    np.fill_diagonal(S, -1.0); np.fill_diagonal(H, 0)
    return S, H


def _hops_from_pred(pred):
    n = pred.shape[0]
    H = np.zeros((n, n), dtype=np.int64)
    for s in range(n):
        p = pred[s]
        h = np.full(n, -1, dtype=np.int64); h[s] = 0
        # resolve depths iteratively
        changed = True
        while changed:
            changed = False
            ok = (h < 0) & (p >= 0)
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                break
            ph = h[p[idx]]
            set_ = ph >= 0
            if set_.any():
                h[idx[set_]] = ph[set_] + 1
                changed = True
        H[s] = np.maximum(h, 0)
    return H
