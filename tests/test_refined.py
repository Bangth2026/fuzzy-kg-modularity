"""Release 5.1.0: Corollary 1 (refined knots make the cut indicator a vertex) and the divergence check."""
import numpy as np
from fuzzykg import FuzzyGraph, fuzzy_modularity
from fuzzykg.simplex import refined_knots, partition_stats, q_of, divergence


def _graph(seed=0, n=60, m=600):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, m); dst = rng.integers(0, n, m); keep = src != dst
    return FuzzyGraph(n, src[keep], dst[keep], rng.uniform(0.05, 1.0, keep.sum()))


def test_vertex_is_indicator():
    G = _graph()
    a = float(np.quantile(G.mu, 0.5))
    knots, ell = refined_knots(G.mu, [a])
    j = int(np.flatnonzero(np.isclose(knots[1:], a))[0])
    rng = np.random.default_rng(1)
    for _ in range(5):
        p = rng.integers(0, 4, G.n)
        A, M = partition_stats(ell, G.src, G.dst, p)
        s = np.full(ell.shape[1], 1e-12); s[j] = 1.0
        q = q_of(s[None, :], A[None, :], M[None, :, :], ell.sum(0))[0, 0]
        H = G.alpha_cut(a)
        assert abs(q - fuzzy_modularity(H, p)) < 1e-8


def test_divergence_zero_for_identical():
    rng = np.random.default_rng(0)
    P = [rng.integers(0, 3, 50) for _ in range(5)]
    d = divergence(P, P)
    assert abs(d["D"] - (min(d["a_SS"], d["a_FF"]) - d["a_FS"])) < 1e-12 and d["a_FS"] == 1.0
