"""Disparity-filter backbone: p-values in [0,1], degree-one edges kept, scale invariance."""
import numpy as np

from fuzzykg import FuzzyGraph
from fuzzykg.backbone import disparity_filter, disparity_pvalues


def _graph(seed=0, n=60, m=400):
    rng = np.random.default_rng(seed)
    edges = set()
    while len(edges) < m:
        u, v = rng.integers(0, n, size=2)
        if u != v:
            edges.add((int(u), int(v)))
    e = sorted(edges)
    mu = rng.uniform(0.05, 1.0, size=len(e))
    return FuzzyGraph.from_edges([(u, v, w) for (u, v), w in zip(e, mu)], n=n)


def test_pvalues_in_unit_interval():
    G = _graph()
    p1, p2 = disparity_pvalues(G)
    assert p1.min() >= 0 and p1.max() <= 1 and p2.min() >= 0 and p2.max() <= 1


def test_degree_one_edges_kept():
    G = FuzzyGraph.from_edges([(0, 1, 0.3), (1, 2, 0.9), (1, 3, 0.1), (1, 4, 0.2)], n=5)
    keep = disparity_filter(G, 0.05)
    assert keep[0]          # (0,1) is the only out-edge of 0


def test_scale_invariance():
    G = _graph(1)
    keep = disparity_filter(G, 0.1)
    G2 = FuzzyGraph(G.n, G.src, G.dst, np.clip(G.mu * 0.5, 1e-3, 1.0))
    assert np.array_equal(keep, disparity_filter(G2, 0.1))
