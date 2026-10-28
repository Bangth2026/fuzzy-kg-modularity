"""Properties of the null models of Algorithm 1."""
import numpy as np

from fuzzykg import FuzzyGraph
from fuzzykg.nulltest import degree_rewire, permute_rows, tilt_memberships


def _graph(seed=0, n=50, m=400):
    rng = np.random.default_rng(seed)
    e = set()
    while len(e) < m:
        u, v = rng.integers(0, n, 2)
        if u != v:
            e.add((int(u), int(v)))
    e = sorted(e)
    return FuzzyGraph.from_edges([(u, v, w) for (u, v), w in zip(e, rng.uniform(0.1, 1, len(e)))], n=n)


def test_row_permutation_keeps_out_strength():
    G = _graph(); rng = np.random.default_rng(1)
    H = permute_rows(G, rng)
    assert np.allclose(G.out_strength(), H.out_strength())
    assert np.allclose(np.sort(G.mu), np.sort(H.mu))


def test_degree_rewire_keeps_degrees():
    G = _graph(2); rng = np.random.default_rng(3)
    H = degree_rewire(G, rng)
    assert (np.bincount(G.src, minlength=G.n) == np.bincount(H.src, minlength=G.n)).all()
    assert (np.bincount(G.dst, minlength=G.n) == np.bincount(H.dst, minlength=G.n)).all()


def test_tilt_keeps_marginal_and_raises_internal():
    G = _graph(4); rng = np.random.default_rng(5)
    memb = np.arange(G.n) % 4
    internal = (memb[G.src] == memb[G.dst]).astype(float)
    mu = tilt_memberships(G, internal, 2.0, rng)
    assert np.allclose(np.sort(mu), np.sort(G.mu))
    assert mu[internal == 1].mean() > mu[internal == 0].mean()
