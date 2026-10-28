"""All-pairs best paths agree with the per-source reference implementations."""
import numpy as np

from fuzzykg import FuzzyGraph
from fuzzykg.paths_fast import strengths
from fuzzykg.tnorms import generator, generator_best_paths, widest_path


def _graph(seed=0, n=40, m=250):
    rng = np.random.default_rng(seed)
    e = set()
    while len(e) < m:
        u, v = rng.integers(0, n, 2)
        if u != v:
            e.add((int(u), int(v)))
    e = sorted(e)
    return FuzzyGraph.from_edges([(u, v, w) for (u, v), w in zip(e, rng.uniform(0.05, 1, len(e)))], n=n)


def test_widest_all_pairs():
    G = _graph()
    S, _ = strengths(G, "min")
    for s in range(G.n):
        ref = widest_path(G, s)[0]
        mask = np.arange(G.n) != s
        assert np.allclose(S[s][mask], ref[mask])


def test_generator_all_pairs_and_hop_limit():
    G = _graph(1)
    for op, p in (("product", None), ("lukasiewicz", None), ("yager", 3.0), ("schweizer_sklar", -2.0)):
        gen = generator(op, p)
        for L in (None, 2, 3):
            S, _ = strengths(G, op, p, max_hops=L)
            for s in range(G.n):
                ref = np.zeros(G.n)
                for t, (v, _) in generator_best_paths(G, s, gen, max_hops=L).items():
                    ref[t] = v
                mask = np.arange(G.n) != s
                assert np.allclose(S[s][mask], ref[mask], atol=1e-12)
