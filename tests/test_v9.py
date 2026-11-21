"""Release 4.6.0: heap widest paths from selected sources, and the grade-preserving baselines."""
import os, sys
import numpy as np
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from fuzzykg import FuzzyGraph
from fuzzykg.paths_fast import strengths


def random_graph(n=40, m=200, seed=0):
    rng = np.random.default_rng(seed)
    e = set()
    while len(e) < m:
        a, b = rng.integers(0, n, 2)
        if a != b:
            e.add((int(a), int(b)))
    e = np.array(sorted(e))
    return FuzzyGraph(n, e[:, 0], e[:, 1], rng.uniform(0.05, 1.0, len(e)))


def test_widest_from_matches_all_pairs_kernel():
    from v9_cross_domain import widest_from
    G = random_graph()
    S_all, H_all = strengths(G, "min")
    A = csr_matrix((G.mu, (G.src, G.dst)), shape=(G.n, G.n))
    src = np.array([0, 3, 7, 21], dtype=np.int64)
    S, H = widest_from(A.indptr, A.indices, A.data, G.n, src)
    for i, s in enumerate(src):
        mask = np.arange(G.n) != s
        assert np.allclose(S[i, mask], np.maximum(S_all[s, mask], 0.0))
        reach = S[i] > 0
        assert np.array_equal(H[i, reach & mask], H_all[s, reach & mask])


def test_graded_link_model_learns_memberships():
    torch = pytest.importorskip("torch")
    from fuzzykg.embeddings import GradedLinkModel
    from scipy.stats import spearmanr
    G = random_graph(n=60, m=600, seed=1)
    M = GradedLinkModel(model="ukge", epochs=40, eval_every=10).fit(G, grid=(("lr", (0.03,)),), refit_seeds=(0,))
    p = M.score(G.src, G.dst)
    rng = np.random.default_rng(2); u = rng.integers(0, G.n, 2000); v = rng.integers(0, G.n, 2000)
    assert p.shape == (G.m,) and np.all((p >= 0) & (p <= 1))
    assert p.mean() > M.score(u, v).mean()          # edges score above random pairs
    assert spearmanr(p, G.mu).statistic > 0.3
