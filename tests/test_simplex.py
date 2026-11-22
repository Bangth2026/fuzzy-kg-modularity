"""Release 5.0.0: Proposition 1 (closed form of Q_F on the cone of rescalings) and Proposition 3."""
import numpy as np
from fuzzykg import FuzzyGraph, fuzzy_modularity
from fuzzykg.audit import _leiden
from fuzzykg.simplex import CutFamily, segment_basis, kappa_star
from fuzzykg.synthetic import planted_partition_fuzzy_graph
from fuzzykg.pathscore import widest, connectivity_level


def _graph():
    rng = np.random.default_rng(0)
    n, m = 60, 600
    src = rng.integers(0, n, m); dst = rng.integers(0, n, m); keep = src != dst
    e = {(int(a), int(b)): float(w) for a, b, w in zip(src[keep], dst[keep], rng.uniform(0.05, 1.0, keep.sum()))}
    return FuzzyGraph.from_edges(((a, b, w) for (a, b), w in e.items()), n=n)


def test_closed_form_matches_direct_modularity():
    G = _graph(); _, ell = segment_basis(G.mu)
    P = [_leiden(G, s) for s in range(2)] + [_leiden(G.alpha_cut(float(np.quantile(G.mu, 0.5))), 0)]
    fam = CutFamily(G, {"p": P})
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = rng.gamma(0.5, 1.0, fam.J) + 1e-6
        Gs = FuzzyGraph(G.n, G.src, G.dst, ell @ s)
        assert np.allclose(fam.q("p", s[None, :])[0], [fuzzy_modularity(Gs, p) for p in P], atol=1e-12)
    # s = 1 is mu_E itself
    assert np.allclose(fam.q("p", np.ones((1, fam.J)))[0], [fuzzy_modularity(G, p) for p in P], atol=1e-12)


def test_kappa_star_box_contains_no_reversal():
    G = _graph(); Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
    parts = {"support": [_leiden(Gs, s) for s in range(5)], "q0.50": [_leiden(G.alpha_cut(float(np.quantile(G.mu, 0.5))), s) for s in range(5)]}
    fam = CutFamily(G, parts)
    r = kappa_star(fam, "q0.50", kmax=50.0)
    kk = min(r["kappa_star"], 50.0)
    S = 1 + (0.99 * kk - 1) * np.random.default_rng(2).random((2000, fam.J))
    _, mg = fam.margin("q0.50", S)
    assert np.all((mg > 0) == (r["decision_mu"] == "flagged"))


def test_maxmin_ignores_memberships_below_its_value():
    G = _graph(); S = widest(G)
    lev = connectivity_level(G, share=1.0, S=S)["alpha_c"]
    low = G.mu < lev
    G2 = FuzzyGraph(G.n, G.src, G.dst, np.where(low, lev * np.random.default_rng(3).uniform(0.01, 1.0, G.m), G.mu))
    assert np.allclose(widest(G2), S)
