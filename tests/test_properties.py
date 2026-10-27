"""Property tests mirroring the paper's propositions.  Run with ``pytest``."""

import numpy as np
import pytest

from fuzzykg import FuzzyGraph
from fuzzykg.fuzzification import (
    collapse_type1,
    degeneration_halfwidth,
    embedded_in_footprint,
    footprint,
    max_collapsed_par,
    zpd_gate_it2,
)
from fuzzykg.leiden import fuzzy_leiden
from fuzzykg.modularity import alpha_average_modularity, correlation_gap, fuzzy_modularity
from fuzzykg.synthetic import synthetic_fuzzy_graph
from fuzzykg.tnorms import (
    STANDARD_TNORMS,
    compose_path,
    owa_weights_from_orness,
    orness,
    schweizer_sklar,
    t_lukasiewicz,
    widest_path,
    yager,
)


@pytest.fixture(scope="module")
def G():
    return synthetic_fuzzy_graph(300, 0.8, seed=3)


@pytest.fixture(scope="module")
def partition(G):
    return fuzzy_leiden(G, 0.5, seed=1, n_iterations=8)


# ------------------------------------------------------------ Proposition 2
def test_boundedness_random_partitions(G):
    rng = np.random.default_rng(0)
    for K in (1, 3, 10, G.n):
        for _ in range(10):
            q = fuzzy_modularity(G, rng.integers(0, K, G.n), gamma=0.7)
            assert -0.7 - 1e-12 <= q <= 1 + 1e-12


def test_trivial_partition_is_one_minus_gamma(G):
    for gamma in (0.5, 1.0, 2.0):
        assert fuzzy_modularity(G, np.zeros(G.n, dtype=int), gamma) == pytest.approx(1 - gamma)


def test_monotone_and_connected(partition):
    assert partition.monotone_violations == 0
    assert partition.connected_violations == 0


def test_reduction_to_leidenalg_quality(G, partition):
    import leidenalg as la
    g = G.to_igraph()
    part = la.RBConfigurationVertexPartition(g, weights="weight", resolution_parameter=0.5,
                                             initial_membership=partition.membership.tolist())
    assert part.quality() / G.W == pytest.approx(partition.Q_F, abs=1e-10)


# ------------------------------------------------------------ Proposition 3
def test_alpha_cut_domination(G, partition):
    cg = correlation_gap(G, partition.membership, 0.5)
    assert cg["gap"] >= -1e-12
    assert cg["gap"] == pytest.approx(cg["gamma_sum_cov"], abs=1e-9)
    assert cg["gap"] <= cg["cauchy_schwarz_bound"] + 1e-12 <= cg["am_gm_bound"] + 1e-12


def test_equality_in_crisp_case(G, partition):
    Gc = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
    assert fuzzy_modularity(Gc, partition.membership, 0.5) == pytest.approx(
        alpha_average_modularity(Gc, partition.membership, 0.5), abs=1e-12)


# -------------------------------------------------------------- Section 5
def test_collapse_properties():
    ps = np.linspace(0, 1, 101)
    for h in (0.0, 0.05, 0.1, 0.2, 0.3):
        u, r, m = collapse_type1(ps, h)
        assert np.allclose(u + r + m, 1, atol=1e-9)
        assert u.min() >= 0 and m.max() <= 1
        assert np.all(np.diff(m) >= -1e-12) and np.all(np.diff(u) <= 1e-12)
    assert np.allclose(collapse_type1(0.95, 0.0), (0, 0, 1))
    assert np.allclose(collapse_type1(0.05, 0.0), (1, 0, 0))
    assert np.allclose(collapse_type1(0.60, 0.0), (0, 1, 0))


def test_footprint_well_formed_and_collapse_defect():
    ps = np.linspace(0, 1, 201); hs = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    P, H = np.meshgrid(ps, hs, indexing="ij")
    f = footprint(P, H)
    for k, (lo, hi) in f.items():
        assert np.all(lo <= hi + 1e-12) and lo.min() >= 0 and hi.max() <= 1
    assert np.all(f["par"][0][:, 0] == f["par"][1][:, 0])          # point at h = 0
    emb = embedded_in_footprint(P, H)
    assert emb["unk"].all() and emb["mas"].all()
    assert emb["par"].mean() == pytest.approx(0.927, abs=0.001)    # 7.3% leave the footprint


def test_degeneration_threshold():
    assert degeneration_halfwidth(0.4) == pytest.approx(0.075)
    ps = np.linspace(0, 1, 2001)
    for h in (0.05, 0.074, 0.076, 0.10):
        _, r, _ = collapse_type1(ps, h)
        assert r.max() == pytest.approx(max_collapsed_par(h), abs=2e-3)
        assert (r.max() >= 0.4) == (h < 0.075)


def test_it2_gate_three_valued():
    g = zpd_gate_it2(np.linspace(0, 1, 201), 0.1)
    assert set(np.unique(g)) <= {-1, 0, 1}
    assert (g == -1).any()


# -------------------------------------------------------------- Section 6
def test_fact1_and_identities():
    rng = np.random.default_rng(1)
    paths = [rng.uniform(0.05, 1, rng.integers(2, 6)) for _ in range(500)]
    Ts = list(STANDARD_TNORMS.values()) + [yager(p) for p in (1.5, 3, 10)] + [schweizer_sklar(p) for p in (-4, -1, 0.5)]
    for T in Ts:
        for ms in paths:
            assert compose_path(ms, T) <= min(ms) + 1e-12
    g = np.linspace(0.01, 1, 50); X, Y = np.meshgrid(g, g)
    assert np.abs(yager(1)(X, Y) - t_lukasiewicz(X, Y)).max() < 1e-12
    assert np.abs(schweizer_sklar(1)(X, Y) - t_lukasiewicz(X, Y)).max() < 1e-12


def test_widest_path_is_maxmin(G):
    from fuzzykg.tnorms import enumerate_best_paths, t_min
    s, _ = widest_path(G, 0)
    b = enumerate_best_paths(G, 0, 8, t_min); b.pop("__expanded__")
    for v, (st, hop) in b.items():
        assert s[v] >= st - 1e-12


def test_owa_orness():
    assert orness([0.5, 0.3, 0.2]) == pytest.approx(0.65)
    w = owa_weights_from_orness(0.65, 3)
    assert orness(w) == pytest.approx(0.65, abs=1e-6) and w.sum() == pytest.approx(1)


# ------------------------------ generators, planted partitions, thresholds, gates
from fuzzykg.tnorms import generator, generator_best_paths, enumerate_best_paths
from fuzzykg.synthetic import planted_partition_fuzzy_graph
from fuzzykg.modularity import bipartition_threshold_general, community_vs_rest_candidates, _community_sums
from fuzzykg.fuzzification import gate_interval_image_type1, gate_point_estimate


@pytest.mark.parametrize("fam,p", [("product", None), ("lukasiewicz", None), ("yager", 2.0),
                                   ("schweizer_sklar", -2.0), ("schweizer_sklar", 0.5)])
def test_generator_matches_closed_form_and_enumeration(G, fam, p):
    gen = generator(fam, p)
    T = {"product": lambda: STANDARD_TNORMS["product"], "lukasiewicz": lambda: t_lukasiewicz,
         "yager": lambda: yager(p), "schweizer_sklar": lambda: schweizer_sklar(p)}[fam]()
    x = np.random.default_rng(0).random((500, 2))
    assert np.allclose(gen.tnorm()(x[:, 0], x[:, 1]), T(x[:, 0], x[:, 1]), atol=1e-12)
    for s in (0, 7, 42):
        e = enumerate_best_paths(G, s, 4, T); e.pop("__expanded__", None)
        b = generator_best_paths(G, s, gen, max_hops=4)
        assert set(e) == set(b)
        assert all(abs(e[k][0] - b[k][0]) < 1e-12 for k in e)


def test_maxmin_is_ordinally_invariant(G):
    rng = np.random.default_rng(1)
    inc = rng.dirichlet(np.ones(20)); grid = np.linspace(0, 1, 21); vals = np.concatenate([[0], np.cumsum(inc)])
    Gx = FuzzyGraph(G.n, G.src, G.dst, np.clip(np.interp(G.mu, grid, vals), 1e-9, 1))
    for s in (0, 7):
        a, _ = widest_path(G, s); b, _ = widest_path(Gx, s)
        reach = a > 0
        assert np.array_equal(np.argsort(a[reach], kind="stable"), np.argsort(b[reach], kind="stable"))


def test_constant_membership_gap_identity():
    G0, truth = planted_partition_fuzzy_graph(300, 4, 6.0, 0.9, 0.5, 0.1, 0.0, seed=1)
    for c in (0.25, 0.5):
        Gc = FuzzyGraph(G0.n, G0.src, G0.dst, np.full(G0.m, c))
        gap = correlation_gap(Gc, truth, 1.0)
        _, So, Si = _community_sums(Gc, truth); W = Gc.W
        assert abs(gap["gap"] - (1 / c - 1) * np.sum(So / W * Si / W)) < 1e-9
        assert abs(gap["Q_F"] - fuzzy_modularity(FuzzyGraph(Gc.n, Gc.src, Gc.dst, np.ones(Gc.m)), truth)) < 1e-12


def test_general_threshold_reduces_to_balanced_form():
    G, truth = planted_partition_fuzzy_graph(400, 2, 6.0, 0.8, 0.6, 0.1, 0.0, seed=2)
    th = bipartition_threshold_general(G, [truth])
    row = th["thresholds"][0]
    assert abs(row["theta"] - row["two_c_over_W"]) < 0.05       # near-balanced planted split
    cands = community_vs_rest_candidates(np.repeat(np.arange(4), 100))
    assert bipartition_threshold_general(G, cands)["min_theta"] > 0


def test_interval_image_gate_is_it2_pass_rule():
    ps = np.linspace(0, 1, 201)
    for h in (0.0, 0.05, 0.1, 0.2):
        H = np.full_like(ps, h)
        assert np.array_equal(gate_interval_image_type1(ps, H), zpd_gate_it2(ps, H) == 1)
        assert gate_point_estimate(ps, H).mean() == gate_point_estimate(ps, np.zeros_like(ps)).mean()


# ------------------------------------------------------------ Section 5.5: level-set modularity
def test_levelset_objective_matches_optimiser_quality():
    from fuzzykg.synthetic import planted_partition_fuzzy_graph
    from fuzzykg.multislice import multislice_leiden, qbar, level_grid
    G, truth = planted_partition_fuzzy_graph(300, 6, 8.0, 0.7, 0.5, 0.15, 0.3, seed=0)
    r = multislice_leiden(G, seed=0)                       # asserts optimiser quality == mean of per-level Q
    assert abs(r.Qbar - qbar(G, r.membership, r.alphas)) < 1e-10
    assert abs(r.Q_F - fuzzy_modularity(G, r.membership, 1.0)) < 1e-12


def test_levelset_single_cut_is_special_case():
    """With one level the objective is the modularity of that level set."""
    from fuzzykg.synthetic import planted_partition_fuzzy_graph
    from fuzzykg.multislice import qbar
    G, truth = planted_partition_fuzzy_graph(300, 6, 8.0, 0.7, 0.5, 0.15, 0.3, seed=1)
    a = float(np.quantile(G.mu, 0.5))
    r = fuzzy_leiden(G.alpha_cut(a), 1.0, seed=0, n_iterations=0)
    assert abs(qbar(G, r.membership, [a]) - r.Q_F) < 1e-12


def test_lemma3_identity_mass_weights():
    """Lemma 3: mass-weighted Qbar equals sum_k [ sum_l |E_l cap k^2| / M - gamma sum_l f g / r_l ]."""
    from fuzzykg.synthetic import planted_partition_fuzzy_graph
    from fuzzykg.multislice import qbar, level_grid
    G, truth = planted_partition_fuzzy_graph(300, 6, 8.0, 0.7, 0.5, 0.15, 0.3, seed=2)
    alphas = level_grid(G)
    memb = truth
    M = sum(int((G.mu >= a).sum()) for a in alphas)
    total = 0.0
    for a in alphas:
        keep = G.mu >= a
        m = int(keep.sum()); r = m / M
        for k in np.unique(memb):
            inside = keep & (memb[G.src] == k) & (memb[G.dst] == k)
            s_out = (keep & (memb[G.src] == k)).sum() / M
            s_in = (keep & (memb[G.dst] == k)).sum() / M
            total += inside.sum() / M - s_out * s_in / r
    assert abs(total - qbar(G, memb, alphas)) < 1e-9


def test_quantile_grid_is_scale_invariant():
    from fuzzykg.synthetic import planted_partition_fuzzy_graph
    from fuzzykg.multislice import level_grid, qbar
    G, truth = planted_partition_fuzzy_graph(300, 6, 8.0, 0.7, 0.5, 0.15, 0.3, seed=3)
    phi = lambda x: x ** 3                                  # strictly increasing rescaling
    G2 = FuzzyGraph(G.n, G.src, G.dst, phi(G.mu))
    assert abs(qbar(G, truth, level_grid(G)) - qbar(G2, truth, level_grid(G2))) < 1e-12


def test_lfr_style_generator_shapes():
    from fuzzykg.synthetic import lfr_style_fuzzy_graph
    G, truth = lfr_style_fuzzy_graph(500, 0.3, seed=1)
    assert G.n == 500 and truth.size == 500
    assert np.bincount(truth).min() >= 10
    assert 0 < G.mu.min() and G.mu.max() <= 1


def test_order_preserving_null_keeps_orientation():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from null_models_extended import null_draw, curriculum_order
    G = synthetic_fuzzy_graph(200, 0.8, seed=5)            # acyclic by construction
    rank = curriculum_order(G)
    rng = np.random.default_rng(0)
    Gn = null_draw(G, "order", rng, rank, None)
    assert Gn.m == G.m
    assert np.all(rank[Gn.src] < rank[Gn.dst])
    assert np.array_equal(np.bincount(Gn.src, minlength=G.n), np.bincount(G.src, minlength=G.n))
