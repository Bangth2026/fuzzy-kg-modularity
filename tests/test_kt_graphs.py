"""Unit tests of the graphs used by the knowledge-tracing experiments (kt/src): the order-preserving support graph of
Eq. (2), the degree-preserving rewiring, the Erdos-Renyi null, the unit-preserving null and the two constructions of
release 3.13.0. Run with: PYTHONPATH=. pytest tests/test_kt_graphs.py (needs torch for the import of kt/src)."""
import os, sys
import numpy as np, pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kt', 'src'))
torch = pytest.importorskip('torch')
from r5_controls import transition_support, rewire          # noqa: E402
from r6_controls import erdos_renyi                          # noqa: E402
import r14_controls as R14                                   # noqa: E402
import r16_controls as R16                                   # noqa: E402


def logs(seed=0, n=60, K=12, L=40):
    rng = np.random.default_rng(seed); rows = []
    for u in range(n):
        q = np.sort(rng.integers(0, K, L)) if u % 2 else rng.integers(0, K, L)   # half the learners follow the order 0..K-1
        for t, k in enumerate(q): rows.append(dict(user=f'u{u}', kc=int(k), correct=int(rng.random() < .6 + .02 * k), order=t))
    return pd.DataFrame(rows), K


def test_support_is_antisymmetric_and_zero_diagonal():
    df, K = logs(); mu = transition_support(df, K)
    assert np.all(np.diag(mu) == 0)
    assert not np.any((mu > 0) & (mu.T > 0))           # only the dominant direction of a pair can be an edge
    assert np.all((mu >= 0) & (mu <= 1))


def test_support_threshold():
    df, K = logs(); T = np.zeros((K, K))
    for _, g in df.groupby('user', sort=False):
        q = g.kc.values; r = q[np.r_[True, q[1:] != q[:-1]]]; np.add.at(T, (r[:-1], r[1:]), 1)
    mu = transition_support(df, K, s=10)
    assert np.all(mu[(T + T.T) < 10] == 0)


def test_rewire_keeps_degrees():
    df, K = logs(); A = (transition_support(df, K) > 0).astype(np.float32)
    R, done = rewire(A, seed=1)
    assert R.sum() == A.sum() and done > 0
    assert np.array_equal(R.sum(0), A.sum(0)) and np.array_equal(R.sum(1), A.sum(1))
    assert np.all(np.diag(R) == 0) and not np.any((R > 0) & (R.T > 0))


def test_erdos_renyi_size():
    A = erdos_renyi(20, 37, seed=3)
    assert A.sum() == 37 and np.all(np.diag(A) == 0) and not np.any((A > 0) & (A.T > 0))


def test_unit_random_keeps_within_unit_count():
    df, K = logs(); sup = (transition_support(df, K) > 0).astype(np.float32)
    grp = np.array([str(k // 4) for k in range(K)]); same = grp[:, None] == grp[None, :]
    U = R14.unit_random(sup, grp, R14.first_encounter_rank(df, K), seed=7)
    assert U.sum() == sup.sum() and (U * same).sum() == (sup * same).sum()


def test_top_pairs_and_corr_graph():
    W = np.random.default_rng(0).random((9, 9)); np.fill_diagonal(W, 0)
    A = R16.top_pairs(W, 11)
    assert A.sum() == 11 and not np.any((A > 0) & (A.T > 0)) and np.all(np.diag(A) == 0)
    df, K = logs(); C = R16.corr_graph(df, K, 15, min_common=5)
    assert C.sum() == 15 and not np.any((C > 0) & (C.T > 0))
