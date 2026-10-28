"""The compiled counting kernels give the same graph as the reference implementation."""
import numpy as np
import pandas as pd

from fuzzykg.construction import ConstructionConfig, build_fuzzy_graph, build_transition_graph, preprocess


def _log(seed=0, users=60, kcs=8, length=60):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(users):
        k = rng.integers(0, kcs)
        for t in range(length):
            if rng.random() < 0.4:
                k = (k + rng.integers(-1, 3)) % kcs
            rows.append((f"u{u}", f"k{k}", int(rng.random() < 0.6), t))
    return pd.DataFrame(rows, columns=["user_id", "kc", "correct", "order"])


def test_fast_equals_reference():
    df = _log()
    cfg = ConstructionConfig(min_support=5)
    G1, t1 = build_fuzzy_graph(df, cfg, fast=True)
    G2, t2 = build_fuzzy_graph(df, cfg, fast=False)
    a = t1.assign(src=t1.src.astype(str), dst=t1.dst.astype(str)).sort_values(["src", "dst"]).reset_index(drop=True)
    b = t2.assign(src=t2.src.astype(str), dst=t2.dst.astype(str)).sort_values(["src", "dst"]).reset_index(drop=True)
    assert len(a) == len(b) and (a.src == b.src).all() and (a.dst == b.dst).all()
    assert np.allclose(a.mu, b.mu, atol=1e-12)


def test_transition_graph_is_antisymmetric():
    G, tab = build_transition_graph(_log(1))
    pairs = set(zip(G.src.tolist(), G.dst.tolist()))
    assert all((v, u) not in pairs for u, v in pairs)
    assert (G.mu >= 0.2).all() and (G.mu <= 1).all()
