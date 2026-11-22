"""Version 4.0.0: structure-matched planted partitions, the three-valued outcome, the
categorical node coding and the audit tool (Algorithm 2)."""
import numpy as np
import pandas as pd

from fuzzykg import FuzzyGraph
from fuzzykg.fastcount import encode
from fuzzykg.nulltest import planted_dc_like, outcome
from fuzzykg.audit import rank_code, cut_audit, audit, report_markdown


def _graph(seed=0, n=60, m=500):
    rng = np.random.default_rng(seed)
    e = set()
    while len(e) < m:
        u = int(rng.integers(0, n)); v = int(rng.integers(0, n))
        if u != v:
            e.add((u, v))
    e = sorted(e)
    return FuzzyGraph.from_edges([(u, v, w) for (u, v), w in zip(e, rng.uniform(0.1, 1, len(e)))], n=n)


def test_dc_planted_keeps_out_degrees_and_sizes():
    G = _graph(1); rng = np.random.default_rng(2)
    sizes = [10, 20, 30]
    H, truth = planted_dc_like(G, sizes, 0.8, rng)
    assert sorted(np.bincount(truth).tolist()) == sizes
    assert (np.bincount(H.src, minlength=G.n) == np.bincount(G.src, minlength=G.n)).all()
    assert (H.src != H.dst).all()
    internal = (truth[H.src] == truth[H.dst]).mean()
    assert internal > 0.6


def test_outcome_rule():
    assert outcome(2.0, 0.5, 0.1) == "no structure detected"
    assert outcome(4.0, 0.05, 0.1) == "excess below floor"
    assert outcome(4.0, 0.2, 0.1) == "structure"
    assert outcome(4.0, 0.2, None) == "floor undefined"


def test_categorical_encode_matches_strings():
    rng = np.random.default_rng(3)
    k = rng.choice(["b", "a", "c", "aa", "d"], 200)
    d1 = pd.DataFrame({"user_id": np.repeat(np.arange(20), 10), "kc": k, "correct": rng.integers(0, 2, 200)})
    d2 = d1.assign(kc=pd.Categorical(k, categories=["d", "c", "b", "aa", "a"]))
    p1, c1, r1, l1 = encode(d1); p2, c2, r2, l2 = encode(d2)
    assert (p1 == p2).all() and (c1 == c2).all() and list(l1) == list(l2)


def test_rank_code_is_rescaling_invariant():
    mu = np.random.default_rng(4).uniform(0.05, 1, 300)
    assert np.allclose(rank_code(mu), rank_code(mu ** 3))


def test_cut_audit_flags_a_cut_that_removes_most_edges():
    G = _graph(5, n=80, m=900)
    rep = cut_audit(G, {"q0.90": float(np.quantile(G.mu, 0.9))}, seeds=range(4))
    r = rep["cuts"]["q0.90"]
    assert r["retained_edges"] < 0.15 and r["low_degree_warning"]
    assert r["qf_drop"] > rep["cuts"]["support"]["qf_drop"]


def test_audit_runs_without_null_test_and_renders():
    G = _graph(6)
    rep = audit(G, null_test=False, seeds=range(3))
    md = report_markdown(rep)
    assert "| q0.50 |" in md and "Rule:" in md


def test_cut_audit_minimum_excess_blocks_tiny_drops():
    G = _graph(7, n=80, m=900)
    cuts = {"q0.90": float(np.quantile(G.mu, 0.9))}
    strict = cut_audit(G, cuts, seeds=range(3), min_excess=10.0)
    assert not strict["cuts"]["q0.90"]["flag_qf_drop"] and "10.0" in strict["rule"]


def test_step3_decision_depends_on_the_unit():
    from fuzzykg.audit import decide
    assert decide(True, False, "meaningful") == "flagged"
    assert decide(False, True, "meaningful") == "cut"
    assert decide(True, False, "arbitrary") == "undecided"
    assert decide(False, True, "arbitrary") == "undecided"
    assert decide(True, True, "arbitrary") == "flagged"
    assert decide(False, False, "arbitrary") == "cut"
    assert decide(True, False, None) == "undecided"


def test_step4_no_loss_and_reference():
    from fuzzykg.audit import step4_no_loss, grouping_reference
    assert step4_no_loss({"ari_grouping": 0.60, "ari_grouping_sd": 0.01}, 0.61, 0.01)
    assert not step4_no_loss({"ari_grouping": 0.50, "ari_grouping_sd": 0.01}, 0.61, 0.01)
    G = _graph(7, n=60, m=500)
    gl = np.arange(G.n) % 3; has = np.ones(G.n, bool)
    ref = grouping_reference(G, gl, has, draws=5)
    assert set(ref) >= {"ari_best3", "perm_p95", "informative"} and ref["draws"] == 5


def test_cut_audit_reports_relative_flags_and_grouping_reference():
    G = _graph(7, n=60, m=500)
    cuts = {"q0.50": float(np.quantile(G.mu, 0.5))}
    grp = np.array([str(i % 3) for i in range(G.n)], dtype=object)
    r = cut_audit(G, cuts, seeds=range(3), grouping=grp, unit="arbitrary", perm_draws=3)
    assert "flag_qf_drop_relative" in r["cuts"]["q0.50"] and "grouping_reference" in r


def test_cut_audit_gamma_default_reproduces_and_changes_partitions():
    from fuzzykg.audit import _leiden
    G = _graph(7, n=80, m=900)
    cuts = {"q0.50": float(np.quantile(G.mu, 0.5))}
    a = cut_audit(G, cuts, seeds=range(3))
    b = cut_audit(G, cuts, seeds=range(3), gamma=1.0)
    assert a["gamma"] == 1.0 and a["cuts"]["q0.50"]["qf_drop"] == b["cuts"]["q0.50"]["qf_drop"]
    assert len(set(_leiden(G, 0, 4.0))) >= len(set(_leiden(G, 0, 0.25)))
    c = cut_audit(G, cuts, seeds=range(3), gamma=2.0)
    assert c["gamma"] == 2.0 and "decision" in c["cuts"]["q0.50"]
