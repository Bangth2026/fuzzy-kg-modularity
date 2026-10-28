#!/usr/bin/env python3
"""Held-out next-concept prediction as an external criterion for the
composition operators (Section 8.5, Table 13).

Learners are split 70 / 10 / 20 (train / validation / test, by a hash of the
learner id).  The fuzzy graph is built on the training learners only, with the
co-practice construction of Section 7.1 or the order-preserving transition
construction.  For every test learner and every transition c_prev -> c_next in
the log with c_prev != c_next and both in the graph, the candidates are all
other nodes, ranked by the best-path strength from c_prev under each composition
operator (max-min widest path; generator shortest path for the Archimedean
families), optionally with a hop limit L and a floor tau_path on the edges a path
may use (the configuration of Equation (14)).  Ties are broken at random
(expected reciprocal rank over the tied positions).

Reported: mean reciprocal rank per test learner averaged over learners, with a
95% bootstrap interval over learners; hit@10 over the transitions; the Yager and
Schweizer-Sklar parameters selected on validation data; baselines (direct edge,
training popularity, training bigram N_train(c_prev -> .), uniform random);
diagnostics (dropped transitions, unreachable targets, one-hop share, mean hop
count of the best path to the true target, tau-b against the direct-edge ranking,
max-min tie share); and paired bootstrap comparisons over learners (each
operator minus the direct edge; the bigram minus the best operator) with Holm
adjustment over the comparisons of one graph.

Usage: python scripts/heldout_transitions.py --release 2009 --level kc
       [--construction copractice|transition] [--max-hops L] [--tau-path 0.3] [--out ...]
       python scripts/heldout_transitions.py --release junyi --construction transition
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from data.loaders import load_assistments, load_junyi  # noqa: E402
from fuzzykg.construction import (ConstructionConfig, TransitionConfig, build_fuzzy_graph,  # noqa: E402
                                  build_transition_graph, preprocess)
from fuzzykg.paths_fast import strengths  # noqa: E402

OPERATORS = {
    "min": ("min", None),
    "min_hopbreak": ("min_hopbreak", None),
    "product": ("product", None),
    "lukasiewicz": ("lukasiewicz", None),
    "yager_2": ("yager", 2.0), "yager_5": ("yager", 5.0), "yager_10": ("yager", 10.0),
    "ss_-8": ("schweizer_sklar", -8.0), "ss_-2": ("schweizer_sklar", -2.0), "ss_0.5": ("schweizer_sklar", 0.5),
}
YAGER_GRID = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
SS_GRID = [-50.0, -20.0, -8.0, -4.0, -2.0, -1.0, 0.5, 1.0]


def split_id(uid) -> str:
    h = int(hashlib.md5(str(uid).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("val" if h < 80 else "test")


def strength_matrix(G, name, p, max_hops=None, tau_path=0.0):
    if name == "min_hopbreak":
        S, H = strengths(G, "min", None, max_hops, tau_path)
        # max-min strength, ties broken by the hop count of the widest path (fewer first);
        # 1e-6 per hop never crosses a strength level
        S2 = np.where(S > 0, S - 1e-6 * H, S)
        return S2, H
    return strengths(G, name, p, max_hops, tau_path)


def rank_stats(S, src, tgt):
    """Vectorised expected reciprocal rank and hit@10 under random tie-breaking."""
    rr = np.empty(src.size); h10 = np.empty(src.size)
    n = S.shape[1]
    harm = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 1))])
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        row = np.sort(S[s])
        v = S[s, tgt[idx]]
        above = n - np.searchsorted(row, v, side="right")
        ties = np.searchsorted(row, v, side="right") - np.searchsorted(row, v, side="left")
        rr[idx] = (harm[above + ties] - harm[above]) / ties
        # P(rank <= 10) with the target uniform among its ties
        h10[idx] = np.clip(10 - above, 0, ties) / ties
    return rr, h10


def evaluate(S, tr, A=None, H=None, B=1000, rng_seed=0):
    src, tgt, uid = tr
    rr, h10 = rank_stats(S, src, tgt)
    df = pd.DataFrame({"u": uid, "rr": rr})
    per = df.groupby("u", sort=True)["rr"].mean()
    rng = np.random.default_rng(rng_seed)
    m = per.to_numpy()
    boots = m[rng.integers(0, m.size, (B, m.size))].mean(axis=1)
    out = {"mrr": float(rr.mean()), "mrr_learner_mean": float(m.mean()),
           "mrr_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
           "hit10": float(h10.mean()), "n_transitions": int(rr.size)}
    if A is not None:
        a = A[src, tgt]
        out["onehop_share"] = float(np.mean((a > 0) & (np.abs(S[src, tgt] - a) <= 1e-6 * np.maximum(1.0, a) + 1e-9)))
        taus = []
        for s in np.unique(src):
            mask = np.ones(S.shape[1], dtype=bool); mask[s] = False
            x, y = S[s][mask], A[s][mask]
            if np.unique(x).size > 1 and np.unique(y).size > 1:
                taus.append(kendalltau(x, y).statistic)
        out["tau_vs_direct_edge"] = float(np.nanmean(taus)) if taus else None
    if H is not None:
        hh = H[src, tgt]
        out["mean_hops_to_target"] = float(hh[hh > 0].mean()) if (hh > 0).any() else None
    return out, per


def transitions_of(d, node_index, uids):
    sub = d[d.user_id.isin(set(uids))]
    k = sub["kc"].astype(str).to_numpy(); u = sub["user_id"].to_numpy()
    same = u[1:] == u[:-1]
    diff = k[1:] != k[:-1]
    sel = np.flatnonzero(same & diff)
    a, b, uu = k[sel], k[sel + 1], u[sel]
    ia = np.array([node_index.get(x, -1) for x in a]); ib = np.array([node_index.get(x, -1) for x in b])
    ok = (ia >= 0) & (ib >= 0)
    return (ia[ok], ib[ok], uu[ok]), int((~ok).sum())


def paired(per_a, per_b, B=2000, rng_seed=1):
    """Mean over learners of (a - b), 95% bootstrap interval and two-sided bootstrap p."""
    j = per_a.index.intersection(per_b.index)
    d = (per_a.loc[j] - per_b.loc[j]).to_numpy()
    rng = np.random.default_rng(rng_seed)
    boots = d[rng.integers(0, d.size, (B, d.size))].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(d.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p": float(max(p, 1.0 / B))}


def holm(ps):
    keys = sorted(ps, key=ps.get); m = len(keys); out = {}; prev = 0.0
    for i, k in enumerate(keys):
        v = min(1.0, max(prev, (m - i) * ps[k])); out[k] = v; prev = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--level", default="kc")
    ap.add_argument("--construction", default="copractice", choices=["copractice", "transition"])
    ap.add_argument("--max-hops", type=int, default=None)
    ap.add_argument("--tau-path", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--min-support", type=int, default=None)
    ap.add_argument("--max-test-learners", type=int, default=2000)
    ap.add_argument("--no-grid", action="store_true", help="skip the validation grids of the two families")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = f"{args.release}_{args.level}" if args.release != "junyi" else "junyi"
    if args.construction != "copractice":
        tag += f"_{args.construction}"
    if args.max_hops:
        tag += f"_L{args.max_hops}"
    if args.tau_path:
        tag += f"_tau{args.tau_path:g}"
    out = args.out or f"results/heldout_{tag}.json"
    cfg = ConstructionConfig()
    if args.horizon:
        cfg.horizon = args.horizon
    if args.min_support:
        cfg.min_support = args.min_support
    df = load_junyi() if args.release == "junyi" else load_assistments(args.release, args.level)
    d = preprocess(df, cfg)
    d["split"] = d["user_id"].map(split_id)
    train = d[d.split == "train"].drop(columns=["split"])
    t0 = time.perf_counter()
    if args.construction == "copractice":
        G, _ = build_fuzzy_graph(train, cfg, preprocessed=True)
    else:
        tc = TransitionConfig()
        if args.min_support:
            tc.min_support = args.min_support
        G, _ = build_transition_graph(train, tc, preprocessed=True)
    build_time = time.perf_counter() - t0
    node_index = {lab: i for i, lab in enumerate(G.labels)}
    val_uids = sorted(d[d.split == "val"].user_id.unique())
    test_uids = sorted(d[d.split == "test"].user_id.unique())[: args.max_test_learners]
    val_tr, _ = transitions_of(d, node_index, val_uids)
    test_tr, test_dropped = transitions_of(d, node_index, test_uids)
    train_tr, _ = transitions_of(d, node_index, train.user_id.unique())
    print(f"{tag}: graph on train {G.summary()} ({build_time:.1f}s); val {val_tr[0].size}; test {test_tr[0].size} / {len(test_uids)}", flush=True)
    res = {"release": args.release, "level": args.level, "construction": args.construction,
           "max_hops": args.max_hops, "tau_path": args.tau_path, "horizon": cfg.horizon, "min_support": cfg.min_support,
           "graph_train": G.summary(), "n_nodes": G.n, "n_edges": G.m,
           "n_val_transitions": int(val_tr[0].size), "n_test_transitions": int(test_tr[0].size),
           "n_test_learners": int(len(set(test_tr[2]))), "n_test_transitions_dropped": test_dropped,
           "n_nodes_all_learners": int(d["kc"].astype(str).nunique()),
           "operators": {}, "baselines": {}, "selected": {}, "paired": {}}
    A = G.adjacency().toarray()
    keepA = A >= args.tau_path
    A = np.where(keepA, A, 0.0); np.fill_diagonal(A, -1.0)
    S_min, _ = strength_matrix(G, "min", None, args.max_hops, args.tau_path)
    res["unreachable_share_test"] = float(np.mean(S_min[test_tr[0], test_tr[1]] <= 0))
    per = {}
    for key, (name, p) in OPERATORS.items():
        S, H = strength_matrix(G, name, p, args.max_hops, args.tau_path)
        r, per[key] = evaluate(S, test_tr, A=A, H=H)
        r["val_mrr"] = evaluate(S, val_tr)[0]["mrr"]
        res["operators"][key] = r
        print(key, round(r["mrr_learner_mean"], 4), flush=True)
    r, per["direct_edge"] = evaluate(A, test_tr, A=A); res["baselines"]["direct_edge"] = r
    Bm = np.zeros((G.n, G.n)); np.add.at(Bm, (train_tr[0], train_tr[1]), 1.0); np.fill_diagonal(Bm, -1.0)
    r, per["bigram"] = evaluate(Bm, test_tr, A=A); r["val_mrr"] = evaluate(Bm, val_tr)[0]["mrr"]; res["baselines"]["bigram"] = r
    pop = train["kc"].astype(str).value_counts()
    P = np.tile(np.array([pop.get(lab, 0) for lab in G.labels], dtype=float), (G.n, 1)); np.fill_diagonal(P, -1.0)
    r, per["popularity"] = evaluate(P, test_tr); res["baselines"]["popularity"] = r
    res["baselines"]["random_expected_rr"] = float(np.mean(1.0 / np.arange(1, G.n)))
    # coverage-only baseline: unweighted hop distance on the training graph (no membership)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    Ah = csr_matrix((np.ones(G.m), (G.src, G.dst)), shape=(G.n, G.n)) if not args.tau_path else csr_matrix(
        (np.ones(int((G.mu >= args.tau_path).sum())), (G.src[G.mu >= args.tau_path], G.dst[G.mu >= args.tau_path])), shape=(G.n, G.n))
    Dh = shortest_path(Ah, directed=True, unweighted=True)
    if args.max_hops:
        Dh[Dh > args.max_hops] = np.inf
    Hs = np.where(np.isfinite(Dh), 1.0 / (1.0 + Dh), 0.0); np.fill_diagonal(Hs, -1.0)
    r, per["hop_baseline"] = evaluate(Hs, test_tr, A=A); r["val_mrr"] = evaluate(Hs, val_tr)[0]["mrr"]; res["baselines"]["hop_baseline"] = r
    print("bigram", round(res["baselines"]["bigram"]["mrr_learner_mean"], 4), "direct", round(res["baselines"]["direct_edge"]["mrr_learner_mean"], 4), flush=True)
    if not args.no_grid:
        for fam, grid in (("yager", YAGER_GRID), ("schweizer_sklar", SS_GRID)):
            curve = {}
            for p in grid:
                S, _ = strength_matrix(G, fam, p, args.max_hops, args.tau_path)
                curve[p] = {"val_mrr": evaluate(S, val_tr)[0]["mrr"], "test": evaluate(S, test_tr)[0]}
            best = max(curve, key=lambda q: curve[q]["val_mrr"])
            res["selected"][fam] = {"p": best, "curve": {str(p): {"val_mrr": c["val_mrr"], "test_mrr": c["test"]["mrr"]} for p, c in curve.items()},
                                    "test": curve[best]["test"]}
    # paired comparisons
    ps = {}
    for key in OPERATORS:
        c = paired(per[key], per["direct_edge"]); res["paired"][f"{key}-direct_edge"] = c; ps[f"{key}-direct_edge"] = c["p"]
    best_op = max(OPERATORS, key=lambda k: res["operators"][k]["mrr_learner_mean"])
    c = paired(per["bigram"], per[best_op]); res["paired"][f"bigram-{best_op}"] = c; ps[f"bigram-{best_op}"] = c["p"]
    c = paired(per["min"], per["direct_edge"]); ps["min-direct_edge"] = c["p"]
    # operator chosen on validation learners, evaluated on test learners
    val_op = max(OPERATORS, key=lambda k: res["operators"][k]["val_mrr"])
    res["validation_selected_operator"] = val_op
    c = paired(per[val_op], per["hop_baseline"]); res["paired"][f"{val_op}-hop_baseline"] = c; ps[f"{val_op}-hop_baseline"] = c["p"]
    c = paired(per["hop_baseline"], per["direct_edge"]); res["paired"]["hop_baseline-direct_edge"] = c; ps["hop_baseline-direct_edge"] = c["p"]
    for k, v in holm(ps).items():
        res["paired"][k]["p_holm"] = v
    res["best_operator"] = best_op
    ties = []
    for s in np.unique(test_tr[0]):
        v = S_min[s]; v = v[v > 0]
        if v.size >= 2:
            _, cnt = np.unique(v, return_counts=True)
            ties.append(float((cnt * (cnt - 1) / 2).sum() / (v.size * (v.size - 1) / 2)))
    res["maxmin_tie_share_test_sources"] = float(np.mean(ties)) if ties else None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
