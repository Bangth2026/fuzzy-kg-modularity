#!/usr/bin/env python3
"""Held-out next-concept prediction as an external criterion for the
composition operators (Section 9.3, Table 12).

Learners of an ASSISTments release are split 70 / 10 / 20 (train / validation /
test, by a hash of the learner id).  The fuzzy graph is built on the training
learners only, with the construction of Section 8.1.  For every test learner
and every transition c_prev -> c_next in the log with c_prev != c_next and both
in the graph, the candidates are all other nodes, ranked by the best-path
strength from c_prev under each composition operator (max-min widest path;
generator shortest path for the Archimedean families).  Ties are broken at random (expected reciprocal rank over the tied positions).  Reported: mean reciprocal rank over the test transitions and its learner-level
mean (each test learner weighted equally) with a 95% bootstrap interval over
test learners, and hit@1 / hit@5 / hit@10 over the transitions; the parameters
p of the Yager and Schweizer-Sklar families are chosen on the validation split
and the chosen value is reported with the test score.  Baselines: the direct
edge membership mu_E(c_prev, .) (one hop only), training popularity of the
target, and the uniform random ranking (expected reciprocal rank).

Usage: python scripts/heldout_transitions.py --release 2009 --level kc [--out results/heldout_2009_kc.json]
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

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from data.loaders import load_assistments  # noqa: E402
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.construction import ConstructionConfig, build_fuzzy_graph, preprocess  # noqa: E402
from fuzzykg.tnorms import widest_path, generator, generator_best_paths  # noqa: E402

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


def split_id(uid: str) -> str:
    h = int(hashlib.md5(str(uid).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("val" if h < 80 else "test")


def strength_matrix(G: FuzzyGraph, name: str, p):
    """S[s, t] = best-path strength from s to t under the operator (0 if unreachable)."""
    S = np.zeros((G.n, G.n))
    if name == "min":
        for s in range(G.n):
            S[s] = widest_path(G, s)[0]
    elif name == "min_hopbreak":
        # max-min strength, ties broken by the hop count of the widest path (shorter first);
        # the lexicographic order is encoded as strength - hops * 1e-6, which never crosses a strength level
        for s in range(G.n):
            st, pred = widest_path(G, s)
            hops = np.zeros(G.n)
            for v in range(G.n):
                if st[v] > 0:
                    h, u = 0, v
                    while u != s and u >= 0:
                        u = pred[u]; h += 1
                    hops[v] = h
            S[s] = np.where(st > 0, st - 1e-6 * hops, 0.0)
    else:
        gen = generator(name, p)
        for s in range(G.n):
            for t, (v, _) in generator_best_paths(G, s, gen).items():
                S[s, t] = v
    np.fill_diagonal(S, -1.0)        # the source itself is never a candidate
    return S


def rank_stats(scores: np.ndarray, target: int):
    """Expected reciprocal rank and hit@k of ``target`` under random tie-breaking:
    the target's rank is uniform on {higher+1, ..., higher+ties}."""
    v = scores[target]
    higher = int(np.sum(scores > v))
    ties = int(np.sum(scores == v))           # includes the target
    ranks = np.arange(higher + 1, higher + ties + 1)
    return float(np.mean(1.0 / ranks)), float(np.mean(ranks <= 1)), float(np.mean(ranks <= 5)), float(np.mean(ranks <= 10))


def evaluate(S: np.ndarray, transitions, learners):
    rr, h1, h5, h10, per_learner = [], [], [], [], {}
    for (s, t, uid) in transitions:
        a, b, c, d = rank_stats(S[s], t)
        rr.append(a); h1.append(b); h5.append(c); h10.append(d)
        per_learner.setdefault(uid, []).append(a)
    rr = np.asarray(rr)
    rng = np.random.default_rng(0)
    means = np.array([np.mean(per_learner[u]) for u in learners if u in per_learner])
    boots = [np.mean(rng.choice(means, size=means.size, replace=True)) for _ in range(1000)]
    return {"mrr": float(rr.mean()), "mrr_learner_mean": float(means.mean()),
            "mrr_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "hit1": float(np.mean(h1)), "hit5": float(np.mean(h5)), "hit10": float(np.mean(h10)),
            "n_transitions": int(rr.size)}


def transitions_of(d: pd.DataFrame, node_index, uids):
    out = []
    for uid, g in d[d.user_id.isin(uids)].groupby("user_id", sort=False):
        kcs = [str(x) for x in g["kc"].tolist()]
        for a, b in zip(kcs[:-1], kcs[1:]):
            if a != b and a in node_index and b in node_index:
                out.append((node_index[a], node_index[b], uid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--level", default="kc")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-test-learners", type=int, default=2000)
    args = ap.parse_args()
    out = args.out or f"results/heldout_{args.release}_{args.level}.json"
    cfg = ConstructionConfig()
    df = load_assistments(args.release, args.level)
    d = preprocess(df, cfg)
    d["split"] = d["user_id"].map(split_id)
    train = d[d.split == "train"].drop(columns=["split"])
    t0 = time.perf_counter()
    G, _ = build_fuzzy_graph(train, cfg)
    build_time = time.perf_counter() - t0
    node_index = {lab: i for i, lab in enumerate(G.labels)}
    val_uids = sorted(d[d.split == "val"].user_id.unique())
    test_uids = sorted(d[d.split == "test"].user_id.unique())[: args.max_test_learners]
    val_tr = transitions_of(d, node_index, val_uids)
    test_tr = transitions_of(d, node_index, test_uids)
    print(f"{args.release} {args.level}: graph on train {G.summary()} ({build_time:.1f}s); "
          f"val {len(val_tr)} transitions / {len(val_uids)} learners; test {len(test_tr)} / {len(test_uids)}", flush=True)
    res = {"release": args.release, "level": args.level, "graph_train": G.summary(), "n_nodes": G.n,
           "n_val_transitions": len(val_tr), "n_test_transitions": len(test_tr),
           "n_test_learners": len(test_uids), "operators": {}, "baselines": {}, "selected": {}}
    # operators
    for key, (name, p) in OPERATORS.items():
        S = strength_matrix(G, name, p)
        res["operators"][key] = evaluate(S, test_tr, test_uids)
        res["operators"][key]["val_mrr"] = evaluate(S, val_tr, val_uids)["mrr"]
        print(key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["operators"][key].items() if k != "mrr_ci"}), flush=True)
    # baselines
    A = G.adjacency().toarray(); np.fill_diagonal(A, -1.0)
    res["baselines"]["direct_edge"] = evaluate(A, test_tr, test_uids)
    pop = train["kc"].value_counts()
    pop = {str(k): v for k, v in pop.items()}
    P = np.tile(np.array([pop.get(lab, 0) for lab in G.labels], dtype=float), (G.n, 1)); np.fill_diagonal(P, -1.0)
    res["baselines"]["popularity"] = evaluate(P, test_tr, test_uids)
    res["baselines"]["random_expected_rr"] = float(np.mean(1.0 / np.arange(1, G.n)))
    # p selected on validation
    for fam, grid in (("yager", YAGER_GRID), ("schweizer_sklar", SS_GRID)):
        curve = {}
        for p in grid:
            S = strength_matrix(G, fam, p)
            curve[p] = {"val_mrr": evaluate(S, val_tr, val_uids)["mrr"], "test": evaluate(S, test_tr, test_uids)}
        best = max(curve, key=lambda q: curve[q]["val_mrr"])
        res["selected"][fam] = {"p": best, "curve": {str(p): {"val_mrr": c["val_mrr"], "test_mrr": c["test"]["mrr"]} for p, c in curve.items()},
                                "test": curve[best]["test"]}
        print(fam, "selected p =", best, "test MRR", round(curve[best]["test"]["mrr"], 4), flush=True)
    # tie share under max-min on the test sources
    S = strength_matrix(G, "min", None)
    ties = []
    for s in sorted(set(t[0] for t in test_tr)):
        v = S[s]; v = v[v > 0]
        if v.size >= 2:
            _, cnt = np.unique(v, return_counts=True)
            ties.append(float((cnt * (cnt - 1) / 2).sum() / (v.size * (v.size - 1) / 2)))
    res["maxmin_tie_share_test_sources"] = float(np.mean(ties)) if ties else None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
