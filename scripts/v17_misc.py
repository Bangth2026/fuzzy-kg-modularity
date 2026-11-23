#!/usr/bin/env python3
"""Release 5.2.0. Analyses of the revised manuscript that need no new learner log.

1. tau_crossfit   Junyi transition at the five pruning thresholds (data/graphs/junyi_transition_tau*),
                  fixed 1,954 annotated pairs. Best crisp representation = hop score 1/(1 + d) at the
                  tau_E that maximises rho; best graded = product best-path score at its best tau_E.
                  Point values, and 200 random halvings of the exercises: both thresholds chosen on the
                  pairs inside one half and evaluated on the pairs inside the other (held-out
                  difference graded - crisp, and the thresholds chosen).
2. block_null     Partition value of the grades against a grouping-class null that permutes the
                  memberships separately among the edges inside one group and among the edges between
                  groups: if the observed ARI is reached by this null, the partition value is explained
                  by grades being higher (or lower) inside the groups, not by which edge carries which
                  grade. Junyi transition, Junyi transition without suggested attempts, EdNet co-practice;
                  ten draws, best of three seeds. Also the full decomposition (3) on the unsuggested graph.
3. sparse_delta   Robust decisions on the 23 real graphs with the sparse minimum excess set to 0.03
                  (the value selected most often by the cross-validation) instead of 0.0075.
Writes results/v17/misc.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.sparse.csgraph import shortest_path
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden  # noqa: E402
from fuzzykg.nulltest import permute_global, permute_rows  # noqa: E402
from fuzzykg.paths_fast import strengths  # noqa: E402
from data.loaders import load_junyi_expert  # noqa: E402
from v15_simplex_real import groups_for  # noqa: E402

OUT = "results/v17/misc.json"
TAGS = ["0", "0.05", "0.1", "0.2", "0.3"]


def load_tau(tag):
    e = pd.read_csv(f"data/graphs/junyi_transition_tau{tag}_edges.csv"); nd = pd.read_csv(f"data/graphs/junyi_transition_tau{tag}_nodes.csv")
    return FuzzyGraph(len(nd), e.src.to_numpy(), e.dst.to_numpy(), e.mu.to_numpy()), nd.label.astype(str).tolist()


def tau_crossfit(rng):
    _, ann = load_junyi_expert()
    A = ann.Exercise_A.astype(str).to_numpy(); B = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    X = {}
    for t in TAGS:
        G, lab = load_tau(t); idx = {l: i for i, l in enumerate(lab)}
        ok = np.array([a in idx and b in idx for a, b in zip(A, B)])
        ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in B])
        Adj = sp.csr_matrix((np.ones(G.m), (G.src, G.dst)), shape=(G.n, G.n))
        D = shortest_path(Adj, directed=True, unweighted=True); H = np.where(np.isfinite(D), 1 / (1 + D), 0.0)
        P, _ = strengths(G, "product"); P = np.maximum(P, 0.0)
        X[("hop", t)] = np.where(ok, H[ib, ia], 0.0); X[("product", t)] = np.where(ok, P[ib, ia], 0.0)
    rho = lambda x, m: float(spearmanr(x[m], y[m]).statistic)
    allm = np.ones(len(y), bool)
    point = {f"{k}|{t}": rho(X[(k, t)], allm) for (k, t) in X}
    exs = np.unique(np.r_[A, B]); pos = {e: i for i, e in enumerate(exs)}
    pa = np.array([pos[a] for a in A]); pb = np.array([pos[b] for b in B])
    diffs, choice = [], []
    for _ in range(200):
        h = rng.random(exs.size) < 0.5
        mA = h[pa] & h[pb]; mB = ~h[pa] & ~h[pb]
        tc = max(TAGS, key=lambda t: rho(X[("hop", t)], mA)); tg = max(TAGS, key=lambda t: rho(X[("product", t)], mA))
        diffs.append(rho(X[("product", tg)], mB) - rho(X[("hop", tc)], mB)); choice.append((tc, tg))
    from collections import Counter
    return {"point": point, "best_crisp": max(TAGS, key=lambda t: point[f"hop|{t}"]), "best_graded": max(TAGS, key=lambda t: point[f"product|{t}"]),
            "heldout_diff": [float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_positive": float(np.mean(np.array(diffs) > 0)),
            "choices": {f"{a}|{b}": c / 200 for (a, b), c in Counter(choice).items()}}


def permute_within(G, cls, rng):
    mu = G.mu.copy()
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c); mu[idx] = mu[rng.permutation(idx)]
    return FuzzyGraph(G.n, G.src, G.dst, mu)


def best3(H):
    P = [_leiden(H, k) for k in range(3)]
    return max(P, key=lambda p: fuzzy_modularity(H, p))


def block_null(name, rng):
    e = pd.read_csv(f"data/graphs/{name}_edges.csv")
    G = FuzzyGraph.from_edges(zip(e.src, e.dst, e.mu)); Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
    gr = groups_for(name); has = np.array([x is not None for x in gr]); gl = pd.factorize(pd.Series(gr[has]).astype(str))[0]
    g_all = np.array([str(x) if x is not None else f"__none{i}" for i, x in enumerate(gr)])
    cls = (g_all[G.src] == g_all[G.dst]).astype(int)
    a = lambda p: float(ari(gl, p[has]))
    out = {"support": a(best3(Gs)), "observed": a(best3(G)), "share_internal_edges": float(cls.mean()),
           "mu_internal_mean": float(G.mu[cls == 1].mean()), "mu_between_mean": float(G.mu[cls == 0].mean())}
    for kind, fn in (("global", lambda: permute_global(G, rng)), ("row", lambda: permute_rows(G, rng)), ("block", lambda: permute_within(G, cls, rng))):
        v = [a(best3(fn())) for _ in range(10)]
        out[kind] = float(np.mean(v)); out[kind + "_sd"] = float(np.std(v, ddof=1))
    out.update({"V_total": out["observed"] - out["support"], "V_marg": out["global"] - out["support"], "V_strength": out["row"] - out["global"],
                "V_resid": out["observed"] - out["row"], "V_beyond_block": out["observed"] - out["block"], "block_minus_support": out["block"] - out["support"]})
    return out


def sparse_delta():
    R = json.load(open("results/v15/simplex_real.json"))["graphs"]
    cnt = {"cut": 0, "flagged": 0, "undecided": 0}; changed = []
    for n, v in R.items():
        dl = "0.03"
        if v["group"] == "co-practice":
            continue
        for c in ("q0.25", "q0.50", "q0.75"):
            bd = v["gammas"]["1"][c]["by_delta"]
            rob = lambda d: bd[d]["decision"] if bd[d]["kappa_star"] >= 2 else "undecided"
            cnt[rob(dl)] += 1
            if rob(dl) != rob("0.0075"):
                changed.append([n, c, rob("0.0075"), rob(dl)])
    return {"sparse_and_curated_counts_at_0.03": cnt, "changed_vs_0.0075": changed}


def main():
    os.makedirs("results/v17", exist_ok=True)
    rng = np.random.default_rng(5200)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res["tau_crossfit"] = tau_crossfit(rng); json.dump(res, open(OUT, "w"), indent=1)
    print("tau", res["tau_crossfit"]["heldout_diff"], res["tau_crossfit"]["best_crisp"], res["tau_crossfit"]["best_graded"], flush=True)
    res["block_null"] = {}
    for n in ("junyi_transition", "junyi_transition_unsuggested", "ednet_copractice"):
        res["block_null"][n] = block_null(n, rng); json.dump(res, open(OUT, "w"), indent=1)
        print(n, {k: round(v, 3) for k, v in res["block_null"][n].items()}, flush=True)
    res["sparse_delta"] = sparse_delta(); json.dump(res, open(OUT, "w"), indent=1)
    print(res["sparse_delta"])


if __name__ == "__main__":
    main()
