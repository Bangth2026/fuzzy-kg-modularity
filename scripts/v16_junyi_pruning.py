#!/usr/bin/env python3
"""Release 5.1.0. Value of the grades of the Junyi transition graph as a function of the pruning
threshold tau_E of the construction (Section 8.1 of the manuscript, Supplementary Section S4).

The support of the transition graph is itself the alpha-cut of the raw transition relation at
tau_E, so the decomposition (3) at tau_E = 0.2 measures only what the grades add after the
construction has used them to choose the edges. Here the graph is rebuilt at
tau_E in {0 (any positive membership), 0.05, 0.1, 0.2, 0.3} and the decomposition is computed at
the edge and path levels on one fixed set of pairs, the 1,954 crowd-annotated pairs; a pair whose
exercises are not both nodes, or that no path joins, scores 0 under every score.
  edge   direct mu_E(B, A) against the edge indicator;
  path   product best-path score against the hop score 1/(1 + d) on the same support;
nulls: 20 global and 20 row permutations; two-way cluster bootstrap over exercises (500), null
draws fixed. Also the Spearman correlation of every score at every threshold, and the graphs
(written to data/graphs/junyi_transition_tau*_edges.csv).
Writes results/v16/junyi_pruning_value.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.sparse.csgraph import shortest_path
import scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.construction import ConstructionConfig, TransitionConfig, build_transition_graph, preprocess  # noqa: E402
from fuzzykg.nulltest import permute_global, permute_rows  # noqa: E402
from fuzzykg.paths_fast import strengths  # noqa: E402
from data.loaders import load_junyi, load_junyi_expert  # noqa: E402

OUT = "results/v16/junyi_pruning_value.json"
TAUS = (1e-9, 0.05, 0.1, 0.2, 0.3)
N_NULL, B = 20, 500


def wrho(x, y, w):
    k = w > 0
    return spearmanr(np.repeat(x[k], w[k]), np.repeat(y[k], w[k])).statistic


def hop_score(G):
    A = sp.csr_matrix((np.ones(G.m), (G.src, G.dst)), shape=(G.n, G.n))
    D = shortest_path(A, directed=True, unweighted=True)
    S = np.where(np.isfinite(D), 1.0 / (1.0 + D), 0.0); np.fill_diagonal(S, 0.0)
    return S


def prod_score(G):
    S, _ = strengths(G, "product"); S = np.maximum(S, 0.0).copy(); np.fill_diagonal(S, 0.0)
    return S


def main():
    rng = np.random.default_rng(2611)
    _, ann = load_junyi_expert()
    A = ann.Exercise_A.astype(str).to_numpy(); Bn = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    exs = np.unique(np.r_[A, Bn]); pos = {x: i for i, x in enumerate(exs)}
    pa = np.array([pos[x] for x in A]); pb = np.array([pos[x] for x in Bn])
    W = []
    for _ in range(B):
        m = rng.multinomial(exs.size, np.full(exs.size, 1 / exs.size)); W.append(m[pa] * m[pb])
    t0 = time.perf_counter()
    df = load_junyi(); d = preprocess(df, ConstructionConfig()); del df
    print("loaded", f"{time.perf_counter() - t0:.0f}s", flush=True)
    res = {"n_pairs": int(len(y)), "taus": {}}
    for tau in TAUS:
        G, _ = build_transition_graph(d, TransitionConfig(tau_E=tau), preprocessed=True)
        lab = [str(l) for l in G.labels]
        tag = "0" if tau < 1e-6 else f"{tau:g}"
        pd.DataFrame({"src": G.src, "dst": G.dst, "mu": G.mu}).to_csv(f"data/graphs/junyi_transition_tau{tag}_edges.csv", index=False)
        pd.DataFrame({"id": range(G.n), "label": lab}).to_csv(f"data/graphs/junyi_transition_tau{tag}_nodes.csv", index=False)
        idx = {l: i for i, l in enumerate(lab)}
        ok = np.array([a in idx and b in idx for a, b in zip(A, Bn)])
        ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in Bn])
        take = lambda M: np.where(ok, M[ib, ia], 0.0)
        sup = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))

        def direct(H):
            D = np.zeros((H.n, H.n)); D[H.src, H.dst] = H.mu; return take(D)

        X = {"edge": {"support": direct(sup), "observed": direct(G)}, "path": {"support": take(hop_score(sup)), "observed": take(prod_score(G))}}
        nulls = {lev: {"global": [], "row": []} for lev in X}
        for _ in range(N_NULL):
            for kind, fn in (("global", permute_global), ("row", permute_rows)):
                H = fn(G, rng)
                nulls["edge"][kind].append(direct(H)); nulls["path"][kind].append(take(prod_score(H)))

        def comps(w):
            out = {}
            for lev in X:
                q = {k: wrho(X[lev][k], y, w) for k in ("support", "observed")}
                for kind in ("global", "row"):
                    q[kind] = float(np.mean([wrho(x, y, w) for x in nulls[lev][kind]]))
                out[lev] = {**q, "V_total": q["observed"] - q["support"], "V_marg": q["global"] - q["support"],
                            "V_strength": q["row"] - q["global"], "V_resid": q["observed"] - q["row"]}
            return out

        point = comps(np.ones(len(y), int))
        # Monte Carlo SE of the null means (draw-to-draw SD / sqrt(draws))
        for lev in X:
            for kind in ("global", "row"):
                v = [spearmanr(x, y).statistic for x in nulls[lev][kind]]
                point[lev][kind + "_mc_se"] = float(np.std(v, ddof=1) / np.sqrt(len(v)))
        boots = [comps(w) for w in W]
        for lev in point:
            for k in ("V_total", "V_marg", "V_strength", "V_resid"):
                v = [b[lev][k] for b in boots]
                point[lev][k + "_ci"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        # V at this threshold against the hop score of the graph pruned at the default 0.2 is added below
        res["taus"][tag] = {"n": int(G.n), "m": int(G.m), "pairs_in_graph": float(ok.mean()),
                            "mean_out_degree": float(G.m / max(1, G.n)), "value": point,
                            "_boot_obs_path": [float(b["path"]["observed"]) for b in boots],
                            "_boot_sup_path": [float(b["path"]["support"]) for b in boots]}
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        print(tag, G.n, G.m, {lev: {k: round(v, 3) for k, v in point[lev].items() if not k.endswith(("_ci", "_se"))} for lev in point},
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    # differences between thresholds on the same resamples: product at tau minus hop at 0.2
    base = np.array(res["taus"]["0.2"]["_boot_sup_path"])
    for tag, r in res["taus"].items():
        dd = np.array(r["_boot_obs_path"]) - base
        r["product_minus_hop_at_0.2"] = [float(r["value"]["path"]["observed"] - res["taus"]["0.2"]["value"]["path"]["support"]),
                                         float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
    for r in res["taus"].values():
        r.pop("_boot_obs_path"); r.pop("_boot_sup_path")
    json.dump(res, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
