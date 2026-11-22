#!/usr/bin/env python3
"""Release 5.0.0. Value of the grades on the real graphs (Section 6) and the two path results
(Section 5): the connectivity level of max-min and the hop-penalised score.

1. Value of grades (VoG) at three levels, decomposed by nested nulls:
     V_total = Q(observed) - Q(support) = V_marg + V_strength + V_resid,
     V_marg = Q(global) - Q(support), V_strength = Q(row) - Q(global), V_resid = Q(observed) - Q(row),
   with Q a quality (higher is better):
     edge       Spearman of the direct edge mu_E(B, A) (0 without an edge) with the Junyi crowd
                score; support: 1[edge B -> A];
     path       Spearman of the product best-path score; support: the hop distance 1/(1 + d);
     partition  ARI of the best of three Leiden seeds with the Junyi topics or the grouping of the
                graph (best of three seeds by Q_F everywhere; nulls from scripts/v15_partitions.py).
   Nulls: 20 draws (edge, path), 10 draws (partition). Junyi intervals: two-way cluster bootstrap
   over exercises (500 resamples) with the null draws fixed.
2. Connectivity level alpha_c (95% of the connected pairs) and usable share on every graph.
3. Hop-penalised score lam + (1 - lam)(-ln mu) on the Junyi graphs: rho for lam in {0, 0.1, ..., 1};
   lam* cross-fitted over 200 random halvings of the exercises (chosen on pairs inside one half,
   evaluated on pairs inside the other) and a cluster-bootstrap interval of the arg max.
Writes results/v15/value_paths.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.nulltest import permute_global, permute_rows  # noqa: E402
from fuzzykg.paths_fast import strengths  # noqa: E402
from fuzzykg.pathscore import connectivity_level, hop_penalised, score_from_length  # noqa: E402
from data.loaders import load_junyi_expert  # noqa: E402
from v15_partitions import GRAPHS  # noqa: E402
from v15_simplex_real import groups_for  # noqa: E402

OUT = "results/v15/value_paths.json"
LAMS = np.round(np.linspace(0, 1, 11), 2)
JUNYI = ["junyi_transition", "junyi_transition_unsuggested", "junyi_copractice"]


def load(name):
    e = pd.read_csv(f"data/graphs/{name}_edges.csv"); nd = pd.read_csv(f"data/graphs/{name}_nodes.csv")
    return FuzzyGraph.from_edges(zip(e.src, e.dst, e.mu), n=len(nd)), nd.label.astype(str).tolist()


def wrho(x, y, w):
    keep = w > 0
    return spearmanr(np.repeat(x[keep], w[keep]), np.repeat(y[keep], w[keep])).statistic


def product_scores(G):
    S, _ = strengths(G, "product"); S = np.maximum(S, 0.0); return S


def junyi_block(name, ann, rng, B=500, n_null=20):
    G, labels = load(name); idx = {l: i for i, l in enumerate(labels)}
    sel = ann[ann.Exercise_A.isin(idx) & ann.Exercise_B.isin(idx)]
    ia = sel.Exercise_A.map(idx).to_numpy(); ib = sel.Exercise_B.map(idx).to_numpy(); y = sel.Prerequisite_avg.to_numpy()
    K = G.n
    sup = FuzzyGraph(K, G.src, G.dst, np.ones(G.m))

    def direct(H):
        D = np.zeros((K, K)); D[H.src, H.dst] = H.mu; return D[ib, ia]

    X = {"edge": {"support": direct(sup), "observed": direct(G)},
         "path": {"support": score_from_length(hop_penalised(sup, 1.0))[ib, ia], "observed": product_scores(G)[ib, ia]}}
    nulls = {"edge": {"global": [], "row": []}, "path": {"global": [], "row": []}}
    for d in range(n_null):
        for kind, fn in (("global", permute_global), ("row", permute_rows)):
            H = fn(G, rng)
            nulls["edge"][kind].append(direct(H))
            nulls["path"][kind].append(product_scores(H)[ib, ia])
    ex = np.unique(np.r_[ia, ib]); pos = {e: i for i, e in enumerate(ex)}
    pa = np.array([pos[e] for e in ia]); pb = np.array([pos[e] for e in ib])

    def comps(w):
        out = {}
        for lev in ("edge", "path"):
            q = {k: wrho(X[lev][k], y, w) for k in ("support", "observed")}
            for kind in ("global", "row"):
                q[kind] = float(np.mean([wrho(x, y, w) for x in nulls[lev][kind]]))
            out[lev] = {**q, "V_total": q["observed"] - q["support"], "V_marg": q["global"] - q["support"],
                        "V_strength": q["row"] - q["global"], "V_resid": q["observed"] - q["row"]}
        return out

    point = comps(np.ones(len(y), int))
    boots = []
    for _ in range(B):
        m = rng.multinomial(ex.size, np.full(ex.size, 1.0 / ex.size)); boots.append(comps(m[pa] * m[pb]))
    for lev in point:
        for k in ("V_total", "V_marg", "V_strength", "V_resid"):
            v = [b[lev][k] for b in boots]
            point[lev][k + "_ci"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    # hop-penalised family
    R = {}
    L = {lam: score_from_length(hop_penalised(G, float(lam)))[ib, ia] for lam in LAMS}
    R["rho_by_lam"] = {f"{lam:.1f}": float(spearmanr(L[lam], y).statistic) for lam in LAMS}
    sel_l, gain = [], []
    for _ in range(200):
        half = rng.random(ex.size) < 0.5
        inA = half[pa] & half[pb]; inB = ~half[pa] & ~half[pb]
        rA = [spearmanr(L[lam][inA], y[inA]).statistic for lam in LAMS]
        ls = LAMS[int(np.nanargmax(rA))]; sel_l.append(float(ls))
        gain.append(float(spearmanr(L[ls][inB], y[inB]).statistic - spearmanr(L[1.0][inB], y[inB]).statistic))
    R["crossfit_lam"] = {"values_share": {f"{lam:.1f}": float(np.mean(np.isclose(sel_l, lam))) for lam in LAMS},
                         "median": float(np.median(sel_l)),
                         "gain_over_hop_heldout": [float(np.mean(gain)), float(np.percentile(gain, 2.5)), float(np.percentile(gain, 97.5))]}
    am = []
    for _ in range(B):
        m = rng.multinomial(ex.size, np.full(ex.size, 1.0 / ex.size)); w = m[pa] * m[pb]
        am.append(float(LAMS[int(np.nanargmax([wrho(L[lam], y, w) for lam in LAMS]))]))
    R["argmax_lam"] = float(LAMS[int(np.nanargmax(list(R["rho_by_lam"].values())))])
    R["argmax_lam_ci"] = [float(np.percentile(am, 2.5)), float(np.percentile(am, 97.5))]
    R["argmax_lam_share_eq1"] = float(np.mean(np.isclose(am, 1.0)))
    # max-min against the direct edge, and the connectivity level
    S, _ = strengths(G, "min"); S = np.maximum(S, 0.0)
    R["rho_maxmin"] = float(spearmanr(S[ib, ia], y).statistic); R["rho_direct"] = float(spearmanr(X["edge"]["observed"], y).statistic)
    R["rho_product"] = float(spearmanr(X["path"]["observed"], y).statistic); R["rho_hop"] = float(spearmanr(X["path"]["support"], y).statistic)
    Sd = S.copy(); np.fill_diagonal(Sd, 0)
    R["connectivity"] = connectivity_level(G, S=Sd)
    return {"n_pairs": int(len(y)), "value": point, "paths": R}


def partition_value(name):
    gr = groups_for(name)
    if gr is None:
        return None
    store = dict(np.load(f"results/v15/partitions/{name}.npz"))
    e = pd.read_csv(f"data/graphs/{name}_edges.csv")
    G = FuzzyGraph.from_edges(zip(e.src, e.dst, e.mu)); Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
    has = np.array([x is not None for x in gr]); gl = pd.factorize(pd.Series(gr[has]).astype(str))[0]
    b3 = lambda H, w: max([store[f"{w}|1|{s}"] for s in range(3)], key=lambda p: fuzzy_modularity(H, p))
    q = {"support": float(ari(gl, b3(Gs, "support")[has])), "observed": float(ari(gl, b3(G, "F")[has])),
         "global": float(np.mean([ari(gl, store[f"Fglob{d}|1|best3"][has]) for d in range(10)])),
         "row": float(np.mean([ari(gl, store[f"Frow{d}|1|best3"][has]) for d in range(10)]))}
    rowv = [ari(gl, store[f"Frow{d}|1|best3"][has]) for d in range(10)]
    q.update({"V_total": q["observed"] - q["support"], "V_marg": q["global"] - q["support"],
              "V_strength": q["row"] - q["global"], "V_resid": q["observed"] - q["row"],
              "row_sd": float(np.std(rowv, ddof=1)), "row_max": float(np.max(rowv))})
    return q


def main():
    rng = np.random.default_rng(2026)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    _, ann = load_junyi_expert()
    res.setdefault("junyi", {})
    for name in JUNYI:
        if name not in res["junyi"]:
            res["junyi"][name] = junyi_block(name, ann, rng)
            json.dump(res, open(OUT, "w"), indent=1, default=float)
            r = res["junyi"][name]
            print(name, {l: {k: round(v, 3) for k, v in r["value"][l].items() if not k.endswith("_ci")} for l in r["value"]}, flush=True)
            print("  lam", r["paths"]["rho_by_lam"], r["paths"]["crossfit_lam"]["median"], r["paths"]["argmax_lam_ci"], flush=True)
    res["partition"] = {n: v for n in GRAPHS if (v := partition_value(n)) is not None}
    res["connectivity"] = {}
    for n in GRAPHS:
        G, _ = load(n); res["connectivity"][n] = connectivity_level(G)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    for n, v in res["partition"].items():
        print(n, {k: round(x, 3) for k, x in v.items()})


if __name__ == "__main__":
    main()
