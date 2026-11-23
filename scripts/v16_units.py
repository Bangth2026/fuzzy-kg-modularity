#!/usr/bin/env python3
"""Release 5.1.0. Dependence of the path and partition values of the grades on the unit of mu_E
(Section 8.1, Supplementary Section S4).

The edge level is unit-free (Spearman correlations and both permutation nulls are invariant under
an increasing rescaling). The path and partition levels are not: best paths under the product
change with the unit, and so does the Leiden partition on Q_F. Here mu_E is replaced by
phi_s(mu_E) for the 18 vertices of the box B_2 that have one segment steeper or one segment flatter
than the rest (s = 1 + e_j and s = 2 - e_j, knots of Section 4.2), rescaled so that the largest
membership keeps its value (memberships stay in (0, 1]); the whole decomposition (3) is recomputed
on each.
  partition  Junyi transition and EdNet co-practice (the graphs with a positive partition value):
             best of three Leiden seeds on Q_F, ten global and ten row draws (best of three);
  path       Junyi co-practice and Junyi transition: product best-path score against the hop
             distance, 20 draws per null (point values, no bootstrap).
Writes results/v16/units.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden  # noqa: E402
from fuzzykg.nulltest import permute_global, permute_rows  # noqa: E402
from fuzzykg.simplex import segment_basis  # noqa: E402
from fuzzykg.pathscore import hop_penalised, score_from_length  # noqa: E402
from data.loaders import load_junyi_expert  # noqa: E402
from v15_value_paths import load, product_scores  # noqa: E402
from v15_simplex_real import groups_for  # noqa: E402

OUT = "results/v16/units.json"


def vertices(J):
    V = [("mu", np.ones(J))]
    for j in range(J):
        s = np.ones(J); s[j] = 2.0; V.append((f"steep{j + 1}", s))
    for j in range(J):
        s = 2.0 * np.ones(J); s[j] = 1.0; V.append((f"flat{j + 1}", s))
    return V


def rescaled(G, s):
    _, ell = segment_basis(G.mu)
    w = ell @ s
    return FuzzyGraph(G.n, G.src, G.dst, w * (G.mu.max() / w.max()))


def best3(H):
    P = [_leiden(H, k) for k in range(3)]
    return max(P, key=lambda p: fuzzy_modularity(H, p))


def partition_value(name, rng):
    G, _ = load(name); Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
    gr = groups_for(name); has = np.array([x is not None for x in gr]); gl = pd.factorize(pd.Series(gr[has]).astype(str))[0]
    a_sup = ari(gl, best3(Gs)[has])
    out = {}
    for tag, s in vertices(9):
        H = rescaled(G, s)
        obs = ari(gl, best3(H)[has])
        glob = float(np.mean([ari(gl, best3(permute_global(H, rng))[has]) for _ in range(10)]))
        row = float(np.mean([ari(gl, best3(permute_rows(H, rng))[has]) for _ in range(10)]))
        out[tag] = {"support": a_sup, "observed": obs, "global": glob, "row": row, "V_total": obs - a_sup,
                    "V_marg": glob - a_sup, "V_strength": row - glob, "V_resid": obs - row}
        print(name, tag, {k: round(v, 3) for k, v in out[tag].items()}, flush=True)
    return out


def path_value(name, ann, rng, n_null=20):
    G, labels = load(name); idx = {l: i for i, l in enumerate(labels)}
    sel = ann[ann.Exercise_A.isin(idx) & ann.Exercise_B.isin(idx)]
    ia = sel.Exercise_A.map(idx).to_numpy(); ib = sel.Exercise_B.map(idx).to_numpy(); y = sel.Prerequisite_avg.to_numpy()
    sup = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
    q_sup = float(spearmanr(score_from_length(hop_penalised(sup, 1.0))[ib, ia], y).statistic)
    f = lambda H: float(spearmanr(product_scores(H)[ib, ia], y).statistic)
    out = {}
    for tag, s in vertices(9):
        H = rescaled(G, s)
        obs = f(H)
        glob = float(np.mean([f(permute_global(H, rng)) for _ in range(n_null)]))
        row = float(np.mean([f(permute_rows(H, rng)) for _ in range(n_null)]))
        out[tag] = {"support": q_sup, "observed": obs, "global": glob, "row": row, "V_total": obs - q_sup,
                    "V_marg": glob - q_sup, "V_strength": row - glob, "V_resid": obs - row}
        print(name, tag, {k: round(v, 3) for k, v in out[tag].items()}, flush=True)
    return out


def main():
    rng = np.random.default_rng(5111)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"partition": {}, "path": {}}
    for name in ("ednet_copractice", "junyi_transition"):
        if name not in res["partition"]:
            res["partition"][name] = partition_value(name, rng); json.dump(res, open(OUT, "w"), indent=1, default=float)
    _, ann = load_junyi_expert()
    for name in ("junyi_copractice", "junyi_transition"):
        if name not in res["path"]:
            res["path"][name] = path_value(name, ann, rng); json.dump(res, open(OUT, "w"), indent=1, default=float)
    for lev in res:
        for n, d in res[lev].items():
            d = {k: x for k, x in d.items() if not k.startswith("_")}
            v = [x["V_total"] for x in d.values()]; r = [x["V_resid"] for x in d.values()]
            res[lev][n]["_range"] = {"V_total": [float(min(v)), float(max(v))], "V_resid": [float(min(r)), float(max(r))],
                                     "sign_changes": int(sum(np.sign(x) != np.sign(d["mu"]["V_total"]) for x in v))}
    json.dump(res, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
