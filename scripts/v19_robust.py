#!/usr/bin/env python3
"""Release 5.4.0. Permutation tests, noise robustness, scaling and partition predictors
(Section 8.4 and Table 8 of the manuscript; Supplementary Tables S4j, S11b, S11c).

perm      Row-permutation tests of the residual claims: the statistic on the graded graph against
          its distribution under row permutations of the memberships (the null of V_res).
          Junyi co-practice, edge (direct membership) and path (product best-path score) level,
          Spearman rho with the crowd score on the annotated pairs inside the graph, 199 draws;
          Junyi transition and EdNet co-practice, partition level, ARI with the grouping, best of
          seeds 0-2, 99 draws. p = (1 + #null >= observed) / (1 + draws).
noise     Memberships of a fraction p of the edges replaced by a random permutation of themselves
          (p = 0, 0.25, 0.5, 0.75, 1; five noise draws). V_marg, V_str, V_res recomputed (20 row and
          20 global null draws): Junyi co-practice at the edge level, EdNet co-practice at the
          partition level (p = 0, 0.5, 1; three noise draws; ten null draws, best of seeds 0-2).
scaling   Random directed graphs with n = 10^3 to 10^5 nodes and 10 edges per node, Beta(2, 5)
          memberships: wall time of one Leiden run (n <= 3 x 10^4), of the segment basis and the
          closed-form modularity of the support and three quantile cuts over 20 partitions (the
          audit, given the partitions), of one row-null draw, and of product best-path scores from
          100 sources (Dijkstra on -ln mu).
features  For the ten graphs with an informative grouping: nodes, edges, mean out-degree, density,
          reciprocity, coefficient of variation of node strength, partition value V (Table S4d).
Writes results/v19/robust.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden  # noqa: E402
from fuzzykg.nulltest import permute_rows, permute_global  # noqa: E402
from fuzzykg.paths_fast import strengths  # noqa: E402
from fuzzykg.simplex import CutFamily  # noqa: E402
from data.loaders import load_junyi_expert  # noqa: E402
from v15_value_paths import load, product_scores  # noqa: E402
from v15_simplex_real import groups_for  # noqa: E402

OUT = "results/v19/robust.json"
INFORMATIVE = ["assist2009_item", "assist2017_item_transition", "junyi_transition", "algebra2005_copractice", "algebra2005_transition",
               "bridge2006_copractice", "bridge2006_transition", "eedi_transition", "ednet_copractice", "ednet_transition"]


def best3(H, seeds=range(3)):
    P = [_leiden(H, s) for s in seeds]
    return max(P, key=lambda p: fuzzy_modularity(H, p))


def junyi_pairs(name):
    G, labels = load(name); idx = {l: i for i, l in enumerate(labels)}
    _, ann = load_junyi_expert()
    sel = ann[ann.Exercise_A.isin(idx) & ann.Exercise_B.isin(idx)]
    return G, sel.Exercise_A.map(idx).to_numpy(), sel.Exercise_B.map(idx).to_numpy(), sel.Prerequisite_avg.to_numpy()


def direct(H, ia, ib):
    D = np.zeros((H.n, H.n)); D[H.src, H.dst] = H.mu; return D[ib, ia]


def grouping(name):
    gr = groups_for(name); has = np.array([x is not None for x in gr]); return has, pd.factorize(pd.Series(gr[has]).astype(str))[0]


def perm(res):
    rng = np.random.default_rng(5405); out = {}
    G, ia, ib, y = junyi_pairs("junyi_copractice")
    r = lambda x: spearmanr(x, y).statistic
    obs_e, obs_p = r(direct(G, ia, ib)), r(product_scores(G)[ib, ia])
    ne, npth = [], []
    for _ in range(199):
        H = permute_rows(G, rng); ne.append(r(direct(H, ia, ib))); npth.append(r(product_scores(H)[ib, ia]))
    pv = lambda o, nn: float((1 + np.sum(np.array(nn) >= o)) / (1 + len(nn)))
    out["junyi_copractice_edge"] = {"observed": float(obs_e), "null_mean": float(np.mean(ne)), "null_max": float(np.max(ne)), "p": pv(obs_e, ne), "draws": 199}
    out["junyi_copractice_path"] = {"observed": float(obs_p), "null_mean": float(np.mean(npth)), "null_max": float(np.max(npth)), "p": pv(obs_p, npth), "draws": 199}
    print(out, flush=True)
    for name in ("junyi_transition", "ednet_copractice"):
        G, _ = load(name); has, gl = grouping(name)
        o = ari(gl, best3(G)[has]); nn = [ari(gl, best3(permute_rows(G, rng))[has]) for _ in range(99)]
        out[f"{name}_partition"] = {"observed": float(o), "null_mean": float(np.mean(nn)), "null_max": float(np.max(nn)), "p": pv(o, nn), "draws": 99}
        print(name, out[f"{name}_partition"], flush=True)
    res["perm"] = out


def noisy(G, p, rng):
    k = rng.random(G.m) < p; mu = G.mu.copy(); idx = np.flatnonzero(k); mu[idx] = mu[rng.permutation(idx)]
    return FuzzyGraph(G.n, G.src, G.dst, mu)


def noise(res):
    rng = np.random.default_rng(5406); out = {"junyi_copractice_edge": {}, "ednet_copractice_partition": {}}
    G, ia, ib, y = junyi_pairs("junyi_copractice"); r = lambda x: spearmanr(x, y).statistic
    sup = r(direct(FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m)), ia, ib))
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows = []
        for _ in range(5 if p > 0 else 1):
            H = noisy(G, p, rng); o = r(direct(H, ia, ib))
            gl_ = np.mean([r(direct(permute_global(H, rng), ia, ib)) for _ in range(20)]); rw = np.mean([r(direct(permute_rows(H, rng), ia, ib)) for _ in range(20)])
            rows.append([o - sup, gl_ - sup, rw - gl_, o - rw])
        out["junyi_copractice_edge"][f"{p:g}"] = dict(zip(["V", "V_marg", "V_str", "V_res"], np.mean(rows, 0).tolist()))
        print("edge", p, out["junyi_copractice_edge"][f"{p:g}"], flush=True)
    G, _ = load("ednet_copractice"); has, gl = grouping("ednet_copractice")
    sup = ari(gl, best3(FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m)))[has])
    for p in (0.0, 0.5, 1.0):
        rows = []
        for _ in range(3 if p > 0 else 1):
            H = noisy(G, p, rng); o = ari(gl, best3(H)[has])
            gl_ = np.mean([ari(gl, best3(permute_global(H, rng))[has]) for _ in range(10)]); rw = np.mean([ari(gl, best3(permute_rows(H, rng))[has]) for _ in range(10)])
            rows.append([o - sup, gl_ - sup, rw - gl_, o - rw])
        out["ednet_copractice_partition"][f"{p:g}"] = dict(zip(["V", "V_marg", "V_str", "V_res"], np.mean(rows, 0).tolist()))
        print("partition", p, out["ednet_copractice_partition"][f"{p:g}"], flush=True)
    res["noise"] = out


def scaling(res):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    out = {}
    for n in (1000, 3000, 10000, 30000, 100000):
        rng = np.random.default_rng(n); m = 10 * n
        src = rng.integers(0, n, m); dst = rng.integers(0, n, m); k = src != dst
        G = FuzzyGraph(n, src[k], dst[k], np.clip(rng.beta(2, 5, k.sum()), 1e-3, 1))
        r = {"edges": int(G.m)}
        if n <= 30000:
            t = time.time(); P = _leiden(G, 0); r["leiden_s"] = time.time() - t
        else:
            P = rng.integers(0, 100, n)
        parts = {"support": [P] * 20, **{f"q{q}": [P] * 20 for q in (0.25, 0.5, 0.75)}}
        t = time.time(); fam = CutFamily(G, parts); S = np.ones((1, fam.J))
        for c in parts:
            fam.q(c, S)
        r["audit_closed_form_s"] = time.time() - t
        t = time.time(); permute_rows(G, rng); r["row_null_draw_s"] = time.time() - t
        t = time.time(); W = sp.csr_matrix((-np.log(G.mu), (G.src, G.dst)), shape=(n, n))
        np.exp(-dijkstra(W, directed=True, indices=np.arange(100))); r["product_paths_100_sources_s"] = time.time() - t
        out[str(n)] = r; print(n, r, flush=True)
    res["scaling"] = out


def features(res):
    out = {}
    s4d = json.load(open("results/v16/summary.json"))["V_partition_real"] if "V_partition_real" in json.load(open("results/v16/summary.json")) else {}
    for name in INFORMATIVE:
        G, _ = load(name)
        n, m = G.n, G.m; E = set(zip(G.src.tolist(), G.dst.tolist()))
        recip = float(np.mean([(b, a) in E for a, b in E]))
        st = np.bincount(G.src, G.mu, n) + np.bincount(G.dst, G.mu, n)
        out[name] = {"n": int(n), "m": int(m), "mean_out_degree": m / n, "density": m / (n * (n - 1)), "reciprocity": recip,
                     "strength_cv": float(st.std() / st.mean()), "mu_mean": float(G.mu.mean()), "mu_cv": float(G.mu.std() / G.mu.mean())}
        print(name, out[name], flush=True)
    res["features"] = out


def main(parts):
    os.makedirs("results/v19", exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for p in parts:
        {"perm": perm, "noise": noise, "scaling": scaling, "features": features}[p](res)
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["features", "scaling", "perm", "noise"])
