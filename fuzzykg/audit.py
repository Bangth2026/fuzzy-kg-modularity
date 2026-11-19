"""Algorithm 2 of the manuscript as a tool: a pre-deployment audit of a fuzzy prerequisite graph.

``audit(G, ...)`` runs, on one directed fuzzy graph,
  1. a description of the graph and of each candidate level set (density, reciprocity, mean
     out-degree; a level set below mean out-degree 2 is marked, not rejected);
  2. Algorithm 1 (degree null with both decision floors, weights-only nulls with power), unless
     ``null_test=False``;
  3. the cut audit: Q_F drop and rank-coded Q_F drop of every candidate cut over optimiser
     seeds, read against the no-cut (support) value; a cut is flagged when its mean exceeds the
     no-cut mean by more than twice the larger of the two seed SDs;
  4. optionally, agreement of every partition with an external grouping of the nodes;
  5. optionally, best paths for the composition operators scored against external prerequisite
     pairs next to the direct edge and the hop distance on the same support, with a cluster
     bootstrap over items for every difference to the hop distance;
and returns a dictionary with the recommendation of each step. ``report_markdown`` renders it.
The thresholds are those of the manuscript; they are reported, not tuned.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .graph import FuzzyGraph
from .modularity import fuzzy_modularity

CUT_RULE = "flag a cut whose audit exceeds the no-cut value by > 2 x max(seed SD)"


def _leiden(G: FuzzyGraph, seed: int) -> np.ndarray:
    import igraph as ig
    import leidenalg as la
    g = ig.Graph(n=G.n, edges=list(zip(G.src.tolist(), G.dst.tolist())), directed=True)
    g.es["weight"] = G.mu.tolist()
    p = la.RBConfigurationVertexPartition(g, weights="weight", resolution_parameter=1.0)
    o = la.Optimiser(); o.set_rng_seed(int(seed)); o.optimise_partition(p, n_iterations=-1)
    return np.asarray(p.membership)


def rank_code(mu: np.ndarray, L: int = 8) -> np.ndarray:
    """Grid ranks N_A(e)/L on the L-level quantile grid (Lemma 2(ii) of Supplementary material B; Lemma 1 of the main article)."""
    qs = np.unique(np.quantile(mu, np.arange(L) / L))
    return (mu[:, None] >= qs[None, :]).sum(1) / len(qs)


def describe(G: FuzzyGraph) -> Dict:
    act = np.flatnonzero(np.bincount(G.src, minlength=G.n) + np.bincount(G.dst, minlength=G.n))
    return {"n_active": int(act.size), "m": int(G.m), "density": float(G.m / max(1, act.size * (act.size - 1))),
            "reciprocity": float(G.reciprocity()) if hasattr(G, "reciprocity") else None,
            "mean_out_degree": float(G.m / max(1, act.size)), "mu_mean": float(G.mu.mean()), "mu_sd": float(G.mu.std())}


def cut_audit(G: FuzzyGraph, cuts: Dict[str, Optional[float]], seeds: Sequence[int] = range(10),
              grouping: Optional[np.ndarray] = None) -> Dict:
    from sklearn.metrics import adjusted_rand_score as ari
    Gr = FuzzyGraph(G.n, G.src, G.dst, rank_code(G.mu))
    PF = [_leiden(G, s) for s in seeds]; PR = [_leiden(Gr, s) for s in seeds]
    QF = np.array([fuzzy_modularity(G, p) for p in PF]); QR = np.array([fuzzy_modularity(Gr, p) for p in PR])
    has = None if grouping is None else np.array([x is not None for x in grouping])
    if has is not None:
        import pandas as pd
        gl = pd.factorize(pd.Series(grouping[has]).astype(str))[0]
    rows = {}
    allc = {"support": None, **cuts}
    for c, a in allc.items():
        H = FuzzyGraph(G.n, G.src, G.dst, np.ones_like(G.mu)) if a is None else G.alpha_cut(a)
        if H.m == 0:
            rows[c] = {"alpha": a, "empty": True}; continue
        P = [_leiden(H, s) for s in seeds]
        d = QF - np.array([fuzzy_modularity(G, p) for p in P]); dr = QR - np.array([fuzzy_modularity(Gr, p) for p in P])
        r = {"alpha": a, **describe(H), "retained_edges": float(H.m / G.m), "qf_drop": float(d.mean()), "qf_drop_sd": float(d.std(ddof=1)),
             "rank_drop": float(dr.mean()), "rank_drop_sd": float(dr.std(ddof=1))}
        r["low_degree_warning"] = bool(r["mean_out_degree"] < 2.0)
        if has is not None:
            v = np.array([ari(gl, p[has]) for p in P]); r["ari_grouping"] = float(v.mean()); r["ari_grouping_sd"] = float(v.std(ddof=1))
        rows[c] = r
    b = rows["support"]
    for c, r in rows.items():
        if c == "support" or r.get("empty"):
            continue
        for k in ("qf_drop", "rank_drop"):
            r[f"flag_{k}"] = bool(r[k] - b[k] > 2 * max(r[k + "_sd"], b[k + "_sd"]))
    out = {"rule": CUT_RULE, "seeds": len(list(seeds)), "cuts": rows}
    if has is not None:
        v = np.array([ari(gl, p[has]) for p in PF]); out["ari_grouping_PiF"] = float(v.mean())
    return out


def path_audit(G: FuzzyGraph, pairs_src: np.ndarray, pairs_dst: np.ndarray, y: np.ndarray,
               operators=("min", "product", "lukasiewicz", "yager_5"), B: int = 500, seed: int = 0) -> Dict:
    """Spearman correlation of best-path scores from src to dst with an external prerequisite
    score y, next to the direct edge and the hop distance; two-way cluster bootstrap over items."""
    from scipy.stats import spearmanr
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    from .paths_fast import strengths
    spec = {"min": ("min", None), "product": ("product", None), "lukasiewicz": ("lukasiewicz", None), "yager_5": ("yager", 5.0)}
    W = np.zeros((G.n, G.n)); W[G.src, G.dst] = G.mu
    D = shortest_path(csr_matrix((np.ones(G.m), (G.src, G.dst)), shape=(G.n, G.n)), directed=True, unweighted=True)
    X = {"direct_edge": W[pairs_src, pairs_dst], "hop_distance": np.where(np.isfinite(D), 1.0 / (1.0 + D), 0.0)[pairs_src, pairs_dst]}
    for k in operators:
        S, _ = strengths(G, *spec[k]); X[k] = np.maximum(S, 0.0)[pairs_src, pairs_dst]
    items = np.unique(np.r_[pairs_src, pairs_dst]); pos = {x: i for i, x in enumerate(items)}
    pa = np.array([pos[x] for x in pairs_src]); pb = np.array([pos[x] for x in pairs_dst])
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in X}
    for _ in range(B):
        m = rng.multinomial(items.size, np.full(items.size, 1 / items.size)); w = m[pa] * m[pb]; s = w > 0
        for k, x in X.items():
            boots[k].append(spearmanr(np.repeat(x[s], w[s]), np.repeat(y[s], w[s])).statistic)
    res = {k: {"rho": float(spearmanr(x, y).statistic), "reached": float((x > 0).mean())} for k, x in X.items()}
    for k in X:
        if k in ("hop_distance",):
            continue
        d = np.array(boots[k]) - np.array(boots["hop_distance"])
        res[k]["minus_hop"] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        res[k]["beats_hop"] = bool(np.nanpercentile(d, 2.5) > 0)
    best = [k for k in operators if res[k]["beats_hop"]]
    res["recommendation"] = (max(best, key=lambda k: res[k]["rho"]) if best else "hop distance on the support (no operator beats it)")
    res["composition_below_direct_edge"] = [k for k in operators if res[k]["rho"] < res["direct_edge"]["rho"]]
    return res


def audit(G: FuzzyGraph, cuts: Optional[Dict[str, Optional[float]]] = None, grouping: Optional[np.ndarray] = None,
          pairs: Optional[tuple] = None, null_test: bool = True, draws: int = 200, seeds: Sequence[int] = range(10)) -> Dict:
    from .nulltest import best_qf, degree_test, density_matched_floor, structure_matched_floor, weights_only_test, weights_only_power, outcome
    if cuts is None:
        cuts = {f"q{q:.2f}": float(np.quantile(G.mu, q)) for q in (0.25, 0.5, 0.75)}
    rep = {"graph": describe(G)}
    if null_test:
        s5 = (0, 1, 2, 3, 4) if G.m <= 10000 else (0, 1, 2)
        b = best_qf(G, 1.0, s5)
        dg = degree_test(G, draws, 1.0, s5, obs=b.Q_F)
        f1 = density_matched_floor(G, b.n_communities, draws=20)
        f2 = structure_matched_floor(G, b.membership, draws=20)
        wg = weights_only_test(G, "global", draws, 1.0, s5, obs=b.Q_F); wr = weights_only_test(G, "row", draws, 1.0, s5, obs=b.Q_F)
        pr = weights_only_power(G, b.membership, kind="row", draws=30)
        rep["algorithm1"] = {"Q_F": b.Q_F, "k": b.n_communities, "degree_delta_Q": dg["delta_Q"], "degree_z": dg["z"],
                             "floor_density": f1["floor"], "floor_structure_matched": f2["floor"],
                             "outcome_density_floor": outcome(dg["z"], dg["delta_Q"], f1["floor"]),
                             "outcome_structure_matched_floor": outcome(dg["z"], dg["delta_Q"], f2["floor"]),
                             "weights_global_z": wg["z"], "weights_row_z": wr["z"], "row_detectable_r": pr["detectable_r_pb"]}
        rep["partitioning"] = ("stop: no partition to lose" if rep["algorithm1"]["outcome_density_floor"] == "no structure detected"
                               else "partition is structure; audit the cut")
    rep["cuts"] = cut_audit(G, cuts, seeds, grouping)
    if pairs is not None:
        rep["paths"] = path_audit(G, *pairs)
    return rep


def report_markdown(rep: Dict, title: str = "Fuzzy prerequisite graph audit") -> str:
    L = [f"# {title}", "", "## Graph", ""]
    L += [f"- {k}: {v:.4g}" if isinstance(v, float) else f"- {k}: {v}" for k, v in rep["graph"].items()]
    if "algorithm1" in rep:
        a = rep["algorithm1"]; L += ["", "## Algorithm 1", ""]
        L += [f"- {k}: {v:.4g}" if isinstance(v, float) else f"- {k}: {v}" for k, v in a.items()]
        L += [f"- decision: {rep['partitioning']}"]
    L += ["", "## Cuts", "", f"Rule: {rep['cuts']['rule']} ({rep['cuts']['seeds']} seeds).", "",
          "| cut | alpha | retained | mean out-degree | Q_F drop (SD) | rank-coded drop (SD) | flagged | ARI with grouping |", "|---|---|---|---|---|---|---|---|"]
    for c, r in rep["cuts"]["cuts"].items():
        if r.get("empty"):
            L.append(f"| {c} | {r['alpha']} | 0 | - | - | - | - | - |"); continue
        fl = "-" if c == "support" else ("yes" if r["flag_qf_drop"] else "no")
        al = "-" if r["alpha"] is None else f"{r['alpha']:.3f}"
        L.append(f"| {c} | {al} | {r['retained_edges']:.2f} | {r['mean_out_degree']:.2f}{' (low)' if r['low_degree_warning'] else ''} | "
                 f"{r['qf_drop']:.4f} ({r['qf_drop_sd']:.4f}) | {r['rank_drop']:.4f} ({r['rank_drop_sd']:.4f}) | {fl} | {r.get('ari_grouping', float('nan')):.3f} |")
    if "paths" in rep:
        p = rep["paths"]; L += ["", "## Path inference", "", "| score | rho | reached | minus hop (95% CI) |", "|---|---|---|---|"]
        for k, v in p.items():
            if isinstance(v, dict):
                ci = v.get("minus_hop"); L.append(f"| {k} | {v['rho']:.3f} | {v['reached']:.2f} | {'-' if ci is None else f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'} |")
        L += ["", f"Recommendation: {p['recommendation']}.", f"Operators below the direct edge: {', '.join(p['composition_below_direct_edge']) or 'none'}."]
    return "\n".join(L) + "\n"
