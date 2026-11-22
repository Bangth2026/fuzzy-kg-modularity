"""Algorithm 2 of the manuscript as a tool: a pre-deployment audit of a fuzzy prerequisite graph.

``audit(G, ...)`` runs, on one directed fuzzy graph,
  1. a description of the graph and of each candidate level set (density, reciprocity, mean
     out-degree; a level set below mean out-degree 2 is marked, not rejected);
  2. Algorithm 1 (degree null with both decision floors, weights-only nulls with power), unless
     ``null_test=False``;
  3. the cut audit: Q_F drop and rank-coded Q_F drop of every candidate cut over optimiser
     seeds, both partitions of a drop re-drawn for every seed, read against the no-cut (support)
     value; a cut is flagged when its mean exceeds the no-cut mean by more than twice the larger
     of the two seed SDs and by more than ``min_excess`` (0.01, added in release 4.8.0). From
     release 4.9.0 each cut also gets a decision that depends on the unit of mu_E: where the unit
     is meaningful (``unit="meaningful"``: a count share, or a released or calibrated score) the
     Q_F drop on mu_E decides; where it is arbitrary (``unit="arbitrary"``: a weighted combination
     whose weights the builder chose) a cut is cut only when both drops pass, flagged when both
     flag, and "undecided" when they disagree (keep the graded graph unless step 4 shows no loss).
     A no-cut drop above 0.05 is reported as a warning, since the rule is relative to that
     baseline; from release 4.10.0 such a graph is also read on the absolute drops (a cut is
     flagged when its own mean drop exceeds 2 x its seed SD and ``min_excess``), and a cut is
     flagged when either reading flags it. Release 4.11.0 adds ``gamma``, the resolution of every
     Leiden run and Q_F of the step (the manuscript decides at 1; results/v14 reports 0.5 and 2);
  4. optionally, agreement of every partition with an external grouping of the nodes. From
     release 4.10.0 the grouping is first checked against a permutation reference: the ARI of the
     best of three seeds of G_F is compared with the same quantity on ``perm_draws`` row-wise
     permutations of mu_E (out-strengths kept). Only a grouping that the graded partition matches
     above the 95th percentile of that reference can show that an undecided cut loses nothing;
     otherwise the cut stays undecided;
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

CUT_RULE = "flag a cut whose audit exceeds the no-cut value by > 2 x max(seed SD) and by > {min_excess}"
UNIT_RULE = {"meaningful": "the Q_F drop on mu_E decides; the rank-coded drop is reported beside it",
             "arbitrary": "cut when both drops pass, flagged when both flag, undecided when they disagree "
                          "(keep the graded graph unless an independent grouping, matched above its permutation reference, shows no loss)",
             None: "unit not stated: both flags reported, no decision where they disagree"}
BASELINE_WARNING = 0.05  # fixed in release 4.9.0 after the drops under x^3 had been seen


def decide(flag_mu: bool, flag_rank: bool, unit: Optional[str]) -> str:
    """Decision of step 3 for one cut (release 4.9.0)."""
    if unit == "meaningful":
        return "flagged" if flag_mu else "cut"
    if flag_mu and flag_rank:
        return "flagged"
    if not flag_mu and not flag_rank:
        return "cut"
    return "undecided"


def _leiden(G: FuzzyGraph, seed: int, gamma: float = 1.0) -> np.ndarray:
    import igraph as ig
    import leidenalg as la
    g = ig.Graph(n=G.n, edges=list(zip(G.src.tolist(), G.dst.tolist())), directed=True)
    g.es["weight"] = G.mu.tolist()
    p = la.RBConfigurationVertexPartition(g, weights="weight", resolution_parameter=float(gamma))
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


def _best_of(G: FuzzyGraph, seeds: Sequence[int]) -> np.ndarray:
    P = [_leiden(G, s) for s in seeds]
    return max(P, key=lambda p: fuzzy_modularity(G, p))


def grouping_reference(G: FuzzyGraph, gl: np.ndarray, has: np.ndarray, draws: int = 20, seed: int = 3) -> Dict:
    """Permutation reference of step 4 (release 4.10.0): ARI of the best of three seeds of G_F
    with the grouping, against the same quantity on row-wise permutations of mu_E."""
    from sklearn.metrics import adjusted_rand_score as ari
    from .nulltest import permute_rows
    rng = np.random.default_rng(seed)
    obs = float(ari(gl, _best_of(G, range(3))[has]))
    ref = np.array([ari(gl, _best_of(permute_rows(G, rng), range(3))[has]) for _ in range(draws)])
    p95 = float(np.percentile(ref, 95))
    return {"ari_best3": obs, "perm_mean": float(ref.mean()), "perm_p95": p95,
            "n_perm_at_or_above": int((ref >= obs).sum()), "draws": int(draws), "informative": bool(obs > p95)}


def step4_no_loss(r: Dict, piF_mean: float, piF_sd: float) -> bool:
    """A cut loses nothing on the grouping when its mean ARI is not below that of Pi_F by more
    than 2 x the larger seed SD."""
    return bool(piF_mean - r["ari_grouping"] <= 2 * max(piF_sd, r["ari_grouping_sd"]))


def cut_audit(G: FuzzyGraph, cuts: Dict[str, Optional[float]], seeds: Sequence[int] = range(20),
              grouping: Optional[np.ndarray] = None, min_excess: float = 0.01,
              unit: Optional[str] = None, perm_draws: int = 20, gamma: float = 1.0) -> Dict:
    """Step 3 (and step 4 when ``grouping`` is given). ``gamma`` (release 4.11.0) is the resolution
    of every Leiden run and of every Q_F evaluated in the drops; the manuscript's decisions use 1."""
    if unit not in UNIT_RULE:
        raise ValueError("unit must be 'meaningful', 'arbitrary' or None")
    from sklearn.metrics import adjusted_rand_score as ari
    Gr = FuzzyGraph(G.n, G.src, G.dst, rank_code(G.mu))
    PF = [_leiden(G, s, gamma) for s in seeds]; PR = [_leiden(Gr, s, gamma) for s in seeds]
    QF = np.array([fuzzy_modularity(G, p, gamma) for p in PF]); QR = np.array([fuzzy_modularity(Gr, p, gamma) for p in PR])
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
        P = [_leiden(H, s, gamma) for s in seeds]
        d = QF - np.array([fuzzy_modularity(G, p, gamma) for p in P]); dr = QR - np.array([fuzzy_modularity(Gr, p, gamma) for p in P])
        r = {"alpha": a, **describe(H), "retained_edges": float(H.m / G.m), "qf_drop": float(d.mean()), "qf_drop_sd": float(d.std(ddof=1)),
             "rank_drop": float(dr.mean()), "rank_drop_sd": float(dr.std(ddof=1))}
        r["low_degree_warning"] = bool(r["mean_out_degree"] < 2.0)
        if has is not None:
            v = np.array([ari(gl, p[has]) for p in P]); r["ari_grouping"] = float(v.mean()); r["ari_grouping_sd"] = float(v.std(ddof=1))
        rows[c] = r
    b = rows["support"]
    warn = [k for k in ("qf_drop", "rank_drop") if b.get(k, 0.0) > BASELINE_WARNING]
    for c, r in rows.items():
        if c == "support" or r.get("empty"):
            continue
        for k in ("qf_drop", "rank_drop"):
            rel = bool(r[k] - b[k] > max(2 * max(r[k + "_sd"], b[k + "_sd"]), min_excess))
            r[f"flag_{k}_relative"] = rel
            if k in warn:  # release 4.10.0: the relative rule has lost resolution; read the absolute drop too
                r[f"flag_{k}_absolute"] = bool(r[k] > max(2 * r[k + "_sd"], min_excess))
                r[f"flag_{k}"] = rel or r[f"flag_{k}_absolute"]
            else:
                r[f"flag_{k}"] = rel
        r["decision"] = decide(r["flag_qf_drop"], r["flag_rank_drop"], unit)
    out = {"rule": CUT_RULE.format(min_excess=min_excess), "gamma": float(gamma), "unit": unit, "unit_rule": UNIT_RULE[unit],
           "seeds": len(list(seeds)), "cuts": rows,
           "baseline_warning": [f"no-cut {k} {b[k]:.3f} > {BASELINE_WARNING}: the relative rule loses resolution; "
                                f"absolute drops read as well, a cut flagged when either reading flags it" for k in warn]}
    if has is not None:
        v = np.array([ari(gl, p[has]) for p in PF])
        out["ari_grouping_PiF"] = float(v.mean()); out["ari_grouping_PiF_sd"] = float(v.std(ddof=1))
        ref = grouping_reference(G, gl, has, draws=perm_draws)
        out["grouping_reference"] = ref
        for c, r in rows.items():
            if r.get("decision") != "undecided":
                continue
            nl = step4_no_loss(r, out["ari_grouping_PiF"], out["ari_grouping_PiF_sd"])
            r["step4"] = {"informative_grouping": ref["informative"], "no_loss": nl}
            if ref["informative"] and nl:
                r["decision"] = "cut (step 4: no loss)"
            elif ref["informative"]:
                r["decision"] = "flagged (step 4: loss)"
            else:
                r["step4"]["note"] = "grouping not matched above its permutation reference; cut stays undecided, keep the graded graph"
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
          pairs: Optional[tuple] = None, null_test: bool = True, draws: int = 200, seeds: Sequence[int] = range(20),
          unit: Optional[str] = None, perm_draws: int = 20, gamma: float = 1.0) -> Dict:
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
    rep["cuts"] = cut_audit(G, cuts, seeds, grouping, unit=unit, perm_draws=perm_draws, gamma=gamma)
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
    L += ["", "## Cuts", "", f"Rule: {rep['cuts']['rule']} ({rep['cuts']['seeds']} seeds).",
          f"Unit of mu_E: {rep['cuts'].get('unit')}; decision: {rep['cuts'].get('unit_rule', '')}.", ""]
    L += [f"Warning: {w}" for w in rep["cuts"].get("baseline_warning", [])]
    gr = rep["cuts"].get("grouping_reference")
    if gr:
        L += [f"Step 4 permutation reference: ARI of Pi_F (best of 3 seeds) {gr['ari_best3']:.3f} against "
              f"{gr['draws']} row-wise permutations (mean {gr['perm_mean']:.3f}, 95th percentile {gr['perm_p95']:.3f}); "
              f"grouping {'informative' if gr['informative'] else 'not informative: an undecided cut stays undecided'}."]
    L += ["", "| cut | alpha | retained | mean out-degree | Q_F drop (SD) | rank-coded drop (SD) | flagged (Q_F / rank) | decision | ARI with grouping |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c, r in rep["cuts"]["cuts"].items():
        if r.get("empty"):
            L.append(f"| {c} | {r['alpha']} | 0 | - | - | - | - | - | - |"); continue
        yn = lambda x: "yes" if x else "no"
        fl = "-" if c == "support" else f"{yn(r['flag_qf_drop'])} / {yn(r['flag_rank_drop'])}"
        de = "-" if c == "support" else r["decision"]
        al = "-" if r["alpha"] is None else f"{r['alpha']:.3f}"
        L.append(f"| {c} | {al} | {r['retained_edges']:.2f} | {r['mean_out_degree']:.2f}{' (low)' if r['low_degree_warning'] else ''} | "
                 f"{r['qf_drop']:.4f} ({r['qf_drop_sd']:.4f}) | {r['rank_drop']:.4f} ({r['rank_drop_sd']:.4f}) | {fl} | {de} | {r.get('ari_grouping', float('nan')):.3f} |")
    if "paths" in rep:
        p = rep["paths"]; L += ["", "## Path inference", "", "| score | rho | reached | minus hop (95% CI) |", "|---|---|---|---|"]
        for k, v in p.items():
            if isinstance(v, dict):
                ci = v.get("minus_hop"); L.append(f"| {k} | {v['rho']:.3f} | {v['reached']:.2f} | {'-' if ci is None else f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'} |")
        L += ["", f"Recommendation: {p['recommendation']}.", f"Operators below the direct edge: {', '.join(p['composition_below_direct_edge']) or 'none'}."]
    return "\n".join(L) + "\n"
