#!/usr/bin/env python3
"""Release 5.0.0. Semi-synthetic logs with a known answer (Section 7 of the manuscript).

96 worlds (fuzzykg.semisynthetic): community strength p_comm in {0.6, 0.8, 0.95}, traversal noise
in {0.1, 0.4}, spiral review in {0, 0.3}, platform suggestions in {0, 0.2}, learners in {400, 1500},
two seeds. Both constructions of Section 4 run on every log (192 graphs). Evaluation pairs for paths:
every ordered pair of concepts in the same or adjacent communities (the analogue of annotated pairs
of related exercises); edge level: the ordered pairs with an edge in either direction. On every graph:

  partition  support, quantile cuts 0.25 / 0.50 / 0.75 and Leiden on Q_F, ten seeds at gamma = 1;
             ARI of every partition with the true communities;
             the unit-robust audit (fuzzykg.simplex): decision on mu_E and kappa*;
             the release 4.11 audit (fuzzykg.audit.cut_audit, ten seeds) for comparison;
             true loss of a cut = mean ARI(support) - mean ARI(cut);
  value      edge, path and partition value of the grades with the global and row nulls
             (Section 6), against the true prerequisite scores and communities;
  paths      Spearman of every score with the true graded prerequisite score over all ordered
             pairs; retrieval precision@k of the true parents of every concept; alpha_c and the
             usable share of Proposition 3; lam* of the hop-penalised score, cross-fitted on
             halves of the targets.
Writes results/v15/semisynthetic.json (resumable per world).
"""
from __future__ import annotations
import itertools, json, os, sys, time
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden, cut_audit, describe  # noqa: E402
from fuzzykg.construction import build_fuzzy_graph, build_transition_graph  # noqa: E402
from fuzzykg.nulltest import permute_global, permute_rows  # noqa: E402
from fuzzykg.semisynthetic import World, simulate, true_prereq_score  # noqa: E402
from fuzzykg.simplex import CutFamily, kappa_star, simplex_extremes  # noqa: E402
from fuzzykg.pathscore import widest, connectivity_level, hop_penalised, score_from_length  # noqa: E402
from fuzzykg.paths_fast import strengths  # noqa: E402

OUT = "results/v15/semisynthetic.json"  # minimum excess: every value of DELTAS is evaluated
SEEDS = range(10)
CUTS = (0.25, 0.50, 0.75)
DELTAS = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
LAMS = np.round(np.linspace(0, 1, 11), 2)
GRID = list(itertools.product((0.6, 0.8, 0.95), (0.1, 0.4), (0.0, 0.3), (0.0, 0.2), (400, 1500), (0, 1)))


def relabel(G, labels, K):
    """Map graph nodes (indices into labels) back to concept ids 0..K-1; isolated concepts get no node."""
    lab = np.array([int(x) for x in labels])
    return FuzzyGraph(K, lab[G.src], lab[G.dst], G.mu.copy())


def best(G, seeds=range(3)):
    P = [_leiden(G, s) for s in seeds]
    return max(P, key=lambda p: fuzzy_modularity(G, p))


def pair_scores(G, K):
    """Ordered-pair score matrices (rows: prerequisite b, columns: target a)."""
    D = np.zeros((K, K)); D[G.src, G.dst] = G.mu
    sup = FuzzyGraph(K, G.src, G.dst, np.ones(G.m))
    H = hop_penalised(sup, 1.0)
    out = {"direct": D, "hop": score_from_length(H)}
    S, _ = strengths(G, "min"); S = S.copy(); np.fill_diagonal(S, 0); out["maxmin"] = S
    S, _ = strengths(G, "product"); S = S.copy(); np.fill_diagonal(S, 0); out["product"] = S
    S, _ = strengths(G, "lukasiewicz"); S = S.copy(); np.fill_diagonal(S, 0); out["lukasiewicz"] = S
    for lam in LAMS:
        out[f"lam{lam:.1f}"] = score_from_length(hop_penalised(G, float(lam)))
    return out


def rho(score, T, mask):
    return float(spearmanr(score[mask], T[mask]).correlation)


def precision_at_parents(score, truth, K):
    par = {}
    for s, d in zip(truth["dag_src"], truth["dag_dst"]):
        par.setdefault(int(d), set()).add(int(s))
    hits = tot = 0
    rng = np.random.default_rng(0)
    for a, ps in par.items():
        col = score[:, a].astype(float).copy(); col[a] = -np.inf
        col = col + 1e-9 * rng.random(K)  # random tie-breaking
        top = np.argsort(-col)[:len(ps)]
        hits += len(ps & set(top.tolist())); tot += len(ps)
    return hits / tot


def value_decomposition(G, K, T, mask, comm, n_null=5):
    """Section 6: VoG = L(support) - L(observed) split by the global and row nulls; the task
    score here is a quality (higher is better), so components are differences of qualities."""
    rng = np.random.default_rng(7)
    sup = FuzzyGraph(K, G.src, G.dst, np.ones(G.m))

    emask = np.zeros((K, K), bool); emask[G.src, G.dst] = True; emask |= emask.T

    def q_edge(H):
        D = np.zeros((K, K)); D[H.src, H.dst] = H.mu; return rho(D, T, emask)

    def q_path(H):
        if np.all(H.mu == 1.0):   # grades removed: the hop distance on the same support
            return rho(score_from_length(hop_penalised(H, 1.0)), T, mask)
        S, _ = strengths(H, "product"); S = S.copy(); np.fill_diagonal(S, 0); return rho(S, T, mask)

    def q_part(H):
        return float(np.mean([ari(comm, _leiden(H, s)) for s in range(3)]))

    out = {}
    for lev, f in (("edge", q_edge), ("path", q_path), ("partition", q_part)):
        obs, s0 = f(G), f(sup)
        glob = float(np.mean([f(permute_global(G, rng)) for _ in range(n_null)]))
        row = float(np.mean([f(permute_rows(G, rng)) for _ in range(n_null)]))
        out[lev] = {"support": s0, "global": glob, "row": row, "observed": obs,
                    "V_total": obs - s0, "V_marg": glob - s0, "V_strength": row - glob, "V_resid": obs - row}
    return out


def audit_graph(G, K, comm, T, mask, truth, unit):
    rec = {"describe": describe(G)}
    Gs = FuzzyGraph(K, G.src, G.dst, np.ones(G.m))
    parts = {"support": [_leiden(Gs, s) for s in SEEDS]}
    for q in CUTS:
        parts[f"q{q:.2f}"] = [_leiden(G.alpha_cut(float(np.quantile(G.mu, q))), s) for s in SEEDS]
    parts["F"] = [_leiden(G, s) for s in SEEDS]
    ar = {k: [ari(comm, p) for p in v] for k, v in parts.items()}
    rec["ari"] = {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in ar.items()}
    fam = CutFamily(G, {k: v for k, v in parts.items() if k != "F"})
    old = cut_audit(G, {f"q{q:.2f}": float(np.quantile(G.mu, q)) for q in CUTS}, SEEDS, unit=unit)
    rec["cuts"] = {}
    for q in CUTS:
        c = f"q{q:.2f}"
        ks = kappa_star(fam, c, kmax=100.0)
        by_delta = {}
        for dl in DELTAS:
            k2 = kappa_star(fam, c, min_excess=dl, kmax=100.0)
            by_delta[f"{dl:g}"] = {"decision": k2["decision_mu"], "margin": k2["margin_mu"], "kappa_star": k2["kappa_star"]}
        ext = simplex_extremes(fam, c, n_rand=1000)
        ex1, _ = fam.margin(c, np.ones((1, fam.J)))
        # excess at the exact indicator of the cut (the limit of Lemma 2): crisp Q on the level set
        H = G.alpha_cut(float(np.quantile(G.mu, q)))
        ind = float(np.mean([fuzzy_modularity(H, p) for p in parts["support"]]) - np.mean([fuzzy_modularity(H, p) for p in parts[c]]))
        loss_sup = ar["support"]; loss_cut = ar[c]
        rec["cuts"][c] = {**ks, "excess_mu": float(ex1[0]), "simplex": {k: ext[k] for k in ("excess_min", "excess_max", "margin_min", "margin_max")},
                          "excess_at_indicator": ind, "by_delta": by_delta,
                          "old_decision": old["cuts"][c]["decision"],
                          "true_loss_vs_support": float(np.mean(loss_sup) - np.mean(loss_cut)),
                          "true_loss_sd": float(max(np.std(loss_sup, ddof=1), np.std(loss_cut, ddof=1))),
                          "true_loss_vs_F": float(np.mean(ar["F"]) - np.mean(loss_cut)),
                          "mean_out_degree": describe(H)["mean_out_degree"], "retained": float(H.m / G.m)}
    # paths
    sc = pair_scores(G, K)
    rec["paths"] = {k: {"rho": rho(v, T, mask), "rho_all": rho(v, T, ~np.eye(K, dtype=bool)), "prec": precision_at_parents(v, truth, K)} for k, v in sc.items()}
    rec["paths"]["reach"] = {"rho": rho((sc["hop"] > sc["hop"].min()).astype(float), T, mask)}
    rec["connectivity"] = connectivity_level(G, S=sc["maxmin"])
    # lam* cross-fitted over halves of the targets
    rng = np.random.default_rng(11)
    lam_star = []
    for rep in range(20):
        half = rng.random(K) < 0.5
        mA = mask & half[None, :]; mB = mask & ~half[None, :]
        rA = [rho(sc[f"lam{l:.1f}"], T, mA) for l in LAMS]
        lam_star.append(float(LAMS[int(np.nanargmax(rA))]))
    rec["lam_star"] = {"median": float(np.median(lam_star)), "values": lam_star,
                       "rho_by_lam": [rec["paths"][f"lam{l:.1f}"]["rho"] for l in LAMS]}
    rec["value"] = value_decomposition(G, K, T, mask, comm)
    return rec


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"worlds": {}}
    for (pc, noise, il, sug, N, seed) in GRID:
        key = f"p{pc}_n{noise}_i{il}_s{sug}_N{N}_seed{seed}"
        if key in res["worlds"]:
            continue
        t0 = time.perf_counter()
        w = World(p_comm=pc, noise=noise, interleave=il, suggest=sug, N=N, seed=seed)
        log, truth = simulate(w)
        K = w.K; comm = truth["community"]
        T = true_prereq_score(truth, K)
        mask = (np.abs(comm[:, None] - comm[None, :]) <= 1) & ~np.eye(K, dtype=bool)
        rec = {"world": truth["world"], "attempts": int(len(log)), "dag_edges": int(len(truth["dag_src"]))}
        for cname, build, unit in (("co-practice", build_fuzzy_graph, "arbitrary"), ("transition", build_transition_graph, "meaningful")):
            G0, _ = build(log)
            G = relabel(G0, G0.labels, K)
            rec[cname] = audit_graph(G, K, comm, T, mask, truth, unit)
        rec["time_s"] = time.perf_counter() - t0
        res["worlds"][key] = rec
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        print(key, f"{rec['time_s']:.0f}s", {c: [rec[c]["cuts"][q]["decision_mu"][0] + f"{rec[c]['cuts'][q]['kappa_star']:.3g}" for q in rec[c]["cuts"]] for c in ("co-practice", "transition")}, flush=True)


if __name__ == "__main__":
    main()
