#!/usr/bin/env python3
"""Version 4.3.0. Robustness of the community findings to the transition construction (Junyi).

The order-preserving Junyi graph is the one graph on which the memberships carry topic
information (Section 8.2), so its construction is varied for the community question itself:

  conjunction   min(asym, pref) [default], (asym + pref) / 2, asym * pref
  pruning       tau_E in {0.1, 0.2, 0.3}
  successor     pref = T(a->b) / max_c T(a->c) [default], or T(a->b) / sum_c T(a->c)

Support 20 throughout.  For every variant: Algorithm 1 in full (degree-preserving null, 200
draws; density-matched floor; global and row weights-only nulls, 200 draws; power on the
observed topology, 30 draws per tilt, skipped for a null whose test already counts); the topic
test of Section 8.2 (topic ARI of Leiden on Q_F, best of three seeds, against 20 row-wise and 20
global permutations of the memberships, and against crisp Leiden on the support graph, ten
seeds); and the Spearman correlation of the direct edge, the hop distance and the product with
the crowd prerequisite score on the fixed 1,954 annotated pairs.
Needs data/raw/cache/junyi_transition_counts.parquet (scripts/v6_transition_counts.py).
Writes results/v6/transition_sensitivity.json.
Usage: python scripts/v6_transition_sensitivity.py [variant ...]
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from data.loaders import RAW, load_junyi_expert  # noqa: E402
from fuzzykg import FuzzyGraph, fuzzy_leiden  # noqa: E402
from fuzzykg.nulltest import (best_qf, degree_test, weights_only_test, weights_only_power, density_matched_floor,  # noqa: E402
                              membership_degree_correlation, permute_rows, permute_global, outcome)
from composition_robustness import mats  # noqa: E402

OUT = os.environ.get("TS_OUT", "results/v6/transition_sensitivity.json")
SUPPORT = 20
VARIANTS = {f"{c}_{p}_tau{t}": (c, p, t) for p in ("max", "sum") for c in ("min", "mean", "product") for t in (0.1, 0.2, 0.3)}
DEFAULT_ORDER = (["min_max_tau0.2", "mean_max_tau0.2", "product_max_tau0.2", "min_max_tau0.1", "min_max_tau0.3",
                  "mean_max_tau0.1", "mean_max_tau0.3", "product_max_tau0.1", "product_max_tau0.3",
                  "min_sum_tau0.1", "min_sum_tau0.2", "mean_sum_tau0.2"])


def variant_graph(T, conj, pref_kind, tau):
    fwd = T.set_index(["src", "dst"])["T"]
    back = pd.Series(fwd.values, index=pd.MultiIndex.from_arrays([fwd.index.get_level_values(1), fwd.index.get_level_values(0)]))
    t = T.copy()
    t["back"] = back.reindex(pd.MultiIndex.from_arrays([t.src, t.dst])).fillna(0).to_numpy()
    tot = t["T"] + t["back"]
    t = t[(tot >= SUPPORT) & (t["T"] > t["back"])].copy()
    rowmax = T.groupby("src")["T"].max(); rowsum = T.groupby("src")["T"].sum()
    t["asym"] = 2.0 * t["T"] / (t["T"] + t["back"]) - 1.0
    denom = (rowmax if pref_kind == "max" else rowsum).reindex(t.src).to_numpy()
    t["pref"] = t["T"].to_numpy() / denom
    mu = {"min": np.minimum(t.asym, t.pref), "mean": (t.asym + t.pref) / 2.0, "product": t.asym * t.pref}[conj]
    t["mu"] = mu
    t = t[t.mu >= tau]
    labs = sorted(set(t.src.astype(str)) | set(t.dst.astype(str)))
    ix = {l: i for i, l in enumerate(labs)}
    G = FuzzyGraph(len(labs), t.src.astype(str).map(ix).to_numpy(np.int64), t.dst.astype(str).map(ix).to_numpy(np.int64),
                   t.mu.to_numpy(float), labels=labs)
    return G, labs


def topic_test(G, labs, topics, rng):
    truth = np.array([str(topics.get(l, "NA")) for l in labs])
    def best3(H):
        return ari(truth, max((fuzzy_leiden(H, 1.0, seed=s, n_iterations=0) for s in range(3)), key=lambda r: r.Q_F).membership)
    sup = FuzzyGraph(G.n, G.src, G.dst, np.ones_like(G.mu))
    obs10 = [ari(truth, fuzzy_leiden(G, 1.0, seed=s, n_iterations=0).membership) for s in range(10)]
    sup10 = [ari(truth, fuzzy_leiden(sup, 1.0, seed=s, n_iterations=0).membership) for s in range(10)]
    row = [best3(permute_rows(G, rng)) for _ in range(20)]
    glo = [best3(permute_global(G, rng)) for _ in range(20)]
    same = truth[G.src] == truth[G.dst]
    return {"observed_best3": best3(G), "observed_seeds_mean": float(np.mean(obs10)), "support_seeds_mean": float(np.mean(sup10)),
            "row_perm_mean": float(np.mean(row)), "row_perm_sd": float(np.std(row)), "row_perm_max": float(np.max(row)),
            "global_perm_mean": float(np.mean(glo)), "global_perm_sd": float(np.std(glo)),
            "share_row_ge_obs": float(np.mean(np.array(row) >= best3(G))),
            "same_topic_edge_share": float(same.mean()),
            "mu_same_topic": float(G.mu[same].mean()) if same.any() else None,
            "mu_cross_topic": float(G.mu[~same].mean()) if (~same).any() else None}


def expert_rho(G, labs, ann):
    idx = {l: i for i, l in enumerate(labs)}
    A = ann.Exercise_A.astype(str).to_numpy(); B = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    ok = np.array([a in idx and b in idx for a, b in zip(A, B)])
    ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in B])
    M, _ = mats(G)
    return {k: float(spearmanr(np.where(ok, M[k][ib, ia], 0.0), y).statistic) for k in ("direct_edge", "hop_baseline", "min", "product")}


def evaluate(G, labs, topics, ann):
    t0 = time.perf_counter()
    seeds = (0, 1, 2, 3, 4)
    out_deg = np.bincount(G.src, minlength=G.n)
    r = {"n": int(G.n), "m": int(G.m), "mean_out_degree": float(G.m / max(1, np.count_nonzero(out_deg))),
         "mu_mean": float(G.mu.mean()), "mu_sd": float(G.mu.std()), "mu_degree": membership_degree_correlation(G)}
    obs = best_qf(G, 1.0, seeds)
    r["Q_F"] = obs.Q_F; r["k"] = obs.n_communities
    r["degree"] = degree_test(G, 200, 1.0, seeds, obs=obs.Q_F)
    fl = density_matched_floor(G, obs.n_communities, draws=20)
    r["floor"] = {k: fl[k] for k in ("floor", "floor_p_in", "by_p_in")}
    r["outcome"] = outcome(r["degree"]["z"], r["degree"]["delta_Q"], fl["floor"])
    for kind in ("global", "row"):
        w = weights_only_test(G, kind, 200, 1.0, seeds, obs=obs.Q_F)
        r[f"weights_{kind}"] = {k: w[k] for k in ("delta_Q", "z", "p_upper", "p_lower", "null_mean", "null_sd")}
        if abs(w["z"]) < 3:
            p = weights_only_power(G, obs.membership, kind=kind, draws=30)
            r[f"power_{kind}"] = {"detectable_tilt": p["detectable_tilt"], "detectable_r_pb": p["detectable_r_pb"],
                                   "rows": [{kk: row[kk] for kk in row if kk in ("tilt", "r_pb", "z")} for row in p["rows"]]}
    r["topics"] = topic_test(G, labs, topics, np.random.default_rng(3))
    r["expert_rho"] = expert_rho(G, labs, ann)
    r["time_s"] = time.perf_counter() - t0
    return r


def main():
    T = pd.read_parquet(RAW / "cache" / "junyi_transition_counts.parquet")
    topics = json.load(open("data/graphs/junyi_topics.json"))
    _, ann = load_junyi_expert()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    names = sys.argv[1:] or DEFAULT_ORDER
    for name in names:
        if name in res:
            continue
        c, p, t = VARIANTS[name]
        G, labs = variant_graph(T, c, p, t)
        res[name] = {"conjunction": c, "pref": p, "tau_E": t, **evaluate(G, labs, topics, ann)}
        json.dump(res, open(OUT, "w"), indent=1)
        r = res[name]
        print(name, r["n"], r["m"], round(r["Q_F"], 3), r["k"], r["outcome"], round(r["degree"]["delta_Q"], 3), r["floor"]["floor"],
              "glob", round(r["weights_global"]["delta_Q"], 4), round(r["weights_global"]["z"], 1),
              "row", round(r["weights_row"]["delta_Q"], 4), round(r["weights_row"]["z"], 1),
              "topicARI", round(r["topics"]["observed_best3"], 3), round(r["topics"]["row_perm_mean"], 3), round(r["topics"]["support_seeds_mean"], 3),
              {k: round(v, 3) for k, v in r["expert_rho"].items()}, round(r["time_s"]), flush=True)


if __name__ == "__main__":
    main()
