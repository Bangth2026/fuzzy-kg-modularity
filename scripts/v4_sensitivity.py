#!/usr/bin/env python3
"""Version 4.0.0. Sensitivity of the findings to the construction conventions.

One-at-a-time variation around the defaults of Section 4 on five logs (ASSISTments 2009 and
2017 KC level, Junyi, Algebra 2005, Eedi):
  co-practice   horizon h in {5, 10, 20}, support in {10, 20, 40}, tau_E in {0.1, 0.2, 0.3}
  transition    support in {10, 20, 40}, tau_E in {0.1, 0.2, 0.3},
                membership min(asym, pref) [default], asym * pref, (asym + pref) / 2
For every variant: size, density, reciprocity, mean out-degree; Q_F and k (best of five Leiden
runs at gamma = 1); the two weights-only nulls of Algorithm 1 (global and row permutation, 50
draws, three seeds; 20 draws on graphs with more than 50,000 edges); the Q_F drop of the median and
upper-quartile cuts with the no-cut (support) baseline; and on Junyi the Spearman correlation of
the direct edge, hop distance, max-min and product scores with the crowd prerequisite score over
the fixed 1,954 annotated pairs (pairs outside the graph score 0).
Writes results/v4/sensitivity.json.
Usage: python scripts/v4_sensitivity.py 2009 2017 junyi algebra2005 eedi
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from data.loaders import load_any, load_junyi_expert  # noqa: E402
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.construction import ConstructionConfig, TransitionConfig, build_fuzzy_graph, build_transition_graph, preprocess  # noqa: E402
from fuzzykg.nulltest import best_qf, weights_only_test  # noqa: E402
from composition_robustness import mats  # noqa: E402

OUT = "results/v4/sensitivity.json"


def describe(G):
    out_deg = np.bincount(G.src, minlength=G.n)
    act = np.flatnonzero((out_deg > 0) | (np.bincount(G.dst, minlength=G.n) > 0))
    return {"n": int(G.n), "m": int(G.m), "density": float(G.m / max(1, G.n * (G.n - 1))),
            "reciprocity": float(G.reciprocity()) if hasattr(G, "reciprocity") else None,
            "mean_out_degree": float(G.m / max(1, act.size)), "mu_mean": float(G.mu.mean()), "mu_sd": float(G.mu.std())}


def cut_drops(G, PiF_Q, seeds=range(5)):
    out = {}
    for c, a in (("support", None), ("q0.50", float(np.quantile(G.mu, 0.5))), ("q0.75", float(np.quantile(G.mu, 0.75)))):
        H = FuzzyGraph(G.n, G.src, G.dst, np.ones_like(G.mu)) if a is None else G.alpha_cut(a)
        b = best_qf(H, 1.0, tuple(seeds))
        out[c] = float(PiF_Q - fuzzy_modularity(G, b.membership, 1.0))
    return out


def expert(G, ann):
    idx = {str(l): i for i, l in enumerate(G.labels)}
    A = ann.Exercise_A.astype(str).to_numpy(); B = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    ok = np.array([a in idx and b in idx for a, b in zip(A, B)])
    ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in B])
    M, _ = mats(G)
    return {k: float(spearmanr(np.where(ok, M[k][ib, ia], 0.0), y).statistic) for k in ("direct_edge", "hop_baseline", "min", "product")}


def evaluate(G, ann=None):
    r = describe(G)
    t0 = time.perf_counter()
    b = best_qf(G, 1.0)
    r["Q_F"] = b.Q_F; r["k"] = b.n_communities
    draws = 20 if G.m > 50000 else 50
    for kind in ("global", "row"):
        w = weights_only_test(G, kind, draws, 1.0, (0, 1, 2), obs=b.Q_F)
        r[f"weights_{kind}"] = {"delta_Q": w["delta_Q"], "z": w["z"], "draws": draws}
    r["qf_drop"] = cut_drops(G, b.Q_F)
    if ann is not None:
        r["expert_rho"] = expert(G, ann)
    r["time_s"] = time.perf_counter() - t0
    return r


def transition_variant(tab_all, labels_all, support, tau, member):
    t = tab_all[(tab_all.moves + tab_all.moves_back) >= support].copy()
    t = t[t.asym > 0]
    mu = {"min": np.minimum(t.asym, t.pref), "product": t.asym * t.pref, "mean": (t.asym + t.pref) / 2}[member]
    t = t.assign(mu=mu)
    t = t[t.mu >= tau]
    labs = sorted(set(t.src.astype(str)) | set(t.dst.astype(str)))
    ix = {l: i for i, l in enumerate(labs)}
    return FuzzyGraph(len(labs), t.src.astype(str).map(ix).to_numpy(np.int64), t.dst.astype(str).map(ix).to_numpy(np.int64),
                      t.mu.to_numpy(float), labels=labs)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    ex, ann = load_junyi_expert()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tonly = "--transition-only" in sys.argv
    for rel in args:
        df = load_any(rel, "kc")
        d = preprocess(df, ConstructionConfig())
        a = ann if rel == "junyi" else None
        # co-practice
        for key, cfg in [] if tonly else [("default", ConstructionConfig()), ("h5", ConstructionConfig(horizon=5)), ("h20", ConstructionConfig(horizon=20)),
                         ("support10", ConstructionConfig(min_support=10)), ("support40", ConstructionConfig(min_support=40)),
                         ("tau0.1", ConstructionConfig(tau_E=0.1)), ("tau0.3", ConstructionConfig(tau_E=0.3))]:
            k = f"{rel}/copractice/{key}"
            if k in res:
                continue
            G, _ = build_fuzzy_graph(d, cfg, preprocessed=True)
            res[k] = evaluate(G, a); json.dump(res, open(OUT, "w"), indent=1)
            print(k, G.summary(), {x: res[k][x] for x in ("Q_F", "k")}, res[k]["weights_row"]["z"], res[k]["qf_drop"], res[k].get("expert_rho"), flush=True)
        # transition: one count pass at support 10 and tau ~ 0, variants derived from the signals
        G0, tab = build_transition_graph(d, TransitionConfig(min_support=10, tau_E=1e-9), preprocessed=True)
        for key, (sup, tau, mem) in [("default", (20, 0.2, "min")), ("support10", (10, 0.2, "min")), ("support40", (40, 0.2, "min")),
                                     ("tau0.1", (20, 0.1, "min")), ("tau0.3", (20, 0.3, "min")),
                                     ("member_product", (20, 0.2, "product")), ("member_mean", (20, 0.2, "mean"))]:
            k = f"{rel}/transition/{key}"
            if k in res:
                continue
            G = transition_variant(tab, None, sup, tau, mem)
            res[k] = evaluate(G, a); json.dump(res, open(OUT, "w"), indent=1)
            print(k, G.summary(), {x: res[k][x] for x in ("Q_F", "k")}, res[k]["weights_row"]["z"], res[k]["qf_drop"], res[k].get("expert_rho"), flush=True)


if __name__ == "__main__":
    main()
