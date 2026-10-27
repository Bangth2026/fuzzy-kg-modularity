"""Planted-partition benchmark for the binarise-then-partition shortcut (Table 7).

For each setting of the membership distribution (mean, spread, community shift)
on a directed planted-partition graph with known communities:

  * Fuzzy Leiden on the fuzzy graph (gamma scanned, best Q_F kept);
  * binarise at absolute levels alpha and at membership quantiles q, run crisp
    Leiden on the level set with unit weights, with the resolution REFITTED per
    level set (scan over gamma', keep the partition with the best Q_F on the
    fuzzy graph);
  * report NMI/ARI against the planted partition, NMI between the level-set
    partition and the fuzzy partition, the operational loss
        loss(alpha) = Q_F(Pi_fuzzy) - max_gamma' Q_F(Pi_alpha(gamma')),
    and, for reference, the Corollary 1 gap of Pi_fuzzy.

Also checks the constant-membership counterexample: all mu_E = c gives a
Corollary 1 gap of gamma (1/c - 1) sum_k a_k b_k although binarising at any
alpha <= c is lossless.

Usage:  python scripts/benchmark_binarisation.py [--quick] [--out results/binarisation.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden, fuzzy_modularity, correlation_gap  # noqa: E402
from fuzzykg.synthetic import planted_partition_fuzzy_graph  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari, adjusted_mutual_info_score as ami  # noqa: E402

GAMMAS = [0.25, 0.5, 1.0, 2.0, 4.0]
ALPHAS = [0.4, 0.6, 0.8]
QUANTILES = [0.25, 0.5, 0.75]


def best_partition(G: FuzzyGraph, Geval: FuzzyGraph, seeds, practitioner=False):
    """Scan gamma' and seeds on G.  Oracle protocol (default): keep the partition
    with the best Q_F on Geval at gamma = 1.  Practitioner protocol: run crisp
    Leiden on G at the default resolution gamma' = 1 only, keep the seed with the
    best modularity of G itself (what a user who binarises sees), and report that
    partition's Q_F on Geval at gamma = 1."""
    best = None
    gammas = [1.0] if practitioner else GAMMAS
    for g in gammas:
        for s in seeds:
            r = fuzzy_leiden(G, g, seed=s, n_iterations=0)
            q = fuzzy_modularity(Geval, r.membership, 1.0)
            key = r.Q_F if practitioner else q
            if best is None or key > best[3]:
                best = (q, g, r.membership, key)
    return best[:3]


def level_set(G: FuzzyGraph, alpha: float):
    keep = G.mu >= alpha
    if keep.sum() == 0:
        return None
    return FuzzyGraph(G.n, G.src[keep], G.dst[keep], np.ones(int(keep.sum())))


def run_setting(mu_mean, mu_sd, shift, p_in, n, k, seeds, graph_seed):
    G, truth = planted_partition_fuzzy_graph(n, k, 8.0, p_in, mu_mean, mu_sd, shift, seed=graph_seed)
    qf, gf, pf = best_partition(G, G, seeds)
    gap = correlation_gap(G, pf, 1.0)
    row = {
        "mu_mean": mu_mean, "mu_sd": mu_sd, "shift": shift, "p_in": p_in, "graph_seed": graph_seed,
        "mu_mean_obs": float(G.mu.mean()), "mu_sd_obs": float(G.mu.std()),
        "fuzzy": {"Q_F": qf, "gamma": gf, "k": int(len(np.unique(pf))),
                  "NMI_truth": nmi(truth, pf), "ARI_truth": ari(truth, pf), "AMI_truth": ami(truth, pf),
                  "gap_cor1": gap["gap"]},
        "cuts": {},
    }
    cuts = [("alpha", a, a) for a in ALPHAS] + [("quantile", q, float(np.quantile(G.mu, q))) for q in QUANTILES]
    for kind, label, a in cuts:
        Ga = level_set(G, a)
        key = f"{kind}={label}"
        if Ga is None:
            row["cuts"][key] = None
            continue
        qa, ga, pa = best_partition(Ga, G, seeds)
        qp, gp, pp = best_partition(Ga, G, seeds, practitioner=True)
        row["cuts"][key] = {
            "alpha": a, "m_alpha": Ga.m, "edge_frac": Ga.m / G.m,
            "mass_kept": float(G.mu[G.mu >= a].sum() / G.W),
            "mean_degree_alpha": Ga.m / G.n,
            "gamma_refit": ga, "k": int(len(np.unique(pa))),
            "Q_F_on_fuzzy": qa, "loss": qf - qa,
            "NMI_truth": nmi(truth, pa), "ARI_truth": ari(truth, pa), "AMI_truth": ami(truth, pa),
            "NMI_vs_fuzzy": nmi(pf, pa), "ARI_vs_fuzzy": ari(pf, pa),
            "practitioner": {"gamma": gp, "k": int(len(np.unique(pp))), "Q_F_on_fuzzy": qp,
                             "loss": qf - qp, "ARI_truth": ari(truth, pp), "AMI_truth": ami(truth, pp)},
        }
    return row


def constant_membership_check():
    """All memberships equal to c: binarising at alpha <= c is lossless, yet the
    Corollary 1 gap equals gamma (1/c - 1) sum_k a_k b_k."""
    out = []
    for c in (0.25, 0.5, 0.75):
        G0, truth = planted_partition_fuzzy_graph(400, 4, 6.0, 0.9, 0.5, 0.1, 0.0, seed=1)
        G = FuzzyGraph(G0.n, G0.src, G0.dst, np.full(G0.m, c))
        gap = correlation_gap(G, truth, 1.0)
        W = G.W
        from fuzzykg.modularity import _community_sums
        _, S_out, S_in = _community_sums(G, truth)
        predicted = (1 / c - 1) * float(np.sum((S_out / W) * (S_in / W)))
        # the crisp graph (unit weights) has the same modularity for the same partition
        Gc = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
        out.append({"c": c, "Q_F": gap["Q_F"], "Q_crisp_same_partition": fuzzy_modularity(Gc, truth),
                    "gap_cor1": gap["gap"], "predicted_gap": predicted})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--graph-seeds", type=int, default=5, help="graph seeds per setting (1 with --quick)")
    ap.add_argument("--seed-list", type=int, nargs="*", default=None,
                    help="run only these graph seeds (for splitting the run over processes; merge with --merge)")
    ap.add_argument("--merge", nargs="*", default=None, help="merge partial result files into --out and exit")
    ap.add_argument("--out", default="results/binarisation.json")
    ap.add_argument("--edges", nargs="*", default=None,
                    help="real edge lists (src,dst,mu CSV); runs the shortcut with gamma refitted per level set")
    args = ap.parse_args()
    if args.merge:
        rows, seen = [], set()
        for f in args.merge:
            for r in json.load(open(f)):
                key = (r["mu_mean"], r["mu_sd"], r["shift"], r["p_in"], r["graph_seed"])
                if key not in seen:
                    seen.add(key); rows.append(r)
        rows.sort(key=lambda r: (r["mu_mean"], r["mu_sd"], r["shift"], r["p_in"], r["graph_seed"]))
        seeds_ = sorted({r["graph_seed"] for r in rows})
        res = {"settings": {"n": 1000, "k": 20, "out_degree": 8.0, "gammas": GAMMAS, "alphas": ALPHAS,
                            "quantiles": QUANTILES, "seeds": [0, 1, 2], "graph_seeds": seeds_},
               "rows": rows, "constant_membership": constant_membership_check(), "runtime_s": None}
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"merged {len(rows)} rows into {args.out}")
        return
    if args.edges:
        import pandas as pd
        out = []
        for path in args.edges:
            df = pd.read_csv(path)
            G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
            seeds = (0, 1, 2, 3, 4)
            qf, gf, pf = best_partition(G, G, seeds)
            row = {"graph": os.path.basename(path), "summary": G.summary(),
                   "fuzzy": {"Q_F": qf, "gamma": gf, "k": int(len(np.unique(pf))),
                             "gap_cor1": correlation_gap(G, pf, 1.0)["gap"]}, "cuts": {}}
            for a in (0.2, 0.4, 0.6, 0.8):
                Ga = level_set(G, a)
                if Ga is None:
                    row["cuts"][f"alpha={a}"] = None; continue
                qa, ga, pa = best_partition(Ga, G, seeds)
                qp, gp, pp = best_partition(Ga, G, seeds, practitioner=True)
                row["cuts"][f"alpha={a}"] = {"m_alpha": Ga.m, "edge_frac": Ga.m / G.m,
                                             "mass_kept": float(G.mu[G.mu >= a].sum() / G.W),
                                             "gamma_refit": ga, "k": int(len(np.unique(pa))), "Q_F_on_fuzzy": qa,
                                             "loss": qf - qa, "NMI_vs_fuzzy": nmi(pf, pa), "ARI_vs_fuzzy": ari(pf, pa),
                                             "AMI_vs_fuzzy": ami(pf, pa),
                                             "practitioner": {"gamma": gp, "k": int(len(np.unique(pp))), "Q_F_on_fuzzy": qp,
                                                              "loss": qf - qp, "ARI_vs_fuzzy": ari(pf, pp), "AMI_vs_fuzzy": ami(pf, pp)}}
            print(json.dumps(row, indent=1), flush=True)
            out.append(row)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out.replace(".json", "_real.json"), "w"), indent=1)
        return
    n, k = (400, 8) if args.quick else (1000, 20)
    seeds = (0,) if args.quick else (0, 1, 2)
    graph_seeds = (0,) if args.quick else tuple(args.seed_list if args.seed_list is not None else range(args.graph_seeds))
    means = [0.3, 0.5, 0.7, 0.9]
    sds = [0.05, 0.15, 0.25]
    shifts = [0.0, 0.5]
    p_ins = [0.7]
    rows = []
    partial = args.out.replace(".json", "_partial.json") if args.seed_list is None else \
        args.out.replace(".json", "_partial_" + "_".join(map(str, args.seed_list)) + ".json")
    if os.path.exists(partial):                 # resume
        rows = json.load(open(partial))
    done = {(r["mu_mean"], r["mu_sd"], r["shift"], r["p_in"], r["graph_seed"]) for r in rows}
    t0 = time.time()
    for mu_mean, mu_sd, shift, p_in, gs in itertools.product(means, sds, shifts, p_ins, graph_seeds):
        if (mu_mean, mu_sd, shift, p_in, gs) in done:
            continue
        rows.append(run_setting(mu_mean, mu_sd, shift, p_in, n, k, seeds, gs))
        json.dump(rows, open(partial, "w"))
        print(f"mean={mu_mean} sd={mu_sd} shift={shift} seed={gs}: fuzzy NMI={rows[-1]['fuzzy']['NMI_truth']:.3f} "
              f"gap={rows[-1]['fuzzy']['gap_cor1']:.3f} "
              + " ".join(f"{kk}:NMI={v['NMI_truth']:.2f},loss={v['loss']:.3f}" for kk, v in rows[-1]['cuts'].items() if v),
              flush=True)
    res = {"settings": {"n": n, "k": k, "out_degree": 8.0, "gammas": GAMMAS, "alphas": ALPHAS,
                        "quantiles": QUANTILES, "seeds": list(seeds), "graph_seeds": list(graph_seeds)},
           "rows": rows, "constant_membership": constant_membership_check(),
           "runtime_s": time.time() - t0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    if os.path.exists(partial):
        os.remove(partial)
    print("constant-membership check:", json.dumps(res["constant_membership"], indent=1))
    print(f"written {args.out} in {res['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
