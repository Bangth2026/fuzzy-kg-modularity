#!/usr/bin/env python3
"""The rank-coded Q_F drop, a third audit implied by Lemma 2 (Section 8.1).

On the quantile grid with mass weighting, Lemma 2(ii) gives
    Qbar(Pi) = Q_F[N_A/L](Pi) - gamma * C(Pi),
with N_A(e) the number of grid levels that edge e passes and C(Pi) the sum
over communities of the covariance of their out- and in-degree shares across
levels. The first term is an ordinary weighted modularity on rank-coded
memberships, so the drop
    rank_drop(cut) = Q_F[N_A/L](Pi_rank) - Q_F[N_A/L](Pi_cut),
with Pi_rank the Leiden partition of the rank-coded graph, needs one Leiden run
and is invariant to every strictly increasing rescaling of mu_E.

For every benchmark graph the script recomputes the cut partitions with the
seeds of scripts/benchmark_levelset.py (the ARI is stored so that the match
with results/levelset_<stage>.json can be checked) and records, per cut,
rank_drop, Qbar(Pi_cut), Q_F[N_A/L](Pi_cut) and C(Pi_cut). Within one graph the
order of cuts by the cost index is the order by -Qbar(Pi_cut) and the order by
the rank-coded drop is the order by -Q_F[N_A/L](Pi_cut), so the covariance term
changes the order of two cuts exactly when these two orders differ.

Stages: shift (planted sweep, cuts q0.50, q0.75, a0.6, as Table 2), lfr
(LFR-style benchmark, as Table 2), rescale (planted sweep with the quantile cuts
q0.25, q0.50, q0.75 of the 330-pair comparison).

Usage: python scripts/rank_drop_audit.py --stage shift|lfr|rescale [--part i/k]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden, fuzzy_modularity  # noqa: E402
from fuzzykg.multislice import level_grid, qbar  # noqa: E402
from fuzzykg.synthetic import planted_partition_fuzzy_graph, lfr_style_fuzzy_graph  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402


def best(G, seeds):
    return max((fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda r: r.Q_F)


def audit(G, truth, seeds, cuts):
    alphas = level_grid(G)
    N = (G.mu[:, None] >= alphas[None, :]).sum(1).astype(float)
    Gr = FuzzyGraph(G.n, G.src, G.dst, N / alphas.size)
    pr = best(Gr, seeds); f = best(G, seeds)
    rec = {"L": int(alphas.size), "rank_partition_ari": float(ari(truth, pr.membership)), "fuzzy_ari": float(ari(truth, f.membership)), "cuts": {}}
    for name, a in cuts.items():
        try:
            Ga = G.alpha_cut(a)
        except ValueError:
            rec["cuts"][name] = None; continue
        if Ga.m == 0:
            rec["cuts"][name] = None; continue
        m = best(Ga, seeds).membership
        qr = fuzzy_modularity(Gr, m, 1.0); qb = qbar(G, m, alphas)
        rec["cuts"][name] = {"ari": float(ari(truth, m)), "rank_drop": float(pr.Q_F - qr), "qf_drop": float(f.Q_F - fuzzy_modularity(G, m, 1.0)),
                             "qbar_cut": float(qb), "qf_rank_cut": float(qr), "cov_cut": float(qr - qb)}
    return rec


def jobs(stage):
    if stage in ("shift", "rescale"):
        for shift in np.round(np.arange(0, 1.01, 0.1), 1):
            for seed in range(10):
                yield {"shift": float(shift), "seed": seed}
    else:
        for n in (500, 2000, 10000):
            for mixing in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                for shift in (0.0, 0.5):
                    for seed in range(10 if n <= 2000 else 5):
                        yield {"n": n, "mixing": mixing, "shift": shift, "seed": seed}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["shift", "lfr", "rescale"])
    ap.add_argument("--part", default="0/1"); a = ap.parse_args()
    i, k = map(int, a.part.split("/"))
    out = f"results/rank_drop_{a.stage}" + ("" if k == 1 else f"_part{i}") + ".json"
    rows = []
    for j, key in enumerate(jobs(a.stage)):
        if j % k != i:
            continue
        if a.stage == "lfr":
            G, truth = lfr_style_fuzzy_graph(key["n"], key["mixing"], mu_mean=0.5, mu_sd=0.15, shift=key["shift"], seed=key["seed"]); seeds = [0, 1]
        else:
            G, truth = planted_partition_fuzzy_graph(1000, 20, 8.0, 0.5, 0.5, 0.15, key["shift"], seed=key["seed"]); seeds = [0, 1, 2]
        if a.stage == "rescale":
            cuts = {f"{q:.2f}": float(np.quantile(G.mu, q)) for q in (0.25, 0.50, 0.75)}
        else:
            cuts = {"q0.50": float(np.quantile(G.mu, 0.5)), "q0.75": float(np.quantile(G.mu, 0.75)), "a0.6": 0.6}
        rec = audit(G, truth, seeds, cuts); rec.update(key); rows.append(rec)
        print(key, {c: (None if v is None else round(v["rank_drop"], 4)) for c, v in rec["cuts"].items()}, flush=True)
        json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
