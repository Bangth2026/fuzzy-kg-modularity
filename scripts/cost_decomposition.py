#!/usr/bin/env python3
"""Decomposition of the cost index on the real graphs (Section 8.3).

For every graph, with the quantile grid of the multislice reference:

* the *no-cut* index: the cost of the crisp partition of the support graph
  (all edges kept, memberships replaced by 1), Qbar(Pi*) - Qbar(Pi_support);
  it is the part of the index that comes from discarding the grades alone;
* the *edge* part of a quantile cut: Qbar(Pi_support) - Qbar(Pi_cut), so that
  index = no-cut part + edge part;
* the optimiser-seed dispersion of every index (ten seeds; Pi* and the cut
  partition taken from the same seed);
* a weights-only reference: the memberships are permuted over the fixed
  edges (global permutation), Pi* is re-optimised and the no-cut and cut
  indices are recomputed; the observed no-cut index is reported against
  this permutation distribution, which says whether the grades carry
  anything the objective rewards beyond what any assignment of the same
  memberships to the same edges would carry;
* on Junyi, the agreement of every partition with the 40 expert topics,
  per seed.

Usage: python scripts/cost_decomposition.py [--seeds 10] [--perms 10] [names ...]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph, fuzzy_leiden  # noqa: E402
from fuzzykg.multislice import multislice_leiden, level_grid, qbar  # noqa: E402
from levelset_real import load, truth_for, NAMES  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

CUTS = ("q0.25", "q0.50", "q0.75")


def crisp(Ga, seed):
    return fuzzy_leiden(Ga, 1.0, seed=seed, n_iterations=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=NAMES)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--perms", type=int, default=10)
    ap.add_argument("--perm-seeds", type=int, default=3)
    ap.add_argument("--out", default="results/cost_decomposition.json")
    args = ap.parse_args()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for name in args.names:
        if name in res:
            continue
        t0 = time.perf_counter()
        G = load(name)
        alphas = level_grid(G)
        qs = {c: float(np.quantile(G.mu, float(c[1:]))) for c in CUTS}
        sup = G.alpha_cut(float(G.mu.min()))
        seeds = range(args.seeds)
        ms = [multislice_leiden(G, alphas, seed=s) for s in seeds]
        sp = [crisp(sup, s) for s in seeds]
        cu = {c: [crisp(G.alpha_cut(a), s) for s in seeds] for c, a in qs.items()}
        fz = [fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in seeds]
        Qms = np.array([m.Qbar for m in ms])
        Qsp = np.array([qbar(G, r.membership, alphas) for r in sp])
        Qcu = {c: np.array([qbar(G, r.membership, alphas) for r in v]) for c, v in cu.items()}
        # best-of-five protocol of the paper
        b5 = lambda arr, objs, key: objs[int(np.argmax([key(o) for o in objs[:5]]))]
        i_ms = int(np.argmax(Qms[:5])); i_sp = int(np.argmax([r.Q_F for r in sp[:5]]))
        nocut_best = float(Qms[i_ms] - Qsp[i_sp])
        row = {"n": G.n, "m": G.m, "alphas": alphas.tolist(), "qbar_star_sd": float(Qms.std(ddof=1)),
               "nocut": {"index_best5": nocut_best, "per_seed": (Qms - Qsp).tolist(),
                         "sd": float((Qms - Qsp).std(ddof=1))}, "cuts": {}}
        for c in CUTS:
            i_c = int(np.argmax([r.Q_F for r in cu[c][:5]]))
            idx = float(Qms[i_ms] - Qcu[c][i_c])
            row["cuts"][c] = {"alpha": qs[c], "index_best5": idx, "edge_part_best5": float(Qsp[i_sp] - Qcu[c][i_c]),
                              "per_seed": (Qms - Qcu[c]).tolist(), "sd": float((Qms - Qcu[c]).std(ddof=1))}
        # weights-only reference for the indices
        rng = np.random.default_rng(12345)
        null_nocut, null_cut = [], {c: [] for c in CUTS}
        for r in range(args.perms):
            Gr = FuzzyGraph(G.n, G.src, G.dst, rng.permutation(G.mu))
            ar = level_grid(Gr)
            msr = max((multislice_leiden(Gr, ar, seed=s) for s in range(args.perm_seeds)), key=lambda x: x.Qbar)
            spr = max((crisp(sup, s) for s in range(args.perm_seeds)), key=lambda x: x.Q_F)
            null_nocut.append(msr.Qbar - qbar(Gr, spr.membership, ar))
            for c in CUTS:
                cr = max((crisp(Gr.alpha_cut(qs[c]), s) for s in range(args.perm_seeds)), key=lambda x: x.Q_F)
                null_cut[c].append(msr.Qbar - qbar(Gr, cr.membership, ar))
        nn = np.array(null_nocut)
        row["nocut"]["perm_mean"] = float(nn.mean()); row["nocut"]["perm_sd"] = float(nn.std(ddof=1))
        row["nocut"]["perm_p_upper"] = float((np.sum(nn >= nocut_best) + 1) / (len(nn) + 1))
        for c in CUTS:
            v = np.array(null_cut[c]); row["cuts"][c]["perm_mean"] = float(v.mean()); row["cuts"][c]["perm_sd"] = float(v.std(ddof=1))
        truth = truth_for(name, G.n)
        if truth is not None:
            row["topic_ari"] = {"fuzzy": [ari(truth, r.membership) for r in fz], "multislice": [ari(truth, r.membership) for r in ms],
                                "support": [ari(truth, r.membership) for r in sp],
                                **{c: [ari(truth, r.membership) for r in v] for c, v in cu.items()}}
        row["time_s"] = time.perf_counter() - t0
        res[name] = row
        json.dump(res, open(args.out, "w"), indent=1)
        print(name, json.dumps({"nocut": round(nocut_best, 4), "nocut_sd": round(row["nocut"]["sd"], 4),
                                "nocut_perm": (round(row["nocut"]["perm_mean"], 4), round(row["nocut"]["perm_sd"], 4)),
                                "cuts": {c: (round(v["index_best5"], 3), round(v["sd"], 3), round(v["perm_mean"], 3)) for c, v in row["cuts"].items()},
                                "topic": {k: round(float(np.mean(v)), 3) for k, v in row.get("topic_ari", {}).items()},
                                "t": round(row["time_s"])}), flush=True)


if __name__ == "__main__":
    main()
