#!/usr/bin/env python3
"""Release 5.1.0. Analyses of the real graphs that reuse the cached partitions of
scripts/v15_partitions.py (no new Leiden run except where stated).

For every one of the 23 educational graphs, gamma = 1, minimum excess by density class:
  gate         graded-partition gain g = Q_F[mu](Pi_F) - Q_F[mu](Pi_sup) (best of three seeds) against
               the same gain on the ten cached row permutations of mu (Pi_F re-optimised on the
               permuted weights); fires when g exceeds every null draw;
  E_F          Q_F(Pi_F) - Q_F(Pi_cut) on mu_E, seed means;
  refined      decision and kappa* in the family with a knot just below each cut level, in which
               the vertex of the segment ending at the cut is its indicator (Corollary 1);
  kappa_boot   kappa* over 100 bootstrap resamples of the 20 optimiser seeds (support and cut
               partitions resampled jointly), median and 2.5 / 97.5 percentiles;
and on the graphs with a grouping:
  losses       loss of every cut against the support partition and against Pi_F (ARI, 20 seeds,
               rule of Section 6.2);
  value_seeds  partition value of the grades over the six disjoint seed triples (0-2, ..., 15-17)
               and the Monte Carlo SE of the null means (ten draws);
  symmetric    max(ARI(Pi_sup), ARI(Pi_F)) of the best-of-three partitions (selection rule, m8).
For the graphs with a positive partition value also a column null (memberships permuted among the
in-edges of each node; ten draws, best of three seeds; new Leiden runs).
Writes results/v16/real.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden  # noqa: E402
from fuzzykg.simplex import CutFamily, kappa_star  # noqa: E402
from v15_partitions import GRAPHS, load  # noqa: E402
from v15_simplex_real import groups_for, group  # noqa: E402
from v16_semisynthetic import refined_family, loss_rule  # noqa: E402

OUT = "results/v16/real.json"
CUTS = ("q0.25", "q0.50", "q0.75")
DELTA = json.load(open("results/v15/summary.json"))["delta"]
POSITIVE = ("junyi_transition", "ednet_copractice")


def permute_cols(G, rng):
    mu = G.mu.copy(); order = np.argsort(G.dst, kind="stable"); d = G.dst[order]
    for block in np.split(order, np.flatnonzero(np.diff(d)) + 1):
        if block.size > 1:
            mu[block] = mu[rng.permutation(block)]
    return FuzzyGraph(G.n, G.src, G.dst, mu)


class Sub:
    """A CutFamily restricted to a resample of the seeds."""
    def __init__(self, fam, idx):
        self.__dict__.update(fam.__dict__)
        self.stats = {k: (a[idx], M[idx]) for k, (a, M) in fam.stats.items()}
    q = CutFamily.q
    margin = CutFamily.margin


def main(names):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"graphs": {}}
    rng = np.random.default_rng(5110)
    for name in names:
        if name in res["graphs"]:
            continue
        store = dict(np.load(f"results/v15/partitions/{name}.npz"))
        G = load(name); Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
        cls = "dense" if group(name) == "co-practice" else "sparse"; dl = DELTA[cls]
        P = lambda w: [store[f"{w}|1|{s}"] for s in range(20)]
        parts = {w: P(w) for w in ("support", *CUTS)}
        fam = CutFamily(G, parts); famr = refined_family(G, parts)
        rec = {"class": cls, "min_excess": dl, "cuts": {}}
        qF = np.array([fuzzy_modularity(G, p) for p in P("F")])
        for c in CUTS:
            k1 = kappa_star(fam, c, min_excess=dl, kmax=1000.0); k2 = kappa_star(famr, c, min_excess=dl, kmax=1000.0)
            qc = np.array([fuzzy_modularity(G, p) for p in parts[c]])
            kb = []
            for b in range(100):
                idx = rng.integers(0, 20, 20)
                kb.append(min(kappa_star(Sub(fam, idx), c, min_excess=dl, kmax=1000.0)["kappa_star"], 1e3))
            rec["cuts"][c] = {"decision_mu": k1["decision_mu"], "kappa_star": k1["kappa_star"],
                              "decision_refined": k2["decision_mu"], "kappa_star_refined": k2["kappa_star"],
                              "E_F": float(qF.mean() - qc.mean()),
                              "kappa_boot": [float(np.median(kb)), float(np.percentile(kb, 2.5)), float(np.percentile(kb, 97.5))],
                              "kappa_boot_share_ge2": float(np.mean(np.array(kb) >= 2))}
        # gate
        b3 = lambda H, w: max([store[f"{w}|1|{s}"] for s in range(3)], key=lambda p: fuzzy_modularity(H, p))
        pS, pF = b3(Gs, "support"), b3(G, "F")
        g_obs = float(fuzzy_modularity(G, pF) - fuzzy_modularity(G, pS))
        null = []
        for d in range(10):
            Hr = FuzzyGraph(G.n, G.src, G.dst, store[f"Frow{d}|1|mu"])
            null.append(float(fuzzy_modularity(Hr, store[f"Frow{d}|1|best3"]) - fuzzy_modularity(Hr, pS)))
        rec["gate"] = {"g_obs": g_obs, "null_mean": float(np.mean(null)), "null_max": float(np.max(null)),
                       "null_sd": float(np.std(null, ddof=1)), "fires": bool(g_obs > np.max(null))}
        gr = groups_for(name)
        if gr is not None:
            has = np.array([x is not None for x in gr]); gl = pd.factorize(pd.Series(gr[has]).astype(str))[0]
            A = {w: np.array([ari(gl, p[has]) for p in P(w)]) for w in ("support", *CUTS, "F")}
            rec["grouping"] = {"symmetric_max_best3": float(max(ari(gl, pS[has]), ari(gl, pF[has]))),
                               "ari_sup_best3": float(ari(gl, pS[has])), "ari_F_best3": float(ari(gl, pF[has])),
                               "ari_mean": {k: float(v.mean()) for k, v in A.items()}}
            for c in CUTS:
                ls, ds = loss_rule(A["support"], A[c]); lf, df = loss_rule(A["F"], A[c])
                rec["grouping"][c] = {"loss_vs_support": ls, "loss_vs_support_value": ds, "loss_vs_F": lf, "loss_vs_F_value": df}
            # partition value over disjoint seed triples, MC SE of the null means
            glob = [ari(gl, store[f"Fglob{d}|1|best3"][has]) for d in range(10)]
            row = [ari(gl, store[f"Frow{d}|1|best3"][has]) for d in range(10)]
            Vt, Vr = [], []
            for t in range(6):
                tr = range(3 * t, 3 * t + 3)
                ps = max([store[f"support|1|{s}"] for s in tr], key=lambda p: fuzzy_modularity(Gs, p))
                pf = max([store[f"F|1|{s}"] for s in tr], key=lambda p: fuzzy_modularity(G, p))
                Vt.append(ari(gl, pf[has]) - ari(gl, ps[has])); Vr.append(ari(gl, pf[has]) - np.mean(row))
            rec["value_seeds"] = {"V_total_triples": [float(x) for x in Vt], "V_resid_triples": [float(x) for x in Vr],
                                  "glob_mc_se": float(np.std(glob, ddof=1) / np.sqrt(10)), "row_mc_se": float(np.std(row, ddof=1) / np.sqrt(10))}
            if name in POSITIVE:
                col = []
                for d in range(10):
                    Hc = permute_cols(G, rng)
                    pc = max([_leiden(Hc, s) for s in range(3)], key=lambda p: fuzzy_modularity(Hc, p))
                    col.append(ari(gl, pc[has]))
                rec["value_seeds"]["col_mean"] = float(np.mean(col)); rec["value_seeds"]["col_mc_se"] = float(np.std(col, ddof=1) / np.sqrt(10))
                rec["value_seeds"]["glob_mean"] = float(np.mean(glob)); rec["value_seeds"]["row_mean"] = float(np.mean(row))
        res["graphs"][name] = rec
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        print(name, rec["gate"]["fires"], round(g_obs, 4), round(np.max(null), 4),
              {c: (r["decision_mu"][0], round(r["kappa_star"], 2), r["decision_refined"][0], round(r["kappa_star_refined"], 2), [round(x, 2) for x in r["kappa_boot"]]) for c, r in rec["cuts"].items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or GRAPHS)
