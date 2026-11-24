#!/usr/bin/env python3
"""Release 5.4.0. Partition value inside the box B_2, not only at its vertices, with the row null
drawn per seed triple (Section 8.1, Supplementary Table S4i).

ARI is not linear in the slopes s, so its extremes over B_2 need not lie at the vertices. For
Junyi transition and EdNet co-practice: mu_E and 64 points s drawn uniformly from [1, 2]^9 (seed
5402). On each point and for each of the six disjoint seed triples: V = ARI(best of the triple on
phi_s(mu_E)) - ARI(best of the same triple on the support), and V_res = ARI(best of the triple) -
mean ARI of ten row permutations of phi_s(mu_E), each permutation partitioned with the best of the
same triple (so graded and null use the same seeds).
Writes results/v19/interior.json.
"""
from __future__ import annotations
import json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.nulltest import permute_rows  # noqa: E402
from v16_units import rescaled  # noqa: E402
from v15_value_paths import load  # noqa: E402
from v15_simplex_real import groups_for  # noqa: E402
from v18_partition import best, TRIPLES  # noqa: E402

OUT = "results/v19/interior.json"


def _point(args):
    name, s, k = args
    G, _ = load(name)
    gr = groups_for(name); has = np.array([x is not None for x in gr]); gl = pd.factorize(pd.Series(gr[has]).astype(str))[0]
    H = rescaled(G, np.asarray(s, float)); rng = np.random.default_rng(54020 + k)
    obs = [float(ari(gl, best(H, tr)[has])) for tr in TRIPLES]
    nul = [float(np.mean([ari(gl, best(permute_rows(H, rng), tr)[has]) for _ in range(10)])) for tr in TRIPLES]
    return k, obs, nul


def main():
    os.makedirs("results/v19", exist_ok=True)
    rng = np.random.default_rng(5402)
    S = [np.ones(9)] + [rng.uniform(1, 2, 9) for _ in range(64)]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    with Pool(os.cpu_count()) as pool:
        for name in ("junyi_transition", "ednet_copractice"):
            if name in res:
                continue
            G, _ = load(name); Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
            gr = groups_for(name); has = np.array([x is not None for x in gr]); gl = pd.factorize(pd.Series(gr[has]).astype(str))[0]
            sup = [float(ari(gl, best(Gs, tr)[has])) for tr in TRIPLES]
            out = {}
            for k, obs, nul in pool.imap_unordered(_point, [(name, list(s), k) for k, s in enumerate(S)]):
                out[k] = (obs, nul)
            V = np.array([[out[k][0][t] - sup[t] for t in range(6)] for k in range(len(S))])
            R = np.array([[out[k][0][t] - out[k][1][t] for t in range(6)] for k in range(len(S))])
            mV, mR = np.median(V[1:], 1), np.median(R[1:], 1)
            res[name] = {"support_triples": sup, "n_points": len(S) - 1,
                         "mu": {"V_triples": V[0].tolist(), "V_resid_triples": R[0].tolist(), "V_median": float(np.median(V[0])),
                                "V_resid_median": float(np.median(R[0])), "V_resid_range": [float(R[0].min()), float(R[0].max())]},
                         "interior": {"V_median_range": [float(mV.min()), float(mV.max())], "V_median_median": float(np.median(mV)),
                                      "n_median_negative": int((mV < 0).sum()), "n_median_below_0.02": int((mV < 0.02).sum()),
                                      "V_single_range": [float(V[1:].min()), float(V[1:].max())], "n_single_negative": int((V[1:] < 0).sum()),
                                      "V_resid_median_range": [float(mR.min()), float(mR.max())], "n_resid_median_negative": int((mR < 0).sum())},
                         "points": [{"s": list(map(float, S[k])), "V": V[k].tolist(), "V_resid": R[k].tolist()} for k in range(len(S))]}
            json.dump(res, open(OUT, "w"), indent=1)
            print(name, {k: v for k, v in res[name].items() if k != "points"}, flush=True)


if __name__ == "__main__":
    main()
