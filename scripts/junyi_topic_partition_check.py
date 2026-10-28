#!/usr/bin/env python3
"""Does Leiden on Q_F recover the Junyi topics better than crisp Leiden on the support
graph because of the memberships, or because of the node strengths they induce?
Topic ARI of Leiden on Q_F (ten seeds) on the observed graph, on 20 row-wise
permutations of the memberships (out-strengths kept) and on 20 global permutations,
against crisp Leiden on the support graph. Writes results/junyi_topic_partition_check.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph, fuzzy_leiden  # noqa
from fuzzykg.nulltest import permute_rows, permute_global  # noqa
from levelset_real import load, truth_for  # noqa
from sklearn.metrics import adjusted_rand_score as ari  # noqa

def best_ari(G, truth, seeds):
    return ari(truth, max((fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in seeds), key=lambda r: r.Q_F).membership)

out = {}
for name in ("junyi_transition", "junyi_copractice"):
    G = load(name); truth = truth_for(name, G.n); rng = np.random.default_rng(3)
    obs = [ari(truth, fuzzy_leiden(G, 1.0, seed=s, n_iterations=0).membership) for s in range(10)]
    sup = [ari(truth, fuzzy_leiden(G.alpha_cut(float(G.mu.min())), 1.0, seed=s, n_iterations=0).membership) for s in range(10)]
    row = [best_ari(permute_rows(G, rng), truth, range(3)) for _ in range(20)]
    glo = [best_ari(permute_global(G, rng), truth, range(3)) for _ in range(20)]
    out[name] = {"observed": obs, "support": sup, "row_perm_best3": row, "global_perm_best3": glo,
                 "observed_best3": best_ari(G, truth, range(3))}
    print(name, {k: (round(float(np.mean(v)), 3), round(float(np.std(v)), 3)) if isinstance(v, list) else round(v, 3) for k, v in out[name].items()}, flush=True)
json.dump(out, open("results/junyi_topic_partition_check.json", "w"), indent=1)
