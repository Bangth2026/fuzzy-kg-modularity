#!/usr/bin/env python3
"""Equal-size binarisations by four rules on the LFR-style benchmark (Section 8.1).

The design of scripts/nonnested_binarisation.py on LFR-style graphs: n = 2000,
mixing 0.1 to 0.6, shift 0 and 0.5, three graph seeds, two optimiser seeds as
in the LFR stage of scripts/benchmark_levelset.py. Within-graph concordance
with the recovery shortfall over pairs of rules whose shortfalls differ by at
least 0.01; retained mass sign-reversed; all four rules and the three informed
rules. Writes results/nonnested_lfr.json.
"""
import itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph, fuzzy_leiden, fuzzy_modularity  # noqa: E402
from fuzzykg.multislice import multislice_leiden, level_grid, qbar  # noqa: E402
from fuzzykg.synthetic import lfr_style_fuzzy_graph  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402
from nonnested_binarisation import keep_mask, RULES  # noqa: E402


def conc(rows, key, sign, rules, rng):
    v = []; npairs = 0
    for r in rows:
        c = {k: x for k, x in r["cells"].items() if rules is None or k in rules}; s = n = 0
        for a, b in itertools.combinations(c, 2):
            dl = c[a]["shortfall"] - c[b]["shortfall"]
            if abs(dl) < 0.01:
                continue
            n += 1; dd = sign * (c[a][key] - c[b][key]); s += 1 if dl * dd > 0 else (0.5 if dd == 0 else 0)
        if n:
            v.append(s / n); npairs += n
    v = np.array(v); b = [rng.choice(v, v.size).mean() for _ in range(2000)]
    return {"concordance": float(v.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "graphs": int(v.size), "pairs": npairs}


def main(lseeds=(0, 1)):
    rng = np.random.default_rng(19); rows = []
    for mixing in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        for shift in (0.0, 0.5):
            for seed in range(3):
                G, truth = lfr_style_fuzzy_graph(2000, mixing, mu_mean=0.5, mu_sd=0.15, shift=shift, seed=seed)
                alphas = level_grid(G)
                ms = max((multislice_leiden(G, alphas, seed=s) for s in lseeds), key=lambda r: r.Qbar)
                f = max((fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in lseeds), key=lambda r: r.Q_F)
                for frac in (0.5, 0.25):
                    cells = {}
                    for rule in RULES:
                        mask = keep_mask(G, rule, frac, rng)
                        Gb = FuzzyGraph(G.n, G.src[mask], G.dst[mask], np.ones(int(mask.sum())))
                        b = max((fuzzy_leiden(Gb, 1.0, seed=s, n_iterations=0) for s in lseeds), key=lambda r: r.Q_F)
                        cells[rule] = {"shortfall": 1 - ari(truth, b.membership), "index": ms.Qbar - qbar(G, b.membership, alphas),
                                       "qf_drop": f.Q_F - fuzzy_modularity(G, b.membership, 1.0), "retained_mass": float(G.mu[mask].sum() / G.W),
                                       "singleton_share": float((np.bincount(b.membership) == 1).sum() / G.n)}
                    rows.append({"mixing": mixing, "shift": shift, "seed": seed, "frac": frac, "cells": cells})
                print(mixing, shift, seed, {r: round(c["shortfall"], 2) for r, c in rows[-1]["cells"].items()}, flush=True)
                json.dump({"rows": rows}, open("results/nonnested_lfr.json", "w"), indent=1)
    summ = {}
    for name, rules in (("all_rules", None), ("informed_rules", ("global", "per_source", "disparity"))):
        summ[name] = {k: conc(rows, k, sg, rules, rng) for k, sg in (("index", 1), ("qf_drop", 1), ("retained_mass", -1), ("singleton_share", 1))}
    print(json.dumps(summ, indent=1))
    json.dump({"rows": rows, "summary": summ}, open("results/nonnested_lfr.json", "w"), indent=1)


if __name__ == "__main__":
    main()
