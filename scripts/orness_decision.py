#!/usr/bin/env python3
"""Does the unit invariance of the cost index change a decision? (Section 8.3)

The three ASSISTments KC-level co-practice graphs are rebuilt with OWA weights
generated from orness 0.50, 0.65 (the default weights) and 0.80, which moves the
mean membership while the candidate pairs stay the same. On every graph, the
support (no cut), the three quantile cuts and the absolute cuts at 0.4, 0.5
and 0.6 are priced by the cost index on the quantile grid and by the Q_F drop
(best of five optimiser seeds for every partition). Reported per graph and
orness: both prices for every cut, the cut each measure ranks cheapest among
the binarising cuts, the Spearman agreement between the two orderings, and the
agreement of each measure's ordering across the three orness values.

Usage: python scripts/orness_decision.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loaders import load_assistments  # noqa: E402
from fuzzykg import fuzzy_leiden, fuzzy_modularity  # noqa: E402
from fuzzykg.construction import ConstructionConfig, build_fuzzy_graph  # noqa: E402
from fuzzykg.multislice import multislice_leiden, level_grid, qbar  # noqa: E402
from fuzzykg.tnorms import owa_weights_from_orness  # noqa: E402

ORNESS = (0.50, 0.65, 0.80)
SEEDS = range(5)


def price(G):
    alphas = level_grid(G)
    ms = max((multislice_leiden(G, alphas, seed=s) for s in SEEDS), key=lambda r: r.Qbar)
    f = max((fuzzy_leiden(G, 1.0, seed=s, n_iterations=0) for s in SEEDS), key=lambda r: r.Q_F)
    cuts = {"none": float(G.mu.min())}
    for q in (0.25, 0.50, 0.75):
        cuts[f"q{q:.2f}"] = float(np.quantile(G.mu, q))
    for a in (0.4, 0.5, 0.6):
        cuts[f"a{a:.1f}"] = a
    out = {}
    for c, a in cuts.items():
        Ga = G.alpha_cut(a)
        if Ga.m == 0:
            out[c] = None; continue
        b = max((fuzzy_leiden(Ga, 1.0, seed=s, n_iterations=0) for s in SEEDS), key=lambda r: r.Q_F)
        out[c] = {"alpha": a, "retained_edges": float(Ga.m / G.m), "index": float(ms.Qbar - qbar(G, b.membership, alphas)),
                  "qf_drop": float(f.Q_F - fuzzy_modularity(G, b.membership, 1.0))}
    return out


def main():
    res = {}
    for rel in ("2009", "2015", "2017"):
        df = load_assistments(rel, "kc", None)
        res[rel] = {}
        for o in ORNESS:
            cfg = ConstructionConfig()
            if o != 0.65:
                cfg.owa_weights = tuple(owa_weights_from_orness(o, 3))
            G, _ = build_fuzzy_graph(df, cfg)
            p = price(G)
            cut = [c for c in p if c != "none" and p[c] is not None]
            r = {"owa_weights": list(cfg.owa_weights), "n": int(G.n), "m": int(G.m), "mean_mu": float(G.mu.mean()), "cuts": p,
                 "cheapest_index": min(cut, key=lambda c: p[c]["index"]), "cheapest_qf_drop": min(cut, key=lambda c: p[c]["qf_drop"]),
                 "rho_index_vs_drop": float(spearmanr([p[c]["index"] for c in cut], [p[c]["qf_drop"] for c in cut]).statistic)}
            res[rel][str(o)] = r
            print(rel, o, G.summary(), r["cheapest_index"], r["cheapest_qf_drop"], round(r["rho_index_vs_drop"], 2),
                  {c: (round(v["index"], 3), round(v["qf_drop"], 3)) for c, v in p.items() if v}, flush=True)
        for key in ("index", "qf_drop"):
            agree = []
            for o1, o2 in ((0.5, 0.65), (0.65, 0.8), (0.5, 0.8)):
                a, b = res[rel][str(o1)]["cuts"], res[rel][str(o2)]["cuts"]
                cc = [c for c in a if c != "none" and a[c] and b[c]]
                agree.append(float(spearmanr([a[c][key] for c in cc], [b[c][key] for c in cc]).statistic))
            res[rel][f"across_orness_rho_{key}"] = agree
    json.dump(res, open("results/orness_decision.json", "w"), indent=1)


if __name__ == "__main__":
    main()
