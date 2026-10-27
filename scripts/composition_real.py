"""Composition on the real graphs (Section 8.5): per-source Kendall tau-b against
max-min with bootstrap CIs, pooled Spearman, and the tie fraction of the max-min
ranking, on every real graph; hop-count means for the three standard operators
(Table 12).  Usage: python scripts/composition_real.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import kendalltau, spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph
from fuzzykg.tnorms import generator, generator_best_paths, widest_path

NAMES = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2009_item", "assist2015_kc", "assist2017_kc", "assist2017_item"]
FAMS = [("product", None), ("lukasiewicz", None), ("yager", 2.0), ("yager", 5.0), ("yager", 10.0),
        ("schweizer_sklar", -8.0), ("schweizer_sklar", -2.0), ("schweizer_sklar", 0.5)]

def load(name):
    df = pd.read_csv(f"data/graphs/{name}_edges.csv")
    return FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))

def agreement(G, n_sources=60, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    srcs = rng.choice(G.n, min(n_sources, G.n), replace=False)
    base = {int(s): widest_path(G, int(s))[0] for s in srcs}
    # tie fraction of the max-min ranking: share of target pairs with equal strength, per source
    ties = []
    for s, st in base.items():
        v = st[st > 0]
        if v.size >= 3:
            _, c = np.unique(np.round(v, 12), return_counts=True)
            ties.append(float((c * (c - 1) / 2).sum() / (v.size * (v.size - 1) / 2)))
    out = {"n_sources_used": 0, "maxmin_tie_fraction_mean": float(np.mean(ties)) if ties else None,
           "targets_per_source_mean": float(np.mean([int((st > 0).sum()) for st in base.values()]))}
    for name, p in FAMS:
        gen = generator(name, p); taus = []; pa = []; pb = []
        for s, st in base.items():
            reach = st > 0
            if reach.sum() < 3:
                continue
            bp = generator_best_paths(G, s, gen); v = np.zeros(G.n)
            for t, (sv, _) in bp.items():
                v[t] = sv
            taus.append(kendalltau(st[reach], v[reach]).statistic); pa.extend(st[reach]); pb.extend(v[reach])
        taus = np.array([t for t in taus if np.isfinite(t)])
        boots = [np.mean(rng.choice(taus, taus.size, replace=True)) for _ in range(n_boot)]
        out[gen.name] = {"tau_b_mean": float(taus.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         "pooled_spearman": float(spearmanr(pa, pb).statistic), "n_pairs": len(pa)}
        out["n_sources_used"] = int(taus.size)
    return out

def hop_table(G, max_hops=5):
    """Mean best-path strength by hop count of the max-min best path, all sources."""
    rows = {}
    gens = {"min": None, "product": generator("product"), "lukasiewicz": generator("lukasiewicz")}
    for s in range(G.n):
        st, pred = widest_path(G, s)
        hops = {}
        for t in np.where(st > 0)[0]:
            h, u = 0, int(t)
            while u != s and u >= 0:
                u = int(pred[u]); h += 1
            hops[int(t)] = h
        bp = {k: generator_best_paths(G, s, g) for k, g in gens.items() if g is not None}
        for t, h in hops.items():
            if h > max_hops:
                continue
            r = rows.setdefault(h, {"n": 0, "min": 0.0, "product": 0.0, "lukasiewicz": 0.0})
            r["n"] += 1; r["min"] += st[t]
            for k in ("product", "lukasiewicz"):
                r[k] += bp[k].get(t, (0.0, 0))[0]
    return {h: {"n": r["n"], **{k: r[k] / r["n"] for k in ("min", "product", "lukasiewicz")}} for h, r in sorted(rows.items())}

if __name__ == "__main__":
    res = {}
    for name in NAMES:
        G = load(name)
        res[name] = {"agreement": agreement(G)}
        if name == "mooccubex_cs":
            res[name]["hop_table"] = hop_table(G)
        a = res[name]["agreement"]
        print(name, "sources", a["n_sources_used"], "targets/src %.1f" % a["targets_per_source_mean"], "ties %.3f" % a["maxmin_tie_fraction_mean"],
              {k: (round(v["tau_b_mean"], 2), round(v["pooled_spearman"], 2)) for k, v in a.items() if isinstance(v, dict)}, flush=True)
    json.dump(res, open("results/composition_real.json", "w"), indent=1)
