#!/usr/bin/env python3
"""Version 4.0.0: both constructions on five further public corpora (KC level), with the
default thresholds of Section 4 (support 20, tau_E = 0.2, horizon 10). Writes
data/graphs/<corpus>_{copractice,transition}_{edges,nodes,signals}.csv, the external grouping
data/graphs/<corpus>_groups.json, and results/v4/new_corpora_stats.json.
Usage: python scripts/v4_build_new.py assist2012 algebra2005 bridge2006 eedi ednet"""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loaders import cached  # noqa: E402
from fuzzykg.construction import ConstructionConfig, TransitionConfig, build_fuzzy_graph, build_transition_graph, preprocess  # noqa: E402
out = "results/v4/new_corpora_stats.json"; os.makedirs("results/v4", exist_ok=True)
stats = json.load(open(out)) if os.path.exists(out) else {}
for name in sys.argv[1:]:
    t0 = time.perf_counter()
    df, grp = cached(name)
    d = preprocess(df, ConstructionConfig())
    s = {"interactions": int(len(d)), "learners": int(d.user_id.nunique()), "kcs": int(d.kc.nunique()),
         "correct_rate": float(d.correct.mean()), "load_s": time.perf_counter() - t0}
    if grp is not None:
        json.dump(grp, open(f"data/graphs/{name}_groups.json", "w"))
        s["groups"] = int(len(set(grp.values())))
    for c in ("copractice", "transition"):
        t1 = time.perf_counter()
        G, tab = (build_fuzzy_graph(d, ConstructionConfig(), preprocessed=True) if c == "copractice"
                  else build_transition_graph(d, TransitionConfig(), preprocessed=True))
        base = f"data/graphs/{name}_{c}"
        pd.DataFrame({"src": G.src, "dst": G.dst, "mu": G.mu}).to_csv(base + "_edges.csv", index=False)
        pd.DataFrame({"id": range(G.n), "label": G.labels}).to_csv(base + "_nodes.csv", index=False)
        tab.to_csv(base + "_signals.csv", index=False)
        s[c] = {"summary": G.summary(), "n": int(G.n), "m": int(G.m), "build_s": time.perf_counter() - t1}
        print(name, c, G.summary(), flush=True)
    stats[name] = s; json.dump(stats, open(out, "w"), indent=1)
    print(name, {k: v for k, v in s.items() if k not in ("copractice", "transition")}, flush=True)
