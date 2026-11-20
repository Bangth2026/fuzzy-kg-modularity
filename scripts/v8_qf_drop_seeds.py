#!/usr/bin/env python3
"""Version 4.5.0. Seed dispersion of the Q_F drop on the thirteen real graphs (Algorithm 2, step 3).

Step 3 of Algorithm 2 flags a cut whose Q_F drop exceeds the no-cut (support) value by more than
twice the larger of the two seed standard deviations.  That rule was read from Table 4, where it
is applied to the cost index.  This script applies it to the Q_F drop itself: fuzzykg.audit.cut_audit
over ten optimiser seeds on the support partition and on the quantile cuts q = 0.25, 0.50, 0.75 of
every graph, and compares the flags with those of the cost index (results/cost_decomposition.json,
same seeds, same rule).
Writes results/v8/qf_drop_seeds.json.
Usage: python scripts/v8_qf_drop_seeds.py [graph ...]
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.audit import cut_audit  # noqa: E402

OUT = "results/v8/qf_drop_seeds.json"
GRAPHS = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2009_item", "assist2015_kc", "assist2017_kc",
          "assist2017_item", "junyi_copractice", "assist2009_kc_transition", "assist2015_kc_transition",
          "assist2017_kc_transition", "assist2017_item_transition", "junyi_transition"]
CUTS = ("q0.25", "q0.50", "q0.75")


def index_flags(cd):
    """Flags of the cost index under the same rule, from the per-seed values of results/cost_decomposition.json."""
    b = cd["nocut"]
    return {c: bool(cd["cuts"][c]["index_best5"] - b["index_best5"] > 2 * max(cd["cuts"][c]["sd"], b["sd"])) for c in CUTS}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cdall = json.load(open("results/cost_decomposition.json"))
    for name in (sys.argv[1:] or GRAPHS):
        if name in res:
            continue
        t0 = time.perf_counter()
        df = pd.read_csv(f"data/graphs/{name}_edges.csv")
        G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
        cuts = {c: float(np.quantile(G.mu, float(c[1:]))) for c in CUTS}
        a = cut_audit(G, cuts, range(10))
        rows = a["cuts"]
        fi = index_flags(cdall[name])
        res[name] = {"support": {k: rows["support"][k] for k in ("qf_drop", "qf_drop_sd", "rank_drop", "rank_drop_sd")},
                     "cuts": {c: {k: rows[c].get(k) for k in ("alpha", "retained_edges", "mean_out_degree", "qf_drop", "qf_drop_sd",
                                                             "rank_drop", "rank_drop_sd", "flag_qf_drop", "flag_rank_drop")}
                              for c in CUTS if not rows[c].get("empty")},
                     "index_flags": fi, "time_s": time.perf_counter() - t0}
        json.dump(res, open(OUT, "w"), indent=1)
        print(name, {c: (round(res[name]["cuts"][c]["qf_drop"], 3), round(res[name]["cuts"][c]["qf_drop_sd"], 3),
                         res[name]["cuts"][c]["flag_qf_drop"], fi[c]) for c in res[name]["cuts"]},
              round(res[name]["support"]["qf_drop_sd"], 4), round(time.perf_counter() - t0), flush=True)
    for k, r in res.items():   # index flags as in Table 4 (best-of-five index, ten-seed SDs)
        if not k.startswith("_"):
            r["index_flags"] = index_flags(cdall[k])
    # summary: agreement of the Q_F-drop flags with the cost-index flags
    pairs = [(r["cuts"][c]["flag_qf_drop"], r["index_flags"][c]) for k, r in res.items() if not k.startswith("_")
             for c in r["cuts"]]
    sds = [r["cuts"][c]["qf_drop_sd"] for k, r in res.items() if not k.startswith("_") for c in r["cuts"]] + \
          [r["support"]["qf_drop_sd"] for k, r in res.items() if not k.startswith("_")]
    res["_summary"] = {"n_cells": len(pairs), "agree": int(sum(a == b for a, b in pairs)),
                       "flag_qf_only": int(sum(a and not b for a, b in pairs)), "flag_index_only": int(sum(b and not a for a, b in pairs)),
                       "n_flag_qf": int(sum(a for a, _ in pairs)), "n_flag_index": int(sum(b for _, b in pairs)),
                       "qf_drop_sd_range": [float(min(sds)), float(max(sds))]}
    json.dump(res, open(OUT, "w"), indent=1)
    print(res["_summary"])


if __name__ == "__main__":
    main()
