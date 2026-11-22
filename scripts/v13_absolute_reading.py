#!/usr/bin/env python3
"""Version 4.10.0. Two checks behind the eighth review, both read from saved results.

1. Absolute reading where the relative rule loses resolution. Step 3 compares the drop of a cut
   with the no-cut (support) drop. Where that baseline exceeds 0.05 (only under x^3: 2009, 2015 and
   2017 KC transition, Junyi transition), step 3 also reads the absolute drop: a cut is flagged
   when its own mean drop exceeds 2 x its seed SD and the minimum excess 0.01, and a cut counts as
   flagged when either reading flags it. This recounts the sparse lower-quartile flags under x^3
   and re-runs the Junyi topic check (a rescaling is rejected when it does not flag the Junyi
   lower-quartile cut, which loses topic agreement). Source: results/v11/flag_rules.json.

2. Permutation reference of step 4. A grouping can show that an undecided cut loses nothing only
   if the graded partition matches it above partitions under permuted memberships. Compared like
   for like: ARI of the best of three seeds (by Q_F) on G_F against the same quantity on 20
   row-wise and 20 global permutations. Source: results/junyi_topic_partition_check.json.

Writes results/v13/absolute_reading.json.
"""
from __future__ import annotations
import json, os
import numpy as np

WARN, EPS = 0.05, 0.01
RESCALINGS = ("identity", "cube", "cuberoot", "one_minus_cube", "rank")


def main():
    G = json.load(open("results/v11/flag_rules.json"))["graphs"]
    sparse = [g for g in G if G[g]["group"] != "co-practice"]
    out = {"rule": __doc__.split("\n\n")[1].strip(), "absolute_cells": [], "sparse_lower_quartile_flags": {}}
    flags = {}
    for a in RESCALINGS:
        n = 0
        for g in G:
            au = G[g]["audits"][a]; b = au["drops"]["support"]["mean"]
            for c in ("q0.25", "q0.50", "q0.75"):
                rel = au["rule"][c]["flag"]; f = rel
                if b > WARN:
                    d = au["drops"][c]; ab = bool(d["mean"] > max(2 * d["sd"], EPS)); f = rel or ab
                    out["absolute_cells"].append({"graph": g, "audit": a, "cut": c, "no_cut_drop": b, "drop": d["mean"],
                                                  "sd": d["sd"], "flag_relative": rel, "flag_absolute": ab, "flag": f})
                flags[(g, a, c)] = f
            n += g in sparse and flags[(g, a, "q0.25")]
        out["sparse_lower_quartile_flags"][a] = int(n)
    out["sparse_n"] = len(sparse)
    keep = [a for a in RESCALINGS[:4] if flags[("junyi_transition", a, "q0.25")]]
    out["not_rejected_by_junyi"] = keep
    out["rejected_by_junyi"] = [a for a in RESCALINGS if a not in keep]
    v = [out["sparse_lower_quartile_flags"][a] for a in keep]
    out["sparse_lower_quartile_range_not_rejected"] = [min(v), max(v)]
    # co-practice decisions are unchanged: no co-practice graph has a no-cut drop above 0.05
    out["co_practice_warnings"] = [x for x in out["absolute_cells"] if G[x["graph"]]["group"] == "co-practice"]

    T = json.load(open("results/junyi_topic_partition_check.json"))
    ref = {}
    for g, r in T.items():
        o = float(r["observed_best3"]); e = {"ari_best3": o, "ari_seed_mean": float(np.mean(r["observed"]))}
        for k in ("row_perm_best3", "global_perm_best3"):
            p = np.asarray(r[k])
            e[k] = {"mean": float(p.mean()), "p95": float(np.percentile(p, 95)), "n_at_or_above": int((p >= o).sum()), "draws": int(p.size)}
        e["informative"] = bool(o > e["row_perm_best3"]["p95"] and o > e["global_perm_best3"]["p95"])
        ref[g] = e
    out["step4_reference"] = ref
    os.makedirs("results/v13", exist_ok=True)
    json.dump(out, open("results/v13/absolute_reading.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("sparse_lower_quartile_flags", "not_rejected_by_junyi", "rejected_by_junyi",
                                          "sparse_lower_quartile_range_not_rejected")}, indent=1))
    for x in out["absolute_cells"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()})
    print(json.dumps(ref, indent=1))


if __name__ == "__main__":
    main()
