#!/usr/bin/env python3
"""Summary of the rank-coded Q_F drop (Section 8.1).

Reads results/rank_drop_{shift,lfr,rescale}.json and results/rescaling_validity.json.
On the 330 pairs of quantile cuts of the rescaling comparison: pairs ordered
differently by the rank-coded drop and by the cost index, by the Q_F drop, and
the number of those disagreements whose shortfalls differ by at least 0.01 and
the audit that orders them as the shortfall does. On all three stages: pairs of
cuts whose order is changed by the covariance term of Lemma 2 (order by
-Qbar(Pi_cut) against order by -Q_F[N_A/L](Pi_cut)), and the size of that term.
Writes results/rank_drop_summary.json.
"""
import itertools, json
import numpy as np

out = {}
rv = {(r["shift"], r["seed"]): r for r in json.load(open("results/rescaling_validity.json"))["rows"]}
rs = json.load(open("results/rank_drop_rescale.json"))
cmp = {k: {"pairs": 0, "disagree": 0, "disagree_untied": 0, "rank_agrees_with_shortfall": 0} for k in ("index", "drop_identity")}
for r in rs:
    o0 = rv[(r["shift"], r["seed"])]["cuts"]; o = {c: o0[str(float(c))] for c in r["cuts"]}
    cs = {c: v for c, v in r["cuts"].items() if v is not None}
    for a, b in itertools.combinations(cs, 2):
        assert abs((1 - cs[a]["ari"]) - o[a]["shortfall"]) < 1e-9
        ds = o[a]["shortfall"] - o[b]["shortfall"]; dr = cs[a]["rank_drop"] - cs[b]["rank_drop"]
        for k in cmp:
            cmp[k]["pairs"] += 1
            if dr * (o[a][k] - o[b][k]) < 0:
                cmp[k]["disagree"] += 1
                if abs(ds) >= 0.01:
                    cmp[k]["disagree_untied"] += 1; cmp[k]["rank_agrees_with_shortfall"] += int(dr * ds > 0)
out["rescale_pairs"] = cmp
for st in ("shift", "lfr", "rescale"):
    rows = json.load(open(f"results/rank_drop_{st}.json")); n = ch = 0; cov = []
    for r in rows:
        cs = [v for v in r["cuts"].values() if v is not None]
        cov += [abs(v["cov_cut"]) for v in cs]
        for a, b in itertools.combinations(cs, 2):
            n += 1; ch += int((a["qbar_cut"] - b["qbar_cut"]) * (a["qf_rank_cut"] - b["qf_rank_cut"]) < 0)
    out[st] = {"graphs": len(rows), "pairs": n, "covariance_changes_order": ch, "abs_cov_median": float(np.median(cov)), "abs_cov_max": float(np.max(cov))}
print(json.dumps(out, indent=1))
json.dump(out, open("results/rank_drop_summary.json", "w"), indent=1)
