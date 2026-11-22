#!/usr/bin/env python3
"""Version 4.9.0. The unit rule of step 3 of Algorithm 2 applied to results/v11/flag_rules.json.

Release 4.9.0 changes how step 3 decides a cut when the unit of mu_E is arbitrary. The flags
themselves are those of scripts/v11_flag_rules.py (k = 2, eps = 0.01, 20 seeds, both partitions
re-drawn per seed); only the reading changes:

  unit meaningful (a count share, or a released or calibrated score):
      the Q_F drop on mu_E decides; the rank-coded drop is reported beside it.
  unit arbitrary (a weighted combination whose weights the builder chose):
      both drops pass  -> cut;
      both drops flag  -> flagged;
      they disagree    -> undecided: keep the graded graph unless an independent grouping
                          (step 4) shows no loss, or fix a unit first.

The co-practice memberships of Equation (8) are arbitrary in unit; the transition memberships of
Equation (9) are the smaller of two count shares, so each membership equals one of the shares
(the minimum returns one of its arguments) and keeps the unit of a count ratio; the curated
MOOCCubeX confidences are the released scale. A no-cut drop above 0.05 is reported as a warning:
the rule is relative to that baseline and loses resolution when the baseline is large.

Also written: the counts on the sparse graphs under every rescaling that the Junyi topic check does
not reject, the unrounded margins of the cells nearest a threshold, and the per-graph decisions.
Writes results/v12/unit_rule.json.
"""
from __future__ import annotations
import json, os

UNIT = {"co-practice": "arbitrary", "transition": "count share", "expert": "released scale"}
CUTS = ("q0.25", "q0.50", "q0.75")
RESCALINGS = ("identity", "cube", "cuberoot", "one_minus_cube")
WARN = 0.05


def main():
    src = json.load(open("results/v11/flag_rules.json"))
    G = src["graphs"]
    out = {"rule": __doc__.split("\n\n")[1].strip(), "graphs": {}, "counts": {}, "warnings": [], "near_threshold": []}
    for g, r in G.items():
        grp = r["group"]; unit = UNIT[grp]
        mu = r["audits"]["identity"]; rk = r["audits"]["rank"]
        dec = {}
        for c in CUTS:
            fm = mu["rule"][c]["flag"]; fr = rk["rule"][c]["flag"]
            if unit == "arbitrary":
                d = "flagged" if fm and fr else ("cut" if not fm and not fr else "undecided")
            else:
                d = "flagged" if fm else "cut"
            dec[c] = {"flag_mu": fm, "flag_rank": fr, "decision": d,
                      "excess_mu": mu["rule"][c]["excess"], "excess_rank": rk["rule"][c]["excess"],
                      "margin_eps_mu": mu["rule"][c]["margin_eps"], "margin_sd_mu": mu["rule"][c]["margin_sd"],
                      "margin_eps_rank": rk["rule"][c]["margin_eps"], "margin_sd_rank": rk["rule"][c]["margin_sd"]}
            for a, x in (("mu_E", dec[c]["margin_eps_mu"]), ("rank", dec[c]["margin_eps_rank"])):
                m_sd = dec[c]["margin_sd_mu"] if a == "mu_E" else dec[c]["margin_sd_rank"]
                if abs(x) < 0.001 or abs(m_sd) < 0.003:
                    out["near_threshold"].append({"graph": g, "cut": c, "audit": a, "margin_to_min_excess": x, "margin_to_2sd": m_sd})
        for a in RESCALINGS + ("rank",):
            b = r["audits"][a]["drops"]["support"]["mean"]
            if b > WARN:
                out["warnings"].append({"graph": g, "audit": a, "no_cut_drop": b})
        out["graphs"][g] = {"group": grp, "unit": unit, "cuts": dec}
    # counts by construction under the decision rule
    for grp in ("co-practice", "transition", "expert"):
        gs = [g for g in G if G[g]["group"] == grp]
        out["counts"][grp] = {c: {k: sum(out["graphs"][g]["cuts"][c]["decision"] == k for g in gs) for k in ("cut", "flagged", "undecided")}
                              | {"n": len(gs)} for c in CUTS}
    # sparse graphs: lower-quartile flags under each rescaling
    sparse = [g for g in G if G[g]["group"] != "co-practice"]
    out["sparse_lower_quartile_flags"] = {a: sum(G[g]["audits"][a]["rule"]["q0.25"]["flag"] for g in sparse) for a in RESCALINGS + ("rank",)}
    out["sparse_n"] = len(sparse)
    # rescalings the Junyi topic check does not reject: those that flag the Junyi lower-quartile cut
    keep = [a for a in RESCALINGS if G["junyi_transition"]["audits"][a]["rule"]["q0.25"]["flag"]]
    out["not_rejected_by_junyi"] = keep
    v = [out["sparse_lower_quartile_flags"][a] for a in keep]
    out["sparse_lower_quartile_range_not_rejected"] = [min(v), max(v)]
    os.makedirs("results/v12", exist_ok=True)
    json.dump(out, open("results/v12/unit_rule.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("counts", "sparse_lower_quartile_flags", "not_rejected_by_junyi",
                                          "sparse_lower_quartile_range_not_rejected", "warnings")}, indent=1))
    for g, x in out["graphs"].items():
        print(g, x["unit"], {c: x["cuts"][c]["decision"] for c in CUTS})
    for x in out["near_threshold"]:
        print("near", x)


if __name__ == "__main__":
    main()
