#!/usr/bin/env python3
"""Release 5.3.0. Collects the new numbers of the third revision (Sections 1, 8.1, 9 and Table 4 of
the manuscript; Supplementary Tables S4g, S4h, S4i, S9f) into results/v18/summary.json."""
import json
import numpy as np

reach = json.load(open("results/v18/reach.json")); part = json.load(open("results/v18/partition.json")); cf = json.load(open("results/v18/crossfit.json"))
out = {"table4": {t: {"reached": v["coverage"], "auc_reach": v["auc_reach"],
                      "rho_hop_all": v["scores"]["hop_baseline"]["rho_all"], "rho_product_all": v["scores"]["product"]["rho_all"],
                      "rho_hop_reached": v["scores"]["hop_baseline"]["rho_reached"], "rho_product_reached": v["scores"]["product"]["rho_reached"],
                      "product_minus_hop_reached": [v["product_minus_hop_reached"], *v["ci"]["product_minus_hop_reached"]]}
                  for t, v in reach["thresholds"].items()},
       "reach_change_0.2_to_0.05": reach["change_0.2_to_0.05"], "pairs_reached_at_0.2": reach["pairs_reached_at_0.2"],
       "pairs_first_reached_at_0.05": reach["pairs_first_reached_at_0.05"],
       "lukasiewicz_reached_0.05": reach["thresholds"]["0.05"]["lukasiewicz_reached"],
       "crossfit": {k: cf[k] for k in ("both_chosen", "crisp_chosen_product_unpruned", "choices")},
       "unsuggested_common_vertices": {k: v for k, v in part["unsuggested"].items()}, "box_seeds": {}}
for n, r in part["box_seeds"].items():
    V = np.array([r["per_vertex"][str(k)]["V"] for k in range(r["n_vertices"])]); m = np.median(V, 1)
    out["box_seeds"][n] = {"median_over_triples": {"min": float(m.min()), "median": float(np.median(m)), "max": float(m.max()),
                                                   "n_negative": int((m < 0).sum()), "n_below_0.02": int((m < 0.02).sum())},
                           "single_triple_negative": {"n": int((V < 0).sum()), "share": float((V < 0).mean()), "of": int(V.size)},
                           "V_at_mu_triples": r["V_at_mu_triples"], "V_resid_median_over_triples_range": r["V_resid_median_over_triples_range"]}
json.dump(out, open("results/v18/summary.json", "w"), indent=1)
print(json.dumps(out["box_seeds"], indent=1))
