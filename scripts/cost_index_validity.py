#!/usr/bin/env python3
"""Does the cost index measure the cost of a cut, and what does it add to free proxies?
(Section 8.1, Table 5.)

The recovery loss of a single cut is defined from the ground truth alone, so that it
shares no reference partition with the index:

    loss_truth(alpha_0) = ARI(Leiden on Q_F, truth) - ARI(Pi_alpha_0, truth).

On the planted sweep (results/levelset_shift.json) and the LFR-style benchmark
(results/levelset_lfr.json) the script reports Spearman correlations of loss_truth with
the cost index and with the free proxies (retained mass, mean out-degree of the level
set, share of singleton communities, crisp modularity of the level-set partition), all
cells and the cells with loss_truth > 0.02; partial Spearman correlations of the cost
index with loss_truth controlling for the mean out-degree of the level set; and the
accuracy with which each quantity flags the cuts whose loss_truth exceeds 0.10
(ROC AUC), against the rule "flag when the mean out-degree is below 2".

Usage: python scripts/cost_index_validity.py
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score

CUTS = ("q0.50", "q0.75", "a0.6")


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        A = np.c_[np.ones_like(b), b]
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def collect(path):
    rows = json.load(open(path))
    rec = []
    for r in rows:
        for c in CUTS:
            cc = r["cuts"].get(c)
            if cc is None:
                continue
            rec.append({"loss_truth": r["fuzzy"]["ari"] - cc["ari"], "loss_ref": r["multislice"]["ari"] - cc["ari"],
                        "cost": cc["cost_index"], "mass": cc["retained_mass"], "deg": cc["mean_out_degree_cut"],
                        "singleton": cc.get("singleton_share"), "crisp_q": cc.get("crisp_modularity"), "shift": r["shift"]})
    return rec


def analyse(rec, label):
    y = np.array([r["loss_truth"] for r in rec])
    feats = {"cost index": np.array([r["cost"] for r in rec]), "retained mass": -np.array([r["mass"] for r in rec]),
             "mean out-degree": -np.array([r["deg"] for r in rec])}
    if rec[0]["singleton"] is not None:
        feats["singleton share"] = np.array([r["singleton"] for r in rec])
        feats["crisp modularity"] = -np.array([r["crisp_q"] for r in rec])
    out = {"n_cells": int(y.size), "n_loss_gt_0.02": int((y > 0.02).sum()), "n_loss_gt_0.10": int((y > 0.10).sum()),
           "rho_cost_vs_loss_reference": float(spearmanr(feats["cost index"], [r["loss_ref"] for r in rec]).statistic),
           "features": {}}
    sel = y > 0.02
    flag = (y > 0.10).astype(int)
    for k, x in feats.items():
        out["features"][k] = {"rho_all": float(spearmanr(x, y).statistic),
                              "rho_loss_gt_0.02": float(spearmanr(x[sel], y[sel]).statistic) if sel.sum() > 3 else None,
                              "auc_flag_loss_gt_0.10": float(roc_auc_score(flag, x)) if 0 < flag.sum() < flag.size else None}
    deg = np.array([r["deg"] for r in rec])
    out["partial_rho_cost_given_degree"] = partial_spearman(feats["cost index"], y, deg)
    out["partial_rho_cost_given_degree_loss_gt_0.02"] = partial_spearman(feats["cost index"][sel], y[sel], deg[sel]) if sel.sum() > 5 else None
    rule = (deg < 2).astype(int)
    tp = int(((rule == 1) & (flag == 1)).sum()); fp = int(((rule == 1) & (flag == 0)).sum()); fn = int(((rule == 0) & (flag == 1)).sum())
    out["rule_deg_below_2"] = {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
    # the same rule for the cost index, with the threshold that matches the rule's recall
    c = feats["cost index"]
    thr = np.quantile(c, 1 - rule.mean()) if rule.mean() > 0 else np.inf
    rc = (c >= thr).astype(int)
    tp = int(((rc == 1) & (flag == 1)).sum()); fp = int(((rc == 1) & (flag == 0)).sum()); fn = int(((rc == 0) & (flag == 1)).sum())
    out["rule_cost_same_flag_rate"] = {"threshold": float(thr), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
    print(label, json.dumps(out, indent=1))
    return out


def main():
    res = {"planted_sweep": analyse(collect("results/levelset_shift.json"), "planted"),
           "lfr": analyse(collect("results/levelset_lfr.json"), "lfr")}
    json.dump(res, open("results/cost_index_validity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
