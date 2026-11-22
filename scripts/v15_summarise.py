#!/usr/bin/env python3
"""Release 5.0.0. Collects every number of the release 5.0.0 manuscript from results/v15/*.json
into results/v15/summary.json (no value is computed elsewhere or typed by hand).

Rules used throughout:
  true loss of a cut   mean ARI(support) - mean ARI(cut) >= 0.02 and its 95% interval (normal,
                       over optimiser seeds) excludes 0;
  robust decision      the decision on mu_E where kappa* >= KAPPA0 = 2, otherwise undecided;
  informative grouping (real graphs) the support partition matches it at ARI >= 0.1.
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

KAPPA0 = 2.0
CUTS = ("q0.25", "q0.50", "q0.75")
OUT = "results/v15/summary.json"


def loss_flag(m_sup, sd_sup, m_cut, sd_cut, n):
    d = m_sup - m_cut; se = np.sqrt(sd_sup ** 2 / n + sd_cut ** 2 / n)
    return bool(d >= 0.02 and d - 1.96 * se > 0), float(d)


def robust(dec, ks):
    return dec if ks >= KAPPA0 else "undecided"


def confusion(rows, key):
    c = {}
    for r in rows:
        k = f"{r[key]}|{'loss' if r['loss'] else 'none'}"; c[k] = c.get(k, 0) + 1
    dec = [r for r in rows if r[key] != "undecided"]
    acc = float(np.mean([(r[key] == "flagged") == r["loss"] for r in dec])) if dec else float("nan")
    fc = sum(1 for r in rows if r[key] == "cut" and r["loss"]); nl = sum(r["loss"] for r in rows)
    ff = sum(1 for r in rows if r[key] == "flagged" and not r["loss"]); nn = len(rows) - nl
    return {"cells": c, "decided": len(dec), "accuracy_decided": acc, "undecided": len(rows) - len(dec),
            "false_cut_share_of_losses": fc / max(1, nl), "false_flag_share_of_no_loss": ff / max(1, nn)}


DELTAS = ("0.0025", "0.005", "0.0075", "0.01", "0.015", "0.02", "0.03", "0.05")


def select_delta(R, con):
    """Minimum excess maximising the balanced accuracy of the decision on mu_E on the seed-0 worlds."""
    curve_cal, curve_te = [], []
    for d in DELTAS:
        for part, curve in ((0, curve_cal), (1, curve_te)):
            yy, pp = [], []
            for key, w in R.items():
                if int(key[-1]) != part:
                    continue
                g = w[con]
                for c in CUTS:
                    a_s, a_c = g["ari"]["support"], g["ari"][c]
                    yy.append(loss_flag(a_s[0], a_s[1], a_c[0], a_c[1], 10)[0]); pp.append(g["cuts"][c]["by_delta"][d]["decision"] == "flagged")
            curve.append(float(balanced_accuracy_score(yy, pp)))
    b = int(np.argmax(curve_cal))
    return float(DELTAS[b]), {"grid": [float(x) for x in DELTAS], "calibration": curve_cal, "test": curve_te}


def semisynthetic():
    R = json.load(open("results/v15/semisynthetic.json"))["worlds"]
    out = {"worlds": len(R), "graphs": 2 * len(R), "delta": {}}
    for con, cls in (("co-practice", "dense"), ("transition", "sparse")):
        out["delta"][cls], out["delta_curve_" + cls] = select_delta(R, con)
    allrows = []
    for con in ("co-practice", "transition"):
        rows = []
        for key, w in R.items():
            g = w[con]
            dsel = f'{out["delta"]["dense" if con == "co-practice" else "sparse"]:g}'
            for c in CUTS:
                r = dict(g["cuts"][c]); bd = r["by_delta"][dsel]
                r["decision_mu"], r["kappa_star"], r["margin_mu"] = bd["decision"], bd["kappa_star"], bd["margin"]
                a_s, a_c = g["ari"]["support"], g["ari"][c]
                loss, d = loss_flag(a_s[0], a_s[1], a_c[0], a_c[1], 10)
                old = r["old_decision"].split(" ")[0]
                rows.append({"world": key, "seed": int(key[-1]), "cut": c, "loss": loss, "loss_value": d, "excess": r["excess_mu"],
                             "margin": r["margin_mu"], "kappa": r["kappa_star"], "mu": r["decision_mu"],
                             "robust": robust(r["decision_mu"], r["kappa_star"]), "v411": old, "deg": r["mean_out_degree"],
                             "indicator": r["excess_at_indicator"], "smin": r["simplex"]["excess_min"], "smax": r["simplex"]["excess_max"]})
        y = np.array([r["loss"] for r in rows]); ex = np.array([r["excess"] for r in rows])
        s = {"cuts": len(rows), "loss_share": float(y.mean()),
             "auc_excess": float(roc_auc_score(y, ex)), "rho_excess_loss": float(spearmanr(ex, [r["loss_value"] for r in rows]).correlation),
             "auc_minus_outdegree": float(roc_auc_score(y, -np.array([r["deg"] for r in rows]))),
             "mu": confusion(rows, "mu"), "robust": confusion(rows, "robust"), "v411": confusion(rows, "v411"),
             "support_ari_mean": float(np.mean([w[con]["ari"]["support"][0] for w in R.values()])),
             "both_signs_over_family": int(sum(r["smin"] < 0 < r["smax"] for r in rows)),
             "indicator_nonpositive": int(sum(r["indicator"] <= 1e-12 for r in rows))}
        cls = "dense" if con == "co-practice" else "sparse"
        cv = out["delta_curve_" + cls]; b = cv["grid"].index(out["delta"][cls])
        s["min_excess_selected"] = out["delta"][cls]; s["test_ba_at_selected"] = cv["test"][b]
        s["test_ba_at_0.01"] = cv["test"][cv["grid"].index(0.01)]; s["min_excess_curve"] = cv
        te = [r for r in rows if r["seed"] == 1]
        s["test"] = {"robust": confusion(te, "robust"), "mu": confusion(te, "mu")}
        # robust decisions on fragile vs firm cuts
        firm = [r for r in rows if r["kappa"] >= KAPPA0]; frag = [r for r in rows if r["kappa"] < KAPPA0]
        s["accuracy_firm"] = float(np.mean([(r["mu"] == "flagged") == r["loss"] for r in firm])) if firm else None
        s["accuracy_fragile"] = float(np.mean([(r["mu"] == "flagged") == r["loss"] for r in frag])) if frag else None
        # paths
        us = [w[con]["connectivity"]["usable_share"] for w in R.values()]
        dmm = [w[con]["paths"]["maxmin"]["rho"] - w[con]["paths"]["direct"]["rho"] for w in R.values()]
        s["usable_share_range"] = [float(np.min(us)), float(np.median(us)), float(np.max(us))]
        s["rho_usable_vs_maxmin_minus_direct"] = float(spearmanr(us, dmm).correlation)
        s["rho_means"] = {k: float(np.mean([w[con]["paths"][k]["rho"] for w in R.values()])) for k in ("direct", "hop", "product", "maxmin", "lukasiewicz")}
        s["prec_means"] = {k: float(np.mean([w[con]["paths"][k]["prec"] for w in R.values()])) for k in ("direct", "hop", "product", "maxmin", "lam0.5")}
        lam = [w[con]["lam_star"]["median"] for w in R.values()]
        s["lam_star"] = {"share_0": float(np.mean(np.isclose(lam, 0))), "share_1": float(np.mean(np.isclose(lam, 1))),
                         "share_interior": float(np.mean((np.array(lam) > 0) & (np.array(lam) < 1))), "median": float(np.median(lam))}
        s["value"] = {lev: {k: float(np.mean([w[con]["value"][lev][k] for w in R.values()])) for k in ("V_total", "V_marg", "V_strength", "V_resid")}
                      for lev in ("edge", "path", "partition")}
        s["value_share_positive_total"] = {lev: float(np.mean([w[con]["value"][lev]["V_total"] > 0 for w in R.values()])) for lev in ("edge", "path", "partition")}
        if con == "co-practice":
            # a detectability rule: flag a cut whose level set has mean out-degree below D, D chosen on the calibration worlds
            Ds = [2, 4, 6, 8, 10, 12, 15, 20, 30]
            cal = [r for r in rows if r["seed"] == 0]; te = [r for r in rows if r["seed"] == 1]
            ba = lambda rs, D: float(balanced_accuracy_score([r["loss"] for r in rs], [r["deg"] < D for r in rs]))
            bD = Ds[int(np.argmax([ba(cal, D) for D in Ds]))]
            s["degree_rule"] = {"grid": Ds, "calibration": [ba(cal, D) for D in Ds], "selected": bD, "test_ba": ba(te, bD)}
        out[con] = s
        allrows += rows
    out["scatter"] = [{k: r[k] for k in ("excess", "loss_value", "loss", "kappa", "deg")} | {"con": "co-practice" if i < len(allrows) // 2 else "transition"} for i, r in enumerate(allrows)]
    out["paths_scatter"] = [{"con": con, "usable": w[con]["connectivity"]["usable_share"],
                             "d": w[con]["paths"]["maxmin"]["rho"] - w[con]["paths"]["direct"]["rho"]} for w in R.values() for con in ("co-practice", "transition")]
    return out


def real(delta):
    R = json.load(open("results/v15/simplex_real.json"))["graphs"]
    out = {"graphs": {}, "counts": {}}
    for n, v in R.items():
        d = f'{delta["sparse"] if v["group"] != "co-practice" else delta["dense"]:g}'
        for c in CUTS:
            r = v["gammas"]["1"][c]; bd = r["by_delta"][d]
            r["decision_mu"], r["kappa_star"], r["excess_mu"] = bd["decision"], bd["kappa_star"], r["excess_mu"]
        # gamma interval at the class delta
        gs = [0.25, 0.35, 0.5, 0.71, 1.0, 1.41, 2.0, 2.83, 4.0]; i1 = gs.index(1.0)
        for c in CUTS:
            dec = lambda gg: v["gammas"][f"{gg:g}"][c]["by_delta"][d]["decision"]
            lo = i1
            while lo > 0 and dec(gs[lo - 1]) == dec(1.0):
                lo -= 1
            hi = i1
            while hi + 1 < len(gs) and dec(gs[hi + 1]) == dec(1.0):
                hi += 1
            v["gamma_interval"][c] = [gs[lo], gs[hi]]
    agree, tot, diffs = 0, 0, []
    for n, v in R.items():
        g1 = v["gammas"]["1"]
        out["graphs"][n] = {"group": v["group"], "n": v["n"], "m": v["m"],
                            "cuts": {c: {"excess": g1[c]["excess_mu"], "decision": g1[c]["decision_mu"], "kappa": g1[c]["kappa_star"],
                                         "robust": robust(g1[c]["decision_mu"], g1[c]["kappa_star"]), "gamma_interval": v["gamma_interval"][c],
                                         "deg": g1[c]["mean_out_degree_cut"], "simplex": [g1[c]["simplex"]["excess_min"], g1[c]["simplex"]["excess_max"]],
                                         "indicator": g1[c]["excess_at_indicator"]} for c in CUTS}}
        if "v12_decisions" in v:
            for c in CUTS:
                tot += 1; rb = out["graphs"][n]["cuts"][c]["robust"]; old = v["v12_decisions"][c]
                agree += rb == old
                if rb != old:
                    diffs.append({"graph": n, "cut": c, "v411": old, "v5": rb, "kappa": out["graphs"][n]["cuts"][c]["kappa"]})
    out["agreement_with_v411_13graphs"] = {"agree": agree, "total": tot, "differences": diffs}
    for grp in ("co-practice", "transition", "expert"):
        for c in CUTS:
            ds = [g["cuts"][c]["robust"] for g in out["graphs"].values() if g["group"] == grp]
            out["counts"][f"{grp}|{c}"] = {d: ds.count(d) for d in ("cut", "flagged", "undecided")} | {"graphs": len(ds)}
    allc = [g["cuts"][c] for g in out["graphs"].values() for c in CUTS]
    out["family_both_signs"] = int(sum(x["simplex"][0] < 0 < x["simplex"][1] for x in allc)); out["family_cuts"] = len(allc)
    out["indicator_nonpositive"] = int(sum(x["indicator"] <= 1e-12 for x in allc))
    out["gamma_full_interval"] = int(sum(x["gamma_interval"] == [0.25, 4.0] for x in allc))
    out["gamma_full_interval_firm"] = int(sum(x["gamma_interval"] == [0.25, 4.0] for x in allc if x["kappa"] >= KAPPA0))
    out["firm_cuts"] = int(sum(x["kappa"] >= KAPPA0 for x in allc))
    out["kappa0_sensitivity"] = {f"{k0:g}": {d: int(sum((x["decision"] if x["kappa"] >= k0 else "undecided") == d for x in allc)) for d in ("cut", "flagged", "undecided")} for k0 in (1.5, 2.0, 4.0)}
    out["kappa_median"] = {}
    for grp in ("co-practice", "transition", "expert"):
        for c in CUTS:
            ks = [g["cuts"][c]["kappa"] for g in out["graphs"].values() if g["group"] == grp]
            out["kappa_median"][f"{grp}|{c}"] = float(np.median(ks))
    # external check
    rows = []
    for n, v in R.items():
        g = v.get("grouping")
        if not g:
            continue
        a = g["ari"]
        if a["support"][0] < 0.1:
            out.setdefault("grouping_excluded", []).append({"graph": n, "ari_support": a["support"][0]}); continue
        for c in CUTS:
            loss, d = loss_flag(a["support"][0], a["support"][1], a[c][0], a[c][1], 20)
            r = out["graphs"][n]["cuts"][c]
            rows.append({"graph": n, "cut": c, "loss": loss, "loss_value": d, "mu": r["decision"], "robust": r["robust"], "kappa": r["kappa"],
                         "v411": v.get("v12_decisions", {}).get(c)})
    for r in rows:
        r["deg"] = out["graphs"][r["graph"]]["cuts"][r["cut"]]["deg"]; r["dense"] = out["graphs"][r["graph"]]["group"] == "co-practice"
    out["external"] = {"rows": rows, "graphs": len({r["graph"] for r in rows}), "mu": confusion(rows, "mu"), "robust": confusion(rows, "robust")}
    sub = [r for r in rows if r["v411"] is not None]
    out["external"]["v411_subset"] = {"cuts": len(sub), "v411": confusion(sub, "v411"), "robust": confusion(sub, "robust")}
    return out


def bootstrap():
    R = json.load(open("results/v15/learner_bootstrap.json"))
    rows = [(k, c, v["summary"][c]) for k, v in R.items() for c in CUTS]
    firm = [s for _, _, s in rows if s["observed"]["kappa_star"] >= KAPPA0]
    frag = [s for _, _, s in rows if s["observed"]["kappa_star"] < KAPPA0]
    return {"graphs": len(R), "cuts": len(rows),
            "same_decision_mean": float(np.mean([s["share_same_decision"] for _, _, s in rows])),
            "same_decision_firm": float(np.mean([s["share_same_decision"] for s in firm])) if firm else None,
            "same_decision_fragile": float(np.mean([s["share_same_decision"] for s in frag])) if frag else None,
            "n_firm": len(firm), "n_fragile": len(frag),
            "same_robust_mean": float(np.mean([s["share_same_robust"] for _, _, s in rows])),
            "per_cut": {f"{k}|{c}": {"observed": s["observed"], "same": s["share_same_decision"], "same_robust": s["share_same_robust"],
                                      "kappa_q": s["kappa_star_quartiles"]} for k, c, s in rows}}


def main():
    ss = semisynthetic()
    rr = real(ss["delta"])
    D = ss["co-practice"]["degree_rule"]["selected"]; dr = [r for r in rr["external"]["rows"] if r["dense"]]
    rr["external"]["degree_rule_dense"] = {"D": D, "cuts": len(dr), "correct": int(sum((r["deg"] < D) == r["loss"] for r in dr)),
                                          "robust_correct": int(sum((r["robust"] == "flagged") == r["loss"] for r in dr if r["robust"] != "undecided")),
                                          "robust_decided": int(sum(r["robust"] != "undecided" for r in dr))}
    s = {"kappa0": KAPPA0, "delta": ss["delta"], "semisynthetic": ss, "real": rr, "bootstrap": bootstrap(),
         "value_paths": json.load(open("results/v15/value_paths.json"))}
    if os.path.exists("results/v15/crossdomain.json"):
        s["crossdomain"] = json.load(open("results/v15/crossdomain.json"))
    json.dump(s, open(OUT, "w"), indent=1, default=float)
    ss = s["semisynthetic"]
    for con in ("co-practice", "transition"):
        x = ss[con]
        print(con, {k: x[k] for k in ("loss_share", "auc_excess", "rho_excess_loss", "auc_minus_outdegree", "min_excess_selected", "test_ba_at_selected", "test_ba_at_0.01", "accuracy_firm", "accuracy_fragile", "both_signs_over_family", "indicator_nonpositive", "rho_usable_vs_maxmin_minus_direct")}, x["test"])
        for k in ("mu", "robust", "v411"):
            print("  ", k, x[k])
        print("   curve", x["min_excess_curve"])
        print("   lam", x["lam_star"], "value", {l: {k: round(v, 3) for k, v in d.items()} for l, d in x["value"].items()})
    r = s["real"]
    print("real agreement", r["agreement_with_v411_13graphs"]["agree"], r["agreement_with_v411_13graphs"]["total"], "both signs", r["family_both_signs"], r["family_cuts"], "ind<=0", r["indicator_nonpositive"], "gamma full", r["gamma_full_interval"], "firm", r["firm_cuts"], r["gamma_full_interval_firm"])
    for k, v in r["counts"].items():
        print("  ", k, v)
    print("external", r["external"]["graphs"], r["external"]["mu"], r["external"]["robust"], r["external"]["v411_subset"])
    print("kappa0", r["kappa0_sensitivity"], r["kappa_median"])
    print("bootstrap", {k: v for k, v in s["bootstrap"].items() if k != "per_cut"})


if __name__ == "__main__":
    main()
