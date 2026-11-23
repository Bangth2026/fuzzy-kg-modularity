#!/usr/bin/env python3
"""Release 5.1.0. Collects every new number of the revised manuscript into results/v16/summary.json.
Numbers carried over unchanged from release 5.0.0 stay in results/v15/summary.json.

Rules (as in release 5.0.0):
  loss of a cut     mean ARI(reference) - mean ARI(cut) >= 0.02 and its 95% normal interval over
                    seeds excludes 0; reference = the support partition (loss against support) or
                    the graded partition Pi_F (loss against Pi_F, new);
  robust decision   the decision on mu_E where kappa* >= 2, otherwise undecided;
  gate (new)        the graded-partition gain on mu_E exceeds its value on every row permutation.
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef

CUTS = ("q0.25", "q0.50", "q0.75")
DELTAS = ("0.0025", "0.005", "0.0075", "0.01", "0.015", "0.02", "0.03", "0.05")
V15 = json.load(open("results/v15/summary.json"))
DSEL = V15["delta"]
OUT = "results/v16/summary.json"


def loss_flag(m_a, sd_a, m_b, sd_b, n):
    d = m_a - m_b; se = np.sqrt(sd_a ** 2 / n + sd_b ** 2 / n)
    return bool(d >= 0.02 and d - 1.96 * se > 0), float(d)


def metrics(rows, dec="mu", lab="loss"):
    """Sensitivity (flag | loss), specificity (cut | no loss), balanced accuracy, MCC and accuracy of
    the decided rows."""
    r = [x for x in rows if x[dec] != "undecided"]
    if not r:
        return None
    y = np.array([x[lab] for x in r]); p = np.array([x[dec] == "flagged" for x in r])
    sens = float(p[y].mean()) if y.any() else float("nan"); spec = float((~p[~y]).mean()) if (~y).any() else float("nan")
    return {"n": len(r), "losses": int(y.sum()), "sens": sens, "spec": spec, "ba": float(np.nanmean([sens, spec])),
            "mcc": float(matthews_corrcoef(y, p)) if len(set(y)) > 1 and len(set(p)) > 1 else 0.0,
            "acc": float((p == y).mean()), "always_flag_acc": float(y.mean()),
            "cut_decisions": int((~p).sum()), "cut_correct": int((~p & ~y).sum()),
            "false_cut_share_of_losses": float((~p & y).sum() / max(1, y.sum()))}


def ss_rows(R, con, cls, dkey=None):
    rows = []
    for key, w in R.items():
        g = w[con]
        for c in CUTS:
            if dkey:   # release 5.0.0 file
                bd = g["cuts"][c]["by_delta"][dkey]; dec, ks = bd["decision"], bd["kappa_star"]
                a_s, a_c, a_f = g["ari"]["support"], g["ari"][c], g["ari"]["F"]
                loss, lv = loss_flag(a_s[0], a_s[1], a_c[0], a_c[1], 10); lossF, lF = loss_flag(a_f[0], a_f[1], a_c[0], a_c[1], 10)
                ex = g["cuts"][c]["excess_mu"]; ef = None; kr = None
            else:
                cc = g["cuts"][c]; dec, ks = cc["decision_mu"], cc["kappa_star"]
                loss, lv, lossF, lF = cc["loss_vs_support"], cc["loss_vs_support_value"], cc["loss_vs_F"], cc["loss_vs_F_value"]
                ex = cc["margin_mu"]; ef = cc["E_F"]; kr = cc["kappa_star_refined"]
            rows.append({"world": key, "seed": int(key[-1]), "cut": c, "mu": dec, "kappa": ks,
                         "robust": dec if ks >= 2 else "undecided", "loss": loss, "loss_value": lv, "lossF": lossF,
                         "lossF_value": lF, "excess": ex, "E_F": ef, "kappa_refined": kr,
                         "gate": (g["gate"]["g_obs"] > g["gate"]["null_max"]) if "gate" in g else None,
                         "V_part": g["value_partition"]["V_total"] if "value_partition" in g else None})
    return rows


def cv_delta(R, con, folds=8, seed=0):
    keys = sorted(R); perm = np.random.default_rng(seed).permutation(len(keys)); fold = {keys[i]: j % folds for j, i in enumerate(perm)}
    def yp(keyset, d):
        yy, pp = [], []
        for k in keyset:
            g = R[k][con]
            for c in CUTS:
                a_s, a_c = g["ari"]["support"], g["ari"][c]
                yy.append(loss_flag(a_s[0], a_s[1], a_c[0], a_c[1], 10)[0]); pp.append(g["cuts"][c]["by_delta"][d]["decision"] == "flagged")
        return yy, pp
    chosen, Y, P = [], [], []
    for f in range(folds):
        tr = [k for k in keys if fold[k] != f]; te = [k for k in keys if fold[k] == f]
        ba = [balanced_accuracy_score(*yp(tr, d)) for d in DELTAS]
        d = DELTAS[int(np.argmax(ba))]; chosen.append(float(d))
        y, p = yp(te, d); Y += y; P += p
    return {"selected_by_fold": chosen, "oof_balanced_accuracy": float(balanced_accuracy_score(Y, P))}


def semisynthetic_v15():
    R = json.load(open("results/v15/semisynthetic.json"))["worlds"]
    out = {}
    for con, cls in (("co-practice", "dense"), ("transition", "sparse")):
        rows = ss_rows(R, con, cls, f"{DSEL[cls]:g}")
        firm = [r for r in rows if r["kappa"] >= 2]; frag = [r for r in rows if r["kappa"] < 2]
        te = [r for r in rows if r["seed"] == 1]
        dec = [r for r in rows if r["robust"] != "undecided"]
        s = {"all_mu": metrics(rows), "firm_mu": metrics(firm), "fragile_mu": metrics(frag),
             "robust": metrics(rows, "robust"), "test_robust": metrics(te, "robust"), "test_mu": metrics(te),
             "test_firm_mu": metrics([r for r in te if r["kappa"] >= 2]), "test_fragile_mu": metrics([r for r in te if r["kappa"] < 2]),
             "always_flag_all": float(np.mean([r["loss"] for r in rows])),
             "always_flag_on_robust_decided": float(np.mean([r["loss"] for r in dec])),
             "fragile_loss_share": float(np.mean([r["loss"] for r in frag])) if frag else None,
             "firm_decided_losses_passed": [int(sum(r["robust"] == "cut" and r["loss"] for r in dec)), int(sum(r["loss"] for r in dec))],
             # against the graded partition
             "lossF_share": float(np.mean([r["lossF"] for r in rows])),
             "robust_vs_F": metrics(rows, "robust", "lossF"), "mu_vs_F": metrics(rows, "mu", "lossF"),
             "cv_delta": cv_delta(R, con)}
        y = np.array([r["lossF"] for r in rows])
        s["auc_excess_lossF"] = float(roc_auc_score(y, [r["excess"] for r in rows])) if 0 < y.mean() < 1 else None
        out[con] = s
    return out


def semisynthetic_delta():
    out = {}
    for dl in ("0", "0.2", "0.4", "0.6"):
        p = f"results/v16/semisynthetic_delta_{dl}.json"
        if not os.path.exists(p):
            continue
        R = json.load(open(p))["worlds"]
        o = {"worlds": len(R)}
        for con, cls in (("co-practice", "dense"), ("transition", "sparse")):
            rows = ss_rows(R, con, cls)
            W = list(R.values())
            vp = np.array([w[con]["value_partition"]["V_total"] for w in W]); vr = np.array([w[con]["value_partition"]["V_resid"] for w in W])
            gate = np.array([w[con]["gate"]["g_obs"] > w[con]["gate"]["null_max"] for w in W])
            pos = vp > 0.02
            yF = np.array([r["lossF"] for r in rows])
            # combined rule: where the gate fires every cut is flagged, elsewhere the robust decision
            comb = [dict(r, combined="flagged" if r["gate"] else r["robust"]) for r in rows]
            s = {"V_partition_mean": float(vp.mean()), "V_resid_mean": float(vr.mean()), "share_V_positive": float(pos.mean()),
                 "support_ari_mean": float(np.mean([w[con]["ari"]["support"][0] for w in W])),
                 "F_ari_mean": float(np.mean([w[con]["ari"]["F"][0] for w in W])),
                 "loss_share": float(np.mean([r["loss"] for r in rows])), "lossF_share": float(yF.mean()),
                 "robust_vs_support": metrics(rows, "robust"), "robust_vs_F": metrics(rows, "robust", "lossF"),
                 "combined_vs_F": metrics(comb, "combined", "lossF"), "combined_vs_support": metrics(comb, "combined"),
                 "gate_fires": float(gate.mean()), "gate_sensitivity": float(gate[pos].mean()) if pos.any() else None,
                 "gate_false_alarm": float(gate[~pos].mean()) if (~pos).any() else None, "worlds_V_positive": int(pos.sum()),
                 "auc_E_F_lossF": float(roc_auc_score(yF, [r["E_F"] for r in rows])) if 0 < yF.mean() < 1 else None,
                 "auc_margin_lossF": float(roc_auc_score(yF, [r["excess"] for r in rows])) if 0 < yF.mean() < 1 else None,
                 "kappa_refined_same_robust": float(np.mean([(r["kappa"] >= 2) == (r["kappa_refined"] >= 2) for r in rows])),
                 "kappa_refined_le": float(np.mean([r["kappa_refined"] <= r["kappa"] * 1.0001 for r in rows]))}
            o[con] = s
        out[dl] = o
    return out


def real(th=None, Dreal=None):
    p = "results/v16/real.json"
    if not os.path.exists(p):
        return None
    R = json.load(open(p))["graphs"]; S15 = json.load(open("results/v15/simplex_real.json"))["graphs"]
    out = {"graphs": {}, "external": {}}
    rows = []
    for n, v in R.items():
        g = {"gate": v["gate"], "cuts": {}}
        for c in CUTS:
            r = v["cuts"][c]
            g["cuts"][c] = {k: r[k] for k in ("decision_mu", "kappa_star", "decision_refined", "kappa_star_refined", "kappa_boot", "kappa_boot_share_ge2", "E_F")}
            rr = {"graph": n, "cut": c, "mu": r["decision_mu"], "kappa": r["kappa_star"], "robust": r["decision_mu"] if r["kappa_star"] >= 2 else "undecided",
                  "robust_refined": r["decision_refined"] if r["kappa_star_refined"] >= 2 else "undecided",
                  "boot_lo": r["kappa_boot"][1], "boot_hi": r["kappa_boot"][2]}
            if "grouping" in v and S15[n].get("grouping", {}).get("ari", {}).get("support", [0])[0] >= 0.1:
                gg = v["grouping"][c]
                rr.update({"loss": gg["loss_vs_support"], "lossF": gg["loss_vs_F"], "loss_value": gg["loss_vs_support_value"], "lossF_value": gg["loss_vs_F_value"], "gated": v["gate"]["fires"], "diverged": (Dreal[n]["gap"] > th) if Dreal else None})
            rows.append(rr)
        if "grouping" in v:
            g["grouping"] = v["grouping"]; g["value_seeds"] = v.get("value_seeds")
        out["graphs"][n] = g
    ext = [r for r in rows if "loss" in r]
    comb = [dict(r, combined="flagged" if r["gated"] else r["robust"]) for r in ext]
    out["external"] = {"cuts": len(ext), "losses_vs_support": int(sum(r["loss"] for r in ext)), "losses_vs_F": int(sum(r["lossF"] for r in ext)),
                       "robust_vs_support": metrics(ext, "robust"), "robust_vs_F": metrics(ext, "robust", "lossF"),
                       "mu_vs_support": metrics(ext, "mu"), "mu_vs_F": metrics(ext, "mu", "lossF"),
                       "combined_vs_F": metrics(comb, "combined", "lossF"), "combined_vs_support": metrics(comb, "combined"),
                       "rows": ext}
    if Dreal:
        cd = [dict(r, with_div="flagged" if r["diverged"] else r["robust"]) for r in ext]
        out["external"]["with_div_vs_F"] = metrics(cd, "with_div", "lossF"); out["external"]["with_div_vs_support"] = metrics(cd, "with_div")
        out["divergence_fires"] = [n for n in R if Dreal[n]["gap"] > th]
    out["gate_fires"] = [n for n, v in R.items() if v["gate"]["fires"]]
    out["refined_changes_robust"] = [(r["graph"], r["cut"], r["robust"], r["robust_refined"]) for r in rows if r["robust"] != r["robust_refined"]]
    out["kappa_ci_straddles_2"] = [(r["graph"], r["cut"], r["kappa"], r["boot_lo"], r["boot_hi"]) for r in rows if r["boot_lo"] < 2 <= r["boot_hi"]]
    out["n_cuts"] = len(rows)
    return out


def load_injected():
    W = {}
    for p in ("results/v16/injected.json", "results/v16/injected_part0.json", "results/v16/injected_part1.json"):
        if os.path.exists(p):
            W.update(json.load(open(p))["worlds"])
    return W


GAPS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)


def injected():
    """Worlds whose grades mark the communities by construction (eta > 0); the graded-partition check
    (row-null gate) and the divergence check against the partition value; the audit against Pi_F."""
    W = load_injected(); D = json.load(open("results/v16/divergence.json"))["worlds"] if os.path.exists("results/v16/divergence.json") else {}
    keys = sorted(k for k in W if k in D)
    out = {"worlds": len(keys)}
    # divergence threshold chosen on the calibration worlds (seed 0), all eta and both constructions pooled
    cal = [(D[k][c][e]["gap"], D[k][c][e]["V_partition"] > 0.02) for k in keys if k.endswith("seed0") for c in ("co-practice", "transition") for e in D[k][c]]
    ba = [balanced_accuracy_score([y for _, y in cal], [g > t for g, _ in cal]) for t in GAPS]
    tsel = GAPS[int(np.argmax(ba))]
    out["divergence_threshold"] = {"grid": list(GAPS), "calibration_ba": [float(x) for x in ba], "selected": tsel}
    for con in ("co-practice", "transition"):
        for e in ("0", "0.5", "1", "2"):
            rows = []; G = []
            for k in keys:
                r = W[k][con][e]; d = D[k][con][e]
                gate = r["gate"]["g_obs"] > r["gate"]["null_max"]; div = d["gap"] > tsel
                G.append({"V": r["V_partition"], "gate": gate, "div": div, "seed": int(k[-1]), "gap": d["gap"]})
                for c, cc in r["cuts"].items():
                    rob = cc["decision_mu"] if cc["kappa_star"] >= 2 else "undecided"
                    rows.append({"robust": rob, "loss": cc["loss_vs_support"], "lossF": cc["loss_vs_F"], "seed": int(k[-1]),
                                 "with_gate": "flagged" if gate else rob, "with_div": "flagged" if div else rob})
            V = np.array([g["V"] for g in G]); pos = V > 0.02
            def rate(key, mask):
                return float(np.mean([g[key] for g, m in zip(G, mask) if m])) if mask.any() else None
            te = [r for r in rows if r["seed"] == 1]
            out[f"{con}|{e}"] = {"V_mean": float(V.mean()), "share_V_pos": float(pos.mean()),
                                 "loss_share": float(np.mean([r["loss"] for r in rows])), "lossF_share": float(np.mean([r["lossF"] for r in rows])),
                                 "robust_vs_support": metrics(rows, "robust"), "robust_vs_F": metrics(rows, "robust", "lossF"),
                                 "with_gate_vs_F": metrics(rows, "with_gate", "lossF"), "with_div_vs_F": metrics(rows, "with_div", "lossF"),
                                 "test_robust_vs_F": metrics(te, "robust", "lossF"), "test_with_div_vs_F": metrics(te, "with_div", "lossF"),
                                 "gate_sens": rate("gate", pos), "gate_false": rate("gate", ~pos), "div_sens": rate("div", pos), "div_false": rate("div", ~pos),
                                 "gap_mean": float(np.mean([g["gap"] for g in G]))}
    # pooled over eta > 0, test worlds
    allg = [(D[k][c][e]["gap"] > tsel, D[k][c][e]["V_partition"] > 0.02) for k in keys if k.endswith("seed1") for c in ("co-practice", "transition") for e in D[k][c]]
    y = np.array([b for _, b in allg]); p = np.array([a for a, _ in allg])
    out["divergence_test"] = {"sens": float(p[y].mean()), "false": float(p[~y].mean()), "n_pos": int(y.sum()), "n": int(len(y))}
    gg = [(W[k][c][e]["gate"]["g_obs"] > W[k][c][e]["gate"]["null_max"], W[k][c][e]["V_partition"] > 0.02) for k in keys if k.endswith("seed1") for c in ("co-practice", "transition") for e in W[k][c]]
    y = np.array([b for _, b in gg]); p = np.array([a for a, _ in gg])
    out["gate_test"] = {"sens": float(p[y].mean()), "false": float(p[~y].mean()), "n_pos": int(y.sum()), "n": int(len(y))}
    return out


def main():
    inj = injected()
    Dreal = json.load(open("results/v16/divergence.json"))["real"] if os.path.exists("results/v16/divergence.json") else None
    res = {"semisynthetic_v15": semisynthetic_v15(), "semisynthetic_delta": semisynthetic_delta(), "real": real(inj["divergence_threshold"]["selected"], Dreal), "injected": inj}
    if os.path.exists("results/v16/divergence.json"):
        res["divergence_real"] = json.load(open("results/v16/divergence.json"))["real"]
    for name in ("junyi_pruning_value", "units", "learner_resampling"):
        p = f"results/v16/{name}.json"
        res[name] = json.load(open(p)) if os.path.exists(p) else None
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    r = main()
    import pprint
    for con, s in r["semisynthetic_v15"].items():
        print(con)
        for k in ("all_mu", "firm_mu", "fragile_mu", "robust", "test_robust", "robust_vs_F"):
            print("  ", k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in s[k].items()})
        print("  ", {k: s[k] for k in ("always_flag_all", "always_flag_on_robust_decided", "fragile_loss_share", "firm_decided_losses_passed", "lossF_share", "cv_delta", "auc_excess_lossF")})
