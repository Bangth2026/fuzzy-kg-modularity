#!/usr/bin/env python3
"""Release 5.2.0. New numbers of the revised manuscript -> results/v17/summary.json.

Divergence check by density class: the threshold of the plain divergence D (Eq. 7) and of the
null-referenced divergence D_null is chosen on the calibration worlds (seed 0) of the injected worlds,
separately for dense (co-practice) and sparse (transition) graphs, by the balanced accuracy with which
the check detects a partition value above 0.02; evaluated on the test worlds (seed 1). Then the audit
of every cut against the loss relative to Pi_F, without a check and with every cut of a firing graph
flagged, per construction and eta, with the class counts (BA reported only when both classes occur).
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.metrics import balanced_accuracy_score

GRID_D = (0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
GRID_N = (-0.02, 0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2)


def load(prefix):
    W = {}
    for p in (f"{prefix}.json", f"{prefix}_part0.json", f"{prefix}_part1.json"):
        if os.path.exists(p):
            d = json.load(open(p)); W.update(d.get("worlds", {}))
            if "real" in d:
                W["__real__"] = d["real"]
    return W


def metrics(rows, dec, lab):
    r = [x for x in rows if x[dec] != "undecided"]
    y = np.array([x[lab] for x in r]); p = np.array([x[dec] == "flagged" for x in r])
    out = {"n": len(r), "losses": int(y.sum()), "no_loss": int((~y).sum()), "cut_decisions": int((~p).sum()), "cut_right": int((~p & ~y).sum())}
    sens = float(p[y].mean()) if y.any() else None; spec = float((~p[~y]).mean()) if (~y).any() else None
    out.update({"sens": sens, "spec": spec, "ba": (sens + spec) / 2 if sens is not None and spec is not None else None})
    return out


def main():
    I = load("results/v16/injected"); DN = load("results/v17/divergence_null")
    real = DN.pop("__real__")
    keys = sorted(k for k in I if k in DN)
    out = {"worlds": len(keys), "classes": {}}
    for con, cls in (("co-practice", "dense"), ("transition", "sparse")):
        G = [(k, e, DN[k][con][e]) for k in keys for e in DN[k][con]]
        cal = [g for g in G if g[0].endswith("seed0")]; te = [g for g in G if g[0].endswith("seed1")]
        y = lambda gs: [g[2]["V_partition"] > 0.02 for g in gs]
        rules = {}
        for t in GRID_D:
            rules[f"D>{t:g}"] = lambda d, t=t: d["D"] > t
        for t in GRID_N:
            rules[f"Dnull>{t:g}"] = lambda d, t=t: d["D_null"] > t
        rules["null_min"] = lambda d: d["fires_null"]
        ba_cal = {n: float(balanced_accuracy_score(y(cal), [f(g[2]) for g in cal])) for n, f in rules.items()}
        bestD = max((n for n in rules if n.startswith("D>")), key=lambda n: ba_cal[n])
        bestN = max((n for n in rules if n.startswith("Dnull") or n == "null_min"), key=lambda n: ba_cal[n])
        c = {"calibration_ba": ba_cal, "selected_plain": bestD, "selected_null": bestN}
        for tag, name in (("plain", bestD), ("null", bestN)):
            f = rules[name]; yt = np.array(y(te)); pt = np.array([f(g[2]) for g in te])
            c[f"test_{tag}"] = {"sens": float(pt[yt].mean()) if yt.any() else None, "false": float(pt[~yt].mean()) if (~yt).any() else None,
                                "ba": float(balanced_accuracy_score(yt, pt)), "n_pos": int(yt.sum()), "n": int(len(yt))}
        # audit with the checks, per eta, all worlds and test worlds
        c["audit"] = {}
        for e in ("0", "0.5", "1", "2"):
            rows = []
            for k in keys:
                r = I[k][con][e]; d = DN[k][con][e]
                for cc in r["cuts"].values():
                    rob = cc["decision_mu"] if cc["kappa_star"] >= 2 else "undecided"
                    rows.append({"seed": int(k[-1]), "robust": rob, "loss": cc["loss_vs_support"], "lossF": cc["loss_vs_F"],
                                 "plain": "flagged" if rules[bestD](d) else rob, "null": "flagged" if rules[bestN](d) else rob})
            te_rows = [x for x in rows if x["seed"] == 1]
            c["audit"][e] = {"no_check_vs_F": metrics(rows, "robust", "lossF"), "no_check_vs_support": metrics(rows, "robust", "loss"),
                             "plain_vs_F": metrics(rows, "plain", "lossF"), "null_vs_F": metrics(rows, "null", "lossF"),
                             "test_no_check_vs_F": metrics(te_rows, "robust", "lossF"), "test_plain_vs_F": metrics(te_rows, "plain", "lossF"),
                             "test_null_vs_F": metrics(te_rows, "null", "lossF")}
        out["classes"][cls] = c
    # real graphs
    S15 = json.load(open("results/v15/simplex_real.json"))["graphs"]
    R16 = json.load(open("results/v16/real.json"))["graphs"]
    out["real"] = {}
    for n, d in real.items():
        cls = "dense" if S15[n]["group"] == "co-practice" else "sparse"
        c = out["classes"][cls]
        thrD = float(c["selected_plain"].split(">")[1]); nm = c["selected_null"]
        firesN = d["fires_null"] if nm == "null_min" else d["D_null"] > float(nm.split(">")[1])
        out["real"][n] = {**d, "class": cls, "fires_plain_class": d["D"] > thrD, "fires_null_class": bool(firesN)}
    # external check with the class-specific checks
    V15 = json.load(open("results/v15/value_paths.json"))["partition"]
    ext = []
    for n, v in R16.items():
        if "grouping" not in v or S15[n].get("grouping", {}).get("ari", {}).get("support", [0])[0] < 0.1:
            continue
        for cut, r in v["cuts"].items():
            rob = r["decision_mu"] if r["kappa_star"] >= 2 else "undecided"; g = v["grouping"][cut]
            ext.append({"graph": n, "cut": cut, "robust": rob, "loss": g["loss_vs_support"], "lossF": g["loss_vs_F"],
                        "plain": "flagged" if out["real"][n]["fires_plain_class"] else rob,
                        "null": "flagged" if out["real"][n]["fires_null_class"] else rob,
                        "dense_only": "flagged" if (out["real"][n]["class"] == "dense" and out["real"][n]["fires_plain_class"]) else rob})
    out["external"] = {k: metrics(ext, k, "lossF") for k in ("robust", "plain", "null", "dense_only")}
    out["external_vs_support"] = {k: metrics(ext, k, "loss") for k in ("robust", "plain", "null", "dense_only")}
    out["fires_real_dense_only"] = [n for n, r in out["real"].items() if r["class"] == "dense" and r["fires_plain_class"]]
    out["fires_real"] = {"plain": [n for n, r in out["real"].items() if r["fires_plain_class"]], "null": [n for n, r in out["real"].items() if r["fires_null_class"]]}
    out["V_partition_real"] = {n: V15[n]["V_total"] for n in V15}
    for name in ("misc", "refined", "junyi_fraction"):
        p = f"results/v17/{name}.json"
        out[name] = json.load(open(p)) if os.path.exists(p) else None
    json.dump(out, open("results/v17/summary.json", "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    o = main()
    for cls, c in o["classes"].items():
        print(cls, c["selected_plain"], c["selected_null"], "test plain", c["test_plain"], "test null", c["test_null"])
        for e, a in c["audit"].items():
            f = lambda m: (round(m["ba"], 2) if m["ba"] is not None else None, f"{m['cut_right']}/{m['cut_decisions']}")
            print("  eta", e, "no", f(a["no_check_vs_F"]), "plain", f(a["plain_vs_F"]), "null", f(a["null_vs_F"]))
    print("fires", o["fires_real"])
    for k, m in o["external"].items(): print("ext", k, m)
