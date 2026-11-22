#!/usr/bin/env python3
"""Version 4.8.0. Flagging rules of Algorithm 2 (step 3) applied to results/v11/cut_audit_rescale_*.json.

Drops are paired by seed: drop_c[s] = Q(Pi_opt[s]) - Q(Pi_c[s]), Pi_opt = Pi_F (Q_F drop on phi(mu_E))
or Pi_rank (rank-coded drop).  A cut c is flagged under (k, eps) when
    mean(drop_c) - mean(drop_support) > max(k * max(sd_c, sd_support), eps).
Reported: flags of every graph under the rule of the manuscript (k = 2, eps = 0.01, seeds 0-19), for
the Q_F drop under each rescaling and for the rank-coded drop; counts by construction over the grid
k in {2, 3}, eps in {0, 0.005, 0.01, 0.02}; agreement of the flags of seeds 0-9 with seeds 10-19;
the drop with Pi_F fixed at the best of seeds 0-4 (results/v7 protocol); unrounded margins to both
thresholds; and, on the two Junyi graphs, ARI with the expert topics per seed with a bootstrap
interval for the difference between the support partition and each cut.
Writes results/v11/flag_rules.json.
"""
from __future__ import annotations
import glob, json, os
import numpy as np

GROUP = {"mooccubex_cs": "expert", "mooccubex_cs_full": "expert", "assist2009_kc": "co-practice", "assist2009_item": "co-practice",
         "assist2015_kc": "co-practice", "assist2017_kc": "co-practice", "assist2017_item": "co-practice",
         "junyi_copractice": "co-practice", "assist2009_kc_transition": "transition", "assist2015_kc_transition": "transition",
         "assist2017_kc_transition": "transition", "assist2017_item_transition": "transition", "junyi_transition": "transition"}
CUTS = ("q0.25", "q0.50", "q0.75")
AUDITS = ("identity", "cube", "cuberoot", "one_minus_cube", "rank")


def drops(r, audit, sl=slice(None), fixed=False):
    blk = r["rank"] if audit == "rank" else r["phi"][audit]
    q = np.array(blk["QF_opt"])
    ref = q[:5].max() if fixed else q[sl]
    return {c: ref - np.array(v)[sl] for c, v in blk["Q_cut"].items()}


def stats(d):
    return {c: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))} for c, v in d.items()}


def flags(s, k=2.0, eps=0.01):
    b = s["support"]; out = {}
    for c in CUTS:
        exc = s[c]["mean"] - b["mean"]; thr = k * max(s[c]["sd"], b["sd"])
        out[c] = {"excess": exc, "k_sd": thr, "flag": bool(exc > max(thr, eps)),
                  "margin_sd": exc - thr, "margin_eps": exc - eps}
    return out


def main():
    R = {os.path.basename(f)[len("cut_audit_rescale_"):-5]: json.load(open(f)) for f in sorted(glob.glob("results/v11/cut_audit_rescale_*.json"))}
    out = {"graphs": {}, "grid": {}, "halves": {}, "ari": {}}
    for g, r in R.items():
        e = {"group": GROUP[g], "seeds": r["seeds"], "mean_out_degree": r["mean_out_degree"], "audits": {}}
        for a in AUDITS:
            s = stats(drops(r, a)); e["audits"][a] = {"drops": s, "rule": flags(s)}
        s7 = stats(drops(r, "identity", slice(0, 10), fixed=True)); e["v7_protocol"] = {"drops": s7, "rule": flags(s7)}
        s10 = stats(drops(r, "identity", slice(0, 10))); e["first10"] = {"drops": s10, "rule": flags(s10)}
        out["graphs"][g] = e
    for a in AUDITS:
        for k in (2.0, 3.0):
            for eps in (0.0, 0.005, 0.01, 0.02):
                cnt = {}
                for g, r in R.items():
                    f = flags(stats(drops(r, a)), k, eps)
                    for c in CUTS:
                        key = f"{GROUP[g]}|{c}"; cnt.setdefault(key, [0, 0]); cnt[key][0] += f[c]["flag"]; cnt[key][1] += 1
                out["grid"][f"{a}|k={k:g}|eps={eps:g}"] = cnt
    for a in ("identity", "rank"):
        agree = tot = 0; diff = []
        for g, r in R.items():
            f1 = flags(stats(drops(r, a, slice(0, 10)))); f2 = flags(stats(drops(r, a, slice(10, 20))))
            for c in CUTS:
                tot += 1; agree += f1[c]["flag"] == f2[c]["flag"]
                if f1[c]["flag"] != f2[c]["flag"]:
                    diff.append(f"{g}:{c}")
        out["halves"][a] = {"agree": agree, "cells": tot, "differ": diff}
    rng = np.random.default_rng(2026)
    for g, r in R.items():
        if not r.get("ari"):
            continue
        A = {k: np.array(v) for k, v in r["ari"].items()}
        res = {k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))} for k, v in A.items()}
        for c in ("PiF",) + CUTS:
            d = A["support"].mean() - A[c].mean()
            bt = [rng.choice(A["support"], A["support"].size).mean() - rng.choice(A[c], A[c].size).mean() for _ in range(10000)]
            res[f"support_minus_{c}"] = {"diff": float(d), "ci95": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]}
        out["ari"][g] = res
    os.makedirs("results/v11", exist_ok=True)
    json.dump(out, open("results/v11/flag_rules.json", "w"), indent=1)


if __name__ == "__main__":
    main()
