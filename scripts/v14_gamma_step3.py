#!/usr/bin/env python3
"""Version 4.11.0. Sensitivity of the decisions of step 3 (Algorithm 2) to the Leiden resolution.

Every decision of the manuscript (Table 4, step 3) is taken at gamma = 1. This script re-runs
step 3 exactly as fuzzykg.audit.cut_audit defines it (20 seeds, both partitions of every drop
re-drawn per seed, 2 SD and minimum excess 0.01, the absolute reading where the no-cut drop
exceeds 0.05, the unit rule) on mu_E of the thirteen real graphs at gamma in {0.5, 1, 2}; gamma is
the resolution of every Leiden run and of every Q_F in the drops. Step 4 is not run (no undecided
cut has an informative grouping). The gamma = 1 run reproduces results/v12/unit_rule.json and is
checked against it. Writes results/v14/gamma_step3.json (resumable, one entry per graph and gamma).
Usage: python scripts/v14_gamma_step3.py [graph ...]; then --communities adds the community counts.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.audit import cut_audit  # noqa: E402

OUT = "results/v14/gamma_step3.json"
GRAPHS = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2009_item", "assist2015_kc", "assist2017_kc",
          "assist2017_item", "junyi_copractice", "assist2009_kc_transition", "assist2015_kc_transition",
          "assist2017_kc_transition", "assist2017_item_transition", "junyi_transition"]
GAMMAS = (0.5, 1.0, 2.0)
CUTS = ("q0.25", "q0.50", "q0.75")


def group(name):
    if name.startswith("mooccubex"):
        return "expert"
    return "transition" if name.endswith("transition") else "co-practice"


def unit(name):
    return "arbitrary" if group(name) == "co-practice" else "meaningful"


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    for name in (sys.argv[1:] or GRAPHS):
        df = pd.read_csv(f"data/graphs/{name}_edges.csv"); G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
        cuts = {c: float(np.quantile(G.mu, float(c[1:]))) for c in CUTS}
        for g in GAMMAS:
            key = f"{name}/gamma{g:g}"
            if key in res["runs"]:
                continue
            t0 = time.perf_counter()
            r = cut_audit(G, cuts, range(20), None, unit=unit(name), gamma=g)
            keep = ("qf_drop", "qf_drop_sd", "rank_drop", "rank_drop_sd", "flag_qf_drop", "flag_rank_drop", "decision")
            res["runs"][key] = {"graph": name, "group": group(name), "unit": unit(name), "gamma": g,
                                "cuts": {c: {k: v for k, v in r["cuts"][c].items() if k in keep} for c in ("support", *CUTS)},
                                "baseline_warning": r["baseline_warning"], "time_s": time.perf_counter() - t0}
            json.dump(res, open(OUT, "w"), indent=1)
            print(key, {c: res["runs"][key]["cuts"][c]["decision"] for c in CUTS}, round(res["runs"][key]["time_s"]), flush=True)
    summarise(res)
    json.dump(res, open(OUT, "w"), indent=1)


def summarise(res):
    ref = json.load(open("results/v12/unit_rule.json"))["graphs"]
    R = res["runs"]; names = sorted({v["graph"] for v in R.values()})
    s = {"gamma1_matches_v12": [], "gamma1_differs_from_v12": [], "changes": {}, "counts": {}}
    for n in names:
        if f"{n}/gamma1" not in R:
            continue
        d1 = {c: R[f"{n}/gamma1"]["cuts"][c]["decision"] for c in CUTS}
        dv = {c: ref[n]["cuts"][c]["decision"] for c in CUTS} if n in ref else None
        (s["gamma1_matches_v12"] if d1 == dv else s["gamma1_differs_from_v12"]).append(n if d1 == dv else {"graph": n, "gamma1": d1, "v12": dv})
    for g in GAMMAS:
        cnt = {}; ch = []
        for n in names:
            k = f"{n}/gamma{g:g}"
            if k not in R:
                continue
            for c in CUTS:
                d = R[k]["cuts"][c]["decision"]; cnt.setdefault(R[k]["group"], {}).setdefault(c, {}).setdefault(d, 0)
                cnt[R[k]["group"]][c][d] += 1
                d1 = R[f"{n}/gamma1"]["cuts"][c]["decision"]
                if g != 1.0 and d != d1:
                    ch.append({"graph": n, "cut": c, "gamma1": d1, f"gamma{g:g}": d})
        s["counts"][f"gamma{g:g}"] = cnt
        if g != 1.0:
            s["changes"][f"gamma{g:g}"] = ch
    sparse = [n for n in names if group(n) != "co-practice"]
    s["sparse_lower_quartile_flagged"] = {f"gamma{g:g}": sum(R[f"{n}/gamma{g:g}"]["cuts"]["q0.25"]["decision"] == "flagged" for n in sparse
                                                              if f"{n}/gamma{g:g}" in R) for g in GAMMAS}
    s["sparse_n"] = len(sparse)
    res["summary"] = s
    print(json.dumps(s, indent=1))


def communities():
    """Number of communities of Pi_F (median over seeds 0-4) at each gamma, added to the summary."""
    from fuzzykg.audit import _leiden
    res = json.load(open(OUT)); k = {}
    for name in GRAPHS:
        df = pd.read_csv(f"data/graphs/{name}_edges.csv"); G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
        k[name] = {f"gamma{g:g}": float(np.median([len(set(_leiden(G, s, g))) for s in range(5)])) for g in GAMMAS}
        print(name, k[name], flush=True)
    res["summary"]["communities_PiF_median_seeds0to4"] = k
    res["summary"]["n_changed"] = {g: len(v) for g, v in res["summary"]["changes"].items()}
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--communities"]:
        communities()
    else:
        main()
