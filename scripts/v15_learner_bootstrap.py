#!/usr/bin/env python3
"""Release 5.0.0. Stability of the step-3 decisions under resampling of the learners (Section 8).

The uncertainty of mu_E comes mostly from the sample of learners, which optimiser seeds do not
see. On the three ASSISTments KC-level logs (both constructions, six graphs) the log is resampled
by learner with replacement (20 bootstrap logs; a resampled learner enters as a new learner),
both constructions are rebuilt with the stated thresholds, and the unit-robust audit
(fuzzykg.simplex, ten seeds, minimum excess calibrated by density class) is run on the quantile cuts of every rebuilt graph.
Reports per graph and cut the share of bootstrap graphs that reproduce the decision on the
observed graph, the share that reproduce the robust decision (kappa* >= 2, else undecided), and
the bootstrap distribution of kappa*.
Writes results/v15/learner_bootstrap.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.audit import _leiden  # noqa: E402
from fuzzykg.construction import build_fuzzy_graph, build_transition_graph, preprocess, ConstructionConfig  # noqa: E402
from fuzzykg.simplex import CutFamily, kappa_star  # noqa: E402
from data.loaders import load_assistments  # noqa: E402

OUT = "results/v15/learner_bootstrap.json"
B = 20
CUTS = (0.25, 0.50, 0.75)
SEEDS = range(10)
DELTA = json.load(open("results/v15/summary.json"))["delta"]   # minimum excess by density class (Section 4.4)


def audit(G, delta):
    Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
    parts = {"support": [_leiden(Gs, s) for s in SEEDS]}
    for q in CUTS:
        parts[f"q{q:.2f}"] = [_leiden(G.alpha_cut(float(np.quantile(G.mu, q))), s) for s in SEEDS]
    fam = CutFamily(G, parts)
    out = {}
    for q in CUTS:
        c = f"q{q:.2f}"; k = kappa_star(fam, c, min_excess=delta, kmax=100.0)
        out[c] = {"decision": k["decision_mu"], "kappa_star": k["kappa_star"],
                  "robust": k["decision_mu"] if k["kappa_star"] >= 2 else "undecided"}
    return out


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rng = np.random.default_rng(2026)
    for rel in ("2009", "2015", "2017"):
        d = preprocess(load_assistments(rel, "kc"), ConstructionConfig())
        users = d.user_id.unique()
        groups = {u: g for u, g in d.groupby("user_id", sort=False)}
        for cname, build in (("copractice", build_fuzzy_graph), ("transition", build_transition_graph)):
            key = f"assist{rel}_kc_{cname}"
            if key in res and len(res[key]["boot"]) >= B:
                continue
            t0 = time.perf_counter()
            G, _ = build(d, preprocessed=True)
            delta = DELTA["dense" if cname == "copractice" else "sparse"]
            rec = {"observed": audit(G, delta), "boot": [], "min_excess": delta}
            for b in range(B):
                pick = rng.choice(users, users.size, replace=True)
                parts = []
                for i, u in enumerate(pick):
                    g = groups[u].copy(); g["user_id"] = f"{u}_{i}"; parts.append(g)
                db = pd.concat(parts, ignore_index=True)
                Gb, _ = build(db, preprocessed=True)
                rec["boot"].append(audit(Gb, delta))
            s = {}
            for c in rec["observed"]:
                o = rec["observed"][c]
                s[c] = {"observed": o, "share_same_decision": float(np.mean([r[c]["decision"] == o["decision"] for r in rec["boot"]])),
                        "share_same_robust": float(np.mean([r[c]["robust"] == o["robust"] for r in rec["boot"]])),
                        "kappa_star_quartiles": [float(x) for x in np.percentile([min(r[c]["kappa_star"], 1e3) for r in rec["boot"]], [25, 50, 75])]}
            rec["summary"] = s; rec["time_s"] = time.perf_counter() - t0
            res[key] = rec
            json.dump(res, open(OUT, "w"), indent=1, default=float)
            print(key, {c: (v["observed"]["decision"], round(v["share_same_decision"], 2), round(v["share_same_robust"], 2)) for c, v in s.items()}, f"{rec['time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
