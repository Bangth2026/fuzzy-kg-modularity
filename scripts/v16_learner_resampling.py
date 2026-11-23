#!/usr/bin/env python3
"""Release 5.1.0. Resampling of the learners, extended (Section 8.2, Supplementary Section S7).

The bootstrap of release 5.0.0 (scripts/v15_learner_bootstrap.py) draws learners with
replacement and enters a learner drawn twice as two learners, so a duplicated learner counts
twice towards the support threshold of 20 and the rebuilt graphs can be denser than the observed
one. Two additions:
  subsample  subsamples of 80% of the learners without replacement (no duplicate; the graphs
             can only lose support): 20 on the six ASSISTments KC-level graphs, 10 on Junyi transition;
  bootstrap  the release 5.0.0 bootstrap on Junyi transition (10 draws).
Audit as in release 5.0.0 (ten seeds, minimum excess by density class, kappa* searched up to 100).
Also the number of edges of every rebuilt graph.
Writes results/v16/learner_resampling.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg.construction import build_fuzzy_graph, build_transition_graph, preprocess, ConstructionConfig  # noqa: E402
from data.loaders import load_assistments, load_junyi  # noqa: E402
from v15_learner_bootstrap import audit, DELTA  # noqa: E402

OUT = "results/v16/learner_resampling.json"
B = 20
B_JUNYI = 10   # Junyi: 10 draws of each kind (each rebuild reads 25 million attempts)


def resample(d, starts, ends, pick):
    lens = ends[pick] - starts[pick]
    idx = np.repeat(starts[pick] - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(lens.sum())
    out = d.iloc[idx].copy()
    out["user_id"] = np.repeat(np.arange(pick.size), lens)
    return out.reset_index(drop=True)


def summarise(rec):
    s = {}
    for c in rec["observed"]["audit"]:
        o = rec["observed"]["audit"][c]
        for kind in ("subsample", "bootstrap"):
            if kind not in rec:
                continue
            R = rec[kind]
            s.setdefault(c, {"observed": o})[kind] = {
                "share_same_decision": float(np.mean([r["audit"][c]["decision"] == o["decision"] for r in R])),
                "share_same_robust": float(np.mean([r["audit"][c]["robust"] == o["robust"] for r in R])),
                "kappa_star_quartiles": [float(x) for x in np.percentile([min(r["audit"][c]["kappa_star"], 1e3) for r in R], [25, 50, 75])],
                "edges_median": float(np.median([r["m"] for r in R]))}
    return s


def run(key, d, build, delta, kinds, rng, res, nb=B):
    if key in res and all(k in res[key] for k in kinds):
        return
    t0 = time.perf_counter()
    rec = res.get(key, {})
    G, _ = build(d, preprocessed=True)
    rec["observed"] = {"m": int(G.m), "audit": audit(G, delta)}; rec["min_excess"] = delta
    u = d["user_id"].to_numpy(); ch = np.r_[True, u[1:] != u[:-1]]
    starts = np.flatnonzero(ch); ends = np.r_[starts[1:], len(u)]; nU = starts.size
    for kind in kinds:
        rec[kind] = []
        for b in range(nb):
            pick = np.sort(rng.choice(nU, int(0.8 * nU), replace=False)) if kind == "subsample" else rng.choice(nU, nU, replace=True)
            Gb, _ = build(resample(d, starts, ends, pick), preprocessed=True)
            rec[kind].append({"m": int(Gb.m), "audit": audit(Gb, delta)})
    rec["summary"] = summarise(rec); rec["time_s"] = time.perf_counter() - t0
    res[key] = rec
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(key, G.m, {c: {k: (round(v[k]["share_same_decision"], 2), round(v[k]["share_same_robust"], 2), v[k]["edges_median"])
                         for k in kinds} for c, v in rec["summary"].items()}, f"{rec['time_s']:.0f}s", flush=True)


def main(which):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rng = np.random.default_rng(5112)
    if "assist" in which:
        for rel in ("2009", "2015", "2017"):
            d = preprocess(load_assistments(rel, "kc"), ConstructionConfig())
            for cname, build in (("copractice", build_fuzzy_graph), ("transition", build_transition_graph)):
                run(f"assist{rel}_kc_{cname}", d, build, DELTA["dense" if cname == "copractice" else "sparse"], ("subsample",), rng, res)
    if "junyi" in which:
        df = load_junyi(); d = preprocess(df, ConstructionConfig()); del df
        run("junyi_transition", d, build_transition_graph, DELTA["sparse"], ("subsample", "bootstrap"), rng, res, B_JUNYI)


if __name__ == "__main__":
    main(sys.argv[1:] or ["assist", "junyi"])
