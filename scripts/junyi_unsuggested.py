#!/usr/bin/env python3
"""Robustness of the Junyi transition graph to the platform's own sequencing.

About a fifth of the Junyi attempts are flagged `suggested`: the platform proposed the
exercise from the prerequisite links of its knowledge map.  An order-preserving graph
built from those attempts could agree with the knowledge map because the platform
followed it.  This script drops every suggested attempt, rebuilds the transition graph
with the same thresholds, and recomputes the expert criteria of Section 8.2 and 8.4
(composition operators unlimited in length).

Writes data/graphs/junyi_transition_unsuggested_* and results/junyi_unsuggested.json.
Usage: python scripts/junyi_unsuggested.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from data.loaders import RAW  # noqa: E402
from fuzzykg.construction import ConstructionConfig, TransitionConfig, build_transition_graph, preprocess  # noqa: E402
from build_junyi_graph import write  # noqa: E402
import prerequisite_criterion as pc  # noqa: E402


def main():
    p = RAW / "junyi" / "junyi" / "junyi_ProblemLog_original.csv"
    df = pd.read_csv(p, usecols=["user_id", "exercise", "time_done", "correct", "suggested"],
                     dtype={"user_id": "int64", "exercise": "category", "time_done": "int64", "correct": "string", "suggested": "string"})
    share = float((df.suggested.str.lower() == "true").mean())
    df = df[df.suggested.str.lower() != "true"]
    out = pd.DataFrame({"user_id": df.user_id, "kc": df.exercise.astype(str),
                        "correct": (df.correct.str.lower() == "true").astype("int8"), "order": df.time_done})
    out = out.drop_duplicates(subset=["user_id", "order", "kc"], keep="first").reset_index(drop=True)
    d = preprocess(out, ConstructionConfig())
    G, tab = build_transition_graph(d, TransitionConfig(), preprocessed=True)
    name = "junyi_transition_unsuggested"
    write(G, tab, name, {"suggested_share_dropped": share, "learners": int(d.user_id.nunique()), "interactions": int(len(d))})
    rng = np.random.default_rng(0)
    res = {"suggested_share": share, "learners": int(d.user_id.nunique()), "interactions": int(len(d)),
           "n": int(G.n), "m": int(len(G.mu)), "mu_mean": float(np.mean(G.mu)),
           "criterion": pc.junyi(name, rng)}
    json.dump(res, open("results/junyi_unsuggested.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "criterion"}))
    u = res["criterion"]["configs"]["unlimited"]
    print({k: (round(v["rho"], 3), round(v["map_auc"], 3), round(v["coverage"], 2)) for k, v in u.items()})


if __name__ == "__main__":
    main()
