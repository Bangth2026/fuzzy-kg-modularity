#!/usr/bin/env python3
"""Release 5.2.0. kappa* of the Junyi transition cuts against the share of learners kept.

Subsamples without replacement of 50% and 65% of the learners (five draws each), next to the 80%
subsamples of scripts/v16_learner_resampling.py (ten draws) and the full log; audit as there (ten
seeds, sparse minimum excess, kappa* searched up to 100). Writes results/v17/junyi_fraction.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg.construction import build_transition_graph, preprocess, ConstructionConfig  # noqa: E402
from data.loaders import load_junyi  # noqa: E402
from v15_learner_bootstrap import audit, DELTA  # noqa: E402
from v16_learner_resampling import resample  # noqa: E402


def main():
    rng = np.random.default_rng(5201)
    d = preprocess(load_junyi(), ConstructionConfig())
    u = d["user_id"].to_numpy(); ch = np.r_[True, u[1:] != u[:-1]]
    starts = np.flatnonzero(ch); ends = np.r_[starts[1:], len(u)]; nU = starts.size
    res = {}
    for frac in (0.5, 0.65):
        res[f"{frac:g}"] = []
        for _ in range(5):
            pick = np.sort(rng.choice(nU, int(frac * nU), replace=False))
            G, _ = build_transition_graph(resample(d, starts, ends, pick), preprocessed=True)
            a = audit(G, DELTA["sparse"]); res[f"{frac:g}"].append({"m": int(G.m), "audit": a})
            print(frac, G.m, {c: round(min(v["kappa_star"], 1e3), 2) for c, v in a.items()}, flush=True)
            json.dump(res, open("results/v17/junyi_fraction.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
