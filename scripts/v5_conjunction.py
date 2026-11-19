#!/usr/bin/env python3
"""Which edges does a compensatory conjunction add to the Junyi transition graph?

Builds the transition table of Junyi once, without pruning, and compares the memberships
min(asym, pref) (Equation 3, default) and (asym + pref)/2 (the compensatory variant of Section 8.5)
at tau_E = 0.2: number of edges, share of edges whose endpoints share an expert topic, the
distribution of asym among the edges, and the same for the edges that only the mean admits.
Output: results/v5/conjunction.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.loaders import load_junyi, load_junyi_expert  # noqa: E402
from fuzzykg.construction import ConstructionConfig, TransitionConfig, build_transition_graph, preprocess  # noqa: E402


def describe(t, topic, mu=None):
    same = np.array([topic.get(a) is not None and topic.get(a) == topic.get(b) for a, b in zip(t.src, t.dst)])
    extra = {}
    if mu is not None:
        m = t[mu].to_numpy()
        extra = {"mu_same_topic_mean": float(m[same].mean()) if same.any() else None,
                 "mu_cross_topic_mean": float(m[~same].mean()) if (~same).any() else None,
                 "pref_below_0.2_share": float((t.pref < 0.2).mean())}
    return {**extra, "edges": int(len(t)), "same_topic_share": float(same.mean()),
            "asym_median": float(t.asym.median()), "asym_below_0.2_share": float((t.asym < 0.2).mean()),
            "back_to_forward_ratio_median": float((t.moves_back / t.moves).median())}


def main():
    ex, _ = load_junyi_expert()
    topic = dict(zip(ex.name.astype(str), ex.topic.astype(str)))
    d = preprocess(load_junyi(), ConstructionConfig())
    _, tab = build_transition_graph(d, TransitionConfig(tau_E=1e-9), preprocessed=True)
    tab["mu_min"] = np.minimum(tab.asym, tab.pref); tab["mu_mean"] = (tab.asym + tab.pref) / 2
    tmin = tab[tab.mu_min >= 0.2]; tmean = tab[tab.mu_mean >= 0.2]
    only = tab[(tab.mu_mean >= 0.2) & (tab.mu_min < 0.2)]
    res = {"min": describe(tmin, topic, "mu_min"), "mean": describe(tmean, topic, "mu_mean"), "mean_only": describe(only, topic, "mu_mean")}
    os.makedirs(os.path.join(ROOT, "results/v5"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results/v5/conjunction.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
