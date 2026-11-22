#!/usr/bin/env python3
"""Version 4.8.0. The grade-preserving baselines of scripts/v9_grade_preserving_baselines.py on the
two Junyi graphs, with every evaluation pair kept out of the sampled non-edges.

In release 4.6.0 both models regress 0 on non-edges drawn uniformly, so an annotated pair without
an edge (87% on the transition graph) can be taught as 0.  Here the annotated pairs (both
orientations), the knowledge-map links and their random negative pairs are never drawn as
non-edges, in training or in validation; everything else (grid, epochs, refit seeds, criteria,
bootstrap resamples) is that of release 4.6.0.  Needs PyTorch and the Junyi annotation files in
data/raw/junyi/junyi/.  Writes results/v11/baselines_excluded.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import v9_grade_preserving_baselines as v9  # noqa: E402
from fuzzykg.embeddings import GradedLinkModel  # noqa: E402

OUT = "results/v11/baselines_excluded.json"
_EXCLUDE = {}
_fit = GradedLinkModel.fit


def fit_excluding(self, G, *a, **k):
    k.setdefault("exclude", _EXCLUDE.get("pairs"))
    return _fit(self, G, *a, **k)


def evaluation_pairs(name):
    """The pairs that the criteria of v9.junyi score: annotated (both orientations), map links, map negatives."""
    G, labels = v9.load(name); idx = {l: i for i, l in enumerate(labels)}
    ex, ann = v9.load_junyi_expert()
    sel = ann[ann.Exercise_A.astype(str).isin(idx) & ann.Exercise_B.astype(str).isin(idx)]
    ia = np.array([idx[a] for a in sel.Exercise_A.astype(str)]); ib = np.array([idx[b] for b in sel.Exercise_B.astype(str)])
    mp = set()
    for n_, p in zip(ex.name.astype(str), ex.prerequisites):
        if isinstance(p, str):
            for q in p.split(","):
                if q.strip() in idx and n_ in idx:
                    mp.add((idx[q.strip()], idx[n_]))
    lk = np.array(sorted(mp)); nrng = np.random.default_rng(0); neg = []
    while len(neg) < 5 * len(lk):          # the same draw as v9.junyi
        a, b = nrng.integers(0, G.n, 2)
        if a != b and (a, b) not in mp and (b, a) not in mp:
            neg.append((a, b))
    neg = np.array(neg)
    P = np.r_[np.c_[ib, ia], np.c_[ia, ib], lk, neg]
    return np.unique(P, axis=0)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    GradedLinkModel.fit = fit_excluding
    for name in sys.argv[1:] or ["junyi_transition", "junyi_copractice"]:
        if name in res:
            continue
        _EXCLUDE["pairs"] = evaluation_pairs(name)
        r = v9.junyi(name)
        r["n_excluded_pairs"] = int(len(_EXCLUDE["pairs"]))
        res[name] = r
        json.dump(res, open(OUT, "w"), indent=1)
        print(name, {k: (round(v["rho"], 3), round(v["map_auc"], 3)) for k, v in r["scores"].items()}, flush=True)


if __name__ == "__main__":
    main()
