#!/usr/bin/env python3
"""Version 4.7.0. Paired comparison of the grade-preserving models with the best membership-free
heuristic on the hidden MOOCCubeX expert edges.

Re-runs the MOOCCubeX protocol of scripts/v9_grade_preserving_baselines.py (five folds, twenty
random non-edges per hidden edge, rng 7, same draw order), keeps the score of every hidden edge
and non-edge for the GCN, the UKGE-style model, the product path score and the preferential
attachment heuristic (out-degree of the source times in-degree of the target on the training
graph; scripts/v10_link_heuristics.py), and reports AUC differences with a paired bootstrap
(500 resamples of hidden edges and of non-edges, the same resample for every score).
Needs PyTorch. Writes results/v10/gcn_vs_heuristics.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402
torch.set_num_threads(1)
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.embeddings import GradedLinkModel  # noqa: E402
from composition_robustness import mats  # noqa: E402

OUT = "results/v10/gcn_vs_heuristics.json"


def main(folds=5, neg_per_pos=20):
    e = pd.read_csv("data/graphs/mooccubex_cs_full_edges.csv"); nd = pd.read_csv("data/graphs/mooccubex_cs_full_nodes.csv")
    n = len(nd); src = e.src.to_numpy(); dst = e.dst.to_numpy(); mu = e.mu.to_numpy(); m = len(src)
    edges = set(zip(src.tolist(), dst.tolist()))
    rng = np.random.default_rng(7); perm = rng.permutation(m); parts = np.array_split(perm, folds)
    keys = ["product", "preferential_attachment", "ukge", "gnn"]
    P = {k: [] for k in keys}; N = {k: [] for k in keys}
    for f in range(folds):
        test = parts[f]; train = np.setdiff1d(perm, test)
        G = FuzzyGraph(n, src[train], dst[train], mu[train])
        neg = []
        while len(neg) < neg_per_pos * len(test):
            a, b = rng.integers(0, n, 2)
            if a != b and (a, b) not in edges and (b, a) not in edges:
                neg.append((a, b))
        neg = np.array(neg)
        M, _ = mats(G)
        P["product"].extend(M["product"][src[test], dst[test]].tolist()); N["product"].extend(M["product"][neg[:, 0], neg[:, 1]].tolist())
        dout = np.bincount(src[train], minlength=n).astype(float); din = np.bincount(dst[train], minlength=n).astype(float)
        P["preferential_attachment"].extend((dout[src[test]] * din[dst[test]]).tolist())
        N["preferential_attachment"].extend((dout[neg[:, 0]] * din[neg[:, 1]]).tolist())
        for mdl in ("ukge", "gnn"):
            E = GradedLinkModel(model=mdl).fit(G)
            P[mdl].extend(E.score(src[test], dst[test]).tolist()); N[mdl].extend(E.score(neg[:, 0], neg[:, 1]).tolist())
        print("fold", f, flush=True)
    P = {k: np.asarray(v, float) for k, v in P.items()}; N = {k: np.asarray(v, float) for k, v in N.items()}
    lab = np.r_[np.ones(len(P["gnn"])), np.zeros(len(N["gnn"]))]
    auc = {k: float(roc_auc_score(lab, np.r_[P[k], N[k]])) for k in keys}
    brng = np.random.default_rng(13); B = {k: [] for k in keys}
    for _ in range(500):
        i = brng.integers(0, len(lab) - len(N["gnn"]), len(P["gnn"])); j = brng.integers(0, len(N["gnn"]), len(N["gnn"]))
        lb = np.r_[np.ones(len(i)), np.zeros(len(j))]
        for k in keys:
            B[k].append(roc_auc_score(lb, np.r_[P[k][i], N[k][j]]))
    B = {k: np.array(v) for k, v in B.items()}
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {"auc": auc, "auc_ci": {k: ci(B[k]) for k in keys},
           "differences": {f"{a}_minus_{b}": {"diff": auc[a] - auc[b], "ci": ci(B[a] - B[b])}
                           for a, b in [("gnn", "preferential_attachment"), ("ukge", "preferential_attachment"),
                                        ("preferential_attachment", "product"), ("gnn", "product")]}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
