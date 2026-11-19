#!/usr/bin/env python3
"""Sensitivity of the cut audit (Algorithm 2, step 3) to the resolution gamma.

Re-runs the protocol of scripts/v4_protocol_external.py on the observed memberships (no tilt) at
gamma in {0.5, 1, 2}, five optimiser seeds per partition, on the eleven graphs whose grouping is
released with this repository (the two ASSISTments item-level graphs need the raw logs for their
skill tags). Reports, per gamma, the 2x2 agreement of each flag with a resolved ARI loss.
Output: results/v5/gamma_audit.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v4_protocol_external as pe  # noqa: E402
from fuzzykg import FuzzyGraph  # noqa: E402

GRAPHS = ["junyi_transition", "junyi_copractice", "eedi_transition", "eedi_copractice", "algebra2005_transition",
          "algebra2005_copractice", "bridge2006_transition", "bridge2006_copractice", "ednet_transition", "ednet_copractice"]
OUT = "results/v5/gamma_audit.json"
GAMMA = [1.0]


def leiden(G, seed):
    import igraph as ig, leidenalg as la
    g = ig.Graph(n=G.n, edges=list(zip(G.src.tolist(), G.dst.tolist())), directed=True); g.es["weight"] = G.mu.tolist()
    p = la.RBConfigurationVertexPartition(g, weights="weight", resolution_parameter=GAMMA[0])
    o = la.Optimiser(); o.set_rng_seed(int(seed)); o.optimise_partition(p, n_iterations=-1)
    return np.asarray(p.membership)


def groups(name):
    lab = pd.read_csv(f"data/graphs/{name}_nodes.csv").label.astype(str).tolist()
    base = name.split("_")[0]
    g = json.load(open("data/graphs/junyi_topics.json" if base == "junyi" else f"data/graphs/{base}_groups.json"))
    return np.array([g.get(l, g.get(l.replace(".0", ""))) for l in lab], dtype=object)


def table(cells, key):
    y = np.array([c["ari_loss_resolved"] for c in cells]); f = np.array([c[key] for c in cells])
    tp, fp, fn, tn = int((y & f).sum()), int((~y & f).sum()), int((y & ~f).sum()), int((~y & ~f).sum())
    n = tp + fp + fn + tn; po = (tp + tn) / n
    pe_ = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=tp / max(1, tp + fn), specificity=tn / max(1, tn + fp),
                kappa=(po - pe_) / (1 - pe_) if pe_ < 1 else None)


def main():
    pe.leiden = leiden
    pe.SEEDS = range(5)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for gamma in (0.5, 1.0, 2.0):
        GAMMA[0] = gamma
        for name in GRAPHS:
            key = f"{name}/gamma{gamma:g}"
            if key in res:
                continue
            df = pd.read_csv(f"data/graphs/{name}_edges.csv"); G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
            rows, info = pe.audit_graph(G, groups(name))
            res[key] = {"info": info, "rows": rows}
            json.dump(res, open(OUT, "w"), indent=1)
            print(key, flush=True)
    summ = {}
    for gamma in (0.5, 1.0, 2.0):
        cells = [r for k, v in res.items() if k.endswith(f"gamma{gamma:g}") for c, r in v["rows"].items()
                 if c not in ("Pi_F", "support") and not r.get("empty")]
        summ[f"gamma{gamma:g}"] = {"cells": len(cells), "losses": int(sum(c["ari_loss_resolved"] for c in cells)),
                                   "qf_drop": table(cells, "flag_qf_drop"), "rank_drop": table(cells, "flag_rank_drop")}
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
