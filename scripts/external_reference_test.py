#!/usr/bin/env python3
"""Do the memberships favour edges inside an external grouping? (Section 8.2)

The weights-only nulls of Algorithm 1 define internality by the partition
optimised on the graph under test. Where the corpus supplies a grouping fixed
independently of the graph (the 40 expert topics of Junyi; the skill tags of the
ASSISTments item-level graphs) the question can be asked against it instead.

For every such graph and reference grouping, over the edges whose two endpoints
carry a label:
* the point-biserial correlation r between mu_E and the indicator that both
  endpoints share a group, and the difference of mean memberships;
* its permutation distribution under the global permutation of the memberships
  and under the within-source permutation (memberships permuted among the
  out-edges of each node, so every out-strength is kept), 5000 draws each;
* the fuzzy-graph modularity Q_F of the reference grouping itself at gamma = 1
  against the same two permutation nulls (a fixed partition, so no optimiser
  enters).

Usage: python scripts/external_reference_test.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.modularity import fuzzy_modularity  # noqa: E402
from real_graph_tables import references, load  # noqa: E402

GRAPHS = [("junyi_transition", "topic"), ("junyi_copractice", "topic"), ("junyi_transition", "area"), ("junyi_copractice", "area"),
          ("assist2009_item", "skill"), ("assist2017_item", "skill"), ("assist2017_item_transition", "skill"),
          ("assist2017_item", "assignment"), ("assist2017_item_transition", "assignment")]


def row_blocks(src):
    order = np.argsort(src, kind="stable")
    bounds = np.flatnonzero(np.diff(src[order])) + 1
    return [b for b in np.split(order, bounds) if b.size > 1]


def perm_rows(mu, blocks, rng):
    out = mu.copy()
    for b in blocks:
        out[b] = mu[rng.permutation(b)]
    return out


def main(draws=5000):
    rng = np.random.default_rng(2026)
    out = {}
    for name, ref in GRAPHS:
        G, labels = load(name)
        lab = np.array(references(name, labels)[ref])
        ok = (lab[G.src] != "NA") & (lab[G.dst] != "NA")
        same = (lab[G.src] == lab[G.dst])[ok].astype(float)
        mu = G.mu[ok]; src = G.src[ok]
        def stat(m):
            return float(np.corrcoef(m, same)[0, 1])
        obs = stat(mu); diff = float(mu[same == 1].mean() - mu[same == 0].mean())
        blocks = row_blocks(src)
        g_null = np.array([stat(rng.permutation(mu)) for _ in range(draws)])
        r_null = np.array([stat(perm_rows(mu, blocks, rng)) for _ in range(draws)])
        # Q_F of the reference partition (unlabelled nodes become singletons)
        codes = {v: i for i, v in enumerate(sorted(set(lab)))}
        memb = np.array([codes[v] if v != "NA" else len(codes) + i for i, v in enumerate(lab)])
        q_obs = fuzzy_modularity(G, memb, 1.0)
        blocks_all = row_blocks(G.src)
        qg = np.array([fuzzy_modularity(FuzzyGraph(G.n, G.src, G.dst, rng.permutation(G.mu)), memb, 1.0) for _ in range(1000)])
        qr = np.array([fuzzy_modularity(FuzzyGraph(G.n, G.src, G.dst, perm_rows(G.mu, blocks_all, rng)), memb, 1.0) for _ in range(1000)])
        def p2(x, null):
            return float((np.sum(np.abs(null - null.mean()) >= abs(x - null.mean())) + 1) / (len(null) + 1))
        r = {"edges_labelled": int(ok.sum()), "share_same": float(same.mean()), "r_pb": obs, "mean_mu_same_minus_diff": diff,
             "global": {"null_mean": float(g_null.mean()), "null_sd": float(g_null.std()), "p_two_sided": p2(obs, g_null)},
             "row": {"null_mean": float(r_null.mean()), "null_sd": float(r_null.std()), "p_two_sided": p2(obs, r_null)},
             "Q_F_reference": q_obs,
             "Q_global": {"delta": float(q_obs - qg.mean()), "z": float((q_obs - qg.mean()) / qg.std()), "p_two_sided": p2(q_obs, qg)},
             "Q_row": {"delta": float(q_obs - qr.mean()), "z": float((q_obs - qr.mean()) / qr.std()), "p_two_sided": p2(q_obs, qr)}}
        out[f"{name}:{ref}"] = r
        print(name, ref, json.dumps({"r": round(obs, 3), "diff": round(diff, 4), "pg": r["global"]["p_two_sided"], "pr": r["row"]["p_two_sided"],
                                     "Qg": (round(r["Q_global"]["delta"], 4), round(r["Q_global"]["z"], 1)),
                                     "Qr": (round(r["Q_row"]["delta"], 4), round(r["Q_row"]["z"], 1))}), flush=True)
    json.dump(out, open("results/external_reference_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
