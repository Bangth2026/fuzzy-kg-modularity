#!/usr/bin/env python3
"""Release 5.3.0. Partition value of the grades over the whole box B_2 and over seed triples, and
the Junyi graph without platform-suggested attempts on the vertex set of the original graph
(Section 8.1, Supplementary Tables S4c, S4f, S4h).

1. box_seeds   Junyi transition and EdNet co-practice. mu_E is replaced by phi_s(mu_E) for every
               vertex of B_2 after the normalisation min s = 1, i.e. s in {1, 2}^9 (512 vertices,
               knots of Section 4.2, largest membership kept). On each vertex: the partition value
               V = ARI(best of a seed triple on phi_s(mu_E)) - ARI(best of the same triple on the
               support) for the six disjoint seed triples (0-2, ..., 15-17), and the row-null mean
               (ten row permutations of phi_s(mu_E), best of seeds 0-2), giving V_res per triple.
               Reported: range of V and V_res over vertices x triples, the share of negative
               values, the minimum over vertices for each triple, and the 18 axis vertices of S4c.
2. unsuggested Junyi transition with and without platform-suggested attempts, both on the 628
               exercises that are nodes of both graphs and have a topic (the original graph has
               633 nodes): V over the six seed triples, the grouping-class null (permutations of
               mu_E separately inside and between topics; ten draws, best of each triple).
3. edges       Edges of the original graph absent from the graph without suggested attempts
               ("suggestion-only"), edges present only without them, and shared edges: number,
               share inside one topic, mean membership.
Writes results/v18/partition.json.
"""
from __future__ import annotations
import itertools, json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden  # noqa: E402
from fuzzykg.nulltest import permute_rows  # noqa: E402
from v16_units import rescaled  # noqa: E402
from v15_value_paths import load  # noqa: E402
from v15_simplex_real import groups_for  # noqa: E402

OUT = "results/v18/partition.json"
TRIPLES = [range(3 * t, 3 * t + 3) for t in range(6)]


def best(H, seeds):
    P = [_leiden(H, s) for s in seeds]
    return max(P, key=lambda p: fuzzy_modularity(H, p))


def _vertex(args):
    name, s, k = args
    G, _ = load(name)
    gr = groups_for(name); has = np.array([x is not None for x in gr]); gl = pd.factorize(pd.Series(gr[has]).astype(str))[0]
    H = rescaled(G, np.asarray(s, float))
    obs = [float(ari(gl, best(H, tr)[has])) for tr in TRIPLES]
    rng = np.random.default_rng(1000 + k)
    row = float(np.mean([ari(gl, best(permute_rows(H, rng), range(3))[has]) for _ in range(10)]))
    return k, obs, row


def box_seeds(name, pool):
    G, _ = load(name); Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
    gr = groups_for(name); has = np.array([x is not None for x in gr]); gl = pd.factorize(pd.Series(gr[has]).astype(str))[0]
    sup = [float(ari(gl, best(Gs, tr)[has])) for tr in TRIPLES]
    S = [tuple(v) for v in itertools.product((1.0, 2.0), repeat=9)]
    out = {}
    for k, obs, row in pool.imap_unordered(_vertex, [(name, s, k) for k, s in enumerate(S)], chunksize=4):
        out[k] = (obs, row)
    V = np.array([[out[k][0][t] - sup[t] for t in range(6)] for k in range(len(S))])
    R = np.array([[out[k][0][t] - out[k][1] for t in range(6)] for k in range(len(S))])
    ax = [k for k, s in enumerate(S) if sum(x == 2.0 for x in s) in (1, 8)]
    mu = S.index(tuple([1.0] * 9))
    res = {"support_triples": sup, "n_vertices": len(S),
           "V_range": [float(V.min()), float(V.max())], "V_resid_range": [float(R.min()), float(R.max())],
           "V_share_negative": float(np.mean(V < 0)), "V_resid_share_negative": float(np.mean(R < 0)),
           "V_min_per_triple": V.min(0).tolist(), "V_max_per_triple": V.max(0).tolist(),
           "V_median_over_triples_range": [float(np.median(V, 1).min()), float(np.median(V, 1).max())],
           "V_median_over_triples_share_negative": float(np.mean(np.median(V, 1) < 0)),
           "V_resid_median_over_triples_range": [float(np.median(R, 1).min()), float(np.median(R, 1).max())],
           "V_at_mu_triples": V[mu].tolist(), "V_resid_at_mu_triples": R[mu].tolist(),
           "axis_vertices": {"V_range_triple0": [float(V[ax, 0].min()), float(V[ax, 0].max())],
                             "V_range_all_triples": [float(V[ax].min()), float(V[ax].max())]},
           "per_vertex": {str(k): {"s": list(S[k]), "V": V[k].tolist(), "V_resid": R[k].tolist()} for k in range(len(S))}}
    return res


def remap(name, labels_target):
    e = pd.read_csv(f"data/graphs/{name}_edges.csv"); nd = pd.read_csv(f"data/graphs/{name}_nodes.csv")
    lab = nd.label.astype(str).tolist(); idx = {l: i for i, l in enumerate(labels_target)}
    keep = [i for i, l in enumerate(lab) if l in idx]
    m = {i: idx[lab[i]] for i in keep}
    k = e.src.isin(m) & e.dst.isin(m)
    return FuzzyGraph(len(labels_target), e.src[k].map(m).to_numpy(), e.dst[k].map(m).to_numpy(), e.mu[k].to_numpy())


def permute_within(G, cls, rng):
    mu = G.mu.copy()
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c); mu[idx] = mu[rng.permutation(idx)]
    return FuzzyGraph(G.n, G.src, G.dst, mu)


def unsuggested():
    l0 = pd.read_csv("data/graphs/junyi_transition_nodes.csv").label.astype(str).tolist()
    l1 = pd.read_csv("data/graphs/junyi_transition_unsuggested_nodes.csv").label.astype(str).tolist()
    ex = pd.read_csv("data/raw/junyi/junyi/junyi_Exercise_table.csv", encoding="utf-8-sig")
    topic = dict(zip(ex["name"].astype(str), ex["topic"].astype(str)))
    common = sorted(set(l0) & set(l1) & set(topic))
    gl = pd.factorize(pd.Series([topic[l] for l in common]))[0]
    out = {"n_original": len(l0), "n_unsuggested": len(l1), "n_common_with_topic": len(common)}
    rng = np.random.default_rng(5301)
    for tag, name in (("original", "junyi_transition"), ("unsuggested", "junyi_transition_unsuggested")):
        G = remap(name, common); Gs = FuzzyGraph(G.n, G.src, G.dst, np.ones(G.m))
        cls = (gl[G.src] == gl[G.dst]).astype(int)
        sup = [float(ari(gl, best(Gs, tr))) for tr in TRIPLES]; obs = [float(ari(gl, best(G, tr))) for tr in TRIPLES]
        blk = [[float(ari(gl, best(permute_within(G, cls, rng), tr))) for tr in TRIPLES] for _ in range(10)]
        blk_mean = np.mean(blk, 0)
        V = np.array(obs) - np.array(sup)
        out[tag] = {"m": int(G.m), "support_triples": sup, "observed_triples": obs, "V_triples": V.tolist(), "V_median": float(np.median(V)),
                    "V_range": [float(V.min()), float(V.max())], "block_triples": blk_mean.tolist(),
                    "V_beyond_block_triples": (np.array(obs) - blk_mean).tolist(),
                    "block_minus_support_triples": (blk_mean - np.array(sup)).tolist(),
                    "V_beyond_block_median": float(np.median(np.array(obs) - blk_mean)),
                    "block_minus_support_median": float(np.median(blk_mean - np.array(sup)))}
        print(tag, {k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in out[tag].items()}, flush=True)
    # edges
    G0 = remap("junyi_transition", common); G1 = remap("junyi_transition_unsuggested", common)
    E0 = {(a, b): m for a, b, m in zip(G0.src, G0.dst, G0.mu)}; E1 = {(a, b): m for a, b, m in zip(G1.src, G1.dst, G1.mu)}
    def stats(keys, E):
        keys = list(keys)
        if not keys:
            return {"n": 0}
        inside = np.array([gl[a] == gl[b] for a, b in keys]); mu = np.array([E[k] for k in keys])
        return {"n": len(keys), "share_inside": float(inside.mean()), "mu_mean": float(mu.mean()),
                "mu_mean_inside": float(mu[inside].mean()) if inside.any() else None,
                "mu_mean_between": float(mu[~inside].mean()) if (~inside).any() else None}
    sh = set(E0) & set(E1)
    out["edges"] = {"suggestion_only": stats(set(E0) - set(E1), E0), "unsuggested_only": stats(set(E1) - set(E0), E1),
                    "shared_original_mu": stats(sh, E0), "shared_unsuggested_mu": stats(sh, E1),
                    "all_original": stats(set(E0), E0)}
    print(out["edges"], flush=True)
    return out


def main(parts):
    os.makedirs("results/v18", exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if "unsuggested" in parts:
        res["unsuggested"] = unsuggested(); json.dump(res, open(OUT, "w"), indent=1)
    if "box" in parts:
        res.setdefault("box_seeds", {})
        with Pool(2) as pool:
            for name in ("junyi_transition", "ednet_copractice"):
                if name in res["box_seeds"]:
                    continue
                res["box_seeds"][name] = box_seeds(name, pool); json.dump(res, open(OUT, "w"), indent=1)
                r = res["box_seeds"][name]
                print(name, {k: v for k, v in r.items() if k != "per_vertex"}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["unsuggested", "box"])
