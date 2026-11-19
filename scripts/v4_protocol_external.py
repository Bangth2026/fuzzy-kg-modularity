#!/usr/bin/env python3
"""Version 4.0.0. Evaluation of the cut audit of Algorithm 2 (step 3) against external groupings,
on real topologies with the observed memberships and with memberships tilted toward the grouping.

Graphs with a grouping of their nodes fixed independently of the graph: Junyi (40 topics), Eedi
(level-2 subject), Algebra 2005 and Bridge to Algebra 2006 (curriculum unit), EdNet (TOEIC part),
ASSISTments 2009 and 2017 item level (skill tag). For every graph and every candidate cut (the
support graph, quantile cuts at 0.25, 0.50, 0.75 and absolute cuts at 0.3, 0.4, 0.5), over ten
optimiser seeds:
  audit    Q_F drop, Q_F(Pi_F) - Q_F(Pi_cut), and the rank-coded Q_F drop (Lemma 2), each read
           against the no-cut (support) value: a cut is flagged when its mean exceeds the no-cut
           mean by more than twice the larger of the two seed SDs (Algorithm 2, step 3);
  truth    ARI of each partition with the grouping (nodes that carry a group), and the ARI loss
           of the cut, ARI(Pi_F) - ARI(Pi_cut).
Tilt: memberships re-assigned by the rank of Z + s * 1[edge internal to the grouping] (marginal
kept exactly, as in the power calibration of Algorithm 1), s in {0 (observed), 0.5, 1, 2, 4}; s = 0
keeps the observed memberships.
Writes results/v4/protocol_external.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.nulltest import tilt_memberships  # noqa: E402

OUT = "results/v4/protocol_external.json"
GRAPHS = ["junyi_transition", "junyi_copractice", "eedi_transition", "eedi_copractice", "algebra2005_transition",
          "algebra2005_copractice", "bridge2006_transition", "bridge2006_copractice", "ednet_transition", "ednet_copractice",
          "assist2009_item", "assist2017_item", "assist2017_item_transition"]
TILTS = (0.0, 0.5, 1.0, 2.0, 4.0)
SEEDS = range(10)


def groups_for(name):
    lab = pd.read_csv(f"data/graphs/{name}_nodes.csv").label.astype(str).tolist()
    if name.startswith("junyi"):
        ex = pd.read_csv("data/raw/junyi/junyi/junyi_Exercise_table.csv", encoding="utf-8-sig")
        g = dict(zip(ex["name"].astype(str), ex["topic"].astype(str)))
    elif name.startswith(("eedi", "algebra2005", "bridge2006", "ednet")):
        g = json.load(open(f"data/graphs/{name.split('_')[0]}_groups.json"))
    elif name.startswith("assist2009_item"):
        d = pd.read_csv("data/raw/assist2009/skill_builder_data.csv", encoding="latin-1", usecols=["problem_id", "skill_id"], low_memory=False).dropna()
        g = d.groupby("problem_id").skill_id.agg(lambda s: s.mode().iloc[0]).astype(int).astype(str)
        g = {str(k): v for k, v in g.items()}
        g.update({f"{float(k)}": v for k, v in list(g.items())})
    elif name.startswith("assist2017_item"):
        d = pd.read_csv("data/raw/assist2017/anonymized_full_release_competition_dataset.csv", usecols=["problemId", "skill"], low_memory=False).dropna()
        g = d.groupby("problemId").skill.agg(lambda s: s.mode().iloc[0])
        g = {str(k): str(v) for k, v in g.items()}
        g.update({f"{float(k)}": v for k, v in list(g.items())})
    return np.array([g.get(l, g.get(str(l).replace(".0", ""), None)) for l in lab], dtype=object)


def leiden(G, seed):
    import igraph as ig, leidenalg as la
    g = ig.Graph(n=G.n, edges=list(zip(G.src.tolist(), G.dst.tolist())), directed=True); g.es["weight"] = G.mu.tolist()
    p = la.RBConfigurationVertexPartition(g, weights="weight", resolution_parameter=1.0)
    o = la.Optimiser(); o.set_rng_seed(int(seed)); o.optimise_partition(p, n_iterations=-1)
    return np.asarray(p.membership)


def rank_code(mu, L=8):
    qs = np.unique(np.quantile(mu, np.arange(L) / L))
    return (mu[:, None] >= qs[None, :]).sum(1) / len(qs)


def audit_graph(G, grp):
    has = np.array([x is not None for x in grp]); gl = pd.factorize(pd.Series(grp[has]).astype(str))[0]
    Gr = FuzzyGraph(G.n, G.src, G.dst, rank_code(G.mu))
    cuts = {"support": None}
    for q in (0.25, 0.5, 0.75):
        cuts[f"q{q:.2f}"] = float(np.quantile(G.mu, q))
    for a in (0.3, 0.4, 0.5):
        cuts[f"abs{a}"] = a
    PF = [leiden(G, s) for s in SEEDS]; PR = [leiden(Gr, s) for s in SEEDS]
    QF = np.array([fuzzy_modularity(G, p) for p in PF]); QR = np.array([fuzzy_modularity(Gr, p) for p in PR])
    aF = np.array([ari(gl, p[has]) for p in PF])
    rows = {"Pi_F": {"ari_mean": float(aF.mean()), "ari_sd": float(aF.std(ddof=1))}}
    for c, a in cuts.items():
        H = FuzzyGraph(G.n, G.src, G.dst, np.ones_like(G.mu)) if a is None else G.alpha_cut(a)
        if H.m == 0:
            rows[c] = {"empty": True}; continue
        P = [leiden(H, s) for s in SEEDS]
        d = QF - np.array([fuzzy_modularity(G, p) for p in P]); dr = QR - np.array([fuzzy_modularity(Gr, p) for p in P])
        ac = np.array([ari(gl, p[has]) for p in P])
        kept = H.m / G.m; act = np.flatnonzero(np.bincount(H.src, minlength=G.n) + np.bincount(H.dst, minlength=G.n))
        rows[c] = {"alpha": a, "retained_edges": float(kept), "mean_out_degree": float(H.m / max(1, act.size)),
                   "qf_drop": float(d.mean()), "qf_drop_sd": float(d.std(ddof=1)), "rank_drop": float(dr.mean()), "rank_drop_sd": float(dr.std(ddof=1)),
                   "ari_mean": float(ac.mean()), "ari_sd": float(ac.std(ddof=1)), "ari_loss": float(aF.mean() - ac.mean())}
    b = rows["support"]
    for c, r in rows.items():
        if c in ("Pi_F", "support") or r.get("empty"):
            continue
        for k in ("qf_drop", "rank_drop"):
            r[f"flag_{k}"] = bool(r[k] - b[k] > 2 * max(r[k + "_sd"], b[k + "_sd"]))
        r["ari_loss_vs_support"] = float(b["ari_mean"] - r["ari_mean"])
        r["ari_loss_resolved"] = bool(r["ari_loss_vs_support"] > 2 * max(r["ari_sd"], b["ari_sd"]))
    return rows, {"nodes_with_group": int(has.sum()), "groups": int(gl.max() + 1)}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    names = sys.argv[1:] or GRAPHS
    for name in names:
        df = pd.read_csv(f"data/graphs/{name}_edges.csv"); G0 = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
        grp = groups_for(name)
        gi = pd.factorize(pd.Series(grp).astype(str))[0]
        internal = ((gi[G0.src] == gi[G0.dst]) & (grp[G0.src] != None) & (grp[G0.dst] != None)).astype(float)  # noqa: E711
        for s in TILTS:
            key = f"{name}/tilt{s:g}"
            if key in res:
                continue
            t0 = time.perf_counter()
            mu = G0.mu if s == 0 else tilt_memberships(G0, internal, s, np.random.default_rng(7))
            G = FuzzyGraph(G0.n, G0.src, G0.dst, mu)
            rows, info = audit_graph(G, grp)
            info.update({"tilt": s, "r_pb_internal": float(np.corrcoef(mu, internal)[0, 1]) if internal.std() > 0 else None,
                         "internal_share": float(internal.mean()), "m": int(G.m), "n": int(G.n), "time_s": time.perf_counter() - t0})
            res[key] = {"info": info, "rows": rows}; json.dump(res, open(OUT, "w"), indent=1)
            print(key, f"r_pb={info['r_pb_internal']}", "PiF ARI", round(rows["Pi_F"]["ari_mean"], 3),
                  {c: (round(r.get("qf_drop", 0), 3), r.get("flag_qf_drop"), round(r.get("ari_loss_vs_support", 0), 3), r.get("ari_loss_resolved"))
                   for c, r in rows.items() if c not in ("Pi_F",) and not r.get("empty")}, f"{info['time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
