#!/usr/bin/env python3
"""Release 5.0.0. The unit-robust cut audit on the 23 real graphs (Sections 4 and 8).

Reads the partitions of scripts/v15_partitions.py. For every graph, resolution gamma and quantile
cut (0.25, 0.50, 0.75):
  excess_mu      E(1): mean Q_F(Pi_support) - mean Q_F(Pi_cut) on mu_E (seed means)
  decision_mu    flagged when E(1) exceeds twice the larger seed SD and 0.01, else cut
  kappa_star     largest slope ratio of a piecewise-linear increasing rescaling (knots at the
                 eight quantiles) under which the decision holds for every rescaling
  simplex        range of the excess over every rescaling of that family (Proposition 2)
  indicator      excess on the cut's own indicator (the limit of the family at the cut level)
Then per graph and cut: the interval of gamma around 1 on which decision_mu does not change, and
on graphs whose nodes carry a grouping fixed independently of the graph, the ARI of each partition
with it and the permutation reference of step 4 (row-permuted memberships, from the cache).
Writes results/v15/simplex_real.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.simplex import CutFamily, kappa_star, simplex_extremes  # noqa: E402
from v15_partitions import GRAPHS, GRAPHS13, GAMMAS, load  # noqa: E402

OUT = "results/v15/simplex_real.json"
CUTS = ("q0.25", "q0.50", "q0.75")
DELTAS = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)


def group(name):
    if name.startswith("mooccubex"):
        return "expert"
    return "transition" if name.endswith("transition") else "co-practice"


def groups_for(name):
    lab = pd.read_csv(f"data/graphs/{name}_nodes.csv").label.astype(str).tolist()
    if name.startswith("junyi"):
        ex = pd.read_csv("data/raw/junyi/junyi/junyi_Exercise_table.csv", encoding="utf-8-sig")
        g = dict(zip(ex["name"].astype(str), ex["topic"].astype(str)))
    elif name.startswith(("eedi", "algebra2005", "bridge2006", "ednet")):
        g = json.load(open(f"data/graphs/{name.split('_')[0]}_groups.json"))
    elif name == "assist2009_item":
        d = pd.read_csv("data/raw/assist2009/skill_builder_data.csv", encoding="latin-1", usecols=["problem_id", "skill_id"], low_memory=False).dropna()
        g = d.groupby("problem_id").skill_id.agg(lambda s: s.mode().iloc[0]).astype(int).astype(str)
        g = {str(k): v for k, v in g.items()}; g.update({f"{float(k)}": v for k, v in list(g.items())})
    elif name.startswith("assist2017_item"):
        d = pd.read_csv("data/raw/assist2017/anonymized_full_release_competition_dataset.csv", usecols=["problemId", "skill"], low_memory=False).dropna()
        g = d.groupby("problemId").skill.agg(lambda s: s.mode().iloc[0])
        g = {str(k): str(v) for k, v in g.items()}; g.update({f"{float(k)}": v for k, v in list(g.items())})
    else:
        return None
    return np.array([g.get(l, g.get(str(l).replace(".0", ""), None)) for l in lab], dtype=object)


def parts_at(store, which, g):
    seeds = range(20) if g == 1.0 else range(10)
    return [store[f"{which}|{g:g}|{s}"] for s in seeds]


def main(names):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"graphs": {}}
    v12 = json.load(open("results/v12/unit_rule.json"))["graphs"]
    for name in names:
        path = f"results/v15/partitions/{name}.npz"
        if not os.path.exists(path):
            print("missing partitions", name); continue
        store = dict(np.load(path))
        need = [f"{w}|{g:g}|0" for g in GAMMAS for w in ("support", *CUTS)]
        if not all(k in store for k in need):
            print("incomplete partitions", name); continue
        G = load(name)
        rec = {"group": group(name), "n": int(G.n), "m": int(G.m), "gammas": {}}
        for g in GAMMAS:
            fam = CutFamily(G, {w: parts_at(store, w, g) for w in ("support", *CUTS)}, gamma=g)
            row = {}
            for c in CUTS:
                ks = kappa_star(fam, c, kmax=1000.0) if g == 1.0 else None
                ex1, mg1 = fam.margin(c, np.ones((1, fam.J)))
                r = {"excess_mu": float(ex1[0]), "margin_mu": float(mg1[0]), "decision_mu": "flagged" if mg1[0] > 0 else "cut"}
                r["by_delta"] = {}
                for dl in DELTAS:
                    _, mgd = fam.margin(c, np.ones((1, fam.J)), dl)
                    r["by_delta"][f"{dl:g}"] = {"decision": "flagged" if mgd[0] > 0 else "cut", "margin": float(mgd[0])}
                    if g == 1.0:
                        r["by_delta"][f"{dl:g}"]["kappa_star"] = kappa_star(fam, c, min_excess=dl, kmax=1000.0)["kappa_star"]
                if g == 1.0:
                    r["kappa_star"] = ks["kappa_star"]
                    ext = simplex_extremes(fam, c)
                    r["simplex"] = {k: ext[k] for k in ("excess_min", "excess_max", "margin_min", "margin_max", "vertex_excess")}
                    q = float(c[1:])
                    H = G.alpha_cut(float(np.quantile(G.mu, q)))
                    r["excess_at_indicator"] = float(np.mean([fuzzy_modularity(H, p) for p in parts_at(store, "support", g)])
                                                     - np.mean([fuzzy_modularity(H, p) for p in parts_at(store, c, g)]))
                    r["mean_out_degree_cut"] = float(H.m / max(1, len(np.unique(np.concatenate([H.src, H.dst])))))
                    r["retained"] = float(H.m / G.m)
                row[c] = r
            rec["gammas"][f"{g:g}"] = row
        # gamma interval around 1 on which the decision on mu_E holds
        gs = list(GAMMAS); i1 = gs.index(1.0)
        rec["gamma_interval"] = {}
        for c in CUTS:
            d1 = rec["gammas"]["1"][c]["decision_mu"]
            lo = i1
            while lo - 1 >= 0 and rec["gammas"][f"{gs[lo - 1]:g}"][c]["decision_mu"] == d1:
                lo -= 1
            hi = i1
            while hi + 1 < len(gs) and rec["gammas"][f"{gs[hi + 1]:g}"][c]["decision_mu"] == d1:
                hi += 1
            rec["gamma_interval"][c] = [gs[lo], gs[hi]]
        if name in v12:
            rec["v12_decisions"] = {c: v12[name]["cuts"][c]["decision"] for c in CUTS}
        # external grouping
        gr = groups_for(name)
        if gr is not None:
            has = np.array([x is not None for x in gr])
            gl = pd.factorize(pd.Series(gr[has]).astype(str))[0]
            A = {w: [ari(gl, p[has]) for p in parts_at(store, w, 1.0)] for w in ("support", *CUTS, "F")}
            obs3 = max(parts_at(store, "F", 1.0)[:3], key=lambda p: fuzzy_modularity(G, p))
            ref = [ari(gl, store[f"Frow{d}|1|best3"][has]) for d in range(10)]
            refg = [ari(gl, store[f"Fglob{d}|1|best3"][has]) for d in range(10)]
            rec["grouping"] = {"nodes_with_group": int(has.sum()), "groups": int(gl.max() + 1),
                               "ari": {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in A.items()},
                               "ari_F_best3": float(ari(gl, obs3[has])),
                               "row_perm": [float(np.mean(ref)), float(np.percentile(ref, 95))],
                               "glob_perm": [float(np.mean(refg)), float(np.percentile(refg, 95))],
                               "informative": bool(ari(gl, obs3[has]) > np.percentile(ref, 95))}
            for c in CUTS:
                loss = A["support"]; cut = A[c]
                d = float(np.mean(loss) - np.mean(cut)); sd = float(max(np.std(loss, ddof=1), np.std(cut, ddof=1)))
                rec["grouping"][c] = {"loss_vs_support": d, "sd": sd, "loss": bool(d > max(2 * sd, 0.02))}
        res["graphs"][name] = rec
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        print(name, {c: (rec["gammas"]["1"][c]["decision_mu"], round(rec["gammas"]["1"][c]["kappa_star"], 2), rec["gamma_interval"][c]) for c in CUTS}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or GRAPHS)
