#!/usr/bin/env python3
"""Version 4.8.0. The cut audit of Algorithm 2 (step 3) on the thirteen real graphs, with both
partitions of every drop re-drawn per seed, under three increasing rescalings of mu_E.

For every graph and seed s = 0..19:
  * Pi_F^phi[s]: Leiden on Q_F of the graph with memberships phi(mu_E), for phi in
    {identity, x^3, x^(1/3), 1-(1-x)^3};
  * Pi_rank[s]: Leiden on the rank-coded graph N_A/L (eight-level quantile grid; invariant to phi);
  * Pi_c[s]: crisp Leiden on the support (no cut) and on the quantile cuts q = 0.25, 0.50, 0.75,
    whose edge sets and partitions do not change under phi.
Stored per seed: Q_F^phi(Pi_F^phi[s]), Q_F^phi(Pi_c[s]), Q_F[N_A/L](Pi_rank[s]), Q_F[N_A/L](Pi_c[s]),
and, where a grouping of the nodes exists, ARI(Pi, grouping) for Pi_F (identity), Pi_rank and Pi_c.
The Q_F drop of seed s is Q_F(Pi_F[s]) - Q_F(Pi_c[s]) (paired by seed), so its SD contains the
dispersion of both partitions; the drop with Pi_F fixed at the best of seeds 0-4 (as in
results/v7/qf_drop_seed_sd.json) is kept for comparison.  The flagging rules are applied by
scripts/v11_flag_rules.py.  Writes results/v11/cut_audit_rescale.json (one entry per graph).
Usage: python scripts/v11_cut_audit_rescale.py [graph ...]
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden, rank_code  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

OUT = os.environ.get("V11_OUT", "results/v11/cut_audit_rescale.json")
GRAPHS = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2009_item", "assist2015_kc", "assist2017_kc",
          "assist2017_item", "junyi_copractice", "assist2009_kc_transition", "assist2015_kc_transition",
          "assist2017_kc_transition", "assist2017_item_transition", "junyi_transition"]
PHI = {"identity": lambda x: x, "cube": lambda x: x ** 3, "cuberoot": lambda x: x ** (1 / 3),
       "one_minus_cube": lambda x: 1 - (1 - x) ** 3}
CUTS = ("q0.25", "q0.50", "q0.75")
SEEDS = range(int(os.environ.get("V11_SEEDS", "20")))


def grouping(name, n):
    if not name.startswith("junyi"):
        return None
    lab = pd.read_csv(f"data/graphs/{name}_nodes.csv")
    topics = json.load(open("data/graphs/junyi_topics.json"))
    col = [c for c in lab.columns if c != "id"][0]
    g = np.array([None] * n, dtype=object)
    for i, l in zip(lab["id"], lab[col]):
        g[int(i)] = topics.get(str(l))
    return g


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for name in (sys.argv[1:] or GRAPHS):
        out_g = OUT.replace(".json", f"_{name}.json")
        if os.path.exists(out_g):
            continue
        t0 = time.perf_counter()
        df = pd.read_csv(f"data/graphs/{name}_edges.csv")
        G = FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))
        Gr = FuzzyGraph(G.n, G.src, G.dst, rank_code(G.mu))
        alphas = {c: float(np.quantile(G.mu, float(c[1:]))) for c in CUTS}
        H = {"support": FuzzyGraph(G.n, G.src, G.dst, np.ones_like(G.mu)), **{c: G.alpha_cut(a) for c, a in alphas.items()}}
        grp = grouping(name, G.n)
        has = None if grp is None else np.array([x is not None for x in grp])
        gl = None if grp is None else pd.factorize(pd.Series(grp[has]).astype(str))[0]
        rec = {"n": int(G.n), "m": int(G.m), "alpha": alphas,
               "retained_edges": {c: float(h.m / G.m) for c, h in H.items()},
               "mean_out_degree": {c: float(h.m / max(1, len(np.unique(np.r_[h.src, h.dst])))) for c, h in H.items()},
               "seeds": len(SEEDS), "phi": {}, "rank": {}, "ari": {}}
        Pc = {c: [_leiden(h, s) for s in SEEDS] for c, h in H.items()}
        Pr = [_leiden(Gr, s) for s in SEEDS]
        rec["rank"] = {"QF_opt": [float(fuzzy_modularity(Gr, p)) for p in Pr],
                       "Q_cut": {c: [float(fuzzy_modularity(Gr, p)) for p in P] for c, P in Pc.items()}}
        PF_id = None
        for k, f in PHI.items():
            Gp = FuzzyGraph(G.n, G.src, G.dst, np.clip(f(G.mu), 1e-12, None))
            PF = [_leiden(Gp, s) for s in SEEDS]
            if k == "identity":
                PF_id = PF
            rec["phi"][k] = {"QF_opt": [float(fuzzy_modularity(Gp, p)) for p in PF],
                             "Q_cut": {c: [float(fuzzy_modularity(Gp, p)) for p in P] for c, P in Pc.items()}}
        if grp is not None:
            rec["ari"] = {"PiF": [float(ari(gl, p[has])) for p in PF_id], "Pi_rank": [float(ari(gl, p[has])) for p in Pr],
                          **{c: [float(ari(gl, p[has])) for p in P] for c, P in Pc.items()}}
        rec["time_s"] = time.perf_counter() - t0
        json.dump(rec, open(out_g, "w"))
        d = {c: round(float(np.mean(np.array(rec["phi"]["identity"]["QF_opt"]) - np.array(rec["phi"]["identity"]["Q_cut"][c]))), 4) for c in H}
        print(name, d, round(rec["time_s"]), flush=True)


if __name__ == "__main__":
    main()
