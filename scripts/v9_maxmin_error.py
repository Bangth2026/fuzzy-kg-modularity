#!/usr/bin/env python3
"""Version 4.6.0. Why max-min erases the graded evidence on the dense Junyi co-practice graph.

The max-min strength from a to b is the largest alpha at which b is reachable from a in the
alpha-cut.  On a graph whose cuts stay strongly connected up to a high level, almost every pair
gets that level, whatever its own edge says.  The script measures this and separates the
candidate causes (near-equal reciprocal memberships, density, hub paths) by counterfactual graphs.

  (a) reciprocity: |mu(a,b) - mu(b,a)| over reciprocal pairs, their correlation, and how often
      the stronger direction of a knowledge-map link is the expert direction;
  (b) percolation: the share of ordered pairs reachable in the alpha-cut, for alpha on a grid,
      and the level alpha_50 at which it falls below one half; the share of annotated pairs
      whose widest-path strength lies within 0.02 of the most common value;
  (c) widest paths of the annotated pairs: share stronger than the direct edge, hops, SD,
      number of distinct values, and how concentrated they are around their most common value;
  (d) counterfactual graphs, each scored by max-min and the product against the crowd score
      (Spearman rho on the 1,954 pairs, two-way cluster-bootstrap interval over exercises):
        oriented     only the stronger direction of every reciprocal pair (reciprocity 0)
        top-q        the edges at or above the q-quantile of mu_E, q in 0.5, 0.75, 0.9, 0.95
        no hubs      the 10% of exercises with the largest out-strength removed as path
                     interiors (paths may start or end at them but not pass through them)
        L <= 2       paths of at most two hops
Writes results/v9/maxmin_error.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.paths_fast import strengths  # noqa: E402
from data.loaders import load_junyi_expert  # noqa: E402

OUT = "results/v9/maxmin_error.json"


def load(name):
    e = pd.read_csv(f"data/graphs/{name}_edges.csv"); nd = pd.read_csv(f"data/graphs/{name}_nodes.csv")
    return FuzzyGraph.from_edges(zip(e.src, e.dst, e.mu), n=len(nd)), nd.label.astype(str).tolist()


def reach_share(n, src, dst):
    if len(src) == 0:
        return 0.0, 0
    A = csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    k, lab = connected_components(A, directed=True, connection="strong")
    # ordered pairs (a, b), a != b, with b reachable from a: exact by BFS on the condensation is
    # unnecessary at this size; use boolean matrix closure via scipy shortest paths
    from scipy.sparse.csgraph import shortest_path
    D = shortest_path(A, directed=True, unweighted=True)
    R = np.isfinite(D); np.fill_diagonal(R, False)
    return float(R.sum() / (n * (n - 1))), int(np.bincount(lab).max())


def main(B=500):
    G, labels = load("junyi_copractice"); n = G.n; idx = {l: i for i, l in enumerate(labels)}
    ex, ann = load_junyi_expert()
    sel = ann[ann.Exercise_A.astype(str).isin(idx) & ann.Exercise_B.astype(str).isin(idx)]
    A_ = sel.Exercise_A.astype(str).to_numpy(); B_ = sel.Exercise_B.astype(str).to_numpy(); y = sel.Prerequisite_avg.to_numpy()
    ia = np.array([idx[a] for a in A_]); ib = np.array([idx[b] for b in B_])
    rng = np.random.default_rng(31)
    exs = np.unique(np.r_[A_, B_]); pos = {x: i for i, x in enumerate(exs)}
    pa = np.array([pos[x] for x in A_]); pb = np.array([pos[x] for x in B_])
    Wb = []
    for _ in range(B):
        m = rng.multinomial(exs.size, np.full(exs.size, 1 / exs.size)); Wb.append(m[pa] * m[pb])

    def rho_ci(x):
        r = float(spearmanr(x, y).statistic)
        bs = []
        for w in Wb:
            k = w > 0; bs.append(spearmanr(np.repeat(x[k], w[k]), np.repeat(y[k], w[k])).statistic)
        return {"rho": r, "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}

    W = np.zeros((n, n)); W[G.src, G.dst] = G.mu
    res = {"graph": G.summary()}
    # (a) reciprocity
    rec = (W > 0) & (W.T > 0); iu = np.triu_indices(n, 1); m2 = rec[iu]
    a, b = W[iu][m2], W.T[iu][m2]
    mp = set()
    for nm, p in zip(ex.name.astype(str), ex.prerequisites):
        if isinstance(p, str):
            for q in p.split(","):
                if q.strip() in idx and nm in idx:
                    mp.add((idx[q.strip()], idx[nm]))
    lk = np.array(sorted(mp))
    both = (W[lk[:, 0], lk[:, 1]] > 0) & (W[lk[:, 1], lk[:, 0]] > 0)
    res["reciprocity"] = {"reciprocal_pairs": int(m2.sum()), "abs_diff_mean": float(np.abs(a - b).mean()),
                          "abs_diff_median": float(np.median(np.abs(a - b))), "abs_diff_q90": float(np.quantile(np.abs(a - b), 0.9)),
                          "corr_forward_backward": float(np.corrcoef(a, b)[0, 1]), "membership_sd": float(G.mu.std()),
                          "map_links_reciprocal": int(both.sum()),
                          "map_links_expert_direction_stronger": float(np.mean(W[lk[both, 0], lk[both, 1]] > W[lk[both, 1], lk[both, 0]]))}
    # (b) percolation of the alpha-cuts
    grid = np.round(np.arange(0.20, 0.81, 0.05), 2); perc = []
    for al in grid:
        k = G.mu >= al; s, gs = reach_share(n, G.src[k], G.dst[k])
        perc.append({"alpha": float(al), "edges": int(k.sum()), "reachable_share": s, "largest_scc": gs})
    res["percolation"] = perc
    # (c) widest paths of the annotated pairs
    S, H = strengths(G, "min"); S = np.maximum(S, 0)
    w = S[ib, ia]; d = W[ib, ia]; h = H[ib, ia]
    vals, cnt = np.unique(np.round(w, 6), return_counts=True); mode = vals[cnt.argmax()]
    outs = np.zeros(n); np.add.at(outs, G.src, G.mu)
    res["widest"] = {"stronger_than_direct": float(np.mean(w > d + 1e-12)), "mean_hops": float(h.mean()),
                     "sd_widest": float(w.std()), "sd_direct": float(d.std()), "distinct_values": int(vals.size),
                     "mode": float(mode), "share_within_0.02_of_mode": float(np.mean(np.abs(w - mode) <= 0.02)),
                     "quantile_of_mode_in_mu": float(np.mean(G.mu < mode)),
                     "rho_widest_with_direct": float(spearmanr(w, d).statistic)}
    # (d) counterfactual graphs
    cf = {}

    def score(Gc, label, max_hops=None):
        Sm = np.maximum(strengths(Gc, "min", None, max_hops)[0], 0)[ib, ia]
        Sp = np.maximum(strengths(Gc, "product", None, max_hops)[0], 0)[ib, ia]
        cf[label] = {"edges": int(Gc.m), "mean_out_degree": float(Gc.m / n), "reached": float((Sm > 0).mean()),
                     "max_min": rho_ci(Sm), "product": rho_ci(Sp), "direct_edge": rho_ci(Wc(Gc)[ib, ia]),
                     "sd_maxmin": float(Sm.std())}
        print(label, {k: (v["rho"] if isinstance(v, dict) else v) for k, v in cf[label].items()}, flush=True)

    def Wc(Gc):
        M = np.zeros((n, n)); M[Gc.src, Gc.dst] = Gc.mu; return M

    score(G, "observed")
    keep = W[G.src, G.dst] > W[G.dst, G.src]          # the stronger direction (ties drop both)
    score(FuzzyGraph(n, G.src[keep], G.dst[keep], G.mu[keep]), "oriented")
    for q in (0.5, 0.75, 0.9, 0.95):
        a0 = float(np.quantile(G.mu, q)); k = G.mu >= a0
        score(FuzzyGraph(n, G.src[k], G.dst[k], G.mu[k]), f"top_{q}")
    hubs = set(np.argsort(-outs)[: int(round(0.1 * n))].tolist())
    # remove hubs as interiors: split each hub h into h_out (keeps out-edges) and h_in (keeps in-edges)
    src2, dst2 = G.src.copy(), G.dst.copy()
    hub_in = {h: n + i for i, h in enumerate(sorted(hubs))}
    dst2 = np.array([hub_in.get(v, v) for v in dst2])
    n2 = n + len(hub_in); G2 = FuzzyGraph(n2, src2, dst2, G.mu)
    S2m = np.maximum(strengths(G2, "min")[0], 0); S2p = np.maximum(strengths(G2, "product")[0], 0)
    tgt = np.array([hub_in.get(v, v) for v in ia])
    xm = S2m[ib, tgt]; xp = S2p[ib, tgt]
    cf["no_hub_interiors"] = {"hubs": len(hubs), "max_min": rho_ci(xm), "product": rho_ci(xp), "reached": float((xm > 0).mean()),
                              "sd_maxmin": float(xm.std())}
    print("no hubs", cf["no_hub_interiors"]["max_min"], cf["no_hub_interiors"]["product"], flush=True)
    score(G, "L<=2", max_hops=2)
    res["counterfactuals"] = cf
    # the same diagnostics on the sparse transition graph, for contrast
    Gt, lt = load("junyi_transition"); it = {l: i for i, l in enumerate(lt)}
    st = ann[ann.Exercise_A.astype(str).isin(it) & ann.Exercise_B.astype(str).isin(it)]
    ta = st.Exercise_A.astype(str).map(it).to_numpy(); tb = st.Exercise_B.astype(str).map(it).to_numpy()
    St, Ht = strengths(Gt, "min"); wt = np.maximum(St, 0)[tb, ta]; r = wt > 0
    Wt = np.zeros((Gt.n, Gt.n)); Wt[Gt.src, Gt.dst] = Gt.mu
    res["transition_contrast"] = {"reached": float(r.mean()), "sd_widest_reached": float(wt[r].std()),
                                  "distinct_values": int(np.unique(np.round(wt[r], 6)).size), "mean_hops_reached": float(Ht[tb, ta][r].mean()),
                                  "stronger_than_direct": float(np.mean(wt > Wt[tb, ta] + 1e-12)),
                                  "percolation": [{"alpha": float(al), "reachable_share": reach_share(Gt.n, Gt.src[Gt.mu >= al], Gt.dst[Gt.mu >= al])[0]}
                                                  for al in (0.2, 0.3, 0.4, 0.5)]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("reciprocity", "widest")}, indent=1))
    print([(p["alpha"], round(p["reachable_share"], 3)) for p in perc])


if __name__ == "__main__":
    main()
