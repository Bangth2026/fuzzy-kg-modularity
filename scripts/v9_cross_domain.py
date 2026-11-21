#!/usr/bin/env python3
"""Version 4.6.0. The path half of the audit on graded knowledge graphs outside education.

Two uncertain knowledge graphs released with UKGE (Chen et al., AAAI 2019) and their fixed
train/validation/test split (https://github.com/stasl0217/UKGE, data/):
  ppi5k   protein-protein interactions with STRING confidence scores (biomedical): 4,999
          proteins, seven interaction types, every interaction listed in both directions;
  cn15k   a 15,000-concept subgraph of ConceptNet with confidence weights (commonsense):
          36 relation types, directed.
Relation types are collapsed to one graded relation by the maximum confidence over the types of
an ordered pair, so mu_E(u, v) is the strongest evidence that u is linked to v; all three UKGE
splits are pooled, because the UKGE split is by triple and most test triples share their pair
with a training triple of another type.  As for the hidden MOOCCubeX edges of Section 8.4, 10%
of the pairs are held out (seed 2026; for ppi5k unordered pairs, both directions removed) and
scored from the graph of the remaining 90%, with their confidence as the graded target.

For every held-out pair (h, t) the score from h to t is computed under max-min, the product,
Lukasiewicz and Yager p = 5 (exact best paths: heap widest path for max-min, Dijkstra on the
additive generator otherwise), the hop distance on the support, and the two grade-preserving
models of fuzzykg.embeddings (on these larger graphs with dimension 64, at most 100 epochs and the
learning rate chosen from 0.003, 0.01 and 0.03 by validation error).  Reported: share reached; Spearman rho with the held-out
confidence (unreached pairs score 0); rho on the pairs every path operator reaches; AUC of
held-out pairs against five times as many random pairs that are not edges of the full graph;
cluster-bootstrap intervals over source nodes (200 resamples) for rho and for its difference to the hop
distance; the spread and hop count of widest paths.

Writes results/v9/cross_domain_paths.json; graphs for the community half are written to
data/graphs/<name>_edges.csv (full graph, all splits) when --write-graphs is given.
"""
from __future__ import annotations
import argparse, heapq, json, os, sys, time
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from numba import njit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.tnorms import generator  # noqa: E402

RAW = "data/raw/ukge"
OUT = "results/v9/cross_domain_paths.json"
OPS = {"min": None, "product": ("product", None), "lukasiewicz": ("lukasiewicz", None), "yager_5": ("yager", 5.0)}


def load_split(name):
    cols = ["h", "r", "t", "w"]
    parts = {s: pd.read_csv(f"{RAW}/{name}/{s}.tsv", sep="\t", header=None, names=cols) for s in ("train", "val", "test")}
    ent = sorted(set().union(*[set(p.h) | set(p.t) for p in parts.values()]))
    ix = {e: i for i, e in enumerate(ent)}
    for p in parts.values():
        p["u"] = p.h.map(ix); p["v"] = p.t.map(ix)
    return parts, len(ent)


def collapse(df):
    g = df.groupby(["u", "v"]).w.max().reset_index()
    return g.u.to_numpy(np.int64), g.v.to_numpy(np.int64), g.w.to_numpy(float)


@njit(cache=True)
def widest_from(indptr, indices, data, n, sources):
    """Max-min best-path strength and hop count of a widest path with fewest hops, from each source."""
    S = np.zeros((len(sources), n)); H = np.zeros((len(sources), n), dtype=np.int64)
    for i in range(len(sources)):
        s = sources[i]
        best = np.zeros(n); hops = np.zeros(n, dtype=np.int64); done = np.zeros(n, dtype=np.bool_)
        best[s] = 2.0
        heap = [(-2.0, np.int64(0), np.int64(s))]
        while len(heap) > 0:
            nb, hu, u = heapq.heappop(heap)
            if done[u]:
                continue
            done[u] = True
            bu = -nb
            for k in range(indptr[u], indptr[u + 1]):
                v = np.int64(indices[k])
                c = min(bu, data[k])
                if not done[v] and (c > best[v] or (c == best[v] and hu + 1 < hops[v])):
                    best[v] = c; hops[v] = hu + 1
                    heapq.heappush(heap, (-c, hu + 1, v))
        for v in range(n):
            if v != s:
                S[i, v] = best[v]; H[i, v] = hops[v] if best[v] > 0 else 0
    return S, H


def path_scores(n, src, dst, mu, qh, qt, chunk=500):
    """Scores of the query pairs (qh -> qt) under every operator and the hop distance."""
    A = csr_matrix((mu, (src, dst)), shape=(n, n)); A.sum_duplicates()
    U = csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    heads = np.unique(qh); pos = {h: i for i, h in enumerate(heads)}
    out = {k: np.zeros(len(qh)) for k in list(OPS) + ["hop_distance"]}
    whops = np.zeros(len(qh))
    timing = {k: 0.0 for k in out}
    for c in range(0, len(heads), chunk):
        hs = heads[c:c + chunk]; sel = np.where(np.isin(qh, hs))[0]
        row = np.array([pos[h] - c for h in qh[sel]]); col = qt[sel]
        t = time.perf_counter(); S, H = widest_from(A.indptr, A.indices, A.data, n, hs.astype(np.int64)); timing["min"] += time.perf_counter() - t
        out["min"][sel] = S[row, col]; whops[sel] = H[row, col]
        for k, spec in OPS.items():
            if spec is None:
                continue
            gen = generator(*spec); L = A.copy(); L.data = gen.g(A.data); L.data = np.where(L.data <= 0, 1e-300, L.data)
            t = time.perf_counter(); D = dijkstra(L, directed=True, indices=hs); timing[k] += time.perf_counter() - t
            d = D[row, col]
            out[k][sel] = np.where(np.isfinite(d), np.maximum(gen.pseudo_inverse(np.where(np.isfinite(d), d, 0.0)), 0.0), 0.0)
        t = time.perf_counter(); D = shortest_path(U, directed=True, unweighted=True, indices=hs); timing["hop_distance"] += time.perf_counter() - t
        d = D[row, col]; out["hop_distance"][sel] = np.where(np.isfinite(d), 1.0 / (1.0 + d), 0.0)
    return out, whops, timing


def node_cluster_weights(qh, qt, B, rng):
    """Cluster bootstrap over the source nodes of the held-out pairs: a pair enters with the
    multiplicity of its source.  (A two-way scheme over both endpoints, as used for the Junyi
    exercises, is biased here because most nodes occur in a single held-out pair.)"""
    nodes = np.unique(qh); pos = {x: i for i, x in enumerate(nodes)}
    pa = np.array([pos[x] for x in qh])
    for _ in range(B):
        m = rng.multinomial(nodes.size, np.full(nodes.size, 1 / nodes.size)); yield m[pa]


def wrho(x, y, w):
    k = w > 0
    return spearmanr(np.repeat(x[k], w[k]), np.repeat(y[k], w[k])).statistic


def run(name, B=200, models=("ukge", "gnn"), write_graphs=False, heldout=0.1):
    parts, n = load_split(name)
    full = pd.concat([parts["train"], parts["val"], parts["test"]]); fs, fd, fm = collapse(full)
    sym = name == "ppi5k"
    # held-out pairs: 10% of the pairs of the collapsed graph (unordered pairs, both directions, for ppi5k)
    hrng = np.random.default_rng(2026)
    if sym:
        lo = np.minimum(fs, fd); hi = np.maximum(fs, fd); key = lo * n + hi
        uk = np.unique(key); hide = set(hrng.choice(uk, int(round(heldout * len(uk))), replace=False).tolist())
        isout = np.array([k in hide for k in key.tolist()])
    else:
        isout = np.zeros(len(fs), bool); isout[hrng.choice(len(fs), int(round(heldout * len(fs))), replace=False)] = True
    src, dst, mu = fs[~isout], fd[~isout], fm[~isout]
    if write_graphs:
        pd.DataFrame({"src": fs, "dst": fd, "mu": fm}).to_csv(f"data/graphs/{name}_edges.csv", index=False)
    have = set((src * n + dst).tolist())
    te = pd.DataFrame({"u": fs[isout], "v": fd[isout], "w": fm[isout]})
    if sym:
        te = te[te.u < te.v]
    qh = te.u.to_numpy(np.int64); qt = te.v.to_numpy(np.int64); y = te.w.to_numpy(float)
    rng = np.random.default_rng(11)
    allk = set((fs * n + fd).tolist()); neg = []
    while len(neg) < 5 * len(qh):
        a, b = rng.integers(0, n, 2)
        if a != b and (a * n + b) not in allk and (b * n + a) not in allk:
            neg.append((a, b))
    neg = np.array(neg)
    G = FuzzyGraph(n, src, dst, mu)
    rec = {"graph": {"n": int(n), "m": int(len(src)), "density": float(len(src) / (n * (n - 1))),
                     "reciprocity": float(np.mean([(v * n + u) in have for u, v in zip(src, dst)])),
                     "mean_out_degree": float(len(src) / n), "mu_mean": float(mu.mean()), "mu_sd": float(mu.std())},
           "n_heldout_pairs": int(len(qh)), "n_negatives": int(len(neg)), "scores": {}, "models": {}}
    print(name, rec["graph"], "held-out", len(qh), flush=True)
    X, whops, timing = path_scores(n, src, dst, mu, qh, qt)
    XN, _, _ = path_scores(n, src, dst, mu, neg[:, 0], neg[:, 1])
    rec["path_seconds"] = {k: round(v, 2) for k, v in timing.items()}
    for mdl in models:
        from fuzzykg.embeddings import GradedLinkModel
        # reduced selection grid on these larger graphs: lr in {0.003, 0.01, 0.03}, dim 64, at most 100 epochs
        t = time.perf_counter(); E = GradedLinkModel(model=mdl, dim=64, epochs=100).fit(G, grid=(("lr", (0.003, 0.01, 0.03)),))
        X[mdl] = E.score(qh, qt); XN[mdl] = E.score(neg[:, 0], neg[:, 1])
        rec["models"][mdl] = {"lr": E.cfg.lr, "dim": E.cfg.dim, "epochs": E.best_epoch, "val_mse": E.val_mse,
                              "rho_per_seed": [float(spearmanr(s, y).statistic) for s in E.score_each(qh, qt)],
                              "seconds": round(time.perf_counter() - t, 1)}
        print(name, mdl, rec["models"][mdl], flush=True)
    W = list(node_cluster_weights(qh, qt, B, np.random.default_rng(31)))
    bs = {k: np.array([wrho(x, y, w) for w in W]) for k, x in X.items()}
    every = np.all([X[k] > 0 for k in OPS], axis=0)
    lab = np.r_[np.ones(len(qh)), np.zeros(len(neg))]
    for k, x in X.items():
        r = {"reached": float((x > 0).mean()), "rho": float(spearmanr(x, y).statistic),
             "rho_ci": [float(np.nanpercentile(bs[k], 2.5)), float(np.nanpercentile(bs[k], 97.5))],
             "rho_reached_by_all": float(spearmanr(x[every], y[every]).statistic) if every.sum() > 10 else None,
             "auc": float(roc_auc_score(lab, np.r_[x, XN[k]]))}
        if k != "hop_distance":
            d = bs[k] - bs["hop_distance"]
            r["minus_hop"] = float(r["rho"] - spearmanr(X["hop_distance"], y).statistic)
            r["minus_hop_ci"] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        if k != "product":
            d = bs[k] - bs["product"]
            r["minus_product_ci"] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        rec["scores"][k] = r
    reached = X["min"] > 0
    rec["maxmin_diagnostics"] = {"pairs_reached_by_all": int(every.sum()),
                                 "widest_sd": float(X["min"][reached].std()), "membership_sd": float(mu.std()),
                                 "widest_mean_hops": float(whops[reached].mean()), "heldout_conf_sd": float(y.std()),
                                 "widest_distinct_values": int(np.unique(np.round(X["min"][reached], 12)).size)}
    print(name, {k: (round(v["reached"], 2), round(v["rho"], 3), round(v["auc"], 3)) for k, v in rec["scores"].items()},
          rec["maxmin_diagnostics"], flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="*", default=["ppi5k", "cn15k"])
    ap.add_argument("--write-graphs", action="store_true"); ap.add_argument("--no-models", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for nm in a.names:
        res[nm] = run(nm, models=() if a.no_models else ("ukge", "gnn"), write_graphs=a.write_graphs)
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
