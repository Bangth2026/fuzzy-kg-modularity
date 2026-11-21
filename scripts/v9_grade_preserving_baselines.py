#!/usr/bin/env python3
"""Version 4.6.0. Grade-preserving baselines against the expert criteria of Section 8.4.

The two models of fuzzykg.embeddings (a UKGE-style uncertain-KG embedding and a GCN encoder with
a bilinear decoder) are trained on each graph to regress its memberships, with lr and dimension
chosen by validation MSE on held-out edges of the graph itself and three refitted seeds averaged.
They score every ordered pair, so they have no reachability limit.

  Junyi transition (tau_E = 0.2, 1,873 pairs) and Junyi co-practice (1,954 pairs): Spearman rho
      and Kendall tau_b of the score of (B, A) with the crowd score that B is a prerequisite of A,
      knowledge-map AUC against five times as many random pairs linked in neither direction (the
      negative pairs of Section 8.4), two-way cluster-bootstrap intervals over exercises (500
      resamples, rng 31) for rho and for its difference to the hop distance, the product and the
      direct edge; the seed SD of rho over the three refits.
  MOOCCubeX full: five-fold hidden expert edges against twenty times as many random non-edges
      (the protocol of Section 8.4), AUC with the path operators recomputed on the same folds.

Writes results/v9/grade_preserving_baselines.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.embeddings import GradedLinkModel  # noqa: E402
from data.loaders import load_junyi_expert  # noqa: E402
from composition_robustness import mats  # noqa: E402

OUT = "results/v9/grade_preserving_baselines.json"
PATH_KEYS = ["direct_edge", "hop_baseline", "min", "product", "lukasiewicz", "yager_5"]
MODELS = ["ukge", "gnn"]


def load(name):
    e = pd.read_csv(f"data/graphs/{name}_edges.csv"); nd = pd.read_csv(f"data/graphs/{name}_nodes.csv")
    return FuzzyGraph.from_edges(zip(e.src, e.dst, e.mu), n=len(nd)), nd.label.astype(str).tolist()


def wstat(x, y, w, f):
    k = w > 0
    return f(np.repeat(x[k], w[k]), np.repeat(y[k], w[k])).statistic


def ci(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def junyi(name, B=500):
    G, labels = load(name); idx = {l: i for i, l in enumerate(labels)}
    ex, ann = load_junyi_expert()
    sel = ann[ann.Exercise_A.astype(str).isin(idx) & ann.Exercise_B.astype(str).isin(idx)]
    A = sel.Exercise_A.astype(str).to_numpy(); Bn = sel.Exercise_B.astype(str).to_numpy(); y = sel.Prerequisite_avg.to_numpy()
    ia = np.array([idx[a] for a in A]); ib = np.array([idx[b] for b in Bn])
    rng = np.random.default_rng(31)
    exs = np.unique(np.r_[A, Bn]); pos = {x: i for i, x in enumerate(exs)}
    pa = np.array([pos[x] for x in A]); pb = np.array([pos[x] for x in Bn])
    W = [rng.multinomial(exs.size, np.full(exs.size, 1 / exs.size)) for _ in range(B)]
    W = [m[pa] * m[pb] for m in W]
    mp = set()
    for n_, p in zip(ex.name.astype(str), ex.prerequisites):
        if isinstance(p, str):
            for q in p.split(","):
                if q.strip() in idx and n_ in idx:
                    mp.add((idx[q.strip()], idx[n_]))
    lk = np.array(sorted(mp)); nrng = np.random.default_rng(0); neg = []
    while len(neg) < 5 * len(lk):
        a, b = nrng.integers(0, G.n, 2)
        if a != b and (a, b) not in mp and (b, a) not in mp:
            neg.append((a, b))
    neg = np.array(neg)
    M, _ = mats(G)
    X = {k: M[k][ib, ia] for k in PATH_KEYS}
    MAP = {k: np.r_[M[k][lk[:, 0], lk[:, 1]], M[k][neg[:, 0], neg[:, 1]]] for k in PATH_KEYS}
    res = {"n_pairs": int(len(y)), "n_map_links": int(len(lk)), "graph": G.summary(), "models": {}, "scores": {}}
    seeds_rho = {}
    for mdl in MODELS:
        t = time.perf_counter(); E = GradedLinkModel(model=mdl).fit(G)
        each = E.score_each(ib, ia)
        X[mdl] = np.mean(each, axis=0)
        MAP[mdl] = np.r_[E.score(lk[:, 0], lk[:, 1]), E.score(neg[:, 0], neg[:, 1])]
        seeds_rho[mdl] = [float(spearmanr(s, y).statistic) for s in each]
        edge_fit = E.score(G.src, G.dst)
        res["models"][mdl] = {"lr": E.cfg.lr, "dim": E.cfg.dim, "epochs": E.best_epoch, "val_mse": E.val_mse,
                              "grid": E.grid_results, "fit_rho_edges": float(spearmanr(edge_fit, G.mu).statistic),
                              "seconds": round(time.perf_counter() - t, 1)}
        print(name, mdl, res["models"][mdl]["lr"], res["models"][mdl]["dim"], res["models"][mdl]["epochs"],
              "rho", round(float(spearmanr(X[mdl], y).statistic), 3), f"{res['models'][mdl]['seconds']}s", flush=True)
    lab = np.r_[np.ones(len(lk)), np.zeros(len(neg))]
    bs = {k: np.array([wstat(x, y, W[b], spearmanr) for b in range(B)]) for k, x in X.items()}
    for k, x in X.items():
        r = {"reached": float((x > 0).mean()), "rho": float(spearmanr(x, y).statistic), "rho_ci": ci(bs[k]),
             "tau_b": float(kendalltau(x, y).statistic), "map_auc": float(roc_auc_score(lab, MAP[k]))}
        for ref in ("hop_baseline", "product", "direct_edge"):
            if k != ref:
                r[f"minus_{ref}"] = float(r["rho"] - spearmanr(X[ref], y).statistic); r[f"minus_{ref}_ci"] = ci(bs[k] - bs[ref])
        if k in seeds_rho:
            r["rho_per_seed"] = seeds_rho[k]; r["rho_seed_sd"] = float(np.std(seeds_rho[k], ddof=1))
        res["scores"][k] = r
    return res


def mooccubex(folds=5, neg_per_pos=20):
    from fuzzykg.paths_fast import strengths
    e = pd.read_csv("data/graphs/mooccubex_cs_full_edges.csv"); nd = pd.read_csv("data/graphs/mooccubex_cs_full_nodes.csv")
    n = len(nd); src = e.src.to_numpy(); dst = e.dst.to_numpy(); mu = e.mu.to_numpy(); m = len(src)
    edges = set(zip(src.tolist(), dst.tolist()))
    rng = np.random.default_rng(7); perm = rng.permutation(m); parts = np.array_split(perm, folds)
    keys = ["hop_baseline", "min", "product", "lukasiewicz", "yager_5"] + MODELS
    P = {k: [] for k in keys}; N = {k: [] for k in keys}; info = {}
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
        for k in keys[:5]:
            P[k].extend(M[k][src[test], dst[test]].tolist()); N[k].extend(M[k][neg[:, 0], neg[:, 1]].tolist())
        for mdl in MODELS:
            E = GradedLinkModel(model=mdl).fit(G)
            P[mdl].extend(E.score(src[test], dst[test]).tolist()); N[mdl].extend(E.score(neg[:, 0], neg[:, 1]).tolist())
            info.setdefault(mdl, []).append({"lr": E.cfg.lr, "dim": E.cfg.dim, "epochs": E.best_epoch})
        print("mooccubex fold", f, flush=True)
    out = {"n_nodes": n, "n_edges": m, "folds": folds, "models": info, "scores": {}}
    for k in keys:
        p = np.asarray(P[k]); q = np.asarray(N[k]); lab = np.r_[np.ones(len(p)), np.zeros(len(q))]
        bt = []
        for _ in range(500):
            i = rng.integers(0, len(p), len(p)); j = rng.integers(0, len(q), len(q))
            bt.append(roc_auc_score(np.r_[np.ones(len(i)), np.zeros(len(j))], np.r_[p[i], q[j]]))
        out["scores"][k] = {"auc": float(roc_auc_score(lab, np.r_[p, q])), "auc_ci": ci(bt), "hidden_reached": float((p > 0).mean())}
    print("mooccubex", {k: round(v["auc"], 3) for k, v in out["scores"].items()}, flush=True)
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in sys.argv[1:] or ["junyi_transition", "mooccubex_full", "junyi_copractice"]:
        if name in res:
            continue
        res[name] = mooccubex() if name == "mooccubex_full" else junyi(name)
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
