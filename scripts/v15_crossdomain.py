#!/usr/bin/env python3
"""Release 5.0.0. The path results of Section 5 and the path value of Section 6 on the two graded
knowledge graphs outside education (PPI5k, CN15k), with the held-out protocol of
scripts/v9_cross_domain.py: relation types collapsed by the maximum confidence of an ordered pair,
10% of the pairs held out (seed 2026; unordered pairs for PPI5k) and scored from the rest, their
confidence the graded target. Raw file: data/raw/ukge/<name>/data.tsv (all UKGE splits pooled).

Reported: Spearman rho with the held-out confidence (unreached pairs below every reached pair) of
the hop distance, the product, max-min and the hop-penalised score for lam in {0, 0.1, ..., 1};
lam* cross-fitted over 100 random halvings of the source nodes; the path value of the grades
(product against the hop distance, with five global and five row permutations); the connectivity
level alpha_c of Proposition 3 from 1,000 random sources.
Writes results/v15/crossdomain.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.nulltest import permute_global, permute_rows  # noqa: E402
from v9_cross_domain import widest_from  # noqa: E402

OUT = "results/v15/crossdomain.json"
LAMS = np.round(np.linspace(0, 1, 11), 2)


def load(name, heldout=0.1):
    d = pd.read_csv(f"data/raw/ukge/{name}/data.tsv", sep="\t", header=None, names=["h", "r", "t", "w"])
    ent = sorted(set(d.h) | set(d.t)); ix = {e: i for i, e in enumerate(ent)}; n = len(ent)
    d["u"] = d.h.map(ix); d["v"] = d.t.map(ix)
    g = d.groupby(["u", "v"]).w.max().reset_index()
    g = g[g.u != g.v]
    fs, fd, fm = g.u.to_numpy(np.int64), g.v.to_numpy(np.int64), g.w.to_numpy(float)
    fm = np.clip(fm, 1e-6, 1.0)
    hrng = np.random.default_rng(2026)
    if name == "ppi5k":
        key = np.minimum(fs, fd) * n + np.maximum(fs, fd)
        uk = np.unique(key); hide = set(hrng.choice(uk, int(round(heldout * len(uk))), replace=False).tolist())
        isout = np.array([k in hide for k in key.tolist()])
    else:
        isout = np.zeros(len(fs), bool); isout[hrng.choice(len(fs), int(round(heldout * len(fs))), replace=False)] = True
    te = pd.DataFrame({"u": fs[isout], "v": fd[isout], "w": fm[isout]})
    if name == "ppi5k":
        te = te[te.u < te.v]
    return n, FuzzyGraph(n, fs[~isout], fd[~isout], fm[~isout]), te.u.to_numpy(np.int64), te.v.to_numpy(np.int64), te.w.to_numpy(float)


def lengths(G, qh, qt, lam=None, chunk=400):
    """Shortest-path length of each query pair; lam None: hop count."""
    if lam is None:
        w = np.ones(G.m)
    else:
        w = lam + (1 - lam) * (-np.log(G.mu)); w = np.where(w <= 0, 1e-12, w)
    A = csr_matrix((w, (G.src, G.dst)), shape=(G.n, G.n))
    heads = np.unique(qh); pos = {h: i for i, h in enumerate(heads)}
    out = np.full(len(qh), np.inf)
    for c in range(0, len(heads), chunk):
        hs = heads[c:c + chunk]; sel = np.where(np.isin(qh, hs))[0]
        D = dijkstra(A, directed=True, indices=hs)
        out[sel] = D[np.array([pos[h] - c for h in qh[sel]]), qt[sel]]
    return out


def score(d):
    s = -d.copy(); fin = np.isfinite(d); s[~fin] = (s[fin].min() - 1) if fin.any() else -1; return s


def widest_scores(G, qh, qt, chunk=400):
    A = csr_matrix((G.mu, (G.src, G.dst)), shape=(G.n, G.n)); A.sum_duplicates()
    heads = np.unique(qh); pos = {h: i for i, h in enumerate(heads)}; out = np.zeros(len(qh))
    for c in range(0, len(heads), chunk):
        hs = heads[c:c + chunk]; sel = np.where(np.isin(qh, hs))[0]
        S, _ = widest_from(A.indptr, A.indices, A.data, G.n, hs.astype(np.int64))
        out[sel] = S[np.array([pos[h] - c for h in qh[sel]]), qt[sel]]
    return out


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in ("ppi5k", "cn15k"):
        if name in res:
            continue
        t0 = time.perf_counter()
        n, G, qh, qt, y = load(name)
        rho = lambda x: float(spearmanr(x, y).statistic)
        hop = score(lengths(G, qh, qt)); prod = score(lengths(G, qh, qt, 0.0))
        L = {f"{l:.1f}": score(lengths(G, qh, qt, float(l))) for l in LAMS}
        mm = widest_scores(G, qh, qt)
        rec = {"n": int(n), "m": int(G.m), "pairs": int(len(y)), "rho_hop": rho(hop), "rho_product": rho(prod), "rho_maxmin": rho(mm),
               "rho_by_lam": {k: rho(v) for k, v in L.items()}}
        rng = np.random.default_rng(5)
        heads = np.unique(qh); sel_l, gain = [], []
        for _ in range(100):
            half = set(heads[rng.random(heads.size) < 0.5].tolist()); inA = np.array([h in half for h in qh]); inB = ~inA
            rA = [spearmanr(L[f"{l:.1f}"][inA], y[inA]).statistic for l in LAMS]; ls = LAMS[int(np.nanargmax(rA))]
            sel_l.append(float(ls)); gain.append(float(spearmanr(L[f"{ls:.1f}"][inB], y[inB]).statistic - spearmanr(L["1.0"][inB], y[inB]).statistic))
        rec["crossfit_lam"] = {"median": float(np.median(sel_l)), "share_eq0": float(np.mean(np.isclose(sel_l, 0))),
                               "share_eq1": float(np.mean(np.isclose(sel_l, 1))), "values": sel_l,
                               "gain_over_hop_heldout": [float(np.mean(gain)), float(np.percentile(gain, 2.5)), float(np.percentile(gain, 97.5))]}
        q = {"support": rec["rho_hop"], "observed": rec["rho_product"]}
        for kind, fn in (("global", permute_global), ("row", permute_rows)):
            q[kind] = float(np.mean([rho(score(lengths(fn(G, rng), qh, qt, 0.0))) for _ in range(5)]))
        rec["value_path"] = {**q, "V_total": q["observed"] - q["support"], "V_marg": q["global"] - q["support"],
                             "V_strength": q["row"] - q["global"], "V_resid": q["observed"] - q["row"]}
        srcs = rng.choice(n, min(1000, n), replace=False).astype(np.int64)
        A = csr_matrix((G.mu, (G.src, G.dst)), shape=(n, n)); A.sum_duplicates()
        S, _ = widest_from(A.indptr, A.indices, A.data, n, srcs)
        S[np.arange(len(srcs)), srcs] = 0
        v = S[S > 0]; a = float(np.quantile(v, 0.05, method="lower"))
        rec["connectivity"] = {"alpha_c": a, "usable_share": float((G.mu >= a).mean()), "alpha_hard": float(v.min()), "widest_sd": float(v.std())}
        rec["time_s"] = time.perf_counter() - t0
        res[name] = rec
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("rho_by_lam", "crossfit_lam")}, flush=True)
        print("  lam", {k: round(v, 3) for k, v in rec["rho_by_lam"].items()}, rec["crossfit_lam"]["median"], rec["crossfit_lam"]["gain_over_hop_heldout"], flush=True)


if __name__ == "__main__":
    main()
