#!/usr/bin/env python3
"""Version 4.3.0. Where does the weight channel of the weighted SBM act?

For every graph of results/v5/weighted_sbm.json, the blocks of the weighted fit
(results/v6/sbm_partitions/<graph>.npy, second column, from the fit of results/v6/weighted_sbm_fair.json;
results/v5/sbm_partitions/<graph>.npy where the graph was not refitted) are held fixed and three statistics of
y = logit(mu) are compared with 1,000 row-wise permutations of the memberships (out-strength kept):
  assortative   mean y inside blocks minus mean y between blocks (what modularity can use);
  directional   mean over block pairs (r, s) with edges both ways of |mean y(r->s) - mean y(s->r)|;
  eta2          share of the variance of y explained by the ordered block pair.
z = (observed - null mean) / null SD.  Writes results/v6/sbm_where.json.
"""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def logit(mu):
    m = np.clip(mu, 1e-4, 1 - 1e-4)
    return np.log(m / (1 - m))


def row_perm(src, y, rng):
    out = y.copy()
    order = np.argsort(src, kind="stable")
    cuts = np.flatnonzero(np.diff(src[order])) + 1
    for blk in np.split(order, cuts):
        out[blk] = y[rng.permutation(blk)]
    return out


def stats(y, br, bs, B):
    inside = br == bs
    a = (y[inside].mean() - y[~inside].mean()) if inside.any() and (~inside).any() else np.nan
    code = br * B + bs
    s = pd.Series(y).groupby(code).mean()
    gaps = [abs(s[r * B + q] - s[q * B + r]) for r in range(B) for q in range(r + 1, B) if r * B + q in s.index and q * B + r in s.index]
    d = float(np.mean(gaps)) if gaps else np.nan
    cnt = pd.Series(y).groupby(code).transform("mean").to_numpy()
    eta = float(np.sum((cnt - y.mean()) ** 2) / np.sum((y - y.mean()) ** 2)) if y.var() > 0 else np.nan
    return a, d, eta


def main():
    # version 4.5.0: WHERE_PART_DIR (partitions to hold fixed), WHERE_OUT and graph names on the command line,
    # for the fits of results/v8/weighted_sbm_100*.json
    import sys
    part_dir = os.environ.get("WHERE_PART_DIR")
    out_path = os.environ.get("WHERE_OUT", os.path.join(ROOT, "results/v6/sbm_where.json"))
    sbm = json.load(open(os.path.join(ROOT, "results/v5/weighted_sbm.json")))
    names = sys.argv[1:] or list(sbm)
    out = {}
    for name in names:
        e = pd.read_csv(os.path.join(ROOT, f"data/graphs/{name}_edges.csv"))
        p6 = os.path.join(ROOT, f"results/v6/sbm_partitions/{name}.npy")
        src = p6 if os.path.exists(p6) else os.path.join(ROOT, f"results/v5/sbm_partitions/{name}.npy")
        if part_dir:
            src = os.path.join(ROOT, part_dir, f"{name}.npy")
        b = np.load(src)[:, 1]
        _, b = np.unique(b, return_inverse=True)
        B = int(b.max()) + 1
        s, d, y = e.src.to_numpy(), e.dst.to_numpy(), logit(e.mu.to_numpy(float))
        br, bs = b[s], b[d]
        obs = stats(y, br, bs, B)
        rng = np.random.default_rng(2026)
        null = np.array([stats(row_perm(s, y, rng), br, bs, B) for _ in range(1000)])
        r = {"blocks": B, "between_share": float((br != bs).mean()), "partition": os.path.relpath(src, ROOT)}
        for i, k in enumerate(("assortative", "directional", "eta2")):
            col = null[:, i]; col = col[~np.isnan(col)]
            ok = not np.isnan(obs[i]) and col.size > 1 and col.std() > 0
            r[k] = {"observed": None if np.isnan(obs[i]) else float(obs[i]),
                    "null_mean": float(col.mean()) if col.size else None,
                    "z": float((obs[i] - col.mean()) / col.std()) if ok else None}
        out[name] = r
        print(name, B, {k: (None if r[k]["z"] is None else round(r[k]["z"], 1)) for k in ("assortative", "directional", "eta2")}, flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
