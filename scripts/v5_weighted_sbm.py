#!/usr/bin/env python3
"""Weighted stochastic block model check: do the memberships carry block information beyond topology?

For each graph, a directed degree-corrected SBM is fitted by minimum description length
(graph-tool, Peixoto 2018 nonparametric weighted SBM), once on the topology alone and once with the
memberships as an edge covariate (logit(mu) modelled as real-normal within each pair of blocks).
The weight channel is judged by its description length against two nulls on the fixed topology,
refitted from scratch for every draw:
  row null     memberships permuted within the out-edges of each source (out-strength kept);
  global null  memberships permuted over all edges.
z = (mean DL under the null - observed DL) / SD under the null; a positive z means the observed
memberships compress better given blocks than permuted ones, i.e. they carry block information.
Also reported: number of blocks, ARI between the topology-only and the weighted partitions, and
their ARI with the independent grouping where one exists.

SBM_NESTED=1 fits the hierarchical (nested) SBM and compares level-0 partitions; output weighted_sbm_nested.json.
Requires graph-tool >= 2.45 (conda-forge). Output: results/v5/weighted_sbm.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

import graph_tool.all as gt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS = ["eedi_transition", "eedi_copractice", "junyi_transition", "algebra2005_transition",
          "algebra2005_copractice", "ednet_transition", "assist2017_kc", "assist2017_kc_transition",
          "assist2015_kc_transition", "bridge2006_transition"]
N_NULL = int(os.environ.get("SBM_NULL", 20))
N_INIT = int(os.environ.get("SBM_INIT", 5))
NESTED = os.environ.get("SBM_NESTED", "0") == "1"   # hierarchical (nested) SBM instead of a flat one
# version 4.3.0: every null refit gets the same number of initialisations as the observed fit
# (SBM_NULL_INIT; release 4.2.0 used one), and the output goes to SBM_OUT
N_NULL_INIT = int(os.environ.get("SBM_NULL_INIT", 1))
OUT_PATH = os.environ.get("SBM_OUT")
# version 4.5.0: the logit is taken of memberships clipped to [SBM_CLIP, 1 - SBM_CLIP] (default 1e-4,
# the value of every earlier release; no released membership reaches that bound, so the default
# clip never acts), and partitions can be written elsewhere (SBM_PART_DIR)
CLIP = float(os.environ.get("SBM_CLIP", 1e-4))
PART_DIR = os.environ.get("SBM_PART_DIR", "results/v5/sbm_partitions")
KINDS = tuple(os.environ.get("SBM_KINDS", "row,global").split(","))   # version 4.5.0: which nulls to refit


def load(name):
    e = pd.read_csv(os.path.join(ROOT, f"data/graphs/{name}_edges.csv"))
    nodes = pd.read_csv(os.path.join(ROOT, f"data/graphs/{name}_nodes.csv"))
    return len(nodes), e.src.to_numpy(), e.dst.to_numpy(), e.mu.to_numpy(float), nodes.label.astype(str).tolist()


def groups(name, labels):
    base = name.split("_")[0]
    p = os.path.join(ROOT, f"data/graphs/{base}_groups.json")
    if base == "junyi":
        p = os.path.join(ROOT, "data/graphs/junyi_topics.json")
    if not os.path.exists(p):
        return None
    g = json.load(open(p))
    return np.array([g.get(l, g.get(l.replace(".0", ""))) for l in labels], dtype=object)


def build(n, s, d, y=None):
    g = gt.Graph(directed=True)
    g.add_vertex(n)
    g.add_edge_list(np.c_[s, d])
    if y is not None:
        w = g.new_ep("double")
        w.a = y
        return g, w
    return g, None


def fit(g, w, seed):
    gt.seed_rng(seed)
    np.random.seed(seed)
    args = dict(deg_corr=True)
    kw = {}
    if w is not None:
        # the edge covariate needs the WeightedBlockState class (graph-tool >= 2.9x); passing
        # recs to BlockState is silently ignored
        args.update(rec=[w], rec_types=["real-normal"])
        kw = dict(state=gt.WeightedBlockState) if not NESTED else {}
    if NESTED:
        # version 4.5.0: the nested fit takes the covariate through base_state / base_state_args.  Releases
        # 4.2.0 and 4.3.0 passed base_type, rec and rec_types inside state_args, which graph-tool accepts
        # with a UserWarning and ignores, so their nested "weighted" fits never saw the memberships
        # (results/v5/weighted_sbm_nested.json and results/v6/weighted_sbm_nested.json are superseded).
        if w is not None:
            st = gt.minimize_nested_blockmodel_dl(g, base_state=gt.WeightedBlockState, base_state_args=args)
        else:
            st = gt.minimize_nested_blockmodel_dl(g, base_state_args=args)
        for _ in range(3):
            st.multiflip_mcmc_sweep(beta=np.inf, niter=10)
        return st
    st = gt.minimize_blockmodel_dl(g, state_args=args, **kw)
    for _ in range(3):  # a few merge-split sweeps to refine
        st.multiflip_mcmc_sweep(beta=np.inf, niter=10)
    return st


def blocks(st):
    if NESTED:
        return np.asarray(st.get_levels()[0].get_blocks().a)
    return np.asarray(st.get_blocks().a)


def best(g, w, seeds):
    sts = [fit(g, w, s) for s in seeds]
    i = int(np.argmin([st.entropy() for st in sts]))
    return sts[i]


def logit(mu):
    m = np.clip(mu, CLIP, 1 - CLIP)
    return np.log(m / (1 - m))


def row_perm(src, y, rng):
    out = y.copy()
    order = np.argsort(src, kind="stable")
    ss = src[order]
    cuts = np.flatnonzero(np.diff(ss)) + 1
    for blk in np.split(order, cuts):
        out[blk] = y[rng.permutation(blk)]
    return out


def run(name):
    t0 = time.time()
    n, s, d, mu, labels = load(name)
    y = logit(mu)
    g0, _ = build(n, s, d)
    st0 = best(g0, None, range(N_INIT))
    g, w = build(n, s, d, y)
    st1 = best(g, w, range(N_INIT))
    b0 = blocks(st0); b1 = blocks(st1)
    dl_obs = st1.entropy()
    # version 4.5.0 guard: the weighted state must carry the covariate (its levels or its base state has rec)
    base = st1.get_levels()[0] if NESTED else st1
    assert len(getattr(base, "rec", [])) >= 1, "weighted fit does not carry the membership covariate"
    rng = np.random.default_rng(2026)
    nulls = {}
    for kind in KINDS:
        vals = []
        for r in range(N_NULL):
            yp = row_perm(s, y, rng) if kind == "row" else rng.permutation(y)
            gp, wp = build(n, s, d, yp)
            vals.append(best(gp, wp, [100 + 10 * r + j for j in range(N_NULL_INIT)]).entropy())
        vals = np.array(vals)
        nulls[kind] = dict(mean=float(vals.mean()), sd=float(vals.std(ddof=1)),
                           z=(float((vals.mean() - dl_obs) / vals.std(ddof=1)) if vals.std(ddof=1) > 1e-9 else None),
                           delta_bits=float((vals.mean() - dl_obs) / np.log(2)),
                           share_null_below_obs=float((vals <= dl_obs).mean()),
                           # version 4.5.0: one-sided empirical p, (1 + #null DL <= observed) / (N + 1)
                           p_empirical=float((1 + (vals <= dl_obs).sum()) / (vals.size + 1)),
                           values=[float(v) for v in vals])
    out = dict(n=int(n), m=int(len(mu)), B_topology=int(len(np.unique(b0))), B_weighted=int(len(np.unique(b1))),
               ari_topology_vs_weighted=float(ari(b0, b1)), dl_topology_nats=float(st0.entropy()),
               dl_weighted_nats=float(dl_obs), nulls=nulls, n_null=N_NULL, n_init=N_INIT, n_null_init=N_NULL_INIT,
               dl_weighted_single_inits=[float(fit(g, w, s_).entropy()) for s_ in range(N_INIT)])
    # where does the weight channel act? mean logit membership inside blocks, between blocks, and the
    # directional gap between block pairs (r, s) and (s, r)
    br, bs = b1[s], b1[d]
    inside = br == bs
    pair = {}
    for r_, s_, v in zip(br[~inside], bs[~inside], y[~inside]):
        pair.setdefault((int(r_), int(s_)), []).append(v)
    gaps = [abs(np.mean(v) - np.mean(pair[(b, a)])) for (a, b), v in pair.items() if (b, a) in pair and a < b]
    out["weight_inside_mean"] = float(y[inside].mean()) if inside.any() else None
    out["weight_between_mean"] = float(y[~inside].mean()) if (~inside).any() else None
    out["between_share"] = float((~inside).mean())
    out["directional_gap_mean"] = float(np.mean(gaps)) if gaps else None
    out["clip"] = CLIP
    out["n_clipped"] = int(((mu < CLIP) | (mu > 1 - CLIP)).sum())
    os.makedirs(os.path.join(ROOT, PART_DIR), exist_ok=True)
    np.save(os.path.join(ROOT, PART_DIR, f"{name}{'_nested' if NESTED else ''}.npy"), np.c_[b0, b1])
    grp = groups(name, labels)
    if grp is not None:
        has = np.array([x is not None for x in grp])
        gl = pd.factorize(pd.Series(grp[has]).astype(str))[0]
        out["ari_grouping_topology"] = float(ari(gl, b0[has]))
        out["ari_grouping_weighted"] = float(ari(gl, b1[has]))
    out["time_s"] = time.time() - t0
    return out


def main():
    names = sys.argv[1:] or GRAPHS
    path = OUT_PATH or os.path.join(ROOT, "results/v5/weighted_sbm" + ("_nested" if NESTED else "") + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    res = json.load(open(path)) if os.path.exists(path) else {}
    for nm in names:
        res[nm] = run(nm)
        r = res[nm]
        print(nm, r["B_topology"], r["B_weighted"], round(r["ari_topology_vs_weighted"], 3),
              {k: (round(v["z"], 2) if v["z"] is not None else None) for k, v in r["nulls"].items()}, r.get("ari_grouping_topology"),
              r.get("ari_grouping_weighted"), round(r["time_s"]), flush=True)
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
