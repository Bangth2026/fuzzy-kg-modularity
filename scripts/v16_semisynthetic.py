#!/usr/bin/env python3
"""Release 5.1.0. Semi-synthetic worlds in which the grades may mark communities.

The 96 worlds of scripts/v15_semisynthetic.py are re-run with the community information delta of
fuzzykg.semisynthetic.World (a prerequisite inside the community of its target has strength
U(0.3 + delta, 1), one outside U(0.3, 1 - delta)), delta in {0, 0.2, 0.4, 0.6}; delta = 0 is the
generator of release 5.0.0 and reproduces its logs. On every graph:

  partitions  support, quantile cuts 0.25 / 0.50 / 0.75 and Leiden on Q_F (ten seeds, gamma = 1);
              ARI of each with the true communities;
  audit       decision on mu_E and kappa* at the minimum excess selected in release 5.0.0
              (0.03 dense, 0.0075 sparse), family of Section 4.2 and the family with a knot just
              below each cut level (``refined``);
  losses      true loss of a cut against the support partition and against the graded partition
              Pi_F (mean ARI over seeds, same loss rule as Section 6.2);
  E_F         Q_F(Pi_F) - Q_F(Pi_cut) on mu_E (seed means);
  gate        graded-partition gain g = Q_F[mu](Pi_F) - Q_F[mu](Pi_sup) (best of three seeds) against
              the same gain on ten row permutations of mu (Pi_F re-optimised on the permuted weights);
              z = (g - mean null) / sd null; also the partition value of the grades with its components.
Writes results/v16/semisynthetic_delta.json (resumable).  Usage: v16_semisynthetic.py DELTA [DELTA ...]
"""
from __future__ import annotations
import itertools, json, os, sys, time
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_modularity  # noqa: E402
from fuzzykg.audit import _leiden, describe  # noqa: E402
from fuzzykg.construction import build_fuzzy_graph, build_transition_graph  # noqa: E402
from fuzzykg.nulltest import permute_global, permute_rows  # noqa: E402
from fuzzykg.semisynthetic import World, simulate  # noqa: E402
from fuzzykg.simplex import CutFamily, kappa_star  # noqa: E402

OUT = "results/v16/semisynthetic_delta.json"
SEEDS = range(10)
CUTS = (0.25, 0.50, 0.75)
DSEL = {"co-practice": 0.03, "transition": 0.0075}
GRID = list(itertools.product((0.6, 0.8, 0.95), (0.1, 0.4), (0.0, 0.3), (0.0, 0.2), (400, 1500), (0, 1)))
N_NULL = 10


def relabel(G, labels, K):
    lab = np.array([int(x) for x in labels])
    return FuzzyGraph(K, lab[G.src], lab[G.dst], G.mu.copy())


def best3(G):
    P = [_leiden(G, s) for s in range(3)]
    return max(P, key=lambda p: fuzzy_modularity(G, p))


def refined_family(G, parts):
    """CutFamily whose knots also include, for each cut, the largest membership below the cut level,
    so that the vertex of the segment ending at the cut is the cut's indicator (Corollary 1)."""
    fam = CutFamily(G, parts)
    extra = []
    for q in CUTS:
        a = float(np.quantile(G.mu, q)); below = G.mu[G.mu < a]
        if below.size:
            extra.append(float(below.max()))
    knots = np.unique(np.concatenate([fam.knots, extra]))
    lo, hi = knots[:-1], knots[1:]
    fam.knots = knots
    fam.ell = np.clip(G.mu[:, None] - lo[None, :], 0.0, (hi - lo)[None, :])
    fam.m = fam.ell.sum(0); fam.J = fam.ell.shape[1]
    from fuzzykg.simplex import partition_stats
    fam.stats = {}
    for name, plist in parts.items():
        A, Ms = zip(*[partition_stats(fam.ell, G.src, G.dst, p) for p in plist])
        fam.stats[name] = (np.array(A), np.array(Ms))
    return fam


def loss_rule(a, b):
    d = float(np.mean(a) - np.mean(b)); se = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
    return bool(d >= 0.02 and d - 1.96 * se > 0), d


def graph_record(G, K, comm, con):
    rng = np.random.default_rng(7)
    Gs = FuzzyGraph(K, G.src, G.dst, np.ones(G.m))
    parts = {"support": [_leiden(Gs, s) for s in SEEDS]}
    for q in CUTS:
        parts[f"q{q:.2f}"] = [_leiden(G.alpha_cut(float(np.quantile(G.mu, q))), s) for s in SEEDS]
    parts["F"] = [_leiden(G, s) for s in SEEDS]
    ar = {k: [ari(comm, p) for p in v] for k, v in parts.items()}
    rec = {"describe": describe(G), "ari": {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in ar.items()}}
    fam = CutFamily(G, {k: v for k, v in parts.items() if k != "F"})
    famr = refined_family(G, {k: v for k, v in parts.items() if k != "F"})
    qF = np.array([fuzzy_modularity(G, p) for p in parts["F"]])
    rec["cuts"] = {}
    for q in CUTS:
        c = f"q{q:.2f}"
        k1 = kappa_star(fam, c, min_excess=DSEL[con], kmax=100.0)
        k2 = kappa_star(famr, c, min_excess=DSEL[con], kmax=100.0)
        qc = np.array([fuzzy_modularity(G, p) for p in parts[c]])
        ls, ds = loss_rule(ar["support"], ar[c]); lf, df = loss_rule(ar["F"], ar[c])
        rec["cuts"][c] = {"decision_mu": k1["decision_mu"], "kappa_star": k1["kappa_star"], "margin_mu": k1["margin_mu"],
                          "kappa_star_refined": k2["kappa_star"], "decision_refined": k2["decision_mu"],
                          "loss_vs_support": ls, "loss_vs_support_value": ds, "loss_vs_F": lf, "loss_vs_F_value": df,
                          "E_F": float(qF.mean() - qc.mean())}
    # gate and partition value with the row and global nulls
    pF, pS = best3(G), best3(Gs)
    g_obs = float(fuzzy_modularity(G, pF) - fuzzy_modularity(G, pS))
    null_row, ari_row, ari_glob = [], [], []
    for d in range(N_NULL):
        Hr = permute_rows(G, rng); pr = best3(Hr)
        null_row.append(float(fuzzy_modularity(Hr, pr) - fuzzy_modularity(Hr, pS))); ari_row.append(ari(comm, pr))
        Hg = permute_global(G, rng); ari_glob.append(ari(comm, best3(Hg)))
    a_sup, a_obs = ari(comm, pS), ari(comm, pF)
    rec["gate"] = {"g_obs": g_obs, "null_mean": float(np.mean(null_row)), "null_sd": float(np.std(null_row, ddof=1)),
                   "null_max": float(np.max(null_row)),
                   "z": float((g_obs - np.mean(null_row)) / max(np.std(null_row, ddof=1), 1e-12))}
    rec["value_partition"] = {"support": a_sup, "observed": a_obs, "global": float(np.mean(ari_glob)), "row": float(np.mean(ari_row)),
                              "V_total": a_obs - a_sup, "V_marg": float(np.mean(ari_glob)) - a_sup,
                              "V_strength": float(np.mean(ari_row) - np.mean(ari_glob)), "V_resid": a_obs - float(np.mean(ari_row))}
    return rec


def main(deltas):
    os.makedirs("results/v16", exist_ok=True)
    for dl in deltas:
        part = os.environ.get("PART")          # optional split of the grid over two processes: PART=0/2, 1/2
        out = OUT.replace(".json", f"_{dl:g}.json" if part is None else f"_{dl:g}_part{part.split('/')[0]}.json")
        res = json.load(open(out)) if os.path.exists(out) else {"delta": dl, "worlds": {}}
        for gi, (pc, noise, il, sug, N, seed) in enumerate(GRID):
            key = f"p{pc}_n{noise}_i{il}_s{sug}_N{N}_seed{seed}"
            if part is not None and gi % int(part.split("/")[1]) != int(part.split("/")[0]):
                continue
            if key in res["worlds"]:
                continue
            t0 = time.perf_counter()
            w = World(p_comm=pc, noise=noise, interleave=il, suggest=sug, N=N, seed=seed, delta=dl)
            log, truth = simulate(w)
            K = w.K; comm = truth["community"]
            rec = {"attempts": int(len(log))}
            for con, build in (("co-practice", build_fuzzy_graph), ("transition", build_transition_graph)):
                G0, _ = build(log)
                rec[con] = graph_record(relabel(G0, G0.labels, K), K, comm, con)
            rec["time_s"] = time.perf_counter() - t0
            res["worlds"][key] = rec
            json.dump(res, open(out, "w"), indent=1, default=float)
            print(dl, key, f"{rec['time_s']:.0f}s", {c: round(rec[c]["value_partition"]["V_total"], 3) for c in ("co-practice", "transition")},
                  {c: round(rec[c]["gate"]["z"], 1) for c in ("co-practice", "transition")}, flush=True)


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or [0.0, 0.2, 0.4, 0.6])
