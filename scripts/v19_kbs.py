#!/usr/bin/env python3
"""Release 5.4.0. End-to-end use of the decision guide (Section 8.4 of the manuscript,
Supplementary Section S10).

A. Prerequisite recommendation on Junyi. For every target exercise with at least two annotated
   candidate prerequisites (351 targets), the candidates are ranked by a path score from the
   candidate to the target; ties are broken at random (mean over 20 tie-breaks). Outcomes per
   target: nDCG@3 with the crowd score as gain, precision@1 (top candidate has crowd score >= 5,
   the midpoint of the 1-9 scale) and the unnecessary-prerequisite rate at k = 3 (share of the
   three recommended candidates with crowd score < 5), averaged over targets. Policies:
     crisp-default   hop distance on the graph pruned at the default 0.2;
     graded-fixed    product best-path score on the unpruned graph;
     guide (D7)      on one half of the targets: pruning threshold of hop count and of the product
                     chosen by nDCG@3, then the graded representation kept only if it beats the
                     best crisp one on that half; the chosen representation is evaluated on the
                     other half.
   200 random halvings of the targets (seed 5403); reported: held-out means, the differences
   guide - crisp-default and guide - graded-fixed with 2.5/97.5 percentiles over halvings, the
   choices made, and the number of edges the chosen graph stores.
B. Next-concept recommendation on held-out learners (results/heldout_*.json of release 4, eleven
   graphs of Junyi and ASSISTments). Policies: crisp (hop distance), graded (product), and the
   guide's choice among hop distance and the nine t-norm scores by mean reciprocal rank on the
   validation learners; test MRR of each and regret against the best on test.
C. Bootstrap p-values (two-way cluster bootstrap over exercises, 2,000 resamples, seed 5404) for
   the path differences of Table 4 and Section 8.1 that enter the primary-results table.
Writes results/v19/kbs.json.
"""
from __future__ import annotations
import glob, json, os, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from data.loaders import load_junyi_expert  # noqa: E402
from composition_robustness import mats  # noqa: E402
from v18_reach import load_tau, TAGS  # noqa: E402


def ndcg_parts(score, y, groups, rng, k=3, draws=20):
    """Per-target nDCG@k, P@1 (y >= 5) and unnecessary rate at k (y < 5) for one score vector."""
    out = {}
    for g, idx in groups.items():
        yy = y[idx]; ideal = np.sort(yy)[::-1][:k]; disc = 1 / np.log2(np.arange(2, k + 2))
        idcg = (ideal * disc[:len(ideal)]).sum(); nd = p1 = un = 0.0
        for _ in range(draws):
            order = np.lexsort((rng.random(len(idx)), -score[idx]))
            top = yy[order][:k]
            nd += (top * disc[:len(top)]).sum() / idcg; p1 += float(top[0] >= 5); un += float(np.mean(top < 5))
        out[g] = (nd / draws, p1 / draws, un / draws)
    return out


def main():
    os.makedirs("results/v19", exist_ok=True)
    _, ann = load_junyi_expert()
    A = ann.Exercise_A.astype(str).to_numpy(); Bn = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    X, E = {}, {}
    for t in TAGS:
        G, lab = load_tau(t); idx = {l: i for i, l in enumerate(lab)}; E[t] = int(G.m)
        ok = np.array([a in idx and b in idx for a, b in zip(A, Bn)])
        ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in Bn])
        M, _ = mats(G)
        for k in ("hop_baseline", "product"):
            X[(k, t)] = np.where(ok, M[k][ib, ia], 0.0)
    tg = pd.Series(range(len(Bn))).groupby(Bn).apply(list)
    groups = {g: np.array(v) for g, v in tg.items() if len(v) >= 2}
    keys = sorted(groups)
    rng = np.random.default_rng(5403)
    per = {kt: ndcg_parts(X[kt], y, groups, rng) for kt in X}
    mean = lambda kt, sub, j=0: float(np.mean([per[kt][g][j] for g in sub]))
    full = {f"{k}|{t}": [mean((k, t), keys, j) for j in range(3)] for (k, t) in X}
    res = {"A": {"n_targets": len(keys), "full_sample": full, "edges": E}}
    rs = np.random.default_rng(5403)
    rows = []; choices = Counter()
    for _ in range(200):
        h = rs.random(len(keys)) < 0.5; hA = [g for g, b in zip(keys, h) if b]; hB = [g for g, b in zip(keys, h) if not b]
        tc = max(TAGS, key=lambda t: mean(("hop_baseline", t), hA)); tgd = max(TAGS, key=lambda t: mean(("product", t), hA))
        pick = ("product", tgd) if mean(("product", tgd), hA) > mean(("hop_baseline", tc), hA) else ("hop_baseline", tc)
        choices[f"{pick[0]}|{pick[1]}"] += 1
        rows.append([[mean(kt, hB, j) for j in range(3)] for kt in (("hop_baseline", "0.2"), ("product", "0"), pick)] + [[E[pick[1]]]])
    arr = np.array([[r[0], r[1], r[2]] for r in rows])  # halvings x policy x metric
    pct = lambda v: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.mean(v > 0))]
    names = ["nDCG@3", "P@1", "unnecessary@3"]
    res["A"]["heldout"] = {pol: {names[j]: float(arr[:, i, j].mean()) for j in range(3)} for i, pol in enumerate(["crisp-default", "graded-fixed", "guide"])}
    res["A"]["guide_minus_crisp_default"] = {names[j]: pct(arr[:, 2, j] - arr[:, 0, j]) for j in range(3)}
    res["A"]["guide_minus_graded_fixed"] = {names[j]: pct(arr[:, 2, j] - arr[:, 1, j]) for j in range(3)}
    res["A"]["choices"] = dict(choices); res["A"]["edges_stored_mean"] = float(np.mean([r[3][0] for r in rows]))
    print(json.dumps(res["A"], indent=1)[:2500], flush=True)
    # B. next-concept
    B = {}
    for f in sorted(glob.glob("results/heldout_*.json")):
        if "_L" in f or "support" in f:
            continue
        r = json.load(open(f)); hop = r["baselines"]["hop_baseline"]
        cand = {"hop": (hop["val_mrr"], hop["mrr_learner_mean"])}
        for k, v in r["operators"].items():
            cand[k] = (v["val_mrr"], v["mrr_learner_mean"])
        sel = max(cand, key=lambda k: cand[k][0]); best = max(cand, key=lambda k: cand[k][1])
        name = os.path.basename(f)[8:-5]
        B[name] = {"construction": r["construction"], "crisp_hop": cand["hop"][1], "graded_product": cand["product"][1],
                   "guide_choice": sel, "guide": cand[sel][1], "best_on_test": best, "best": cand[best][1],
                   "regret_guide": cand[best][1] - cand[sel][1], "regret_crisp": cand[best][1] - cand["hop"][1],
                   "regret_graded": cand[best][1] - cand["product"][1]}
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in B[name].items()}, flush=True)
    res["B"] = B
    for pol in ("guide", "crisp", "graded"):
        res.setdefault("B_summary", {})[pol] = {"mean_regret": float(np.mean([v[f"regret_{pol}"] for v in B.values()])),
                                                "max_regret": float(np.max([v[f"regret_{pol}"] for v in B.values()]))}
    print(res["B_summary"])
    # C. bootstrap p-values for path differences
    rb = np.random.default_rng(5404)
    exs = np.unique(np.r_[A, Bn]); pos = {e: i for i, e in enumerate(exs)}
    pa = np.array([pos[a] for a in A]); pb = np.array([pos[b] for b in Bn])
    comps = {"threshold_gain_hop_0.05_minus_0.2": (("hop_baseline", "0.05"), ("hop_baseline", "0.2")),
             "best_graded_minus_best_crisp": (("product", "0"), ("hop_baseline", "0.05")),
             "product_minus_hop_at_0.05": (("product", "0.05"), ("hop_baseline", "0.05"))}
    D = {k: [] for k in comps}
    for _ in range(2000):
        m = rb.multinomial(exs.size, np.full(exs.size, 1 / exs.size)); w = m[pa] * m[pb]; kk = w > 0; yy = np.repeat(y[kk], w[kk])
        cache = {}
        for name, (a, b) in comps.items():
            for kt in (a, b):
                if kt not in cache:
                    cache[kt] = spearmanr(np.repeat(X[kt][kk], w[kk]), yy).statistic
            D[name].append(cache[a] - cache[b])
    C = {}
    for name, (a, b) in comps.items():
        d = np.array(D[name]); est = spearmanr(X[a], y).statistic - spearmanr(X[b], y).statistic
        p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))); p = max(p, 1 / len(d))
        C[name] = {"estimate": float(est), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_boot": float(p)}
    res["C"] = C; print(C)
    json.dump(res, open("results/v19/kbs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
