"""Agreement of the cut audit (Algorithm 2, step 3) with ARI loss against independent groupings.

Reads results/v4/protocol_external.json (no recomputation of partitions) and reports, for each
tilt, the 2x2 table of flagged vs. resolved ARI loss, sensitivity, specificity, precision,
balanced accuracy, Cohen's kappa and a one-sided Fisher exact p-value, with 95% intervals from a
bootstrap over graphs (cells of one graph are resampled together). Two baselines are scored in
the same way: flag-all, and retained mass (flag a cut whose retained membership mass is below the
median retained mass of the cuts of its graph).

Output: results/v5/audit_agreement.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTS = ["q0.25", "q0.50", "q0.75", "abs0.3", "abs0.4", "abs0.5"]


def cells(d, tilt):
    out = {}
    for key, v in d.items():
        g, t = key.split("/")
        if t != f"tilt{tilt}":
            continue
        rows = v["rows"]
        ret = np.array([rows[c]["retained_edges"] for c in CUTS])
        med = np.median(ret)
        out[g] = [
            dict(loss=bool(rows[c]["ari_loss_resolved"]), qf=bool(rows[c]["flag_qf_drop"]),
                 rank=bool(rows[c]["flag_rank_drop"]), all=True, mass=bool(ret[i] < med),
                 qf_excess=rows[c]["qf_drop"] - rows["support"]["qf_drop"],
                 ari_loss=rows[c]["ari_loss_vs_support"])
            for i, c in enumerate(CUTS)
        ]
    return out


def stats(cs, rule):
    y = np.array([c["loss"] for c in cs]); f = np.array([c[rule] for c in cs])
    tp = int((y & f).sum()); fp = int((~y & f).sum()); fn = int((y & ~f).sum()); tn = int((~y & ~f).sum())
    n = tp + fp + fn + tn
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    po = (tp + tn) / n
    pe = ((tp + fp) / n) * ((tp + fn) / n) + ((fn + tn) / n) * ((fp + tn) / n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=n, base_rate=(tp + fn) / n, flag_rate=(tp + fp) / n,
                sensitivity=sens, specificity=spec, precision=prec,
                balanced_accuracy=np.nanmean([sens, spec]), kappa=kappa)


def boot(by_graph, rule, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    gs = list(by_graph)
    keys = ["sensitivity", "specificity", "precision", "balanced_accuracy", "kappa"]
    draws = {k: [] for k in keys}
    for _ in range(B):
        pick = rng.choice(len(gs), len(gs), replace=True)
        cs = [c for i in pick for c in by_graph[gs[i]]]
        s = stats(cs, rule)
        for k in keys:
            draws[k].append(s[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def main():
    d = json.load(open(os.path.join(ROOT, "results/v4/protocol_external.json")))
    res = {}
    for tilt in ["0", "1", "4"]:
        bg = cells(d, tilt)
        allc = [c for g in bg for c in bg[g]]
        res[f"tilt{tilt}"] = {"graphs": len(bg)}
        for rule in ["qf", "rank", "all", "mass"]:
            s = stats(allc, rule)
            y = np.array([c["loss"] for c in allc]); f = np.array([c[rule] for c in allc])
            if rule != "all":
                s["fisher_one_sided_p"] = float(fisher_exact([[int((y & f).sum()), int((~y & f).sum())],
                                                              [int((y & ~f).sum()), int((~y & ~f).sum())]],
                                                             alternative="greater")[1])
            s["ci95_graph_bootstrap"] = boot(bg, rule)
            res[f"tilt{tilt}"][rule] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in s.items()}
    os.makedirs(os.path.join(ROOT, "results/v5"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results/v5/audit_agreement.json"), "w"), indent=1)
    for t, r in res.items():
        for rule in ["qf", "rank", "all", "mass"]:
            s = r[rule]; ci = s["ci95_graph_bootstrap"]
            print(t, rule, f"TP{s['tp']} FP{s['fp']} FN{s['fn']} TN{s['tn']} sens {s['sensitivity']:.2f} spec {s['specificity']:.2f} "
                  f"prec {s['precision']:.2f} BA {s['balanced_accuracy']:.2f} {ci['balanced_accuracy']} kappa {s['kappa']:.2f} {ci['kappa']} "
                  f"p {s.get('fisher_one_sided_p', float('nan')):.4f}")


if __name__ == "__main__":
    sys.exit(main())
