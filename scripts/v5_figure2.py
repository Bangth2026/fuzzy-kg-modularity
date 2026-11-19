#!/usr/bin/env python3
"""Figure 2 of the KBS manuscript (release 4.2.0).

(a) Junyi transition graph rebuilt at five pruning thresholds, all scored on the same 1,954
    annotated pairs: Spearman rho with the crowd prerequisite score for each score
    (results/junyi_pruning_fixed_pairs.json).
(b) Q_F drop of the quantile cuts above the no-cut value, on the thirteen graphs with an
    independent grouping, by construction (results/v4/protocol_external.json, observed memberships).
Writes docs/figure2.png (600 dpi) and docs/figure2.pdf.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED = "#1f1f1e", "#6b6a64"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})

d = json.load(open(os.path.join(ROOT, "results/junyi_pruning_fixed_pairs.json")))["thresholds"]
taus = ["none", "0.05", "0.1", "0.2", "0.3"]
xs = np.arange(len(taus))
ops = [("product", "Product", "o", "-"), ("hop_baseline", "Hop distance", "s", "--"), ("yager_5", "Yager p = 5", "^", "-."),
       ("lukasiewicz", "Łukasiewicz", "D", ":"), ("min", "Max-min", "v", "-"), ("direct_edge", "Direct edge", "x", "--")]
for i, (k, lab, mk, ls) in enumerate(ops):
    y = [d[t]["scores"][k]["rho"] for t in taus]
    a.plot(xs, y, color=C[i], marker=mk, ls=ls, lw=1.6, ms=4.5, label=lab)
a.set_xticks(xs, ["none", "0.05", "0.10", "0.20", "0.30"])
a.set_xlabel("Pruning threshold τ_E (transition graph)")
a.set_ylabel("Spearman ρ with crowd score")
a.set_title("(a) Junyi, 1,954 fixed pairs", loc="left", fontsize=8.5, color=INK)
a.grid(axis="y", color="#e4e3dd", lw=0.6)
a.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
a.set_ylim(0.2, 0.65)

p = json.load(open(os.path.join(ROOT, "results/v4/protocol_external.json")))
cuts = ["q0.25", "q0.50", "q0.75"]
for name in sorted({k.split("/")[0] for k in p}):
    rows = p[f"{name}/tilt0"]["rows"]
    base = rows["support"]["qf_drop"]
    y = [rows[c]["qf_drop"] - base for c in cuts]
    tr = "transition" in name or name == "assist2017_item_transition"
    col = C[1] if tr else C[0]
    b.plot([0, 1, 2], y, color=col, lw=1.2, marker="o" if not tr else "s", ms=3.5, alpha=0.9)
b.plot([], [], color=C[0], marker="o", ms=3.5, lw=1.2, label="co-practice (7 graphs)")
b.plot([], [], color=C[1], marker="s", ms=3.5, lw=1.2, label="transition (6 graphs)")
b.set_xticks([0, 1, 2], ["lower quartile", "median", "upper quartile"])
b.set_xlabel("Quantile cut of μ_E")
b.set_ylabel("Q_F drop above the no-cut value")
b.set_title("(b) Cost of a cut, 13 graphs", loc="left", fontsize=8.5, color=INK)
b.grid(axis="y", color="#e4e3dd", lw=0.6)
b.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "docs/figure2.png"), dpi=600)
fig.savefig(os.path.join(ROOT, "docs/figure2.pdf"))
print("ok")
