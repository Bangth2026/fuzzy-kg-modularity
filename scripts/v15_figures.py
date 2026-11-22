#!/usr/bin/env python3
"""Release 5.0.0. Figures 1-3 of the manuscript from results/v15/summary.json and crossdomain.json.
Palette: categorical slots 1-4 of the dataviz reference palette (validated: lightness, chroma,
CVD separation and normal-vision floor pass on the light surface); identity is always also
carried by marker shape or a direct label."""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUT = "docs"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})
S = json.load(open("results/v15/summary.json")); CD = json.load(open("results/v15/crossdomain.json"))


def save(fig, name):
    fig.savefig(f"{OUT}/v5_{name}.png", dpi=400, bbox_inches="tight"); fig.savefig(f"{OUT}/v5_{name}.pdf", bbox_inches="tight"); plt.close(fig)


def fig1():
    fig, ax = plt.subplots(figsize=(7.48, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc="white", ec=col, lw=1.1))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=5.9, color=INK2, linespacing=1.35)
    box(0.5, 12, 17.5, 22, "Learner log", "two constructions:\nco-practice,\ntransition\n→ fuzzy graph\nG_F = (C, E, μ_E)", INK2)
    box(23, 25, 24, 13, "Value of grades (Sec. 3)", "V = S(G_F) − S(support)\n= V_marg + V_str + V_res\nedge · path · partition", BLUE)
    box(23, 2, 24, 20, "Robust cut audit (Sec. 4)", "rescalings φ_s, s in a cone\nQ_F[φ_s∘μ](Π) in closed form\n(Prop. 1); no unit-free\nverdict (Prop. 2); κ*: largest\nslope ratio keeping the decision", ORANGE)
    box(52, 25, 22, 13, "Paths (Sec. 5)", "max-min reads only μ ≥ α_c\n(Prop. 3, usable share u)\nλ + (1 − λ)(−ln μ): λ̂", AQUA)
    box(52, 2, 22, 20, "Validation (Sec. 7, 8)", "192 semi-synthetic graphs\n(known DAG, communities);\n23 educational graphs,\n10 with a grouping;\nPPI5k, CN15k", YELLOW)
    box(79, 12, 20, 22, "Algorithm 2", "per use: keep grades,\ndrop them, or keep\nthem unmodified;\nper cut: cut / flagged\n/ undecided; path score", INK2)
    for (x0, y0, x1, y1) in ((18.6, 26, 22.4, 31), (18.6, 20, 22.4, 12), (47.6, 31, 51.4, 31), (47.6, 12, 51.4, 12), (74.6, 31, 78.4, 26), (74.6, 12, 78.4, 20)):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    save(fig, "figure1")


def fig2():
    sc = S["semisynthetic"]["scatter"]
    fig, axs = plt.subplots(1, 3, figsize=(7.48, 2.35), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, con, col, title in ((axs[0], "transition", BLUE, "(a) transition (sparse)"), (axs[1], "co-practice", ORANGE, "co-practice (dense)")):
        r = [x for x in sc if x["con"] == con]
        e = np.array([x["excess"] for x in r]); l = np.array([x["loss_value"] for x in r]); k = np.array([x["kappa"] for x in r])
        firm = k >= 2
        ax.scatter(e[firm], l[firm], s=9, c=col, edgecolors="white", linewidths=0.3, label="κ* ≥ 2")
        ax.scatter(e[~firm], l[~firm], s=9, facecolors="none", edgecolors=col, linewidths=0.6, label="κ* < 2")
        ax.axhline(0.02, color=INK2, lw=0.5, ls=(0, (3, 2))); ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel("excess E on μ_E"); ax.set_title(title, fontsize=7.5, loc="left", color=INK)
        ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    axs[0].set_ylabel("true ARI loss of the cut"); axs[0].legend(loc="upper left", fontsize=6.4, handletextpad=0.2)
    ax = axs[2]
    vals = [(S["semisynthetic"]["transition"]["accuracy_firm"], S["semisynthetic"]["transition"]["accuracy_fragile"]),
            (S["semisynthetic"]["co-practice"]["accuracy_firm"], S["semisynthetic"]["co-practice"]["accuracy_fragile"])]
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, [v[0] for v in vals], w, color=[BLUE, ORANGE], label="κ* ≥ 2")
    b2 = ax.bar(x + w / 2 + 0.01, [v[1] for v in vals], w, color="white", edgecolor=[BLUE, ORANGE], lw=0.9, hatch="////", label="κ* < 2")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center", fontsize=6.2, color=INK)
    ax.axhline(0.5, color=INK2, lw=0.5, ls=(0, (3, 2))); ax.set_ylim(0, 1.05); ax.set_xticks(x, ["transition", "co-practice"])
    ax.set_ylabel("accuracy of the decision on μ_E"); ax.set_title("(b)", fontsize=7.5, loc="left"); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.legend(fontsize=6.2, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.2)
    fig.tight_layout(w_pad=1.2); save(fig, "figure2")


def fig3():
    P = S["semisynthetic"]["paths_scatter"]; J = S["value_paths"]["junyi"]
    fig, axs = plt.subplots(1, 2, figsize=(7.48, 2.5))
    ax = axs[0]
    for con, col, mk, lab in (("co-practice", ORANGE, "o", "semi-synthetic co-practice"), ("transition", BLUE, "s", "semi-synthetic transition")):
        r = [p for p in P if p["con"] == con]
        ax.scatter([p["usable"] for p in r], [p["d"] for p in r], s=10, marker=mk, c=col, edgecolors="white", linewidths=0.3, alpha=0.85, label=lab)
    for nm, lab in (("junyi_copractice", "Junyi co-practice"), ("junyi_transition", "Junyi transition"), ("junyi_transition_unsuggested", "Junyi transition, unsuggested")):
        p = J[nm]["paths"]; u = p["connectivity"]["usable_share"]; d = p["rho_maxmin"] - p["rho_direct"]
        ax.scatter([u], [d], s=26, marker="D", c=INK, edgecolors="white", linewidths=0.5, zorder=4)
        ax.annotate(lab, (u, d), xytext=((-4, -10) if "unsugg" in nm else (-4, 5)) if "transition" in nm else (6, 2), ha="right" if "transition" in nm else "left", textcoords="offset points", fontsize=6.2, color=INK)
    ax.axhline(0, color=INK2, lw=0.5); ax.set_xlabel("usable share u (Proposition 3)"); ax.set_ylabel("ρ(max-min) − ρ(direct edge)")
    ax.set_title("(a)", fontsize=7.5, loc="left"); ax.legend(fontsize=6.2, loc="lower right"); ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax = axs[1]; lam = np.round(np.linspace(0, 1, 11), 1)
    series = [("Junyi transition", [J["junyi_transition"]["paths"]["rho_by_lam"][f"{l:.1f}"] for l in lam], BLUE, "o"),
              ("Junyi co-practice", [J["junyi_copractice"]["paths"]["rho_by_lam"][f"{l:.1f}"] for l in lam], ORANGE, "s"),
              ("PPI5k", [CD["ppi5k"]["rho_by_lam"][f"{l:.1f}"] for l in lam], AQUA, "^"),
              ("CN15k", [CD["cn15k"]["rho_by_lam"][f"{l:.1f}"] for l in lam], YELLOW, "v")]
    for lab, y, col, mk in series:
        ax.plot(lam, y, color=col, lw=1.4, marker=mk, ms=3.5, markeredgecolor="white", markeredgewidth=0.4)
        ax.annotate(lab, (lam[-2], y[-2]), xytext=(-2, 5), textcoords="offset points", ha="right", fontsize=6.2, color=INK)
    ax.set_xlabel("λ (0: product t-norm, 1: hop count)"); ax.set_ylabel("Spearman ρ with the criterion")
    ax.set_title("(b)", fontsize=7.5, loc="left"); ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.5); save(fig, "figure3")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); fig1(); fig2(); fig3(); print("figures written")
