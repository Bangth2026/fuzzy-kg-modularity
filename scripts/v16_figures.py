#!/usr/bin/env python3
"""Release 5.1.0. Figures of the revised manuscript (docs/v51_figure*.{png,pdf}).
Fig. 1: design (adds the graded-partition check); Fig. 2: as release 5.0.0 with panel (b) in
balanced accuracy; Fig. 3: unchanged (scripts/v15_figures.py); Fig. 4 (new): value of the grades of the
Junyi transition graph against the pruning threshold tau_E."""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import v15_figures as F  # noqa: E402
from v15_figures import plt, FancyBboxPatch, BLUE, ORANGE, AQUA, YELLOW, INK, INK2, GRID  # noqa: E402

S16 = json.load(open("results/v16/summary.json"))


def save(fig, name):
    fig.savefig(f"docs/v51_{name}.png", dpi=400, bbox_inches="tight"); fig.savefig(f"docs/v51_{name}.pdf", bbox_inches="tight"); plt.close(fig)


def fig1():
    fig, ax = plt.subplots(figsize=(7.48, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc="white", ec=col, lw=1.1))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=5.9, color=INK2, linespacing=1.35)
    box(0.5, 12, 17.5, 22, "Learner log", "two constructions:\nco-practice,\ntransition (pruned\nat τ_E: a first cut)\n→ G_F = (C, E, μ_E)", INK2)
    box(23, 25, 24, 13, "Value of grades (Sec. 3)", "V = S(G_F) − S(support)\n= V_marg + V_str + V_res\nedge · path · partition", BLUE)
    box(23, 2, 24, 20, "Robust cut audit (Sec. 4)", "Q_F[φ_s∘μ](Π) in closed form\n(Prop. 1); no unit-free verdict\n(Prop. 2, Cor. 1); κ*: largest\nslope ratio keeping the decision;\ndivergence check of Π_F (4.5)", ORANGE)
    box(52, 25, 22, 13, "Paths (Sec. 5)", "max-min reads only μ ≥ α_c\n(Prop. 3, usable share u)\nλ + (1 − λ)(−ln μ): λ̂", AQUA)
    box(52, 2, 22, 20, "Validation (Sec. 7, 8)", "192 semi-synthetic graphs,\nalso with grades that mark\nthe communities;\n23 educational graphs,\n10 with a grouping; PPI5k, CN15k", YELLOW)
    box(79, 12, 20, 22, "Algorithm 2", "per use: keep grades,\ndrop them, or keep\nthem unmodified;\nper cut: cut / flagged\n/ undecided; path score", INK2)
    for (x0, y0, x1, y1) in ((18.6, 26, 22.4, 31), (18.6, 20, 22.4, 12), (47.6, 31, 51.4, 31), (47.6, 12, 51.4, 12), (74.6, 31, 78.4, 26), (74.6, 12, 78.4, 20)):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    save(fig, "figure1")


def fig2():
    S = F.S; ss = S16["semisynthetic_v15"]
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
    vals = [(ss["transition"]["firm_mu"]["ba"], ss["transition"]["fragile_mu"]["ba"]), (ss["co-practice"]["firm_mu"]["ba"], ss["co-practice"]["fragile_mu"]["ba"])]
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, [v[0] for v in vals], w, color=[BLUE, ORANGE], label="κ* ≥ 2")
    b2 = ax.bar(x + w / 2 + 0.01, [v[1] for v in vals], w, color="white", edgecolor=[BLUE, ORANGE], lw=0.9, hatch="////", label="κ* < 2")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center", fontsize=6.2, color=INK)
    ax.axhline(0.5, color=INK2, lw=0.5, ls=(0, (3, 2)))
    ax.set_xticks(x, ["transition", "co-practice"])
    ax.set_ylabel("balanced accuracy (decision on μ_E)"); ax.set_title("(b)", fontsize=7.5, loc="left"); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(fontsize=6.2, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.2)
    fig.tight_layout(w_pad=1.2); save(fig, "figure2")


def fig4():
    P = json.load(open("results/v16/junyi_pruning_value.json"))["taus"]
    tags = ["0", "0.05", "0.1", "0.2", "0.3"]; x = np.arange(len(tags))
    fig, axs = plt.subplots(1, 2, figsize=(7.48, 2.3))
    for ax, lev, title in ((axs[0], "edge", "(a) edge level"), (axs[1], "path", "(b) path level")):
        for k, col, mk, lab, off in (("V_total", INK, "o", "V (total)", -0.12), ("V_resid", BLUE, "s", "V_res", 0.0), ("V_marg", ORANGE, "v", "V_marg", 0.12)):
            y = np.array([P[t]["value"][lev][k] for t in tags]); lo = np.array([P[t]["value"][lev][k + "_ci"][0] for t in tags]); hi = np.array([P[t]["value"][lev][k + "_ci"][1] for t in tags])
            ax.errorbar(x + off, y, yerr=[y - lo, hi - y], color=col, marker=mk, ms=3.5, lw=1.1, capsize=1.5, label=lab)
        ax.axhline(0, color=INK2, lw=0.5); ax.set_xticks(x, ["none", "0.05", "0.1", "0.2\n(default)", "0.3"])
        ax.set_xlabel("pruning threshold τ_E of the construction"); ax.set_title(title, fontsize=7.5, loc="left")
        ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    axs[0].set_ylabel("value of grades (Spearman ρ)"); axs[0].legend(fontsize=6.2, loc="upper right")
    fig.tight_layout(w_pad=1.5); save(fig, "figure4")


if __name__ == "__main__":
    fig1(); fig2(); F.fig3(); fig4()
    import shutil
    for ext in ("png", "pdf"):
        shutil.copy(f"docs/v5_figure3.{ext}", f"docs/v51_figure3.{ext}")
    print("figures written")
