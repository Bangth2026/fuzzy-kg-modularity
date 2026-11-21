"""Fig. 4 of the KBS manuscript (version 4.6.0): from a learner log to a decision.

The audit of Algorithm 2 as a pipeline; D1 to D9 refer to the rows of the decision guide
(Table 7).  Drawn for full-width (190 mm) placement with text of at least 7 pt."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8})
W, H = 7.48, 3.9
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
INK = "#1f2933"; BLUE = "#2f6b9a"; ORANGE = "#b8651b"; GREEN = "#3a8a5c"; GREY = "#52606d"; RED = "#9b2c2c"
FILL = {"w": "white", "b": "#eef4fa", "o": "#fbf0e6", "g": "#ecf6ef", "s": "#f3f5f8", "r": "#fbeeee"}


def box(x, y, w, h, text, edge=INK, fill="w", fs=7.6, lw=1.1, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc=FILL[fill], ec=edge, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.22,
            weight="bold" if bold else "normal")


def arrow(p, q, c=GREY, label=None, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=c, lw=1.0, shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=7, color=c, ha="center", va="center")


# top row: from the log to the described graph
box(1, 42, 15, 8, "learner log\n(attempts in order)", fill="s", edge=GREY)
box(20, 42, 22, 8, "construction (Section 4)\nco-practice or transition;\nsupport, pruning $\\tau_E$")
box(46, 42, 18, 8, "fuzzy prerequisite\ngraph $G_F=(C,E,\\mu_E)$")
box(68, 42, 31, 8, "step 1: describe $G_F$ and its cuts:\ndensity, reciprocity, out-degree,\nlevel where the cuts disconnect")
arrow((16.3, 46), (19.7, 46)); arrow((42.3, 46), (45.7, 46)); arrow((64.3, 46), (67.7, 46))

ax.text(1, 38.6, "Partitioning", ha="left", fontsize=9.5, weight="bold", color=BLUE)
ax.text(52, 38.6, "Path inference", ha="left", fontsize=9.5, weight="bold", color=GREEN)
ax.plot([50, 50], [1, 37], color="#c9d6e3", lw=1)
arrow((55, 41.6), (20, 36.3), c=BLUE); arrow((80, 41.6), (70, 36.3), c=GREEN)

# partitioning lane
box(1, 27, 22, 9, "step 2: Algorithm 1\nstructure? memberships\nbeyond topology (power $r^*$)?", edge=BLUE, fill="b")
box(27, 28.5, 21, 6, "D1: no structure detected\n→ do not partition", edge=RED, fill="r")
arrow((23.3, 31.5), (26.7, 31.5), c=RED)
box(1, 14.5, 22, 9.5, "step 3: $Q_F$ drop of each\ncut (rank-coded if the\nunit is arbitrary) vs. the\nsupport baseline", edge=BLUE, fill="b")
arrow((12, 26.7), (12, 24.3), c=BLUE, label="structure", lx=4.2)
box(27, 14.5, 21, 9.5, "D2–D5: keep $\\Pi_F$, or a\ncut within 2 seed SD\nand 0.01 of the baseline", edge=BLUE, fill="w", fs=7.4)
arrow((23.3, 19.2), (26.7, 19.2), c=BLUE)
box(1, 2, 47, 9, "step 4: if a grouping fixed independently\nof the graph exists, compare $\\Pi_F$ and the\nunflagged cuts with it, against permuted\nmemberships", edge=GREY, fill="s", fs=7.4)
arrow((12, 14.1), (12, 11.3), c=GREY)

# path lane
box(52, 27, 22, 9, "step 5: best paths by\nDijkstra on the generator;\ndirect edge, hop distance,\na model, degree heuristic", edge=GREEN, fill="g")
box(78, 27, 21, 9, "D6: cuts connected\nup to a high α?\nthen not max-min", edge=RED, fill="r")
arrow((74.3, 31.5), (77.7, 31.5), c=RED)
box(52, 14.5, 22, 9.5, "step 6: an operator\nmust beat hop distance\n(cluster bootstrap);\nthreshold cross-fitted", edge=GREEN, fill="g")
arrow((63, 26.7), (63, 24.3), c=GREEN)
box(78, 14.5, 21, 9.5, "D7–D8: hop count,\nproduct unpruned, or\na model (check degree)\nwhere no path reaches", edge=GREEN, fill="w", fs=7.4)
arrow((74.3, 19.2), (77.7, 19.2), c=GREEN)
box(52, 2, 47, 9, "step 7: before next-step use, compare\npath scores with a bigram of the logs\n(D9: the graph explains and checks;\na sequence model recommends)", edge=ORANGE, fill="o", fs=7.4)
arrow((63, 14.1), (63, 11.3), c=ORANGE)

fig.savefig("docs/figure4.png", dpi=600)
fig.savefig("docs/figure4.pdf")
