"""Fig. 1 of the KBS manuscript, redrawn for a full-width (190 mm) placement with text >= 7 pt."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8})
W, H = 7.48, 3.74
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
INK = "#1f2933"; BLUE = "#2f6b9a"; ORANGE = "#b8651b"; GREEN = "#3a8a5c"; GREY = "#52606d"
FILL = {"w": "white", "b": "#eef4fa", "o": "#fbf0e6", "g": "#ecf6ef", "s": "#f3f5f8"}

def box(x, y, w, h, text, edge=INK, fill="w", fs=8, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc=FILL[fill], ec=edge, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)

def arrow(p, q, c=GREY, both=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=8, color=c, lw=1.0, shrinkA=0, shrinkB=0))

ax.text(25, 48, "Binarising before partitioning", ha="center", va="center", fontsize=10, weight="bold", color=INK)
ax.text(76, 48, "Composing memberships along paths", ha="center", va="center", fontsize=10, weight="bold", color=INK)
ax.plot([51.2, 51.2], [2, 46], color="#c9d6e3", lw=1)

# left panel
box(1, 35, 14, 8.5, "fuzzy graph\n$G_F=(C,E,\\mu_E)$")
for i, lab in enumerate(["$E_{\\alpha_1}$", "$E_{\\alpha_2}$", "…", "$E_{\\alpha_L}$"]):
    box(19 + i * 7.6, 36 - i * 0.6, 6.4, 6.5, lab, edge=BLUE, fill="b")
arrow((15.3, 39.3), (18.6, 39.3))
ax.text(29, 32.6, "nested α-cuts (quantile grid)", ha="center", va="center", fontsize=8, color=GREY)
box(1, 20, 13, 8.5, "$\\Pi_F$: Leiden\non $Q_F$")
box(17.5, 20, 15.5, 8.5, "$\\Pi^*$: multislice\nreference $\\bar{Q}$, one\nshared partition", edge=BLUE, fill="b")
box(35, 20, 15.5, 8.5, "$\\Pi_{\\alpha_0}$: crisp\nLeiden on one cut", edge=ORANGE, fill="o")
ax.text(12.7, 30.0, "Lemma 2", fontsize=8, color=GREY)
arrow((8, 34.6), (8, 28.9), c=GREY)
arrow((14.2, 24.2), (17.3, 24.2), both=True)
arrow((26, 31.6), (25.2, 28.9), c=BLUE)
arrow((45, 35.2), (45, 28.9), c=ORANGE)
box(3, 8.4, 45, 9.2, "audit (default): $Q_F(\\Pi_F)-Q_F(\\Pi_{\\alpha_0})$\nunit-invariant: $Q_F$ drop on rank-coded $\\mu_E$ (Lemma 2)\ncost index $\\bar{Q}(\\Pi^*)-\\bar{Q}(\\Pi_{\\alpha_0})$; vs. shortfall")
arrow((7.5, 19.6), (12, 17.8), c=INK); arrow((25.2, 19.6), (25.2, 17.8), c=BLUE); arrow((42.7, 19.6), (38, 17.8), c=ORANGE)
box(1, 0.8, 49, 6.6, "Algorithm 1: degree-preserving null with a density-matched\nfloor; weights-only nulls (global, row) with calibrated power;\nchecked by a weighted stochastic block model", edge=GREY, fill="s", fs=7.6)

# right panel
box(53, 35, 20.5, 8.5, "path score $S_T(\\pi)$\n$T$ = min or Archimedean")
box(76.5, 35, 22.8, 8.5, "best path = shortest path\nthrough the generator $g$;\nmax-min ordinally invariant")
arrow((73.8, 39.3), (76.2, 39.3))
box(53, 16, 21, 13, "prerequisite criteria\nJunyi: expert scores,\nknowledge-map links;\nbaseline: hop distance\non the pruned support", edge=GREEN, fill="g")
box(77.5, 16, 21.5, 13, "next-concept criterion\nheld-out learners;\nbaselines: direct edge,\nhop distance, bigram", edge=ORANGE, fill="o")
arrow((63.5, 34.6), (63.5, 29.4), c=GREEN); arrow((88.2, 34.6), (88.2, 29.4), c=ORANGE)
box(53, 0.8, 46, 11, "both constructions of Section 4 on every log:\nco-practice (dense, reciprocal) vs.\norder-preserving transition (sparse)\nAlgorithm 2: both halves as a pre-deployment audit", edge=GREY, fill="s")
arrow((63.5, 15.6), (70, 11.8)); arrow((88.2, 15.6), (82, 11.8))

fig.savefig("docs/figure1.png", dpi=600)
fig.savefig("docs/figure1.pdf")
