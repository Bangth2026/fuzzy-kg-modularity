# -*- coding: utf-8 -*-
"""ms_v2.md -> ms_v3.md: fill the four [CẦN CHẠY] placeholders with the real-graph runs."""
import json, numpy as np
s = open("ms_v2.md", encoding="utf-8").read()
NM = json.load(open("/home/claude/fuzzy-kg-modularity/results/null_model.json"))
BR = json.load(open("/home/claude/fuzzy-kg-modularity/results/binarisation_real.json"))
TN = json.load(open("/home/claude/fuzzy-kg-modularity/results/tnorm_paths_assist2009_item.json"))
ST = json.load(open("/home/claude/fuzzy-kg-modularity/results/graph_stats.json"))

def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)

NAMES = {"mooccubex_cs_edges.csv": "MOOCCubeX-CS", "assist2009_kc_edges.csv": "2009, KC level",
         "assist2009_item_edges.csv": "2009, item level", "assist2015_kc_edges.csv": "2015, KC level",
         "assist2017_kc_edges.csv": "2017, KC level", "assist2017_item_edges.csv": "2017, item level"}

# ---------------------------------------------------------------- Algorithm 1 steps
rep("""    8:  Q_obs ← best Q_F over S at γ1
    9:  for i = 1..N: rewire G_F preserving degrees, permute the memberships
            over the edges, run Fuzzy Leiden at γ1 over S, record the best Q_F
    10: z ← (Q_obs − mean of the N null values) / their standard deviation
    11: if z < z_min: return "no structure detected"
    Output: resolutions (γ1, γ2) and the partition at γ1, or "no structure detected\"""",
"""    8:  Q_obs ← best Q_F over S at the reference resolution γ_ref (default 1)
    9:  for i = 1..N: rewire G_F preserving in- and out-degrees, permute the
            memberships over the edges, run Fuzzy Leiden at γ_ref over S,
            record the best Q_F
    10: ΔQ ← Q_obs − mean of the N null values;  z ← ΔQ / their standard deviation
    11: if z < z_min or ΔQ < ΔQ_min: return "no structure detected"
    Output: resolutions (γ1, γ2) and the partition at γ1, or "no structure detected\"""")
rep("""            seeds S; grid factor g = 2; stability threshold θ_ARI = 0.8;
            null draws N (default 20); separation z_min (default 3)""",
"""            seeds S; grid factor g = 2; stability threshold θ_ARI = 0.8;
            null draws N (default 20); separation z_min (default 3);
            effect floor ΔQ_min (default 0.10); reference resolution γ_ref (default 1)""")
rep("""The loop of steps 4 to 7 is capped at twelve doublings in the released
implementation; no real graph in Section 9 reached the cap.""",
"""The null test is run at a reference resolution rather than at the
accepted γ₁, because on a graph without structure γ₁ ends far above 1,
where both the observed and the null partitions are near-singleton and
a z-score compares two degenerate numbers; and it carries an
effect-size floor, because on dense graphs the null standard deviation
is of order 10\\^{−3}, so a z-score alone certifies differences of a few
thousandths of Q_F (Table 7). The loop of steps 4 to 7 is capped at
twelve doublings in the released implementation; no real graph in
Section 9 reached the cap.""")

# ---------------------------------------------------------------- 9.1 null paragraph + Table 7
nm = {x["graph"]: x for x in NM}
def row(name, x, label=None):
    n = x["null"]; z = n["z"]; zs = "\\> 10²" if z > 100 else f"{z:.1f}"
    return (f"  {label or name:<26s} {x['gamma1']:<12.3f} {x['k_accepted']:<12d} {x['Q_F_accepted']:<10.3f} "
            f"{n['Q_F_obs']:<9.3f} {n['null_mean']:<9.3f} {n['delta_Q']:<7.3f} {zs:<8s} {x['outcome']}\n\n")
t7 = ("  " + "-" * 112 + "\n"
      "  Graph                      γ₁ accepted  Communities  Q_F at γ₁  Q_F(1)    null(1)   ΔQ      z        Outcome\n"
      "  -------------------------- ------------ ------------ ---------- --------- --------- ------- -------- ---------------------\n")
t7 += row("", nm["planted, p_in=0.8"], "Planted, p_in = 0.8")
t7 += row("", nm["planted, p_in=1/k (no structure)"], "Planted, p_in = 1/12")
t7 += row("", nm["dense co-practice-like, n=110, 50 edges/node"], "Dense co-practice-like")
for k, v in NAMES.items():
    t7 += row("", nm[k], v)
t7 = t7.rstrip("\n") + "\n  " + "-" * 112 + "\n"
old_t7 = s[s.index("**Table 7.** The revised Algorithm 1 on planted and synthetic graphs"):s.index("**Table 5.** Seeding steps of Algorithm 1")]
new_t7 = """**Table 7.** The revised Algorithm 1 on planted and synthetic graphs
and on the six real graphs (ρ₁ = 1.25, ρ₂ = 2, five seeds; null model:
20 degree-preserving rewirings with memberships permuted, three seeds
each, at γ_ref = 1; z_min = 3, ΔQ_min = 0.10). Q_F(1) and null(1) are
the best observed and the mean null Q_F at γ = 1. The real-graph rows
were produced by the repository pipeline (see the note in Data
availability).

""" + t7 + "\n\n"
s = s.replace(old_t7, new_t7)

rep("""on a dense co-practice-like
graph of 110 nodes and 50 edges per node it accepts a 109-community
partition with Q_F = −0.041 and the null test again returns "no
structure detected" (z = −0.1). The same test on the six real graphs is
run by the released script; its outcome on the KC-level graphs decides
whether the co-practice reading below stands. \\[CẦN CHẠY: `python
scripts/null_model_check.py data/graphs/*_edges.csv` trên sáu đồ thị
thật và điền z, outcome vào Bảng 7 trước khi nộp.\\]""",
"""on a dense co-practice-like
graph of 110 nodes and 50 edges per node it accepts a 109-community
partition with Q_F = −0.041 and the null test again returns "no
structure detected" (ΔQ = 0.001). On the six real graphs the test
separates the two granularities cleanly. The item-level graphs carry
structure the null model cannot produce: ΔQ = 0.71 (2009) and 0.48
(2017) at γ = 1. The three KC-level graphs do not: their excess over
the null is 0.056, 0.031 and 0.008, statistically separated (the null
standard deviation is 10\\^{−3}) but below any useful floor, so the
outcome is "no structure detected" and the 74- to 87-community
partitions of Table 5 are not communities. The surprise is
MOOCCubeX-CS. Its expert-curated graph has Q_F = 0.524 at γ = 1, and a
degree-preserving null with the same memberships has 0.527 ± 0.018:
the modularity of this graph is what any 59-node directed graph with
91 edges and these degrees has, and Tables 2 and 10 measure the
optimiser on it, not community structure in it. The earlier version
treated that graph as the curated reference; it is a sparse near-tree
on which modularity is uninformative, which Section 9.2 already
identified as the unstable regime.""")

# ---------------------------------------------------------------- components finding
rep("""Whether the 28 communities of the 2009 item-level
graph at every resolution are prerequisite structure or the
skill-builder problem sets the items belong to is a question we could
not settle from Q_F alone; the released code now reports the weakly
connected components and the problem-set partition of every graph next
to its partitions, and Section 10.3 lists the comparison as a
limitation until it is run against a reference partition. The
co-practice reading of the KC-level graphs is the following:""",
"""The 28 communities of the 2009
item-level graph at every resolution are its 28 weakly connected
components: the released code reports the component structure of
every graph (MOOCCubeX-CS 2 components; 2009 KC level 2, of sizes 108
and 2; 2009 item level 28, the largest of 44 nodes; 2015 and 2017 KC
level 1; 2017 item level 7, the largest of 849 nodes), and the
Fuzzy-Leiden partition of the 2009 item-level graph coincides with the
components (NMI 1.00), as does the two-community split of the 2009
KC-level graph at γ = 0.5. Items of one skill-builder problem set
co-occur and items of different sets do not, so Q_F ≈ 0.94 on that
graph reflects the construction, and Fuzzy Leiden recovers components.
The 2017 item-level graph is different: 849 of its 867 nodes lie in
one component and the 16-community partition at γ = 0.5 has NMI 0.16
with the components, so there the partition is not a component
labelling. The co-practice reading of the KC-level graphs is the
following:""")

# ---------------------------------------------------------------- Table 8 refit
br = {x["graph"]: x for x in BR}
t8 = ("  " + "-" * 118 + "\n"
      "  Graph              Fuzzy (Q_F; γ; k)   α = 0.2                 α = 0.4                 α = 0.6                 α = 0.8\n"
      "  ------------------ ------------------- ----------------------- ----------------------- ----------------------- -----------------------\n")
for k, v in NAMES.items():
    x = br[k]; f = x["fuzzy"]
    line = f"  {v:<18s} {f['Q_F']:.3f}; {f['gamma']:g}; {f['k']:<8d} "
    for a in ("alpha=0.2", "alpha=0.4", "alpha=0.6", "alpha=0.8"):
        c = x["cuts"][a]
        line += ("---".ljust(24) if c is None else
                 f"{c['loss']:.3f} / {c['NMI_vs_fuzzy']:.2f} / {c['mass_kept']:.2f} / {c['k']}".ljust(24))
    t8 += line.rstrip() + "\n\n"
t8 = t8.rstrip("\n") + "\n  " + "-" * 118 + "\n"
old_t8 = s[s.index("Table 8 repeats the shortcut on the six real graphs."):s.index("The real graphs tell the same story in retained mass.")]
new_t8 = """Table 8 repeats the shortcut on the six real graphs with the same
protocol: Fuzzy Leiden and crisp Leiden on each level set are scanned
over γ′ ∈ {0.25, 0.5, 1, 2, 4} and five seeds, the partition with the
best Q_F on the fuzzy graph is kept, and each cut reports the loss, the
NMI between Π_α and Π_fuzzy, the retained mass and the number of
communities. The earlier version evaluated the shortcut at the single
Algorithm-1 resolution of Table 5, which on the 2009 item-level graph
at α = 0.6 produced a loss of 0.93; refitting γ reduces that figure to
0.35, so about two thirds of the earlier number was a resolution
artefact, as the referee anticipated, and the remaining third is the
loss proper.

**Table 8.** The binarise-then-partition shortcut on the six real
graphs, resolution refitted per level set. "Fuzzy" gives Q_F, the
refitted γ and the number of communities of Fuzzy Leiden. Each α cell
gives loss(α) / NMI(Π_α, Π_fuzzy) / share of fuzzy mass retained /
communities. A dash means no edge survives the cut.

""" + t8 + "\n"
s = s.replace(old_t8, new_t8)
rep("""The real graphs tell the same story in retained mass. At α = 0.2 the
level set is the whole support graph, since the construction prunes at
τ_E = 0.2, and the crisp partition is within 0.003 of the fuzzy optimum
on every graph; at α = 0.4 it is within 0.004 on all six. At α = 0.6
the 2009 item-level level set keeps 52 of 4950 edges, crisp Leiden
reports Q_α = 0.985 on it, and the same partition scores 0.035 on the
fuzzy graph; part of that 0.93 is a resolution artefact of reusing the
fuzzy-graph γ on a level set of very different density, which the
refitted rows will separate from the loss proper. On MOOCCubeX-CS,
where the mean membership is 0.934, every cut up to α = 0.8 keeps at
least 86 of 91 edges and the loss never exceeds 0.010, while on the
ASSISTments graphs, with mean memberships 0.42--0.47, no edge survives
α = 0.8 at all; these are statements about the mean, not about the
spread.""",
"""The real graphs tell the same story in retained mass. At α = 0.2 the
level set is the whole support graph, since the construction prunes at
τ_E = 0.2, and at α = 0.4 the cut still keeps 73--90% of the mass: the
loss is at most 0.004 on every graph and the item-level partitions
agree with the fuzzy ones at NMI 0.98--1.00. At α = 0.6 the cut keeps
3--25% of the mass and the loss is 0.35 on the 2009 item-level graph,
0.08 on the 2017 item-level graph and 0.03--0.07 on the KC-level
graphs, with NMI against the fuzzy partition falling to 0.07--0.30 on
the KC-level graphs, where the fuzzy partition is itself not
structure (Table 7). On MOOCCubeX-CS every cut up to α = 0.8 keeps at
least 97% of the mass and the loss never exceeds 0.016, while on the
ASSISTments graphs no edge survives α = 0.8; these are statements about
the mean membership, which decides how much mass an absolute level
keeps, not about the spread.""")

# ---------------------------------------------------------------- Table 11 tau-b
tn = TN["per_source_agreement_unbounded"]; tn4 = TN["per_source_agreement_4hop"]
rep("""pairs within four hops (earlier computation; the released script
recomputes the table without the hop limit and with per-source Kendall
τ-b and bootstrap intervals). \\[CẦN CHẠY: `python
scripts/exact_tnorm_paths.py data/graphs/assist2009_item_edges.csv` và
cập nhật cột ρ bằng τ-b theo nguồn kèm khoảng tin cậy bootstrap.\\]""",
"""pairs within four hops. The pooled Spearman ρ of the earlier version is
kept for continuity; Table 11b adds the per-source Kendall τ-b with
95% bootstrap intervals over sources, computed with the exact
shortest-path solutions of Section 7.3 by the repository pipeline (see
Data availability).""")
rows11 = [("product", "product"), ("Łukasiewicz", "lukasiewicz"), ("Yager p = 1.5", "yager_p1.5"),
          ("Yager p = 2", "yager_p2.0"), ("Yager p = 3", "yager_p3.0"), ("Yager p = 5", "yager_p5.0"),
          ("Yager p = 10", "yager_p10.0"), ("Schweizer--Sklar p = −8", "schweizer_sklar_p-8.0"),
          ("Schweizer--Sklar p = −4", "schweizer_sklar_p-4.0"), ("Schweizer--Sklar p = −2", "schweizer_sklar_p-2.0"),
          ("Schweizer--Sklar p = −1", "schweizer_sklar_p-1.0"), ("Schweizer--Sklar p = 0.5", "schweizer_sklar_p0.5")]
t11 = ("  " + "-" * 96 + "\n"
       "  Operator                   τ-b, 4 hops   95% CI            τ-b, no hop limit  95% CI            pooled ρ, no limit\n"
       "  -------------------------- ------------- ----------------- ------------------ ----------------- ------------------\n")
for lab, k in rows11:
    a, b = tn4[k], tn[k]
    t11 += (f"  {lab:<26s} {a['kendall_tau_b_mean']:<13.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]    "
            f"{b['kendall_tau_b_mean']:<18.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]    {b['pooled_spearman']:.3f}\n\n")
t11 = t11.rstrip("\n") + "\n  " + "-" * 96 + "\n"
n_src = tn["product"]["n_sources"]; n_pairs = tn["product"]["n_pairs_pooled"]
rep("""The bound of Fact 1 was checked directly rather than assumed:""",
f"""**Table 11b.** Per-source rank agreement with max-min on the same
graph: Kendall τ-b between the ordering of reachable targets by exact
best-path strength under each operator and under the minimum, mean
over {n_src} sources with at least three reachable targets (of 60
sampled), with 95% bootstrap intervals over sources; with the four-hop
limit of Table 11 and without it ({n_pairs} source--target pairs pooled).

""" + t11 + """

The bound of Fact 1 was checked directly rather than assumed:""")
rep("""First, the operator choice is consequential on this data. Pooled rank
agreement ranges from 0.529 (Łukasiewicz, equivalently Yager p = 1 and
Schweizer--Sklar p = 1) to 0.952 (Yager p = 10). Two cautions apply to
that statistic. It pools pairs across sources and hop lengths, so it is
partly a hop-length and tie effect: Łukasiewicz returns 0 for most 3-
and 4-hop pairs, and the ordering Equation (16) consumes is within a
source, so the per-source Kendall τ-b the released script now reports
is the relevant quantity.""",
f"""First, the operator choice is more consequential than the pooled
statistic suggested. Pooled rank agreement ranges from 0.529
(Łukasiewicz, equivalently Yager p = 1 and Schweizer--Sklar p = 1) to
0.952 (Yager p = 10), but that statistic pools pairs across sources
and hop lengths and is partly a hop-length and tie effect, since
Łukasiewicz returns 0 for most 3- and 4-hop pairs, whereas the
ordering Equation (16) consumes is within a source. The per-source
Kendall τ-b of Table 11b is the relevant quantity, and it is far
lower: {tn['lukasiewicz']['kendall_tau_b_mean']:.2f} for Łukasiewicz,
{tn['product']['kendall_tau_b_mean']:.2f} for the product, and at most
{tn['yager_p10.0']['kendall_tau_b_mean']:.2f} (Yager p = 10) at the near-min end, with bootstrap intervals
of width about 0.1. Within a source, every Archimedean operator
reorders the candidates substantially on this graph; the hop limit
changes the figures by at most 0.03, so the earlier four-hop
truncation did not bias them.""")

# ---------------------------------------------------------------- data availability note
rep("""and compares the printed values listed in the supplement against
its own output, is maintained at
https://github.com/Bangth2026/fuzzy-kg-modularity.""",
"""and compares the printed values listed in the supplement against
its own output, is maintained at
https://github.com/Bangth2026/fuzzy-kg-modularity. \\[CẦN ĐỐI CHIẾU
TRƯỚC KHI NỘP: pipeline trong repo dựng lại sáu đồ thị với cùng số
nút và số cạnh sai khác dưới 0.3% so với Bảng 3 (MOOCCubeX-CS trùng
hoàn toàn), nhưng trung bình μ_E cao hơn Bảng 3 từ 0.01 đến 0.05 (ví
dụ 2009 item: 0.470 so với 0.421) và bước spectral bisection cho c/W
khác Bảng 5; nghĩa là `16_run_experiments.py` dùng cho Bảng 2--5, 11
và mã trong repo chưa đồng nhất về định nghĩa mastery gain/OWA và về
bisection. Bảng 7 (sáu dòng thật), Bảng 8 và Bảng 11b trong bản này
tính bằng pipeline repo. Cần chọn một pipeline làm chuẩn và tính lại
các bảng còn lại bằng nó.\\]""")

assert s.count("CẦN CHẠY") == 0
open("ms_v3.md", "w", encoding="utf-8").write(s)
print("ok; markers left:", s.count("CẦN"))
